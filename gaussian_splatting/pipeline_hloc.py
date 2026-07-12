import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pycolmap
import torch
from transformers import AutoProcessor, GroundingDinoForObjectDetection

from hloc import extract_features, match_features, reconstruction

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_frames(video_path: Path, image_dir: Path, mask_dir: Path, fps_target: int = 5, focus_objects: bool = False,
                   text_prompt: str = "bench. backpack.") -> None:
    """
    Extract frames from a video file at a specified frames per second interval.
    Generates binary bounding-box masks for targeted objects using Grounding DINO.
    """
    logger.info(f"Extracting frames from {video_path} to {image_dir} at {fps_target} FPS")
    image_dir.mkdir(parents=True, exist_ok=True)

    if focus_objects:
        mask_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video <= 0:
        fps_video = 30.0

    frame_interval = max(1, int(fps_video / fps_target))

    frame_idx = 0
    saved_idx = 0

    processor_dino = None
    model_dino = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if focus_objects:
        logger.info(f"Loading Grounding DINO on {device} for bbox mask generation...")
        processor_dino = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        model_dino = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(device)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_name = f"frame_{saved_idx:05d}.jpg"
            mask_name = f"frame_{saved_idx:05d}.png"  # Matches image name for downstream linking

            if focus_objects and model_dino is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                inputs_dino = processor_dino(images=rgb_frame, text=text_prompt, return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs_dino = model_dino(**inputs_dino)

                h, w = frame.shape[:2]

                results_dino = processor_dino.post_process_grounded_object_detection(
                    outputs_dino,
                    inputs_dino.input_ids,
                    threshold=0.3,
                    text_threshold=0.3,
                    target_sizes=[(h, w)]
                )[0]

                boxes = results_dino["boxes"]

                if len(boxes) > 0:
                    combined_mask = np.zeros((h, w), dtype=np.uint8)
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.cpu().numpy())

                        pad_x = int((x2 - x1) * 0.02)
                        pad_y = int((y2 - y1) * 0.02)

                        x1 = max(0, x1 - pad_x)
                        y1 = max(0, y1 - pad_y)
                        x2 = min(w, x2 + pad_x)
                        y2 = min(h, y2 + pad_y)

                        cv2.rectangle(combined_mask, (x1, y1), (x2, y2), 255, -1)

                    cv2.imwrite(str(mask_dir / mask_name), combined_mask)
                else:
                    logger.debug(f"Skipping frame {frame_idx}: No '{text_prompt}' detected.")
                    frame_idx += 1
                    continue

            frame_path = image_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    logger.info(f"Extracted {saved_idx} frames.")


def generate_sequential_pairs(image_dir: Path, output_path: Path, overlap: int = 50) -> None:
    images = sorted([p.name for p in image_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    pairs = []

    for i in range(len(images)):
        for j in range(i + 1, min(i + 1 + overlap, len(images))):
            pairs.append(f"{images[i]} {images[j]}\n")

    with open(output_path, 'w') as f:
        f.writelines(pairs)

    logger.info(f"Generated {len(pairs)} sequential pairs from {len(images)} images (Overlap: {overlap}).")


def run_hloc_pipeline(image_dir: Path, output_dir: Path, mask_dir: Path) -> Path:
    outputs = output_dir / "hloc_outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    sfm_pairs = outputs / "pairs-sequence.txt"
    sfm_dir = outputs / "sfm"
    features = outputs / "features.h5"
    matches = outputs / "matches.h5"

    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superpoint+lightglue']

    logger.info("Step 1/4: Extracting features with SuperPoint...")
    extract_features.main(feature_conf, image_dir, image_list=None, feature_path=features)

    logger.info("Step 2/4: Generating custom sequential pairs...")
    generate_sequential_pairs(image_dir, sfm_pairs, overlap=50)

    logger.info("Step 3/4: Matching features with LightGlue...")
    # LightGlue will naturally only match the keypoints that survived our filtering
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

    logger.info("Step 4/4: Running COLMAP reconstruction (Forcing SINGLE PINHOLE camera)...")
    reconstruction.main(
        sfm_dir=sfm_dir,
        image_dir=image_dir,
        pairs=sfm_pairs,
        features=features,
        matches=matches,
        camera_mode=pycolmap.CameraMode.SINGLE,
        image_options={'camera_model': 'PINHOLE'}
    )

    logger.info(f"Reconstruction completed successfully. Model saved to {sfm_dir}")
    return sfm_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract video frames and run HLOC to output a PINHOLE COLMAP model.")
    parser.add_argument("--video", type=Path, required=True, help="Input .mp4 video file")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for frames and COLMAP data")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second to extract from the video")
    parser.add_argument("--focus_objects", action="store_true",
                        help="Mask out everything except targets using DINO bboxes")
    parser.add_argument("--prompt", type=str, default="bench. backpack.", help="Text prompt for Grounding DINO")
    args = parser.parse_args()

    if args.prompt:
        logger.info(f"Received prompt: '{args.prompt}'")

    image_dir = args.output_dir / "images"
    mask_dir = args.output_dir / "masks"

    extract_frames(args.video, image_dir, mask_dir, args.fps, args.focus_objects, args.prompt)
    run_hloc_pipeline(image_dir, args.output_dir, mask_dir)


if __name__ == "__main__":
    main()
