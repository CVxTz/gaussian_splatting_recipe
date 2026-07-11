import argparse
import logging
import cv2
import pycolmap
from pathlib import Path

# Removed the broken pairs_from_sequence import
from hloc import extract_features, match_features, reconstruction

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_frames(video_path: Path, image_dir: Path, fps_target: int = 5) -> None:
    """
    Extract frames from a video file at a specified frames per second interval.
    """
    logger.info(f"Extracting frames from {video_path} to {image_dir} at {fps_target} FPS")
    image_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file {video_path}")

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if fps_video <= 0:
        fps_video = 30.0

    frame_interval = max(1, int(fps_video / fps_target))

    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            frame_name = f"frame_{saved_idx:05d}.jpg"
            frame_path = image_dir / frame_name
            cv2.imwrite(str(frame_path), frame)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    logger.info(f"Extracted {saved_idx} frames.")


def generate_sequential_pairs(image_dir: Path, output_path: Path, overlap: int = 50) -> None:
    """
    Creates a pairing text file linking each video frame only to its chronological neighbors.
    This completely prevents LightGlue from hallucinating matches across unrelated frames.
    """
    images = sorted([p.name for p in image_dir.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    pairs = []

    for i in range(len(images)):
        # Match current image with the next 'overlap' images
        for j in range(i + 1, min(i + 1 + overlap, len(images))):
            pairs.append(f"{images[i]} {images[j]}\n")

    with open(output_path, 'w') as f:
        f.writelines(pairs)

    logger.info(f"Generated {len(pairs)} sequential pairs from {len(images)} images (Overlap: {overlap}).")


def run_hloc_pipeline(image_dir: Path, output_dir: Path) -> Path:
    """
    Runs the hloc reconstruction pipeline.
    Uses custom sequential matching for video frames and forces a single PINHOLE camera.
    """
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
    # THE FIX: Bypass HLOC and generate the txt file ourselves
    generate_sequential_pairs(image_dir, sfm_pairs, overlap=50)

    logger.info("Step 3/4: Matching features with LightGlue...")
    match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)

    logger.info("Step 4/4: Running COLMAP reconstruction (Forcing SINGLE PINHOLE camera)...")
    reconstruction.main(
        sfm_dir=sfm_dir,
        image_dir=image_dir,
        pairs=sfm_pairs,
        features=features,
        matches=matches,
        camera_mode=pycolmap.CameraMode.SINGLE,  # Forces 1 shared camera for all images
        image_options={'camera_model': 'PINHOLE'}  # Forces zero distortion
    )

    logger.info(f"Reconstruction completed successfully. Model saved to {sfm_dir}")
    return sfm_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract video frames and run HLOC to output a PINHOLE COLMAP model.")
    parser.add_argument("--video", type=Path, required=True, help="Input .mp4 video file")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for frames and COLMAP data")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second to extract from the video")
    args = parser.parse_args()

    image_dir = args.output_dir / "images"

    extract_frames(args.video, image_dir, args.fps)
    run_hloc_pipeline(image_dir, args.output_dir)


if __name__ == "__main__":
    main()