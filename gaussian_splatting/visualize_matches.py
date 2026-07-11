import argparse
import logging
import random
import cv2
import h5py
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_all_pairs(matches_path: Path) -> list[tuple[str, str]]:
    pairs = []
    with h5py.File(matches_path, 'r') as f:
        for img0 in f.keys():
            for img1 in f[img0].keys():
                pairs.append((img0, img1))
    return pairs


def load_keypoints(features_path: Path, image_name: str) -> np.ndarray:
    with h5py.File(features_path, 'r') as f:
        if image_name not in f:
            raise KeyError(f"Image {image_name} not found in {features_path}")
        return f[image_name]['keypoints'][:]


def load_matches(matches_path: Path, img0_name: str, img1_name: str) -> tuple[np.ndarray, bool]:
    with h5py.File(matches_path, 'r') as f:
        # Check standard order
        if img0_name in f and img1_name in f[img0_name]:
            return f[img0_name][img1_name]['matches0'][:], False
        # Check swapped order
        if img1_name in f and img0_name in f[img1_name]:
            return f[img1_name][img0_name]['matches0'][:], True

    raise KeyError(f"Pair {img0_name} and {img1_name} not found in {matches_path}")


def visualize_pair(
        image_dir: Path,
        features_path: Path,
        matches_path: Path,
        img0_name: str,
        img1_name: str,
        output_path: Path,
        max_matches: int = 50
) -> None:
    logger.info("=" * 60)
    logger.info(f"DIAGNOSTICS FOR PAIR: {img0_name} <-> {img1_name}")
    logger.info("=" * 60)

    # 1. Load Images & Verify Shapes
    img0_path = image_dir / img0_name
    img1_path = image_dir / img1_name

    img0 = cv2.imread(str(img0_path))
    img1 = cv2.imread(str(img1_path))

    if img0 is None or img1 is None:
        raise FileNotFoundError("One or both images could not be loaded.")

    logger.info(f"[IMAGE 0] Loaded {img0_name} | Shape: {img0.shape}")
    logger.info(f"[IMAGE 1] Loaded {img1_name} | Shape: {img1.shape}")

    # 2. Load Keypoints
    kpts0 = load_keypoints(features_path, img0_name)
    kpts1 = load_keypoints(features_path, img1_name)

    logger.info(f"[KEYPOINTS 0] Loaded from H5 | Shape: {kpts0.shape}")
    logger.info(f"[KEYPOINTS 1] Loaded from H5 | Shape: {kpts1.shape}")

    # 3. Load Matches & Track Swap Status
    matches, swapped = load_matches(matches_path, img0_name, img1_name)
    logger.info(f"[MATCHES] Loaded matches0 array | Shape: {matches.shape} | H5 Swap Status: {swapped}")

    cv_kpts0 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in kpts0]
    cv_kpts1 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in kpts1]

    cv_matches = []

    # ------------------------------------------------------------------------
    # INDEXING LOGIC
    # If not swapped: matches[i] = j -> kpts0[i] matches kpts1[j]
    # If swapped: matches[j] = i -> kpts1[j] matches kpts0[i]
    # ------------------------------------------------------------------------
    if not swapped:
        for i, j in enumerate(matches):
            if j > -1:
                cv_matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=0))
    else:
        for j, i in enumerate(matches):
            if i > -1:
                cv_matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=0))

    logger.info(f"[MAPPING] Total valid matches parsed: {len(cv_matches)}")

    # 4. Explicit Sanity Check of the HDF5 Data
    if cv_matches:
        logger.info("-" * 40)
        logger.info("EXACT HDF5 COORDINATE MAPPING (First 5 matches):")
        for idx, m in enumerate(cv_matches[:5]):
            pt0 = kpts0[m.queryIdx]
            pt1 = kpts1[m.trainIdx]
            logger.info(
                f"  Match {idx}: [Img 0, Kpt {m.queryIdx}] (x:{pt0[0]:.1f}, y:{pt0[1]:.1f}) ---> [Img 1, Kpt {m.trainIdx}] (x:{pt1[0]:.1f}, y:{pt1[1]:.1f})")
        logger.info("-" * 40)

    # 5. Limit visualization for readability
    if len(cv_matches) > max_matches:
        cv_matches = random.sample(cv_matches, max_matches)
        logger.info(f"[DRAWING] Sampled down to {max_matches} matches for clean visualization.")

    matched_img = cv2.drawMatches(
        img0, cv_kpts0,
        img1, cv_kpts1,
        cv_matches,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_DEFAULT
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), matched_img)
    logger.info(f"Visualization saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize random matched keypoints.")
    parser.add_argument("--image_dir", type=Path, required=True, help="Directory containing images")
    parser.add_argument("--features", type=Path, required=True, help="Path to features.h5")
    parser.add_argument("--matches", type=Path, required=True, help="Path to matches.h5")
    parser.add_argument("--output_dir", type=Path, default=Path("matches_visualizations"))
    parser.add_argument("--num_pairs", type=int, default=5, help="Number of random pairs")

    args = parser.parse_args()

    all_pairs = get_all_pairs(args.matches)
    if not all_pairs:
        logger.error("No matches found in the HDF5 file.")
        return

    num_to_sample = min(args.num_pairs, len(all_pairs))
    selected_pairs = random.sample(all_pairs, num_to_sample)

    for idx, (img0, img1) in enumerate(selected_pairs):
        clean_img0 = img0.replace(".jpg", "").replace(".png", "")
        clean_img1 = img1.replace(".jpg", "").replace(".png", "")
        out_filename = f"match_{idx:03d}_{clean_img0}_to_{clean_img1}.jpg"

        try:
            visualize_pair(
                args.image_dir, args.features, args.matches,
                img0, img1, args.output_dir / out_filename
            )
        except Exception as e:
            logger.error(f"Failed to visualize pair {img0} & {img1}: {e}")


if __name__ == "__main__":
    main()