import argparse
import logging
import random
import cv2
import h5py
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_all_pairs(matches_path: Path) -> list[tuple[str, str]]:
    """
    Parse the HDF5 matches file to retrieve all pairs of matched images.

    Args:
        matches_path: Path to the matches.h5 file.

    Returns:
        A list of tuples, where each tuple contains the names of a matched image pair.
    """
    pairs = []
    with h5py.File(matches_path, 'r') as f:
        for img0 in f.keys():
            for img1 in f[img0].keys():
                pairs.append((img0, img1))
    return pairs


def load_keypoints(features_path: Path, image_name: str) -> np.ndarray:
    """
    Load keypoints for a specific image from the HDF5 features file.

    Args:
        features_path: Path to the features.h5 file.
        image_name: Name of the image.

    Returns:
        An array of keypoints of shape (N, 2).

    Raises:
        KeyError: If the image name is not found in the features file.
    """
    with h5py.File(features_path, 'r') as f:
        if image_name not in f:
            raise KeyError(f"Image {image_name} not found in {features_path}")
        return f[image_name]['keypoints'][:]


def load_matches(matches_path: Path, img0_name: str, img1_name: str) -> tuple[np.ndarray, bool]:
    """
    Load matches between two images from the HDF5 matches file.

    Args:
        matches_path: Path to the matches.h5 file.
        img0_name: Name of the first image.
        img1_name: Name of the second image.

    Returns:
        A tuple containing the matches array and a boolean indicating if the
        images were swapped in the HDF5 hierarchy.

    Raises:
        KeyError: If the match pair is not found in the matches file.
    """
    with h5py.File(matches_path, 'r') as f:
        if img0_name in f and img1_name in f[img0_name]:
            return f[img0_name][img1_name]['matches0'][:], False
        if img1_name in f and img0_name in f[img1_name]:
            return f[img1_name][img0_name]['matches0'][:], True

    raise KeyError(f"Pair {img0_name} and {img1_name} not found in {matches_path}")


def visualize_pair(
        image_dir: Path,
        features_path: Path,
        matches_path: Path,
        img0_name: str,
        img1_name: str,
        output_path: Path
) -> None:
    """
    Visualize matched keypoints between two images and save the result.

    Args:
        image_dir: Directory containing the images.
        features_path: Path to the HDF5 features file.
        matches_path: Path to the HDF5 matches file.
        img0_name: Filename of the first image.
        img1_name: Filename of the second image.
        output_path: Path to save the visualization image.

    Raises:
        FileNotFoundError: If the physical image files cannot be loaded.
    """
    logger.info(f"Visualizing matches between {img0_name} and {img1_name}")

    img0_path = image_dir / img0_name
    img1_path = image_dir / img1_name

    img0 = cv2.imread(str(img0_path))
    img1 = cv2.imread(str(img1_path))

    if img0 is None or img1 is None:
        raise FileNotFoundError("One or both images could not be loaded. Check the image directory and filenames.")

    kpts0 = load_keypoints(features_path, img0_name)
    kpts1 = load_keypoints(features_path, img1_name)

    matches, swapped = load_matches(matches_path, img0_name, img1_name)

    cv_kpts0 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in kpts0]
    cv_kpts1 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in kpts1]

    cv_matches = []
    if not swapped:
        for i, j in enumerate(matches):
            if j > -1:
                cv_matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=0))
    else:
        for j, i in enumerate(matches):
            if i > -1:
                cv_matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=0))

    logger.info(f"Found {len(cv_matches)} valid matches.")

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
    """
    Main entry point for the visualization script.
    """
    parser = argparse.ArgumentParser(description="Visualize random matched keypoints between images.")
    parser.add_argument("--image_dir", type=Path, required=True, help="Directory containing the extracted images")
    parser.add_argument("--features", type=Path, required=True, help="Path to features.h5")
    parser.add_argument("--matches", type=Path, required=True, help="Path to matches.h5")
    parser.add_argument("--output_dir", type=Path, default=Path("matches_visualizations"),
                        help="Output directory for visualizations")
    parser.add_argument("--num_pairs", type=int, default=5, help="Number of random pairs to visualize")

    args = parser.parse_args()

    all_pairs = get_all_pairs(args.matches)

    if not all_pairs:
        logger.error("No matches found in the provided HDF5 file.")
        return

    num_to_sample = min(args.num_pairs, len(all_pairs))
    selected_pairs = random.sample(all_pairs, num_to_sample)

    logger.info(f"Randomly selected {num_to_sample} pairs for visualization.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (img0, img1) in enumerate(selected_pairs):
        clean_img0 = img0.replace(".jpg", "").replace(".png", "")
        clean_img1 = img1.replace(".jpg", "").replace(".png", "")
        out_filename = f"match_{idx:03d}_{clean_img0}_to_{clean_img1}.jpg"
        out_path = args.output_dir / out_filename

        try:
            visualize_pair(
                args.image_dir,
                args.features,
                args.matches,
                img0,
                img1,
                out_path
            )
        except Exception as e:
            logger.error(f"Failed to visualize pair {img0} and {img1}: {e}")


if __name__ == "__main__":
    main()