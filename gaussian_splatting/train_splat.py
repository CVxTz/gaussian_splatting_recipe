import argparse
import logging
import os
from pathlib import Path

import fvdb
import fvdb_reality_capture as frc
from fvdb_reality_capture.radiance_fields import (
    GaussianSplatOptimizerConfig,
    GaussianSplatReconstructionConfig,
    InsertionGrad2dThresholdMode
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
logger = logging.getLogger(__name__)


def _prepare_colmap_structure(trial_dir: Path, hloc_dir: Path, raw_images_path: Path) -> None:
    """
    Ensures sparse/0/ and images/ are present in trial_dir via symlinks.
    """
    sparse_dir = trial_dir / "sparse" / "0"
    sfm_source = hloc_dir / "sfm"

    if not sparse_dir.exists():
        sparse_dir.mkdir(parents=True, exist_ok=True)
        for bin_file in ["cameras.bin", "images.bin", "points3D.bin"]:
            src = sfm_source / bin_file
            if src.exists():
                os.symlink(src, sparse_dir / bin_file)

    # Link/Verify images
    images_dst = trial_dir / "images"
    if not images_dst.exists():
        logger.info(f"Linking raw images from {raw_images_path} to {images_dst}")
        os.symlink(raw_images_path, images_dst)


def train_and_export_splats(dataset_path: Path, output_path: Path, num_epochs: int) -> None:
    """
    Loads SfmScene, configures maximum splat growth and schedule, trains, and exports to .ply.
    """
    logger.info("Loading SfmScene...")
    sfm_scene = frc.sfm_scene.SfmScene.from_colmap(str(dataset_path))

    # Calculate exact steps based on your specific image count
    num_images = len(sfm_scene.images)
    total_steps = num_images * num_epochs
    logger.info(f"Scene loaded with {num_images} images.")
    logger.info(f"Training for {num_epochs} epochs ({total_steps} total steps).")

    logger.info("Configuring Maximum Gaussian Growth (Optimizer)...")
    optimizer_config = GaussianSplatOptimizerConfig(
        max_gaussians=1000000,
        insertion_grad_2d_threshold_mode=InsertionGrad2dThresholdMode.PERCENTILE_EVERY_ITERATION,
        insertion_grad_2d_threshold=0.95,
    )

    logger.info("Configuring Aggressive Refinement Schedule & Overrides...")
    reconstruction_config = GaussianSplatReconstructionConfig(
        max_epochs=num_epochs,
        max_steps=total_steps,
        refine_start_epoch=1,  # Start immediately
        refine_every_epoch=0.5,  # Refine twice as often
        refine_stop_epoch=int(num_epochs * 0.75),  # Stop refining 75% of the way through

        # Disable camera pose optimization
        optimize_camera_poses=False
    )

    logger.info("Initializing Gaussian Splat Reconstruction...")
    runner = frc.radiance_fields.GaussianSplatReconstruction.from_sfm_scene(
        sfm_scene,
        config=reconstruction_config,
        optimizer_config=optimizer_config
    )

    logger.info("Optimizing...")
    runner.optimize()

    # Export
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runner.model.save_ply(str(output_path), metadata=runner.reconstruction_metadata)
    logger.info(f"Export complete: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export Gaussian Splatting with aggressive growth.")
    parser.add_argument("--trial_dir", type=Path, required=True, help="Root folder (e.g., .../colmap/trial1/)")
    parser.add_argument("--images_path", type=Path, required=True, help="Path to raw frames")
    parser.add_argument("--output_splat", type=Path, default=Path("gaussian_model.ply"), help="Output .ply file")
    # Epochs argument defaults to 200
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    args = parser.parse_args()

    _prepare_colmap_structure(args.trial_dir, args.trial_dir / "hloc_outputs", args.images_path)
    train_and_export_splats(args.trial_dir, args.output_splat, args.epochs)


if __name__ == "__main__":
    main()