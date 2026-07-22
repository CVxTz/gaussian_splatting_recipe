import argparse
import logging
from pathlib import Path

import imageio
import numpy as np
import torch
from plyfile import PlyData
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp
from tqdm import tqdm

try:
    import pycolmap
    from gsplat import rasterization
    import imageio_ffmpeg  # Explicit check to ensure the MP4 encoder is installed
except ImportError:
    raise ImportError("Please run: pip install gsplat plyfile pycolmap imageio-ffmpeg")

logging.basicConfig(level=logging.INFO, format="%(levelname)s : %(message)s")
logger = logging.getLogger(__name__)


def interpolate_poses(c2w_matrices: list[np.ndarray], num_frames: int) -> list[np.ndarray]:
    """
    Interpolates a sequence of 4x4 camera-to-world matrices to create a smooth camera path.
    """
    n_keyframes = len(c2w_matrices)
    times = np.linspace(0, 1, n_keyframes)
    eval_times = np.linspace(0, 1, num_frames)

    translations = np.array([pose[:3, 3] for pose in c2w_matrices])
    rotations = R.from_matrix([pose[:3, :3] for pose in c2w_matrices])

    cs = CubicSpline(times, translations)
    interp_translations = cs(eval_times)

    slerp = Slerp(times, rotations)
    interp_rotations = slerp(eval_times).as_matrix()

    interp_poses = []
    for rot, trans in zip(interp_rotations, interp_translations):
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = trans
        interp_poses.append(pose)

    return interp_poses


def load_ply_data(ply_path: str):
    """
    Loads standard 3DGS PLY files and prepares them for gsplat.
    """
    logger.info(f"Unpacking PLY data from {ply_path}...")
    plydata = PlyData.read(ply_path)
    v = plydata.elements[0]

    # Extract properties
    means = np.stack([v['x'], v['y'], v['z']], axis=1)
    scales = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=1)
    quats = np.stack([v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']], axis=1)
    opacities = v['opacity']

    # Extract Spherical Harmonics DC terms and convert to base RGB
    f_dc = np.stack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']], axis=1)
    SH_C0 = 0.28209479177387814
    colors = f_dc * SH_C0 + 0.5

    # Move to GPU and apply necessary activations (exp for scale, sigmoid for opacity)
    means_t = torch.tensor(means, dtype=torch.float32, device="cuda").contiguous()
    scales_t = torch.tensor(np.exp(scales), dtype=torch.float32, device="cuda").contiguous()

    quats_t = torch.tensor(quats, dtype=torch.float32, device="cuda")
    quats_t = (quats_t / quats_t.norm(dim=-1, keepdim=True)).contiguous()

    opacities_t = torch.tensor(1.0 / (1.0 + np.exp(-opacities)), dtype=torch.float32, device="cuda").contiguous()
    colors_t = torch.tensor(colors, dtype=torch.float32, device="cuda").clamp(0, 1).contiguous()

    return means_t, scales_t, quats_t, opacities_t, colors_t


def get_colmap_cameras(sparse_dir: Path):
    """
    Parses COLMAP data using pycolmap to extract intrinsic and extrinsic matrices.
    """
    logger.info(f"Reading COLMAP reconstruction from {sparse_dir}...")
    recon = pycolmap.Reconstruction(str(sparse_dir))

    # Sort images by name to reconstruct the video path sequentially
    images = sorted(recon.images.values(), key=lambda img: img.name)

    c2w_matrices = []
    for img in images:
        # Handle API variations across pycolmap versions dynamically
        if hasattr(img, 'cam_from_world'):
            pose = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
            rot = pose.rotation.matrix()
            trans = pose.translation
        elif hasattr(img, 'qvec'):
            q = img.qvec
            rot = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
            trans = img.tvec
        else:
            raise AttributeError("Could not extract camera pose. Unrecognized pycolmap API structure.")

        w2c = np.eye(4)
        w2c[:3, :3] = rot
        w2c[:3, 3] = trans
        c2w = np.linalg.inv(w2c)
        c2w_matrices.append(c2w)

    # Extract intrinsic matrix from the first camera
    base_cam = recon.cameras[images[0].camera_id]
    width, height = int(base_cam.width), int(base_cam.height)

    # Force dimensions to be even for FFmpeg libx264 compatibility
    width -= width % 2
    height -= height % 2

    if base_cam.model_name in ["PINHOLE", "OPENCV", "OPENCV_FISHEYE"]:
        fx, fy = base_cam.params[0], base_cam.params[1]
        cx, cy = base_cam.params[2], base_cam.params[3]
    elif base_cam.model_name in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
        fx = fy = base_cam.params[0]
        cx, cy = base_cam.params[1], base_cam.params[2]
    else:
        fx = fy = base_cam.params[0]
        cx, cy = width / 2.0, height / 2.0

    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=torch.float32, device="cuda")

    return c2w_matrices, K, width, height


def render_video(dataset_path: Path, ply_path: Path, output_video: Path, seconds: int, fps: int,
                 input_fps: float) -> None:
    # 1. Parse COLMAP completely independent of fVDB
    sparse_path = dataset_path / "sparse" / "0"
    if not sparse_path.exists():
        raise FileNotFoundError(f"Could not find COLMAP sparse model at {sparse_path}")

    c2w_matrices, K, width, height = get_colmap_cameras(sparse_path)

    # CHANGED: Dynamically calculate frame count for natural pacing if seconds is not explicitly given
    if seconds is not None:
        num_frames = fps * seconds
    else:
        num_frames = int(len(c2w_matrices) * (fps / input_fps))
        auto_seconds = num_frames / fps
        logger.info(f"Auto-calculated duration: {auto_seconds:.1f} seconds to match original physical speed.")

    # 2. Interpolate smooth camera path
    logger.info(f"Interpolating {len(c2w_matrices)} keyframes into {num_frames} frames...")
    interpolated_c2w = interpolate_poses(c2w_matrices, num_frames)

    # 3. Load Gaussian Splats
    means, scales, quats, opacities, colors = load_ply_data(str(ply_path))

    # 4. Render using standard gsplat
    writer = imageio.get_writer(
        str(output_video),
        format='FFMPEG',
        fps=fps,
        macro_block_size=None
    )

    K_batched = K.unsqueeze(0).contiguous()  # gsplat expects [1, 3, 3] for intrinsics

    logger.info("Starting render loop with gsplat...")
    for c2w in tqdm(interpolated_c2w, desc="Rendering Video"):
        # gsplat expects World-to-Camera (w2c) as the viewmat
        w2c = np.linalg.inv(c2w)
        viewmat = torch.tensor(w2c, dtype=torch.float32, device="cuda").unsqueeze(0).contiguous()  # [1, 4, 4]

        # Rasterize frame
        render_colors, render_alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=K_batched,
            width=width,
            height=height,
            sh_degree=None,
            render_mode="RGB"
        )

        # Extract RGB tensor, clamp, and write to MP4
        rgb = render_colors[0].clamp(0, 1)
        rgb_np = (rgb.cpu().numpy() * 255).astype(np.uint8)
        writer.append_data(rgb_np)

    writer.close()
    logger.info(f"Video saved successfully to {output_video}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render interpolated video using gsplat and pycolmap.")
    parser.add_argument("--trial_dir", type=Path, required=True, help="Root folder with your COLMAP data")
    parser.add_argument("--ply_path", type=Path, required=True, help="Path to your trained gaussian_model.ply")
    parser.add_argument("--output_video", type=Path, default=Path("rendered_flythrough.mp4"), help="Output video")
    parser.add_argument("--fps", type=int, default=30, help="Output frames per second")

    # CHANGED: Defaults to None for seconds, added input_fps to calculate natural pacing
    parser.add_argument("--seconds", type=int, default=None,
                        help="Force duration in seconds (overrides natural pacing)")
    parser.add_argument("--input_fps", type=float, default=2.0,
                        help="The original FPS used to extract images from video")

    args = parser.parse_args()

    render_video(args.trial_dir, args.ply_path, args.output_video, args.seconds, args.fps, args.input_fps)


if __name__ == "__main__":
    main()