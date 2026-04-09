"""Generate video dataset for training the diffusion prior.

Reuses gsdiff.data.simulation.generate_spi_data() to create SE(2) motion
videos, then saves only gt_frames [T,H,W] to a .pt file.

Usage:
    python scripts/generate_video_dataset.py
    python scripts/generate_video_dataset.py --num_videos 10000 --output data/video_dataset.pt
"""
import argparse, os, sys, itertools
import numpy as np
import torch

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gsdiff.data.simulation import generate_spi_data


IMAGE_SOURCES = [
    "assets/tank.png",
    "assets/NUDT.jpg",
    "assets/usaf_tar_small.png",
    "7", "L", "T", "circle",
]


def parse_args():
    p = argparse.ArgumentParser(description="Generate video dataset for diffusion prior")
    p.add_argument("--output", default="data/video_dataset.pt")
    p.add_argument("--num_videos", type=int, default=5000)
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--W", type=int, default=64)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--vy_range", type=float, nargs=2, default=[-12.0, 12.0])
    p.add_argument("--vx_range", type=float, nargs=2, default=[-12.0, 12.0])
    p.add_argument("--omega_range", type=float, nargs=2, default=[-0.5, 0.5])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    videos = []
    n_sources = len(IMAGE_SOURCES)

    print(f"Generating {args.num_videos} videos ({args.T}x{args.H}x{args.W}) ...")
    for i in range(args.num_videos):
        shape = IMAGE_SOURCES[i % n_sources]
        vy = rng.uniform(*args.vy_range)
        vx = rng.uniform(*args.vx_range)
        omega = rng.uniform(*args.omega_range)

        data = generate_spi_data(
            H=args.H, W=args.W, T=args.T,
            K=4,                            # minimal K (patterns unused)
            pattern_type="random",
            motion_type="custom_se2",
            snr_db=0,                       # no noise on frames (snr_db<=0 skips noise)
            seed=int(rng.randint(0, 2**31)),
            shape=shape,
            motion_mode=2,
            gt_velocity=[vy, vx],
            gt_omega=omega,
        )
        videos.append(torch.from_numpy(data.gt_frames))  # [T, H, W]

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{args.num_videos}")

    dataset = torch.stack(videos)  # [N, T, H, W]
    print(f"Dataset shape: {dataset.shape}, dtype: {dataset.dtype}, "
          f"range: [{dataset.min():.3f}, {dataset.max():.3f}]")

    torch.save({"videos": dataset}, args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
