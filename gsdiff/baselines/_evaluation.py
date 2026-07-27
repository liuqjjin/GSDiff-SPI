"""Explicit legacy evaluation compatibility for non-blind baseline paths."""

import numpy as np
import torch


def evaluate_video(gt_frames, recon):
    """Compatibility adapter for legacy per-frame min-max PSNR."""
    from ..evaluation.metrics import evaluate_video_legacy_per_frame

    gt = (
        gt_frames.cpu().numpy()
        if torch.is_tensor(gt_frames)
        else np.asarray(gt_frames)
    )
    rc = recon.cpu().numpy() if torch.is_tensor(recon) else np.asarray(recon)
    result = evaluate_video_legacy_per_frame(gt, rc)
    return (
        result["per_frame_psnr_legacy_per_frame_minmax"],
        result["psnr_legacy_per_frame_minmax"],
    )
