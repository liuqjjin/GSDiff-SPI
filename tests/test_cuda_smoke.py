from __future__ import annotations

import pytest
import torch

from gsdiff.forward import SPIForwardModel
from gsdiff.motion import SE2Motion
from gsdiff.scene import GaussianScene2D


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available in PyTorch runtime"
)
def test_tiny_gaussian_se2_forward_and_backward_on_cuda():
    device = torch.device("cuda")
    height, width, frames = 4, 7, 3
    scene = GaussianScene2D(M=2, H=height, W=width, init_scale=1.0).to(device)
    motion = SE2Motion(
        img_center=((height - 1) / 2, (width - 1) / 2), enable_rotation=True
    ).to(device)
    model = SPIForwardModel(scene, motion, height, width).to(device)
    patterns = torch.rand(6, height, width, device=device)
    frame_idx = torch.tensor([0, 0, 1, 1, 2, 2], device=device)
    t_grid = torch.linspace(0.0, 1.0, frames, device=device)

    measurements, video = model(patterns, frame_idx, t_grid)
    loss = measurements.square().mean() + video.square().mean()
    loss.backward()

    assert measurements.is_cuda
    assert video.is_cuda
    assert measurements.shape == (6,)
    assert video.shape == (frames, 1, height, width)
    assert torch.isfinite(measurements).all()
    assert torch.isfinite(video).all()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(gradient.is_cuda for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
