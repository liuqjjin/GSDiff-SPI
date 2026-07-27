import math

import pytest
import torch

from gsdiff.prior.tv import (
    TVPrior,
    TVPrior3D,
    _divergence3d,
    _gradient3d,
    isotropic_tv2d_sum,
    isotropic_tv3d_sum,
)


@pytest.mark.parametrize("alpha", [0.0, 0.05, 0.3, 1.0, 2.0])
def test_weighted_gradient_divergence_are_negative_adjoint(alpha):
    torch.manual_seed(0)
    x = torch.randn(4, 5, 7, dtype=torch.float64)
    p = torch.randn(4, 5, 7, 3, dtype=torch.float64)
    p[-1, :, :, 0] = 0
    p[:, -1, :, 1] = 0
    p[:, :, -1, 2] = 0
    lhs = (_gradient3d(x, alpha) * p).sum()
    rhs = -(x * _divergence3d(p, alpha)).sum()
    relative_error = (lhs - rhs).abs() / lhs.abs().clamp_min(1e-15)
    assert relative_error.item() < 1e-10


def test_alpha_zero_matches_framewise_2d_tv():
    torch.manual_seed(0)
    video = torch.rand(3, 1, 8, 10, dtype=torch.float64)
    expected = TVPrior(max_iter=20).proximal(video, weight=0.08)
    actual = TVPrior3D(max_iter=20, temporal_weight=0.0).proximal(
        video, weight=0.08
    )
    assert expected.dtype == video.dtype
    assert actual.dtype == video.dtype
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_tv3d_energy_uses_pointwise_isotropic_norm():
    x = torch.zeros(2, 1, 2, 2)
    x[0, 0, 0, 1] = 1.0
    x[0, 0, 1, 0] = 1.0
    energy = TVPrior3D(temporal_weight=0.0).energy(x)
    assert math.isclose(energy, 2.0 + math.sqrt(2.0), rel_tol=1e-6)


def test_tv2d_energy_uses_pointwise_isotropic_norm():
    x = torch.zeros(1, 1, 2, 2)
    x[0, 0, 0, 1] = 1.0
    x[0, 0, 1, 0] = 1.0
    energy = TVPrior().energy(x)
    assert math.isclose(energy, 2.0 + math.sqrt(2.0), rel_tol=1e-6)


def test_tv3d_alpha_zero_energy_matches_independent_tv2d():
    x = torch.randn(4, 1, 5, 7, dtype=torch.float64)
    expected = sum(isotropic_tv2d_sum(frame[None]) for frame in x)
    actual = isotropic_tv3d_sum(x, temporal_weight=0.0)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("weight", [0.01, 0.2])
def test_tv3d_alpha_zero_prox_matches_independent_tv2d(weight):
    x = torch.randn(4, 1, 5, 7, dtype=torch.float64)
    prior3d = TVPrior3D(temporal_weight=0.0, max_iter=30)
    prior2d = TVPrior(max_iter=30)
    expected = torch.stack([prior2d.proximal(f[None], weight)[0] for f in x])
    actual = prior3d.proximal(x, weight)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
