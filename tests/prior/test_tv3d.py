import pytest
import torch

from gsdiff.prior.tv import TVPrior, TVPrior3D, _divergence3d, _gradient3d


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
