import math

import pytest
import torch

from gsdiff.prior.tv import (
    anisotropic_tv_mean,
    isotropic_tv2d_sum,
    isotropic_tv3d_sum,
)
from gsdiff.solver.admm import _soft_tv
from gsdiff.solver.sgd import tv_loss


def test_theta_and_z_tv_objectives_are_explicitly_distinct():
    x = torch.zeros(2, 1, 2, 2)
    x[0, 0, 0, 1] = 1.0
    x[0, 0, 1, 0] = 1.0
    assert isotropic_tv3d_sum(x, temporal_weight=0.5).item() == pytest.approx(
        math.sqrt(2.0) + 2.0 * math.sqrt(1.25)
    )
    assert anisotropic_tv_mean(x, temporal_weight=0.5).item() == pytest.approx(
        1.25
    )


@pytest.mark.parametrize("temporal_weight", [0.0, 0.5])
def test_solver_tv_wrappers_match_shared_anisotropic_mean(temporal_weight):
    x = torch.randn(3, 1, 4, 5, dtype=torch.float64, requires_grad=True)
    expected = anisotropic_tv_mean(x, temporal_weight)
    torch.testing.assert_close(tv_loss(x, temporal_weight), expected)
    torch.testing.assert_close(_soft_tv(x, temporal_weight), expected)


@pytest.mark.parametrize(
    ("shape", "temporal_weight"),
    [
        ((1, 1, 1, 3), 0.0),
        ((1, 1, 3, 1), 0.0),
        ((1, 1, 1, 1), 0.0),
        ((1, 1, 2, 2), 0.5),
        ((2, 1, 1, 1), 0.5),
    ],
)
def test_anisotropic_tv_mean_is_finite_for_singleton_axes(
    shape, temporal_weight
):
    x = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)
    actual = anisotropic_tv_mean(x, temporal_weight)
    assert actual.dtype == x.dtype
    assert torch.isfinite(actual)


@pytest.mark.parametrize(
    "regularizer",
    [
        isotropic_tv2d_sum,
        lambda x: isotropic_tv3d_sum(x, temporal_weight=0.5),
        lambda x: anisotropic_tv_mean(x, temporal_weight=0.5),
    ],
)
def test_tv_helpers_preserve_dtype_and_autograd(regularizer):
    x = torch.randn(3, 1, 4, 5, dtype=torch.float64, requires_grad=True)
    value = regularizer(x)
    assert value.dtype == x.dtype
    assert value.device == x.device
    value.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
