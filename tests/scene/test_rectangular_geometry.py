import numpy as np
import pytest
import torch
import torch.nn as nn

from gsdiff.baselines.inr import (
    INRForwardModel,
    build_scene,
    normalize_pixel_coordinates,
)
from gsdiff.baselines.recinr import ReCINRCanonicalScene
from gsdiff.forward.spi import SPIForwardModel
from gsdiff.motion.se2 import SE2Motion
from gsdiff.scene.gaussian2d import GaussianScene2D


H, W = 32, 64


class _ConstantOneScene(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(1.0))

    def query(self, x_norm):
        return self.value.expand(x_norm.shape[0]) + 0.0 * x_norm.sum(dim=-1)


class _GridDtypeRecordingINRForwardModel(INRForwardModel):
    def _pixel_grid(self, *args, **kwargs):
        grid = super()._pixel_grid(*args, **kwargs)
        self.render_grid_dtype = grid.dtype
        self.render_grid_device = grid.device
        return grid


def _inr_model(scene):
    motion = SE2Motion(((H - 1) / 2.0, (W - 1) / 2.0))
    return INRForwardModel(scene, motion, H, W)


def _assert_finite_gradients(module):
    gradients = [p.grad for p in module.parameters() if p.requires_grad]
    assert gradients
    assert all(g is not None for g in gradients)
    assert all(torch.isfinite(g).all() for g in gradients)


def test_rectangular_corners_map_to_unit_square():
    coords = torch.tensor([[0.0, 0.0], [31.0, 63.0]])
    center = torch.tensor([15.5, 31.5])

    actual = normalize_pixel_coordinates(coords, center, H=32, W=64)

    expected = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    torch.testing.assert_close(actual, expected)


def test_normalization_preserves_dtype_and_autograd():
    coords = torch.tensor(
        [[0.0, 0.0], [31.0, 63.0]], dtype=torch.float64, requires_grad=True)
    center = torch.tensor([15.5, 31.5], dtype=torch.float64)

    actual = normalize_pixel_coordinates(coords, center, H=H, W=W)
    actual.sum().backward()

    assert actual.dtype == coords.dtype
    assert actual.device == coords.device
    assert coords.grad is not None
    assert torch.isfinite(coords.grad).all()


def test_norm_grid_uses_rectangular_pixel_normalization():
    model = _inr_model(_ConstantOneScene())

    grid = model.norm_grid(torch.device("cpu")).view(H, W, 2)

    torch.testing.assert_close(grid[0, 0], torch.tensor([-1.0, -1.0]))
    torch.testing.assert_close(grid[-1, -1], torch.tensor([1.0, 1.0]))


def test_translation_moving_all_inverse_samples_outside_returns_zero():
    model = _inr_model(_ConstantOneScene())
    with torch.no_grad():
        model.motion.velocity.copy_(torch.tensor([0.0, 128.0]))

    video = model.render_video(torch.tensor([1.0]))
    video.sum().backward()

    torch.testing.assert_close(video, torch.zeros_like(video))
    _assert_finite_gradients(model)


def test_exact_boundary_samples_remain_inside():
    model = _inr_model(_ConstantOneScene())
    with torch.no_grad():
        model.motion.velocity.copy_(torch.tensor([0.0, 63.0]))

    video = model.render_video(torch.tensor([1.0]))

    expected = torch.zeros(1, 1, H, W)
    expected[..., -1] = 1.0
    torch.testing.assert_close(video, expected)


@pytest.mark.parametrize(
    ("kind", "scene_kw"),
    [
        ("siren", {"hidden": 8}),
        ("grid", {}),
        ("lowrank", {"r": 4}),
        ("recinr_se2", {"C": 4, "grid_size": 8}),
    ],
)
def test_built_scene_does_not_extrapolate_outside_field_of_view(kind, scene_kw):
    torch.manual_seed(0)
    model = _inr_model(build_scene(kind, H=H, W=W, **scene_kw))
    with torch.no_grad():
        model.motion.velocity.copy_(torch.tensor([0.0, 128.0]))

    video = model.render_video(torch.tensor([1.0]))
    video.sum().backward()

    torch.testing.assert_close(video, torch.zeros_like(video))
    _assert_finite_gradients(model)


@pytest.mark.parametrize(
    ("kind", "scene_kw"),
    [
        ("siren", {"hidden": 8}),
        ("grid", {}),
        ("lowrank", {"r": 4}),
        ("recinr_se2", {"C": 4, "grid_size": 8}),
    ],
)
def test_real_rectangular_scenes_render_finite_video_and_gradients(kind, scene_kw):
    torch.manual_seed(0)
    model = _inr_model(build_scene(kind, H=H, W=W, **scene_kw))

    video = model.render_video(torch.tensor([0.0, 0.5, 1.0]))
    video.square().mean().backward()

    assert video.shape == (3, 1, H, W)
    assert torch.isfinite(video).all()
    _assert_finite_gradients(model)


def test_real_rectangular_inr_builds_double_grid_before_full_forward():
    torch.manual_seed(0)
    scene = build_scene("siren", H=H, W=W, hidden=8).double()
    motion = SE2Motion(
        ((H - 1) / 2.0, (W - 1) / 2.0)
    ).double()
    model = _GridDtypeRecordingINRForwardModel(
        scene, motion, H, W
    ).double()
    t_grid = torch.tensor([0.0, 1.0], dtype=torch.float64)
    patterns = torch.ones(2, H, W, dtype=torch.float64)
    frame_idx = torch.tensor([0, 1])

    measurements, video = model(patterns, frame_idx, t_grid)
    (measurements.square().mean() + video.square().mean()).backward()

    assert model.render_grid_dtype == torch.float64
    assert model.render_grid_device == t_grid.device
    assert measurements.dtype == torch.float64
    assert video.dtype == torch.float64
    assert video.shape == (2, 1, H, W)
    assert torch.isfinite(measurements).all()
    assert torch.isfinite(video).all()
    _assert_finite_gradients(model)


def test_build_scene_preserves_grid_and_lowrank_rectangular_dimensions():
    grid = build_scene("grid", H=H, W=W)
    lowrank = build_scene("lowrank", H=H, W=W, r=4)

    assert (grid.Hc, grid.Wc) == (H, W)
    assert grid.grid.shape == (1, 1, H, W)
    assert lowrank.U.shape == (H, 4)
    assert lowrank.V.shape == (W, 4)


def test_grid_prefit_validates_size_and_reshapes_rectangular_target():
    scene = build_scene("grid", H=H, W=W)
    target = np.linspace(0.1, 0.9, H * W, dtype=np.float32).reshape(H, W)

    scene.prefit(target, torch.empty(H * W, 2))

    torch.testing.assert_close(
        torch.sigmoid(scene.grid.detach())[0, 0], torch.from_numpy(target))
    with pytest.raises(ValueError, match="2048"):
        scene.prefit(np.zeros(H * W - 1, dtype=np.float32), torch.empty(0, 2))


@pytest.mark.parametrize(
    "shape",
    [(H, W), (H * W,), (1, H, W), (1, 1, H, W)],
)
def test_grid_prefit_accepts_only_supported_image_shapes(shape):
    scene = build_scene("grid", H=H, W=W)
    expected = np.linspace(0.1, 0.9, H * W, dtype=np.float32).reshape(H, W)

    scene.prefit(expected.reshape(shape), torch.empty(H * W, 2))

    torch.testing.assert_close(
        torch.sigmoid(scene.grid.detach())[0, 0], torch.from_numpy(expected))


@pytest.mark.parametrize("shape", [(W, H), (H * W, 1)])
def test_grid_prefit_rejects_same_numel_unsupported_shapes(shape):
    scene = build_scene("grid", H=H, W=W)
    target = np.zeros(shape, dtype=np.float32)

    with pytest.raises(ValueError, match="shape"):
        scene.prefit(target, torch.empty(H * W, 2))


@pytest.mark.parametrize("shape", [(2, H, W), (1, 2, H, W)])
def test_grid_prefit_rejects_non_singleton_batch_or_channel(shape):
    scene = build_scene("grid", H=H, W=W)

    with pytest.raises(ValueError):
        scene.prefit(np.zeros(shape, dtype=np.float32), torch.empty(H * W, 2))


def test_recinr_scalar_grid_size_preserves_rectangular_aspect_ratio():
    scene = build_scene("recinr_se2", H=H, W=W, C=4, grid_size=16)

    assert (scene.gh, scene.gw) == (16, 32)
    assert scene.features.shape == (1, 4, 16, 32)


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        (64, 32, (32, 16)),
        (30, 47, (16, 25)),
        (47, 30, (25, 16)),
    ],
)
def test_recinr_scalar_grid_size_preserves_portrait_and_rounded_aspect(
    height, width, expected
):
    scene = ReCINRCanonicalScene(height, width, C=4, grid_size=16)

    assert (scene.gh, scene.gw) == expected
    assert min(scene.gh, scene.gw) == 16


def test_recinr_accepts_explicit_rectangular_grid_size():
    scene = ReCINRCanonicalScene(H, W, C=4, grid_size=(12, 20))

    assert (scene.gh, scene.gw) == (12, 20)
    assert scene.features.shape == (1, 4, 12, 20)
    assert scene._feat_grid().shape == (1, 4, H, W)


@pytest.mark.parametrize("grid_size", [0, -1, (16,), (16, 0), (16, 32, 64)])
def test_recinr_rejects_invalid_grid_size(grid_size):
    with pytest.raises((TypeError, ValueError)):
        ReCINRCanonicalScene(H, W, C=4, grid_size=grid_size)


def test_gaussian_renderer_is_rectangular_zero_fov_and_has_finite_gradients():
    torch.manual_seed(0)
    scene = GaussianScene2D(2, H, W, init_scale=10.0).double()
    motion = SE2Motion(
        ((H - 1) / 2.0, (W - 1) / 2.0)
    ).double()
    model = SPIForwardModel(scene, motion, H, W).double()
    with torch.no_grad():
        motion.velocity.copy_(torch.tensor([0.0, 64.0], dtype=torch.float64))

    video = model.render_video(torch.tensor([1.0], dtype=torch.float64))
    video.sum().backward()

    assert video.shape == (1, 1, H, W)
    assert video.dtype == torch.float64
    assert torch.isfinite(video).all()
    torch.testing.assert_close(video, torch.zeros_like(video))
    _assert_finite_gradients(model)


def test_gaussian_renderer_includes_exact_inverse_fov_boundary():
    scene = GaussianScene2D(1, H, W, init_scale=10.0).double()
    motion = SE2Motion(
        ((H - 1) / 2.0, (W - 1) / 2.0)
    ).double()
    model = SPIForwardModel(scene, motion, H, W).double()
    with torch.no_grad():
        scene.centers.copy_(
            torch.tensor(
                [[(H - 1) / 2.0, (W - 1) / 2.0]],
                dtype=torch.float64,
            )
        )
        motion.velocity.copy_(torch.tensor([0.0, 63.0], dtype=torch.float64))

    video = model.render_video(torch.tensor([1.0], dtype=torch.float64))

    torch.testing.assert_close(
        video[..., :-1], torch.zeros_like(video[..., :-1])
    )
    assert torch.all(video[..., -1] > 0.0)


def test_gaussian_affine_mask_matches_independent_inverse_domain():
    scene = GaussianScene2D(1, H, W, init_scale=100.0).double()
    motion = SE2Motion(
        ((H - 1) / 2.0, (W - 1) / 2.0),
        enable_affine=True,
    ).double()
    model = SPIForwardModel(scene, motion, H, W).double()
    angle = 0.31
    velocity = torch.tensor([2.0, -3.0], dtype=torch.float64)
    linear = torch.tensor([0.12, -0.08, 0.05], dtype=torch.float64)
    with torch.no_grad():
        scene.centers.copy_(
            torch.tensor(
                [[(H - 1) / 2.0, (W - 1) / 2.0]],
                dtype=torch.float64,
            )
        )
        motion.velocity.copy_(velocity)
        motion.omega.copy_(torch.tensor(angle, dtype=torch.float64))
        motion.lin.copy_(linear)

    video = model.render_video(torch.tensor([1.0], dtype=torch.float64))

    cosine = torch.cos(torch.tensor(angle, dtype=torch.float64))
    sine = torch.sin(torch.tensor(angle, dtype=torch.float64))
    rotation = torch.stack((cosine, -sine, sine, cosine)).reshape(2, 2)
    symmetric = torch.tensor(
        [[1.0 + linear[0], linear[2]],
         [linear[2], 1.0 + linear[1]]],
        dtype=torch.float64,
    )
    transform = rotation @ symmetric
    gy, gx = torch.meshgrid(
        torch.arange(H, dtype=torch.float64),
        torch.arange(W, dtype=torch.float64),
        indexing="ij",
    )
    output_coordinates = torch.stack((gy, gx), dim=-1)
    center = torch.tensor(
        [(H - 1) / 2.0, (W - 1) / 2.0],
        dtype=torch.float64,
    )
    canonical = (
        torch.einsum(
            "ij,hwj->hwi",
            torch.linalg.inv(transform),
            output_coordinates - center - velocity,
        )
        + center
    )
    expected_inside = (
        (canonical[..., 0] >= 0.0)
        & (canonical[..., 0] <= H - 1)
        & (canonical[..., 1] >= 0.0)
        & (canonical[..., 1] <= W - 1)
    )

    assert expected_inside.any()
    assert (~expected_inside).any()
    assert torch.equal(video[0, 0] > 0.0, expected_inside)
    video.sum().backward()
    _assert_finite_gradients(model)
