import math

import numpy as np
import torch
from torch.overrides import TorchFunctionMode

from gsdiff.forward.spi import SPIForwardModel
from gsdiff.motion.se2 import SE2Motion
from gsdiff.scene.gaussian2d import GaussianScene2D


FD_STEP = 1e-6
REL_TOL = 1e-5
NEAR_ZERO = 1e-8
ABS_TOL_NEAR_ZERO = 1e-10


class _CreationRecorder(TorchFunctionMode):
    def __init__(self):
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (torch.arange, torch.eye):
            self.calls.append((func.__name__, dict(kwargs)))
        return func(*args, **kwargs)


def _assert_float64_cpu_creations(recorder, expected_names):
    assert [name for name, _ in recorder.calls] == expected_names
    for _, kwargs in recorder.calls:
        assert kwargs["dtype"] == torch.float64
        assert torch.device(kwargs["device"]) == torch.device("cpu")


def _model():
    scene = GaussianScene2D(M=1, H=4, W=7, init_scale=1.0).double()
    motion = SE2Motion((1.5, 3.0), enable_rotation=True).double()
    with torch.no_grad():
        scene.centers.copy_(torch.tensor([[1.2, 3.4]], dtype=torch.float64))
        scene.log_scales.copy_(
            torch.log(torch.tensor([[0.9, 1.3]], dtype=torch.float64))
        )
        scene.angles.copy_(torch.tensor([0.27], dtype=torch.float64))
        scene.raw_amps.copy_(torch.tensor([0.2], dtype=torch.float64))
        motion.velocity.copy_(torch.tensor([0.31, -0.47], dtype=torch.float64))
        motion.omega.copy_(torch.tensor(0.23, dtype=torch.float64))
    return SPIForwardModel(scene, motion, H=4, W=7).double()


def test_gaussian_render_grid_and_identity_follow_double_scene():
    model = _model()
    recorder = _CreationRecorder()

    with recorder:
        image = model.scene.render()

    assert image.dtype == torch.float64
    _assert_float64_cpu_creations(recorder, ["arange", "arange", "eye"])


def test_forward_render_grid_and_identity_follow_double_video_path():
    model = _model()
    t_grid = torch.tensor([0.0, 0.37, 1.0], dtype=torch.float64)
    recorder = _CreationRecorder()

    with recorder:
        video = model.render_video(t_grid)

    assert video.dtype == torch.float64
    _assert_float64_cpu_creations(
        recorder, ["eye"] + ["arange", "arange"] * 3
    )


def test_no_rotation_identity_follows_double_motion_time():
    motion = SE2Motion((1.5, 3.0), enable_rotation=False).double()
    centers = torch.tensor([[1.2, 3.4]], dtype=torch.float64)
    t_grid = torch.tensor([0.0, 0.37, 1.0], dtype=torch.float64)
    recorder = _CreationRecorder()

    with recorder:
        transformed = motion.transform_centers(centers, t_grid)

    assert transformed.dtype == torch.float64
    _assert_float64_cpu_creations(recorder, ["eye"])


def test_double_motion_converts_registered_center_with_parameters():
    motion = SE2Motion((1.5, 3.0), enable_rotation=True).double()

    assert motion.center.dtype == torch.float64
    assert motion.center.device == motion.velocity.device
    assert motion.center.device == motion.omega.device


def test_no_rotation_identity_follows_meta_participating_device():
    motion = SE2Motion((1.5, 3.0), enable_rotation=False).double().to("meta")
    centers = torch.empty((1, 2), dtype=torch.float64, device="meta")
    t_grid = torch.empty(3, dtype=torch.float64, device="meta")

    transformed = motion.transform_centers(centers, t_grid)

    assert transformed.shape == (3, 1, 2)
    assert transformed.dtype == torch.float64
    assert transformed.device.type == "meta"


def test_affine_identity_follows_double_motion_time():
    motion = SE2Motion(
        (1.5, 3.0), enable_rotation=True, enable_affine=True
    ).double()
    sigma = torch.tensor([[[0.8, 0.1], [0.1, 1.3]]], dtype=torch.float64)
    t_grid = torch.tensor([0.0, 0.37, 1.0], dtype=torch.float64)
    recorder = _CreationRecorder()

    with recorder:
        transformed = motion.transform_covariances(sigma, t_grid)

    assert transformed.dtype == torch.float64
    _assert_float64_cpu_creations(recorder, ["eye"])


def test_affine_identity_follows_meta_participating_device():
    motion = SE2Motion(
        (1.5, 3.0), enable_rotation=True, enable_affine=True
    ).double().to("meta")
    sigma = torch.empty((1, 2, 2), dtype=torch.float64, device="meta")
    t_grid = torch.empty(3, dtype=torch.float64, device="meta")

    transformed = motion.transform_covariances(sigma, t_grid)

    assert transformed.shape == (3, 1, 2, 2)
    assert transformed.dtype == torch.float64
    assert transformed.device.type == "meta"


def _central_difference(parameter, index, loss_fn):
    original = parameter[index].item()
    with torch.no_grad():
        parameter[index] = original + FD_STEP
    plus = loss_fn().item()
    with torch.no_grad():
        parameter[index] = original - FD_STEP
    minus = loss_fn().item()
    with torch.no_grad():
        parameter[index] = original
    return (plus - minus) / (2.0 * FD_STEP)


def _assert_gradient_matches(parameter, index, loss_fn):
    parameter.grad = None
    loss = loss_fn()
    loss.backward()
    autograd_value = parameter.grad[index].item()
    finite_difference = _central_difference(parameter, index, loss_fn)

    assert math.isfinite(autograd_value)
    assert math.isfinite(finite_difference)
    denominator = max(abs(autograd_value), abs(finite_difference))
    if denominator < NEAR_ZERO:
        assert abs(autograd_value - finite_difference) < ABS_TOL_NEAR_ZERO
    else:
        assert abs(autograd_value - finite_difference) / denominator < REL_TOL
    return autograd_value, finite_difference


def test_gaussian_center_gradient_matches_central_difference():
    model = _model()
    spatial_weights = torch.tensor(
        [
            [0.2, -0.1, 0.7, 0.4, -0.3, 0.6, 0.9],
            [-0.5, 0.8, 0.1, -0.4, 0.3, 1.1, -0.2],
            [0.6, -0.7, 0.5, 0.2, 0.9, -0.6, 0.4],
            [-0.3, 0.1, 1.0, -0.8, 0.2, 0.5, -0.4],
        ],
        dtype=torch.float64,
    )

    def loss_fn():
        return (model.scene.render()[0, 0] * spatial_weights).sum()

    autograd_value, finite_difference = _assert_gradient_matches(
        model.scene.centers, (0, 1), loss_fn
    )
    assert max(abs(autograd_value), abs(finite_difference)) > 1e-6


def test_translation_velocity_gradient_matches_central_difference():
    model = _model()
    t_grid = torch.tensor([0.0, 0.37, 1.0], dtype=torch.float64)
    weights = torch.tensor(
        [[[0.3, -0.8]], [[1.2, 0.4]], [[-0.5, 0.9]]],
        dtype=torch.float64,
    )

    def loss_fn():
        centers_t = model.motion.transform_centers(model.scene.centers, t_grid)
        return (centers_t * weights).sum()

    autograd_value, finite_difference = _assert_gradient_matches(
        model.motion.velocity, (1,), loss_fn
    )
    assert max(abs(autograd_value), abs(finite_difference)) > 1e-6


def test_angular_velocity_gradient_matches_central_difference():
    model = _model()
    t_grid = torch.tensor([0.0, 0.37, 1.0], dtype=torch.float64)
    center_weights = torch.tensor(
        [[[0.2, -0.5]], [[0.7, 0.1]], [[-0.3, 0.9]]],
        dtype=torch.float64,
    )
    covariance_weights = torch.tensor(
        [
            [[[0.2, -0.1], [0.4, 0.3]]],
            [[[-0.5, 0.7], [0.1, -0.2]]],
            [[[0.6, 0.2], [-0.3, 0.8]]],
        ],
        dtype=torch.float64,
    )

    def loss_fn():
        centers_t = model.motion.transform_centers(model.scene.centers, t_grid)
        sigma_t = model.motion.transform_covariances(
            model.scene.get_covariances(), t_grid
        )
        return (centers_t * center_weights).sum() + (
            sigma_t * covariance_weights
        ).sum()

    autograd_value, finite_difference = _assert_gradient_matches(
        model.motion.omega, (), loss_fn
    )
    assert max(abs(autograd_value), abs(finite_difference)) > 1e-6


def test_end_to_end_measurement_loss_scene_and_motion_gradients_match():
    model = _model()
    t_grid = torch.tensor([0.0, 0.37, 1.0], dtype=torch.float64)
    pattern_values = torch.arange(5 * 4 * 7, dtype=torch.float64).reshape(5, 4, 7)
    patterns = ((pattern_values % 13) - 4.0) / 9.0
    frame_idx = torch.tensor([0, 0, 1, 2, 2], dtype=torch.long)
    target = torch.tensor([1.1, -0.4, 0.7, 1.8, -0.9], dtype=torch.float64)
    weights = torch.tensor([0.7, 1.1, 0.5, 0.9, 1.3], dtype=torch.float64)

    def loss_fn():
        measurements, _ = model(patterns, frame_idx, t_grid)
        return 0.5 * (weights * (measurements - target).square()).sum()

    for parameter, index in (
        (model.scene.centers, (0, 0)),
        (model.motion.velocity, (0,)),
        (model.motion.omega, ()),
    ):
        autograd_value, finite_difference = _assert_gradient_matches(
            parameter, index, loss_fn
        )
        assert max(abs(autograd_value), abs(finite_difference)) > 1e-6
