import pytest
import torch
from torch import nn

from gsdiff.solver.sgd import SGDSolver


class _ScalarScene(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))


class _ScalarMotion(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))


class _DifferentiableForward(nn.Module):
    """Tiny forward with the same public shape/signature as the real model."""

    H = 1
    W = 1

    def __init__(self, motion_gradient_scale=1.0):
        super().__init__()
        self.scene = _ScalarScene()
        self.motion = _ScalarMotion()
        self.motion_gradient_scale = motion_gradient_scale

    def forward(self, patterns, frame_idx, t_grid):
        scene = self.scene.value
        motion = self.motion_gradient_scale * self.motion.value
        y_pred = torch.stack((scene + motion, 2.0 * scene - motion))
        video = (scene + motion).expand(
            t_grid.shape[0], 1, self.H, self.W
        )
        return y_pred, video

    def render_video(self, t_grid):
        scene = self.scene.value
        motion = self.motion_gradient_scale * self.motion.value
        return (scene + motion).expand(
            t_grid.shape[0], 1, self.H, self.W
        )


@pytest.fixture
def dummy_solver_inputs():
    return {
        "patterns": torch.zeros(2, 1, 1, dtype=torch.float64),
        "y_target": torch.tensor([1.0, -1.0], dtype=torch.float64),
        "frame_idx": torch.tensor([0, 0]),
        "t_grid": torch.tensor([0.0], dtype=torch.float64),
    }


def make_solver(
    dummy_solver_inputs,
    *,
    motion_gradient_scale=1.0,
    freeze_motion=False,
    **kwargs,
):
    fwd = _DifferentiableForward(motion_gradient_scale)
    return SGDSolver(
        fwd,
        **dummy_solver_inputs,
        tv_weight=0.0,
        loss_norm="target_std",
        freeze_motion=freeze_motion,
        **kwargs,
    )


def test_frozen_motion_has_no_grad_and_cannot_change_scene_clipping(
    dummy_solver_inputs,
):
    solver = make_solver(dummy_solver_inputs, freeze_motion=True, n_steps=1)

    solver.step()

    motion_params = list(solver.fwd.motion.parameters())
    assert all(not parameter.requires_grad for parameter in motion_params)
    assert all(parameter.grad is None for parameter in motion_params)


def test_frozen_motion_gradient_scale_cannot_change_scene_update_bytes(
    dummy_solver_inputs,
):
    ordinary = make_solver(
        dummy_solver_inputs,
        motion_gradient_scale=1.0,
        freeze_motion=True,
        n_steps=1,
    )
    extreme = make_solver(
        dummy_solver_inputs,
        motion_gradient_scale=1e6,
        freeze_motion=True,
        n_steps=1,
    )

    ordinary.step()
    extreme.step()

    ordinary_bytes = ordinary.fwd.scene.value.detach().numpy().tobytes()
    extreme_bytes = extreme.fwd.scene.value.detach().numpy().tobytes()
    assert ordinary_bytes == extreme_bytes


def test_sgd_clips_scene_and_motion_gradients_independently(
    dummy_solver_inputs,
):
    ordinary = make_solver(
        dummy_solver_inputs,
        motion_gradient_scale=1.0,
        n_steps=1,
    )
    extreme = make_solver(
        dummy_solver_inputs,
        motion_gradient_scale=1e6,
        n_steps=1,
    )

    ordinary.step()
    extreme.step()

    torch.testing.assert_close(
        extreme.fwd.scene.value.grad,
        ordinary.fwd.scene.value.grad,
        rtol=0.0,
        atol=0.0,
    )
    assert extreme.fwd.motion.value.grad.abs().item() == pytest.approx(5.0)


def test_sgd_motion_warmup_keeps_scene_gradient_and_update_exactly_zero(
    dummy_solver_inputs,
):
    solver = make_solver(
        dummy_solver_inputs,
        motion_gradient_scale=1e6,
        motion_warmup=1,
        n_steps=1,
    )
    initial_scene = solver.fwd.scene.value.detach().numpy().tobytes()

    solver.step()

    assert solver.fwd.scene.value.grad.item() == 0.0
    assert solver.fwd.scene.value.detach().numpy().tobytes() == initial_scene


def test_sgd_omits_frozen_and_empty_groups_in_scene_motion_order(
    dummy_solver_inputs,
):
    joint = make_solver(
        dummy_solver_inputs,
        lr_scene=0.009,
        lr_motion=0.15,
        n_steps=1,
    )
    frozen = make_solver(
        dummy_solver_inputs,
        freeze_motion=True,
        lr_scene=0.009,
        lr_motion=0.15,
        n_steps=1,
    )
    empty_motion = _DifferentiableForward()
    empty_motion.motion = nn.Module()
    empty = SGDSolver(
        empty_motion,
        **dummy_solver_inputs,
        tv_weight=0.0,
        loss_norm="target_std",
        lr_scene=0.009,
        lr_motion=0.15,
        n_steps=1,
    )

    assert [group["lr"] for group in joint.optimizer.param_groups] == [
        0.009,
        0.15,
    ]
    assert [group["lr"] for group in frozen.optimizer.param_groups] == [0.009]
    assert [group["lr"] for group in empty.optimizer.param_groups] == [0.009]


def test_sgd_parameter_groups_follow_exact_four_step_lr_schedule(
    dummy_solver_inputs,
):
    solver = make_solver(
        dummy_solver_inputs,
        lr_scene=0.009,
        lr_motion=0.15,
        n_steps=4,
    )
    expected = [
        [0.009, 0.15],
        [0.007813782463805517, 0.13022970773009196],
        [0.00495, 0.0825],
        [0.0020862175361944825, 0.03477029226990805],
        [0.0009, 0.015],
    ]
    used_lrs = []

    for _ in range(4):
        used_lrs.append(
            [group["lr"] for group in solver.optimizer.param_groups]
        )
        solver.step()

    final_lrs = [group["lr"] for group in solver.optimizer.param_groups]
    for actual, wanted in zip(used_lrs, expected[:-1], strict=True):
        assert actual == pytest.approx(wanted)
    assert final_lrs == pytest.approx(expected[-1])
    assert solver.scheduler.last_epoch == 4
