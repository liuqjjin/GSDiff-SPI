from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gsdiff.prior.tv import TVPrior
from gsdiff.solver.admm import ADMMSolver


class _ScalarScene(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))


class _ScalarMotion(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))


class _DifferentiableForward(nn.Module):
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
def dummy_admm_inputs():
    return {
        "patterns": torch.zeros(2, 1, 1, dtype=torch.float64),
        "y_target": torch.tensor([1.0, -1.0], dtype=torch.float64),
        "frame_idx": torch.tensor([0, 0]),
        "t_grid": torch.tensor([0.0], dtype=torch.float64),
    }


def make_admm_solver(
    dummy_admm_inputs,
    *,
    motion_gradient_scale=1.0,
    fwd=None,
    n_inner=1,
    n_outer=1,
    splitting_warmup_outer=1,
    motion_warmup_outer=0,
    **kwargs,
):
    if fwd is None:
        fwd = _DifferentiableForward(motion_gradient_scale)
    return ADMMSolver(
        fwd,
        TVPrior(max_iter=1),
        **dummy_admm_inputs,
        loss_norm="target_std",
        soft_tv_weight=0.0,
        n_inner=n_inner,
        n_outer=n_outer,
        splitting_warmup_outer=splitting_warmup_outer,
        motion_warmup_outer=motion_warmup_outer,
        **kwargs,
    )


def _parameter_bytes(parameters):
    return tuple(
        parameter.detach().cpu().numpy().tobytes()
        for parameter in parameters
    )


def test_admm_motion_warmup_freezes_only_scene_parameters(
    dummy_admm_inputs,
):
    solver = make_admm_solver(
        dummy_admm_inputs,
        splitting_warmup_outer=0,
        motion_warmup_outer=1,
        n_outer=2,
    )
    scene_before = _parameter_bytes(solver._scene_params)
    motion_before = _parameter_bytes(solver._motion_params)

    first = solver.step()

    assert _parameter_bytes(solver._scene_params) == scene_before
    assert _parameter_bytes(solver._motion_params) != motion_before
    assert first["in_motion_warmup"] is True
    assert first["in_splitting_warmup"] is False
    assert first["is_transition"] is True
    assert solver.rho > 0.1

    second = solver.step()

    assert _parameter_bytes(solver._scene_params) != scene_before
    assert second["in_motion_warmup"] is False
    assert second["in_splitting_warmup"] is False


def test_admm_splitting_warmup_preserves_transition_without_scene_freeze(
    dummy_admm_inputs,
):
    solver = make_admm_solver(
        dummy_admm_inputs,
        splitting_warmup_outer=1,
        motion_warmup_outer=0,
        n_outer=2,
    )
    scene_before = _parameter_bytes(solver._scene_params)
    rho_before = solver.rho

    first = solver.step()

    assert _parameter_bytes(solver._scene_params) != scene_before
    assert first["in_splitting_warmup"] is True
    assert first["in_motion_warmup"] is False
    assert solver.rho == rho_before

    second = solver.step()

    assert second["in_splitting_warmup"] is False
    assert second["is_transition"] is True
    assert solver.rho > rho_before


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("splitting_warmup_outer", True),
        ("splitting_warmup_outer", None),
        ("splitting_warmup_outer", -1),
        ("splitting_warmup_outer", 1.0),
        ("splitting_warmup_outer", 3),
        ("motion_warmup_outer", True),
        ("motion_warmup_outer", None),
        ("motion_warmup_outer", -1),
        ("motion_warmup_outer", 1.0),
        ("motion_warmup_outer", 3),
    ],
)
def test_admm_rejects_invalid_warmup_count(
    dummy_admm_inputs, field, value,
):
    with pytest.raises(ValueError, match="warmup"):
        make_admm_solver(
            dummy_admm_inputs,
            n_outer=2,
            **{field: value},
        )


@pytest.mark.parametrize(
    "field", ("splitting_warmup_outer", "motion_warmup_outer")
)
def test_admm_allows_warmup_equal_to_outer_budget(
    dummy_admm_inputs, field,
):
    solver = make_admm_solver(
        dummy_admm_inputs,
        n_outer=2,
        **{field: 2},
    )
    assert getattr(solver, field) == 2


def test_admm_rejects_legacy_and_new_splitting_warmup_together(
    dummy_admm_inputs,
):
    with pytest.raises(ValueError, match="together"):
        make_admm_solver(
            dummy_admm_inputs,
            splitting_warmup_outer=0,
            n_warmup=0,
        )


def test_admm_legacy_warmup_warns_and_maps_only_to_splitting(
    dummy_admm_inputs,
):
    with pytest.warns(DeprecationWarning, match="splitting_warmup_outer"):
        solver = ADMMSolver(
            _DifferentiableForward(),
            TVPrior(max_iter=1),
            **dummy_admm_inputs,
            loss_norm="target_std",
            n_inner=1,
            n_outer=2,
            n_warmup=1,
            motion_warmup_outer=0,
        )

    assert solver.splitting_warmup_outer == 1
    assert solver.n_warmup == 1
    assert solver.motion_warmup_outer == 0


def test_train_warmup_binding_prefers_independent_registry_fields():
    from train import _admm_warmup_counts

    config = SimpleNamespace(
        splitting_warmup_outer=3,
        motion_warmup_outer=2,
        admm_n_warmup=9,
    )
    assert _admm_warmup_counts(config) == (3, 2)


def test_train_warmup_binding_keeps_legacy_splitting_fallback():
    from train import _admm_warmup_counts

    assert _admm_warmup_counts(
        SimpleNamespace(admm_n_warmup=4)
    ) == (4, 0)


def test_freeze_and_active_parameter_helpers_clear_stale_gradients():
    from gsdiff.solver.gradients import active_parameters, freeze_parameters

    active = nn.Parameter(torch.tensor(1.0))
    frozen = nn.Parameter(torch.tensor(2.0))
    frozen.grad = torch.tensor(7.0)

    freeze_parameters([frozen])

    assert not frozen.requires_grad
    assert frozen.grad is None
    assert active_parameters([frozen, active]) == [active]


def test_clip_grad_groups_ignores_no_grad_frozen_and_empty_groups_in_order():
    from gsdiff.solver.gradients import (
        clip_grad_groups,
        freeze_parameters,
    )

    scene = nn.Parameter(torch.zeros(2))
    motion = nn.Parameter(torch.zeros(2))
    no_grad = nn.Parameter(torch.tensor(0.0))
    frozen = nn.Parameter(torch.tensor(0.0))
    scene.grad = torch.tensor([3.0, 4.0])
    motion.grad = torch.tensor([0.0, 120.0])
    freeze_parameters([frozen])
    frozen.grad = torch.tensor(1e9)

    norms = clip_grad_groups(
        [[motion], [no_grad], [], [frozen], [scene]],
        max_norm=10.0,
    )

    assert [norm.item() for norm in norms] == pytest.approx([120.0, 5.0])
    assert motion.grad.norm().item() == pytest.approx(10.0)
    assert scene.grad.tolist() == pytest.approx([3.0, 4.0])
    assert no_grad.grad is None
    assert frozen.grad.item() == 1e9


def test_admm_clips_scene_and_motion_gradients_independently(
    dummy_admm_inputs,
):
    ordinary = make_admm_solver(
        dummy_admm_inputs,
        motion_gradient_scale=1.0,
    )
    extreme = make_admm_solver(
        dummy_admm_inputs,
        motion_gradient_scale=1e6,
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


def test_admm_omits_empty_logical_groups_without_reordering(
    dummy_admm_inputs,
):
    joint = make_admm_solver(
        dummy_admm_inputs,
        lr_scene=0.009,
        lr_motion=0.15,
        n_outer=2,
    )
    empty_motion = _DifferentiableForward()
    empty_motion.motion = nn.Module()
    empty = make_admm_solver(
        dummy_admm_inputs,
        fwd=empty_motion,
        lr_scene=0.009,
        lr_motion=0.15,
        n_outer=2,
    )

    assert [group["lr"] for group in joint.optimizer.param_groups] == [
        0.009,
        0.15,
    ]
    assert [group["lr"] for group in empty.optimizer.param_groups] == [0.009]


def test_cosine_multiplier_has_exact_capped_endpoints_and_midpoint():
    from gsdiff.solver.gradients import cosine_multiplier

    assert cosine_multiplier(-1, 4) == 1.0
    assert cosine_multiplier(0, 4) == 1.0
    assert cosine_multiplier(2, 4) == pytest.approx(0.55)
    assert cosine_multiplier(4, 4) == 0.1
    assert cosine_multiplier(5, 4) == 0.1


@pytest.mark.parametrize(
    ("total_steps", "final_ratio"),
    [
        (0, 0.1),
        (-1, 0.1),
        (float("nan"), 0.1),
        (float("inf"), 0.1),
        (4, -0.1),
        (4, 1.1),
        (4, float("nan")),
        (4, float("inf")),
    ],
)
def test_cosine_multiplier_rejects_invalid_schedules(
    total_steps, final_ratio
):
    from gsdiff.solver.gradients import cosine_multiplier

    with pytest.raises(ValueError):
        cosine_multiplier(0, total_steps, final_ratio)


def test_admm_parameter_groups_follow_exact_four_step_lr_schedule(
    dummy_admm_inputs,
):
    solver = make_admm_solver(
        dummy_admm_inputs,
        lr_scene=0.009,
        lr_motion=0.15,
        n_inner=2,
        n_outer=2,
        splitting_warmup_outer=2,
    )
    expected = [
        [0.009, 0.15],
        [0.006975, 0.11625],
        [0.002925, 0.04875],
        [0.0009, 0.015],
    ]
    used_lrs = []

    def capture_lrs(optimizer, args, kwargs):
        used_lrs.append([group["lr"] for group in optimizer.param_groups])

    handle = solver.optimizer.register_step_pre_hook(capture_lrs)
    try:
        solver.step()
        solver.step()
    finally:
        handle.remove()

    final_lrs = [group["lr"] for group in solver.optimizer.param_groups]
    assert len(used_lrs) == len(expected)
    for actual, wanted in zip(used_lrs, expected, strict=True):
        assert actual == pytest.approx(wanted)
    assert final_lrs == pytest.approx(expected[-1])
    assert solver.scheduler.last_epoch == 4


def test_admm_single_update_consumes_final_lr_ratio(
    dummy_admm_inputs,
):
    solver = make_admm_solver(
        dummy_admm_inputs,
        lr_scene=0.009,
        lr_motion=0.15,
        n_inner=1,
        n_outer=1,
        splitting_warmup_outer=1,
    )
    used_lrs = []

    def capture_lrs(optimizer, args, kwargs):
        used_lrs.append([group["lr"] for group in optimizer.param_groups])

    handle = solver.optimizer.register_step_pre_hook(capture_lrs)
    try:
        solver.step()
    finally:
        handle.remove()

    assert len(used_lrs) == 1
    assert used_lrs[0] == pytest.approx([0.0009, 0.015])
    assert solver.scheduler.base_lrs == pytest.approx([0.009, 0.15])
    assert [group["lr"] for group in solver.optimizer.param_groups] == (
        pytest.approx([0.0009, 0.015])
    )
    assert solver.scheduler.last_epoch == 1
