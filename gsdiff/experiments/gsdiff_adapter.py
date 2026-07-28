"""Strict, truth-free adapters for the four canonical GSDiff methods."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from gsdiff.data._artifact_identity import ArtifactValidationError
from gsdiff.data._artifact_io import (
    SafeFileSnapshot,
    read_safe_file_snapshot,
    verify_safe_file_snapshot,
)
from gsdiff.data._artifact_models import SPIAcquisitionData

from .adapters import _algorithm_rng, _validate_acquisition
from .child_outputs import MethodChildResult
from .methods import AlgorithmSeed, CANONICAL_METHOD_IDS, ResolvedMethod
from .objectives import heldout_normalized_l2


GSDIFF_METHOD_IDS = (
    "siren",
    "recinr_se2",
    "gsdiff_tv",
    "gsdiff_diffusion",
)
if GSDIFF_METHOD_IDS != CANONICAL_METHOD_IDS[-len(GSDIFF_METHOD_IDS):]:
    raise RuntimeError("GSDiff adapter IDs do not match canonical registry IDs")
_MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class _GSDiffRuntime:
    scene: object
    motion: object
    forward_model: object
    prior: object | None
    solver: object
    dgi: np.ndarray
    checkpoint_snapshots: tuple[SafeFileSnapshot, ...]


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _normalize_01(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    low = float(values.min())
    high = float(values.max())
    if high - low < 1e-8:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def _validate_diffusion_geometry(
    method: ResolvedMethod,
    acquisition: SPIAcquisitionData,
) -> None:
    if method.method_id != "gsdiff_diffusion":
        return
    diffusion_config = _require_mapping(
        method.semantic_config.get("diffusion"), "diffusion"
    )
    channel_mults = diffusion_config.get("channel_mults")
    if (
        not isinstance(channel_mults, (list, tuple))
        or not channel_mults
        or any(type(value) is not int or value <= 0 for value in channel_mults)
    ):
        raise ValueError(
            "diffusion channel_mults must be a nonempty positive-integer sequence"
        )
    pooling_stages = len(channel_mults) - 1
    minimum_extent = 2**pooling_stages
    for dimension, extent in (
        ("T", acquisition.T),
        ("H", acquisition.H),
        ("W", acquisition.W),
    ):
        if extent < minimum_extent:
            raise ValueError(
                f"gsdiff_diffusion requires {dimension} to be at least "
                f"{minimum_extent} for {pooling_stages} factor-two pooling "
                "stages"
            )


def _validate_checkpoint_contract(
    method: ResolvedMethod,
    checkpoint_paths: Mapping[str, Path],
) -> tuple[dict[str, Path], tuple[SafeFileSnapshot, ...]]:
    if not isinstance(checkpoint_paths, Mapping):
        raise TypeError("checkpoint_paths must be a mapping")
    if any(type(key) is not str for key in checkpoint_paths):
        raise TypeError("checkpoint logical IDs must be strings")
    expected = {
        requirement.logical_id: requirement
        for requirement in method.checkpoint_requirements
    }
    actual_ids = set(checkpoint_paths)
    missing = set(expected) - actual_ids
    if missing:
        raise ValueError(
            f"missing checkpoint logical IDs: {sorted(missing)!r}"
        )
    extra = actual_ids - set(expected)
    if extra:
        raise ValueError(f"extra checkpoint logical IDs: {sorted(extra)!r}")

    validated: dict[str, Path] = {}
    snapshots: list[SafeFileSnapshot] = []
    for logical_id, requirement in expected.items():
        path = checkpoint_paths[logical_id]
        if not isinstance(path, Path):
            raise TypeError(f"checkpoint {logical_id!r} path must be a Path")
        try:
            snapshot = read_safe_file_snapshot(
                path,
                max_bytes=_MAX_CHECKPOINT_BYTES,
                noun=f"checkpoint {logical_id!r}",
            )
        except ArtifactValidationError as error:
            raise ValueError(str(error)) from error
        if snapshot.sha256 != requirement.sha256:
            raise ValueError(
                f"checkpoint sha256 mismatch for {logical_id!r}"
            )
        validated[logical_id] = snapshot.path
        snapshots.append(snapshot)
    return validated, tuple(snapshots)


def _construct_gsdiff_runtime(
    method: ResolvedMethod,
    acquisition: SPIAcquisitionData,
    *,
    checkpoint_paths: Mapping[str, Path],
    device: str,
) -> _GSDiffRuntime:
    """Validate capabilities before importing or constructing method models."""
    if type(method) is not ResolvedMethod:
        raise TypeError("method must be a ResolvedMethod")
    if method.method_id not in GSDIFF_METHOD_IDS:
        raise ValueError(f"unsupported GSDiff method: {method.method_id}")
    if method.execution_family != "gsdiff":
        raise ValueError("GSDiff adapter requires the gsdiff execution family")
    if not method.execution_ready:
        blockers = ", ".join(method.execution_blockers)
        raise ValueError(f"method execution is blocked: {blockers}")
    if type(acquisition) is not SPIAcquisitionData:
        raise TypeError("acquisition must be SPIAcquisitionData")
    if type(device) is not str or not device:
        raise ValueError("device must be a nonempty string")
    _validate_acquisition(acquisition, require_holdout=True)
    _validate_diffusion_geometry(method, acquisition)
    validated_paths, snapshots = _validate_checkpoint_contract(
        method, checkpoint_paths
    )
    return _construct_validated_runtime(
        method,
        acquisition,
        checkpoint_paths=validated_paths,
        checkpoint_snapshots=snapshots,
        device=device,
    )


def _construct_validated_runtime(
    method: ResolvedMethod,
    acquisition: SPIAcquisitionData,
    *,
    checkpoint_paths: Mapping[str, Path],
    checkpoint_snapshots: tuple[SafeFileSnapshot, ...],
    device: str,
) -> _GSDiffRuntime:
    import torch

    from gsdiff.baselines.common import (
        calibrate_reconstruction_physical,
        dgi_image,
    )
    from gsdiff.forward.spi import SPIForwardModel
    from gsdiff.motion.se2 import SE2Motion
    from gsdiff.solver.admm import ADMMSolver
    from gsdiff.solver.sgd import SGDSolver

    semantics = method.semantic_config
    scene_config = _require_mapping(semantics.get("scene"), "scene")
    motion_config = _require_mapping(semantics.get("motion"), "motion")
    solver_config = _require_mapping(semantics.get("solver"), "solver")
    torch_device = torch.device(device)

    motion = SE2Motion(
        (
            (acquisition.H - 1) / 2.0,
            (acquisition.W - 1) / 2.0,
        ),
        enable_rotation=motion_config["enable_rotation"],
        poly_degree=motion_config["polynomial_degree"],
        enable_affine=motion_config["enable_affine"],
    ).to(torch_device)

    scene_type = scene_config["scene_type"]
    if method.method_id == "siren":
        if scene_type != "siren":
            raise ValueError("siren method requires scene_type=siren")
        from gsdiff.baselines.inr import INRForwardModel, SIREN

        scene = SIREN(
            hidden=scene_config["hidden"],
            n_hidden=scene_config["hidden_layers"],
            w0=scene_config["w0"],
        ).to(torch_device)
        forward_model = INRForwardModel(
            scene,
            motion,
            acquisition.H,
            acquisition.W,
            time_assignment_mode=acquisition.time_assignment_mode,
        ).to(torch_device)
    elif method.method_id == "recinr_se2":
        if scene_type != "recinr":
            raise ValueError(
                "recinr_se2 method requires registry scene_type=recinr"
            )
        from gsdiff.baselines.inr import INRForwardModel
        from gsdiff.baselines.recinr import ReCINRCanonicalScene

        scene = ReCINRCanonicalScene(
            acquisition.H,
            acquisition.W,
            C=scene_config["channels"],
            render_layers=scene_config["render_layers"],
            grid_size=scene_config["grid_size"],
        ).to(torch_device)
        forward_model = INRForwardModel(
            scene,
            motion,
            acquisition.H,
            acquisition.W,
            time_assignment_mode=acquisition.time_assignment_mode,
        ).to(torch_device)
    else:
        if scene_type != "gaussian":
            raise ValueError(
                f"{method.method_id} requires scene_type=gaussian"
            )
        from gsdiff.scene.gaussian2d import GaussianScene2D

        scene = GaussianScene2D(
            scene_config["gaussian_count"],
            acquisition.H,
            acquisition.W,
            init_scale=scene_config["init_scale"],
            min_scale=scene_config["min_scale"],
        ).to(torch_device)
        forward_model = SPIForwardModel(
            scene,
            motion,
            acquisition.H,
            acquisition.W,
            time_assignment_mode=acquisition.time_assignment_mode,
        ).to(torch_device)

    conditioned_dgi = dgi_image(
        acquisition.patterns, acquisition.measurements
    ).numpy()
    dgi_video = np.repeat(conditioned_dgi[None], acquisition.T, axis=0)
    physical_dgi = calibrate_reconstruction_physical(
        dgi_video,
        acquisition.patterns,
        acquisition.measurements,
        acquisition.frame_indices,
    )[0]
    initialization = scene_config["initialization"]
    if initialization == "random":
        if scene_config["dgi_prefit"] is not False:
            raise ValueError("random scene initialization forbids DGI prefit")
    elif initialization == "dgi_adaptive":
        if method.method_id not in {"gsdiff_tv", "gsdiff_diffusion"}:
            raise ValueError("DGI-adaptive initialization requires Gaussian scene")
        scene.init_adaptive(conditioned_dgi)
        scene.init_from_image(_normalize_01(conditioned_dgi))
    else:
        raise ValueError(f"unsupported scene initialization: {initialization}")

    patterns = torch.tensor(
        np.asarray(acquisition.patterns),
        dtype=torch.float32,
        device=torch_device,
    )
    measurements = torch.tensor(
        np.asarray(acquisition.measurements),
        dtype=torch.float32,
        device=torch_device,
    )
    frame_indices = torch.tensor(
        np.asarray(acquisition.frame_indices),
        dtype=torch.long,
        device=torch_device,
    )
    time_grid = torch.tensor(
        np.asarray(acquisition.time_grid),
        dtype=torch.float32,
        device=torch_device,
    )

    solver_type = solver_config["solver_type"]
    prior = None
    if solver_type == "sgd":
        if method.method_id not in {"siren", "recinr_se2"}:
            raise ValueError("SGD binding is invalid for this GSDiff method")
        solver = SGDSolver(
            forward_model,
            patterns,
            measurements,
            frame_indices,
            time_grid,
            tv_weight=solver_config["tv_weight"],
            lr_scene=solver_config["lr_scene"],
            lr_motion=solver_config["lr_motion"],
            n_steps=solver_config["sgd_steps"],
            loss_norm=solver_config["loss_norm"],
            temporal_tv_weight=(
                solver_config["temporal_tv_weight"]
                if solver_config["use_3dtv"]
                else 0.0
            ),
            red_prior=None,
            red_weight=0.0,
            freeze_motion=False,
            motion_warmup=solver_config["motion_warmup_steps"],
            device=torch_device,
        )
    elif solver_type == "admm":
        if method.method_id not in {"gsdiff_tv", "gsdiff_diffusion"}:
            raise ValueError("ADMM binding is invalid for this GSDiff method")
        outer_iterations = solver_config["outer_iterations"]
        splitting_warmup = solver_config["splitting_warmup_outer"]
        expected_motion_warmup = math.ceil(
            solver_config["motion_warmup_fraction"] * outer_iterations
        )
        motion_warmup = solver_config["motion_warmup_outer"]
        if motion_warmup != expected_motion_warmup:
            raise ValueError(
                "motion_warmup_outer does not match the locked fraction"
            )
        prior_type = solver_config["prior_type"]
        if prior_type == "tv":
            if (
                method.method_id != "gsdiff_tv"
                or solver_config["tv_variant"] != "tv3d_corrected"
            ):
                raise ValueError("TV method requires corrected 3D-TV")
            from gsdiff.prior.tv import TVPrior3D

            prior = TVPrior3D(
                max_iter=solver_config["prior_proximal_iterations"],
                temporal_weight=solver_config["temporal_tv_weight"],
            )
            proximal_weight = solver_config["tv_weight"]
        elif prior_type == "diffusion":
            if method.method_id != "gsdiff_diffusion":
                raise ValueError(
                    "diffusion prior requires gsdiff_diffusion identity"
                )
            diffusion_config = _require_mapping(
                semantics.get("diffusion"), "diffusion"
            )
            requirement = method.checkpoint_requirements[0]
            from gsdiff.prior.diffusion import DiffusionPrior

            prior = DiffusionPrior(
                checkpoint_path=str(checkpoint_paths[requirement.logical_id]),
                device=str(torch_device),
                denoise_steps=diffusion_config["denoise_steps"],
                clamp_range=tuple(diffusion_config["clamp_range"]),
                in_channels=diffusion_config["in_channels"],
                base_channels=diffusion_config["base_channels"],
                channel_mults=list(diffusion_config["channel_mults"]),
                emb_dim=diffusion_config["emb_dim"],
                sigma_min=diffusion_config["sigma_min"],
                sigma_max=diffusion_config["sigma_max"],
                sigma_start=diffusion_config["sigma_start"],
                sigma_end=diffusion_config["sigma_end"],
                renoise=diffusion_config["renoise"],
                ddim_spacing=diffusion_config["ddim_spacing"],
            )
            prior.set_n_steps(outer_iterations - splitting_warmup)
            proximal_weight = solver_config["proximal_weight"]
        else:
            raise ValueError(f"unsupported prior_type: {prior_type}")
        solver = ADMMSolver(
            forward_model,
            prior,
            patterns,
            measurements,
            frame_indices,
            time_grid,
            rho=solver_config["rho"],
            tv_weight=proximal_weight,
            lr_scene=solver_config["lr_scene"],
            lr_motion=solver_config["lr_motion"],
            n_inner=solver_config["inner_iterations"],
            rho_growth=solver_config["rho_growth"],
            loss_norm=solver_config["loss_norm"],
            device=torch_device,
            splitting_warmup_outer=splitting_warmup,
            motion_warmup_outer=motion_warmup,
            n_outer=outer_iterations,
            soft_tv_weight=solver_config["soft_tv_weight"],
            temporal_tv_weight=solver_config["temporal_tv_weight"],
            hqs=False,
        )
    else:
        raise ValueError(f"unsupported solver_type: {solver_type}")

    for snapshot in checkpoint_snapshots:
        verify_safe_file_snapshot(snapshot)
    return _GSDiffRuntime(
        scene=scene,
        motion=motion,
        forward_model=forward_model,
        prior=prior,
        solver=solver,
        dgi=np.asarray(physical_dgi, dtype=np.float32),
        checkpoint_snapshots=checkpoint_snapshots,
    )


def _motion_trajectory(motion: object, time_grid: np.ndarray) -> np.ndarray:
    parameters = motion.get_params_dict()
    times = np.asarray(time_grid, dtype=np.float64)
    velocity = np.asarray(parameters["velocity"], dtype=np.float64)
    acceleration = np.asarray(
        parameters.get("accel", [0.0, 0.0]), dtype=np.float64
    )
    translation = (
        times[:, None] * velocity
        + times[:, None] ** 2 * acceleration
    )
    rotation = (
        times * float(parameters.get("omega", 0.0))
        + times**2 * float(parameters.get("beta", 0.0))
    )
    return np.column_stack((translation, rotation)).astype(np.float32)


def run_gsdiff_method(
    method: ResolvedMethod,
    acquisition: SPIAcquisitionData,
    *,
    algorithm_seed: AlgorithmSeed,
    checkpoint_paths: Mapping[str, Path],
    device: str,
) -> MethodChildResult:
    """Run one canonical GSDiff method without evaluator capabilities."""
    if type(algorithm_seed) is not AlgorithmSeed:
        raise TypeError("algorithm_seed must be an AlgorithmSeed")
    with _algorithm_rng(algorithm_seed):
        runtime = _construct_gsdiff_runtime(
            method,
            acquisition,
            checkpoint_paths=checkpoint_paths,
            device=device,
        )
        solver_config = _require_mapping(
            method.semantic_config.get("solver"), "solver"
        )
        history: list[dict[str, object]] = []
        if solver_config["solver_type"] == "sgd":
            budget = solver_config["sgd_steps"]
            for step in range(1, budget + 1):
                native = runtime.solver.step()
                history.append(
                    {
                        "kind": "step",
                        "step": step,
                        "loss": float(native["loss_total"]),
                        "data_fidelity": float(native["loss_data"]),
                    }
                )
            unit = "sgd-step"
        else:
            budget = solver_config["outer_iterations"]
            for outer_iteration in range(1, budget + 1):
                native = runtime.solver.step()
                history.append(
                    {
                        "kind": "outer-iteration",
                        "outer_iteration": outer_iteration,
                        "loss": float(
                            native["loss_data"] + native["loss_consist"]
                        ),
                        "primal_residual": float(native["prim_res"]),
                        "dual_residual": float(native["dual_res"]),
                    }
                )
            unit = "outer-iteration"

        import torch

        with torch.no_grad():
            rendered = (
                runtime.forward_model.render_video(
                    runtime.solver.t_grid
                )[:, 0]
                .detach()
                .cpu()
                .numpy()
            )
        from gsdiff.baselines.common import (
            calibrate_reconstruction_physical,
            unique_optimizer_parameter_count,
        )

        reconstruction = calibrate_reconstruction_physical(
            rendered,
            acquisition.patterns,
            acquisition.measurements,
            acquisition.frame_indices,
        )
        assert acquisition.holdout_patterns is not None
        assert acquisition.holdout_measurements is not None
        assert acquisition.holdout_frame_indices is not None
        objective = heldout_normalized_l2(
            reconstruction,
            acquisition.holdout_patterns,
            acquisition.holdout_measurements,
            acquisition.holdout_frame_indices,
        )
        history[-1]["objective"] = objective.value
        checkpoint_hashes = [
            {
                "logical_id": requirement.logical_id,
                "sha256": requirement.sha256,
            }
            for requirement in method.checkpoint_requirements
        ]
        info = {
            "parameter_count": unique_optimizer_parameter_count(
                runtime.solver.optimizer
            ),
            "native_iteration_unit": unit,
            "native_iteration_budget": budget,
            "convergence_status": method.convergence_status,
            "selected_hyperparameters": None,
            "selection": None,
            "checkpoint_hashes": checkpoint_hashes,
            "native_motion_model": "se2-polynomial",
        }
        return MethodChildResult(
            method_id=method.method_id,
            reconstruction=np.asarray(reconstruction, dtype=np.float32),
            estimated_motion_trajectory=_motion_trajectory(
                runtime.motion, acquisition.time_grid
            ),
            dgi=runtime.dgi,
            info=info,
            history=tuple(history),
        )
