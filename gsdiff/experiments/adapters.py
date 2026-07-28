"""Capability-safe in-process adapters for the seven baseline methods."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import random

import numpy as np
import torch

from gsdiff.data._artifact_models import SPIAcquisitionData

from .child_outputs import MethodChildResult
from .methods import AlgorithmSeed, ResolvedMethod


BASELINE_METHOD_IDS = (
    "dgi", "static_cs", "perframe_cs", "tv3d", "monin", "gidc3dtv", "recinr",
)


@contextmanager
def _algorithm_rng(seed: AlgorithmSeed) -> Iterator[None]:
    """Seed a method without leaking a changed caller RNG state."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed.seed_u32)
        np.random.seed(seed.seed_u32)
        torch.manual_seed(seed.seed_u32)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed.seed_u32)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _base_info(method: ResolvedMethod, *, parameter_count: int, unit: str, budget: int,
               selected_hyperparameters, selection, native_motion_model: str = "none") -> dict[str, object]:
    return {"parameter_count": parameter_count, "native_iteration_unit": unit,
            "native_iteration_budget": budget, "convergence_status": method.convergence_status,
            "selected_hyperparameters": selected_hyperparameters, "selection": selection,
            "checkpoint_hashes": [], "native_motion_model": native_motion_model}


def run_dgi(acquisition: SPIAcquisitionData, semantic_config, algorithm_seed: AlgorithmSeed, device: str):
    del semantic_config, algorithm_seed
    from gsdiff.baselines.common import dgi_image
    image = dgi_image(acquisition.patterns, acquisition.measurements).to(device)
    video = image.unsqueeze(0).repeat(acquisition.T, 1, 1)
    predicted = np.einsum("khw,khw->k", acquisition.patterns.astype(np.float64), video.detach().cpu().numpy()[acquisition.frame_indices])
    numerator, denominator = float(predicted @ acquisition.measurements), float(predicted @ predicted)
    if denominator > 0:
        scale = numerator / denominator
        video = video * scale
        image = image * scale
    return video.detach().cpu().numpy(), image.detach().cpu().numpy(), None, {"selected_hyperparameters": None, "selection": None}


def run_baseline_method(method: ResolvedMethod, acquisition: SPIAcquisitionData, *, algorithm_seed: AlgorithmSeed, device: str) -> MethodChildResult:
    """Run one canonical baseline without evaluator or truth capabilities."""
    if method.method_id not in BASELINE_METHOD_IDS:
        raise ValueError(f"unsupported baseline method: {method.method_id}")
    if method.execution_family != "baseline":
        raise ValueError("baseline adapter requires the baseline execution family")
    if type(acquisition) is not SPIAcquisitionData:
        raise TypeError("acquisition must be SPIAcquisitionData")
    if type(algorithm_seed) is not AlgorithmSeed:
        raise TypeError("algorithm_seed must be AlgorithmSeed")
    semantic = method.semantic_config
    with _algorithm_rng(algorithm_seed):
        if method.method_id == "dgi":
            reconstruction, dgi, motion, details = run_dgi(acquisition, semantic, algorithm_seed, device)
            info = _base_info(method, parameter_count=0, unit=str(semantic["native_unit"]), budget=int(semantic["native_budget"]), selected_hyperparameters=details["selected_hyperparameters"], selection=details["selection"])
        elif method.method_id in {"static_cs", "perframe_cs"}:
            from gsdiff.baselines import cs
            runner = cs.run_static_cs if method.method_id == "static_cs" else cs.run_perframe_cs
            reconstruction, details = runner(acquisition, semantic, algorithm_seed, device)
            count = acquisition.H * acquisition.W if method.method_id == "static_cs" else acquisition.T * acquisition.H * acquisition.W
            info = _base_info(method, parameter_count=count, unit="admm-iteration", budget=int(semantic["solver"]["n_admm"]), selected_hyperparameters=details["selected_hyperparameters"], selection=details["selection"])
            dgi = motion = None
        elif method.method_id == "tv3d":
            from gsdiff.baselines.tv3d import run_tv3d
            reconstruction, details = run_tv3d(acquisition, semantic, algorithm_seed, device)
            info = _base_info(method, parameter_count=acquisition.T * acquisition.H * acquisition.W, unit="primal-dual-iteration", budget=int(semantic["solver"]["iterations"]), selected_hyperparameters=details["selected_hyperparameters"], selection=details["selection"])
            dgi = motion = None
        elif method.method_id == "monin":
            from gsdiff.baselines.monin import run_monin
            reconstruction, motion, details = run_monin(acquisition, semantic, algorithm_seed, device)
            degree = int(semantic["solver"]["polynomial_degree"])
            info = _base_info(method, parameter_count=acquisition.H * acquisition.W + 2 * (degree + 1), unit="admm-iteration", budget=int(semantic["solver"]["n_admm"]), selected_hyperparameters=details["selected_hyperparameters"], selection=details["selection"], native_motion_model="translation-polynomial")
            dgi = None
        elif method.method_id == "gidc3dtv":
            from gsdiff.baselines.gidc import run_gidc3dtv
            reconstruction, details = run_gidc3dtv(acquisition, semantic, algorithm_seed, device)
            info = _base_info(method, parameter_count=int(details["parameter_count"]), unit="adam-step", budget=int(semantic["solver"]["n_steps"]), selected_hyperparameters=details["selected_hyperparameters"], selection=details["selection"])
            dgi = motion = None
        else:
            from gsdiff.baselines.recinr import run_recinr
            reconstruction, details = run_recinr(acquisition, semantic, algorithm_seed, device)
            solver = semantic["solver"]
            info = _base_info(method, parameter_count=int(details["parameter_count"]), unit="optimization-step", budget=int(solver["warm_steps"] + solver["flow_steps"] + solver["joint_steps"]), selected_hyperparameters=details["selected_hyperparameters"], selection=details["selection"])
            dgi = motion = None
    return MethodChildResult(method_id=method.method_id, reconstruction=np.asarray(reconstruction), estimated_motion_trajectory=None if motion is None else np.asarray(motion), dgi=None if dgi is None else np.asarray(dgi), info=info, history=())
