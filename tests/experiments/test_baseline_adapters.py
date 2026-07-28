from __future__ import annotations

import json
import inspect
import random
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from gsdiff.data._artifact_identity import array_descriptor
from gsdiff.data._artifact_models import SPIAcquisitionData
from gsdiff.experiments.methods import AlgorithmSeed, derive_algorithm_seed, resolve_method_semantics

from gsdiff.experiments.adapters import BASELINE_METHOD_IDS, run_baseline_method
import gsdiff.experiments.child_outputs as child_outputs
import gsdiff.baselines.cs as cs
import gsdiff.baselines.tv3d as tv3d


@pytest.fixture
def blind_acquisition() -> SPIAcquisitionData:
    rng = np.random.default_rng(7)
    T, H, W, rows = 5, 16, 16, 2
    patterns = rng.random((T * rows, H, W), dtype=np.float32)
    frame_indices = np.repeat(np.arange(T, dtype=np.int64), rows)
    source = rng.random((T, H, W), dtype=np.float32)
    measurements = np.einsum("khw,khw->k", patterns, source[frame_indices])
    holdout_patterns = rng.random((T, H, W), dtype=np.float32)
    holdout_frame_indices = np.arange(T, dtype=np.int64)
    holdout_measurements = np.einsum(
        "khw,khw->k", holdout_patterns, source[holdout_frame_indices]
    )
    arrays = {
        "patterns": patterns,
        "measurements": measurements.astype(np.float32),
        "frame_indices": frame_indices,
        "time_grid": np.linspace(0, 1, T, dtype=np.float64),
        "holdout_patterns": holdout_patterns,
        "holdout_measurements": holdout_measurements.astype(np.float32),
        "holdout_frame_indices": holdout_frame_indices,
    }
    return SPIAcquisitionData(
        dataset_identity_sha256="a" * 64,
        **arrays,
        H=H,
        W=W,
        T=T,
        K=patterns.shape[0],
        holdout_K=holdout_patterns.shape[0],
        acquisition={
            "pattern_family": "bernoulli",
            "pattern_values": [0, 1],
            "pattern_order": "sequential",
            "time_assignment": "uniform",
            "holdout_pattern_family": "bernoulli",
            "noise_convention": "absolute-gaussian-sigma",
            "noise_sigma_absolute": 0.0,
        },
        array_descriptors={name: array_descriptor(value) for name, value in arrays.items()},
    )


def resolve_smoke(method_id: str, data: SPIAcquisitionData):
    return resolve_method_semantics(
        method_id,
        method_config_id="smoke-default-v1",
        base_config={},
        measurements_metadata={"H": data.H, "W": data.W, "T": data.T, "K": data.K, "holdout_K": data.holdout_K},
        execution_profile="controller-cpu-smoke-v1",
    )


def derive_for(method, data: SPIAcquisitionData):
    return derive_algorithm_seed(
        cell_seed=11,
        dataset_identity_sha256=data.dataset_identity_sha256,
        method_id=method.method_id,
        method_config_sha256=method.method_config_sha256,
    )


@pytest.mark.parametrize("method_id", BASELINE_METHOD_IDS)
def test_smoke_baseline_accepts_blind_acquisition_only(method_id: str, blind_acquisition: SPIAcquisitionData) -> None:
    method = resolve_smoke(method_id, blind_acquisition)
    result = run_baseline_method(method, blind_acquisition, algorithm_seed=derive_for(method, blind_acquisition), device="cpu")
    assert result.method_id == method_id
    assert result.reconstruction.shape == (blind_acquisition.T, blind_acquisition.H, blind_acquisition.W)
    assert np.isfinite(result.reconstruction).all()
    forbidden = json.dumps(result.info, sort_keys=True).lower()
    assert "psnr" not in forbidden
    assert "ssim" not in forbidden
    assert "ground_truth" not in forbidden
    assert "gt_" not in forbidden
    child_outputs._validate_result(result, blind_acquisition, method)
    assert "gsdiff.evaluation" not in sys.modules
    assert "gsdiff.baselines._evaluation" not in sys.modules


def test_static_cs_selects_on_train_then_refits_all_measurements(blind_acquisition: SPIAcquisitionData, monkeypatch: pytest.MonkeyPatch) -> None:
    observed_rows: list[int] = []
    real = cs.admm_tv

    def recording_admm(A, y, H, W, lam, **kwargs):
        observed_rows.append(int(A.shape[0]))
        return real(A, y, H, W, lam, **kwargs)

    monkeypatch.setattr(cs, "admm_tv", recording_admm)
    method = resolve_smoke("static_cs", blind_acquisition)
    run_baseline_method(method, blind_acquisition, algorithm_seed=derive_for(method, blind_acquisition), device="cpu")
    assert observed_rows[:-1] == [blind_acquisition.K]
    assert observed_rows[-1] == blind_acquisition.K + blind_acquisition.holdout_K


def test_tv3d_refit_constructs_an_all_measurement_operator(blind_acquisition: SPIAcquisitionData, monkeypatch: pytest.MonkeyPatch) -> None:
    observed_rows: list[int] = []
    real = tv3d._chambolle_pock

    def recording_pock(op, *args):
        observed_rows.append(op.M)
        return real(op, *args)

    monkeypatch.setattr(tv3d, "_chambolle_pock", recording_pock)
    method = resolve_smoke("tv3d", blind_acquisition)
    run_baseline_method(method, blind_acquisition, algorithm_seed=derive_for(method, blind_acquisition), device="cpu")
    assert observed_rows[:-1] == [blind_acquisition.K]
    assert observed_rows[-1] == blind_acquisition.K + blind_acquisition.holdout_K


def test_only_monin_reports_native_translation_motion(blind_acquisition: SPIAcquisitionData) -> None:
    monin_method = resolve_smoke("monin", blind_acquisition)
    result = run_baseline_method(monin_method, blind_acquisition, algorithm_seed=derive_for(monin_method, blind_acquisition), device="cpu")
    assert result.estimated_motion_trajectory is not None
    assert result.estimated_motion_trajectory.shape == (blind_acquisition.T, 3)
    assert np.isfinite(result.estimated_motion_trajectory).all()
    assert np.all(result.estimated_motion_trajectory[:, 2] == 0)
    for method_id in set(BASELINE_METHOD_IDS) - {"monin"}:
        method = resolve_smoke(method_id, blind_acquisition)
        result = run_baseline_method(method, blind_acquisition, algorithm_seed=derive_for(method, blind_acquisition), device="cpu")
        assert result.estimated_motion_trajectory is None


def test_seeded_stochastic_baseline_is_deterministic_and_restores_rng(blind_acquisition: SPIAcquisitionData) -> None:
    method = resolve_smoke("gidc3dtv", blind_acquisition)
    seed = derive_for(method, blind_acquisition)
    random.seed(99); np.random.seed(99); torch.manual_seed(99)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    first = run_baseline_method(method, blind_acquisition, algorithm_seed=seed, device="cpu")
    second = run_baseline_method(method, blind_acquisition, algorithm_seed=seed, device="cpu")
    assert np.array_equal(first.reconstruction, second.reconstruction)
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.random.get_rng_state(), torch_state)


@pytest.mark.parametrize("method_id", ("gidc3dtv", "recinr"))
def test_stochastic_initialization_changes_with_algorithm_seed(method_id: str, blind_acquisition: SPIAcquisitionData) -> None:
    method = resolve_smoke(method_id, blind_acquisition)
    seed = derive_for(method, blind_acquisition)
    changed = AlgorithmSeed(seed.derivation_sha256, (seed.seed_u32 + 1) % 2**32)
    first = run_baseline_method(method, blind_acquisition, algorithm_seed=seed, device="cpu")
    second = run_baseline_method(method, blind_acquisition, algorithm_seed=changed, device="cpu")
    assert not np.array_equal(first.reconstruction, second.reconstruction)


def test_strict_core_signatures_expose_no_capability_escape_hatches() -> None:
    from gsdiff.baselines.gidc import run_gidc3dtv
    from gsdiff.baselines.monin import run_monin
    from gsdiff.baselines.recinr import run_recinr
    from gsdiff.experiments.adapters import run_dgi
    for function in (run_dgi, cs.run_static_cs, cs.run_perframe_cs, tv3d.run_tv3d, run_gidc3dtv, run_monin, run_recinr):
        signature = inspect.signature(function)
        assert not any("truth" in parameter.name.lower() or "gt" in parameter.name.lower() or parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())


def test_strict_import_closure_survives_without_evaluator_or_truth_sources(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[2]
    snapshot = tmp_path / "snapshot"
    shutil.copytree(source_root / "gsdiff", snapshot / "gsdiff", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (snapshot / "gsdiff" / "baselines" / "_evaluation.py").unlink()
    (snapshot / "gsdiff" / "data" / "_artifact_truth.py").unlink()
    shutil.rmtree(snapshot / "gsdiff" / "evaluation")
    probe = "import sys; import gsdiff.experiments.adapters; import gsdiff.baselines.cs, gsdiff.baselines.gidc, gsdiff.baselines.monin, gsdiff.baselines.tv3d; assert 'gsdiff.baselines._evaluation' not in sys.modules"
    completed = subprocess.run([sys.executable, "-c", probe], cwd=snapshot, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
