from __future__ import annotations

from dataclasses import replace
import json
import inspect
import os
import random
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from gsdiff.data._artifact_dataset import save_acquisition_data
from gsdiff.data._artifact_identity import array_descriptor
from gsdiff.data._artifact_models import SPIAcquisitionData
from gsdiff.data.simulation import SPIData
from gsdiff.experiments.methods import AlgorithmSeed, derive_algorithm_seed, resolve_method_semantics, thaw_json
from gsdiff.experiments.objectives import heldout_normalized_l2

from gsdiff.experiments.adapters import BASELINE_METHOD_IDS, run_baseline_method
import gsdiff.experiments.child_outputs as child_outputs
import gsdiff.baselines.cs as cs
import gsdiff.baselines.tv3d as tv3d


def _replace_arrays(data: SPIAcquisitionData, **changes) -> SPIAcquisitionData:
    return replace(data, **changes)


def _scaled_measurements(data: SPIAcquisitionData, factor: float) -> SPIAcquisitionData:
    assert data.holdout_measurements is not None
    return _replace_arrays(
        data,
        measurements=np.asarray(data.measurements * factor, dtype=np.float32),
        holdout_measurements=np.asarray(data.holdout_measurements * factor, dtype=np.float32),
    )


def _legacy_data(data: SPIAcquisitionData) -> SPIData:
    assert data.holdout_patterns is not None
    assert data.holdout_measurements is not None
    assert data.holdout_frame_indices is not None
    return SPIData(
        canonical=np.zeros((data.H, data.W), dtype=np.float32),
        gt_frames=np.zeros((data.T, data.H, data.W), dtype=np.float32),
        patterns=np.array(data.patterns, copy=True), measurements=np.array(data.measurements, copy=True),
        frame_idx=np.array(data.frame_indices, copy=True), t_grid=np.array(data.time_grid, copy=True),
        gt_velocity=np.zeros(2), gt_omega=0.0, motion_type="translation",
        H=data.H, W=data.W, T=data.T, K=data.K, snr_db=40.0,
        eval_patterns=np.array(data.holdout_patterns, copy=True),
        eval_measurements=np.array(data.holdout_measurements, copy=True),
        eval_frame_idx=np.array(data.holdout_frame_indices, copy=True),
    )


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

    def recording_pock(op, *args, **kwargs):
        observed_rows.append(op.M)
        return real(op, *args, **kwargs)

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
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    first = run_baseline_method(method, blind_acquisition, algorithm_seed=seed, device="cpu")
    second = run_baseline_method(method, blind_acquisition, algorithm_seed=seed, device="cpu")
    assert np.array_equal(first.reconstruction, second.reconstruction)
    assert random.getstate() == python_state
    _assert_numpy_state_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    if cuda_states is not None:
        assert all(torch.equal(actual, expected) for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_states))


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


def test_strict_runtime_import_closure_executes_all_baselines_without_evaluator_or_truth_sources(
    tmp_path: Path,
    blind_acquisition: SPIAcquisitionData,
) -> None:
    source_root = Path(__file__).resolve().parents[2]
    snapshot = tmp_path / "snapshot"
    shutil.copytree(source_root / "gsdiff", snapshot / "gsdiff", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    registry_path = snapshot / "configs" / "protocols" / "methods-v1.yaml"
    registry_path.parent.mkdir(parents=True)
    shutil.copy2(
        source_root / "configs" / "protocols" / "methods-v1.yaml",
        registry_path,
    )
    (snapshot / "gsdiff" / "baselines" / "_evaluation.py").unlink()
    (snapshot / "gsdiff" / "data" / "_artifact_truth.py").unlink()
    shutil.rmtree(snapshot / "gsdiff" / "evaluation")
    acquisition_path = tmp_path / "blind-acquisition.npz"
    serializable_acquisition = replace(
        blind_acquisition,
        acquisition={
            **dict(blind_acquisition.acquisition),
            "pattern_family": "random",
            "holdout_pattern_family": "uniform-random",
            "noise_convention": "detector-absolute",
        },
    )
    save_acquisition_data(serializable_acquisition, acquisition_path)
    probe = textwrap.dedent(
        """\
        import sys
        from pathlib import Path

        import gsdiff
        from gsdiff.data._artifact_dataset import load_acquisition_data
        from gsdiff.experiments.adapters import (
            BASELINE_METHOD_IDS,
            run_baseline_method,
        )
        from gsdiff.experiments.methods import (
            derive_algorithm_seed,
            resolve_method_semantics,
        )

        EXPECTED_METHOD_IDS = (
            "dgi",
            "static_cs",
            "perframe_cs",
            "tv3d",
            "monin",
            "gidc3dtv",
            "recinr",
        )
        FORBIDDEN_PREFIXES = (
            "gsdiff.evaluation",
            "gsdiff.baselines._evaluation",
            "gsdiff.data._artifact_truth",
        )

        assert Path(gsdiff.__file__).resolve().parent.parent == Path.cwd().resolve()
        assert tuple(BASELINE_METHOD_IDS) == EXPECTED_METHOD_IDS
        acquisition = load_acquisition_data(
            Path(sys.argv[1]),
            expected_dataset_identity_sha256="a" * 64,
        )
        for method_id in EXPECTED_METHOD_IDS:
            method = resolve_method_semantics(
                method_id,
                method_config_id="smoke-default-v1",
                base_config={},
                measurements_metadata={
                    "H": acquisition.H,
                    "W": acquisition.W,
                    "T": acquisition.T,
                    "K": acquisition.K,
                    "holdout_K": acquisition.holdout_K,
                },
                execution_profile="controller-cpu-smoke-v1",
            )
            algorithm_seed = derive_algorithm_seed(
                cell_seed=11,
                dataset_identity_sha256=acquisition.dataset_identity_sha256,
                method_id=method.method_id,
                method_config_sha256=method.method_config_sha256,
            )
            result = run_baseline_method(
                method,
                acquisition,
                algorithm_seed=algorithm_seed,
                device="cpu",
            )
            assert result.method_id == method_id
            loaded_forbidden = sorted(
                name
                for name in sys.modules
                if any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in FORBIDDEN_PREFIXES
                )
            )
            assert not loaded_forbidden, (
                f"{method_id} loaded forbidden modules: {loaded_forbidden}"
            )
        """
    )
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = str(snapshot)
    child_env["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(acquisition_path)],
        cwd=snapshot,
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


@pytest.mark.parametrize("method_id", ("dgi", "gidc3dtv", "recinr"))
def test_conditioned_baselines_are_amplitude_equivariant_on_physical_scale(method_id: str, blind_acquisition: SPIAcquisitionData) -> None:
    method = resolve_smoke(method_id, blind_acquisition)
    seed = derive_for(method, blind_acquisition)
    scaled = _scaled_measurements(blind_acquisition, 7.0)
    original = run_baseline_method(method, blind_acquisition, algorithm_seed=seed, device="cpu")
    amplified = run_baseline_method(method, scaled, algorithm_seed=seed, device="cpu")
    assert np.allclose(amplified.reconstruction, 7.0 * original.reconstruction, rtol=2e-3, atol=2e-3)
    assert blind_acquisition.holdout_patterns is not None
    assert blind_acquisition.holdout_measurements is not None
    assert blind_acquisition.holdout_frame_indices is not None
    base_score = heldout_normalized_l2(original.reconstruction, blind_acquisition.holdout_patterns, blind_acquisition.holdout_measurements, blind_acquisition.holdout_frame_indices)
    scaled_score = heldout_normalized_l2(amplified.reconstruction, scaled.holdout_patterns, scaled.holdout_measurements, scaled.holdout_frame_indices)
    unscaled_score = heldout_normalized_l2(original.reconstruction, scaled.holdout_patterns, scaled.holdout_measurements, scaled.holdout_frame_indices)
    assert scaled_score.value == pytest.approx(base_score.value, rel=2e-3, abs=2e-3)
    assert scaled_score.value < unscaled_score.value


@pytest.mark.parametrize("method_id", ("dgi", "gidc3dtv", "recinr"))
def test_physical_calibration_never_uses_holdout_measurements(method_id: str, blind_acquisition: SPIAcquisitionData) -> None:
    method = resolve_smoke(method_id, blind_acquisition)
    seed = derive_for(method, blind_acquisition)
    assert blind_acquisition.holdout_measurements is not None
    changed_holdout = _replace_arrays(blind_acquisition, holdout_measurements=np.asarray(blind_acquisition.holdout_measurements * 11.0, dtype=np.float32))
    original = run_baseline_method(method, blind_acquisition, algorithm_seed=seed, device="cpu")
    changed = run_baseline_method(method, changed_holdout, algorithm_seed=seed, device="cpu")
    assert np.array_equal(changed.reconstruction, original.reconstruction)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"holdout_patterns": None, "holdout_measurements": None, "holdout_frame_indices": None, "holdout_K": 0}, "holdout"),
        ({"holdout_patterns": np.empty((0, 16, 16), np.float32), "holdout_measurements": np.empty(0, np.float32), "holdout_frame_indices": np.empty(0, np.int64), "holdout_K": 0}, "holdout"),
        ({"holdout_K": 999}, "holdout_K"),
        ({"holdout_frame_indices": np.array([0, 1, 2, 3, 99], dtype=np.int64)}, "frame"),
        ({"holdout_patterns": np.ones((5, 15, 16), dtype=np.float32)}, "shape"),
        ({"holdout_measurements": np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=np.float32)}, "finite"),
        ({"K": 999}, "K"),
    ],
)
def test_shared_validator_rejects_invalid_acquisition_before_solver(changes: dict[str, object], message: str, blind_acquisition: SPIAcquisitionData) -> None:
    method = resolve_smoke("static_cs", blind_acquisition)
    invalid = _replace_arrays(blind_acquisition, **changes)
    with pytest.raises((TypeError, ValueError), match=message):
        run_baseline_method(method, invalid, algorithm_seed=derive_for(method, invalid), device="cpu")


def test_shared_validator_rejects_copied_training_triples_as_holdout(blind_acquisition: SPIAcquisitionData) -> None:
    rows = np.arange(blind_acquisition.holdout_K)
    reused = _replace_arrays(
        blind_acquisition,
        holdout_patterns=np.array(blind_acquisition.patterns[rows], copy=True),
        holdout_measurements=np.array(blind_acquisition.measurements[rows], copy=True),
        holdout_frame_indices=np.array(blind_acquisition.frame_indices[rows], copy=True),
    )
    method = resolve_smoke("static_cs", reused)
    with pytest.raises(ValueError, match="distinct|reuse"):
        run_baseline_method(method, reused, algorithm_seed=derive_for(method, reused), device="cpu")


def test_perframe_cs_partitions_train_and_all_measurements_per_frame(blind_acquisition: SPIAcquisitionData, monkeypatch: pytest.MonkeyPatch) -> None:
    observed_rows: list[int] = []
    real = cs.admm_tv

    def recording_admm(A, y, H, W, lam, **kwargs):
        observed_rows.append(int(A.shape[0]))
        return real(A, y, H, W, lam, **kwargs)

    monkeypatch.setattr(cs, "admm_tv", recording_admm)
    method = resolve_smoke("perframe_cs", blind_acquisition)
    run_baseline_method(method, blind_acquisition, algorithm_seed=derive_for(method, blind_acquisition), device="cpu")
    assert observed_rows[:blind_acquisition.T] == [2] * blind_acquisition.T
    assert observed_rows[blind_acquisition.T:] == [3] * blind_acquisition.T


def test_legacy_cs_and_tv3d_wrappers_accept_real_spidata(blind_acquisition: SPIAcquisitionData) -> None:
    legacy = _legacy_data(blind_acquisition)
    static, _ = cs.static_tvcs(legacy, device="cpu", n_admm=1, lam_grid=[1e-3])
    perframe, _ = cs.perframe_tvcs(legacy, device="cpu", n_admm=1, lam_grid=[1e-3])
    video, _ = tv3d.tv3d(legacy, device="cpu", iters=1, lam_xy_grid=[3e-3], lam_t_grid=[1e-3])
    assert static.shape == perframe.shape == video.shape == (legacy.T, legacy.H, legacy.W)


def test_smoke_histories_report_real_native_observation_counts(blind_acquisition: SPIAcquisitionData) -> None:
    expected = {"dgi": 0, "static_cs": 1, "perframe_cs": 1, "tv3d": 1, "monin": 1, "gidc3dtv": 1, "recinr": 3}
    for method_id, count in expected.items():
        method = resolve_smoke(method_id, blind_acquisition)
        result = run_baseline_method(method, blind_acquisition, algorithm_seed=derive_for(method, blind_acquisition), device="cpu")
        assert len(result.history) == count


@pytest.mark.parametrize("method_id", ("static_cs", "perframe_cs", "tv3d", "monin", "gidc3dtv", "recinr"))
def test_convergence_result_with_21_real_steps_writes_method_info(method_id: str, blind_acquisition: SPIAcquisitionData, tmp_path: Path) -> None:
    smoke = resolve_smoke(method_id, blind_acquisition)
    semantic = thaw_json(smoke.semantic_config)
    solver = semantic["solver"]
    if method_id in {"static_cs", "perframe_cs", "monin"}:
        solver["n_admm"] = 21
    elif method_id == "tv3d":
        solver["iterations"] = 21
    elif method_id == "gidc3dtv":
        solver["n_steps"] = 21
    else:
        solver["warm_steps"], solver["flow_steps"], solver["joint_steps"] = 1, 1, 19
    method = replace(smoke, semantic_config=semantic, convergence_status="convergence-required")
    seed = derive_for(method, blind_acquisition)
    result = run_baseline_method(method, blind_acquisition, algorithm_seed=seed, device="cpu")
    assert len(result.history) >= 21
    child_outputs.write_method_child_outputs_v2(
        tmp_path / method_id, method=method, acquisition=blind_acquisition,
        measurements_file_sha256="b" * 64, algorithm_seed=seed, result=result,
        child_started_at_utc="2026-07-28T00:00:00Z", child_finished_at_utc="2026-07-28T00:00:01Z",
    )
    info = json.loads((tmp_path / method_id / "method-info.json").read_text(encoding="utf-8"))
    assert info["convergence"]["observed_count"] == len(result.history)
    assert info["convergence"]["serialized_count"] == 21


def test_adapter_uses_exact_lazy_runner_table() -> None:
    import gsdiff.experiments.adapters as adapters
    assert tuple(adapters._RUNNER_TABLE) == BASELINE_METHOD_IDS


def _assert_numpy_state_equal(actual, expected) -> None:
    assert actual[0] == expected[0]
    assert np.array_equal(actual[1], expected[1])
    assert actual[2:] == expected[2:]


def test_rng_state_is_restored_when_runner_raises(blind_acquisition: SPIAcquisitionData, monkeypatch: pytest.MonkeyPatch) -> None:
    import gsdiff.experiments.adapters as adapters
    method = resolve_smoke("dgi", blind_acquisition)
    random.seed(17); np.random.seed(17); torch.manual_seed(17)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def fail(*args):
        raise RuntimeError("solver failed")

    monkeypatch.setitem(adapters._RUNNER_TABLE, "dgi", fail)
    with pytest.raises(RuntimeError, match="solver failed"):
        run_baseline_method(method, blind_acquisition, algorithm_seed=derive_for(method, blind_acquisition), device="cpu")
    assert random.getstate() == python_state
    _assert_numpy_state_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    if cuda_states is not None:
        assert all(torch.equal(actual, expected) for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_states))


def test_gidc_one_step_snapshot_is_the_post_update_state(blind_acquisition: SPIAcquisitionData, monkeypatch: pytest.MonkeyPatch) -> None:
    import gsdiff.baselines.gidc as gidc
    from gsdiff.baselines.common import calibrate_reconstruction_physical
    observed: list[np.ndarray] = []
    real_forward = gidc.GIDCUNet2D.forward

    def recording_forward(self, value):
        output = real_forward(self, value)
        observed.append(output[:, 0].detach().cpu().numpy().copy())
        return output

    monkeypatch.setattr(gidc.GIDCUNet2D, "forward", recording_forward)
    method = resolve_smoke("gidc3dtv", blind_acquisition)
    result = run_baseline_method(method, blind_acquisition, algorithm_seed=derive_for(method, blind_acquisition), device="cpu")
    assert len(observed) == 2
    expected = calibrate_reconstruction_physical(
        observed[-1], blind_acquisition.patterns, blind_acquisition.measurements,
        blind_acquisition.frame_indices,
    )
    stale = calibrate_reconstruction_physical(
        observed[0], blind_acquisition.patterns, blind_acquisition.measurements,
        blind_acquisition.frame_indices,
    )
    assert np.array_equal(result.reconstruction, expected)
    assert not np.array_equal(result.reconstruction, stale)


def test_gidc_parameter_count_is_unique_optimizer_owned_count(blind_acquisition: SPIAcquisitionData) -> None:
    import gsdiff.baselines.gidc as gidc
    method = resolve_smoke("gidc3dtv", blind_acquisition)
    network = gidc.GIDCUNet2D(
        in_channels=2, channels=method.semantic_config["solver"]["unet_channels"]
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=0.05)
    expected = sum(
        parameter.numel()
        for parameter in {
            id(parameter): parameter
            for group in optimizer.param_groups for parameter in group["params"]
            if parameter.requires_grad
        }.values()
    )
    result = run_baseline_method(
        method, blind_acquisition,
        algorithm_seed=derive_for(method, blind_acquisition), device="cpu",
    )
    assert result.info["parameter_count"] == expected


def test_recinr_parameter_count_is_unique_optimizer_owned_count(blind_acquisition: SPIAcquisitionData) -> None:
    import gsdiff.baselines.recinr as recinr
    method = resolve_smoke("recinr", blind_acquisition)
    representation = method.semantic_config["representation"]
    solver = method.semantic_config["solver"]
    config = recinr._paper_cfg(
        blind_acquisition.H, blind_acquisition.W, blind_acquisition.T, seed=1,
        hidden=int(representation["hidden_dim"]),
        render_layers=int(representation["render_layers"]),
        warp_arch=str(representation["basis"]),
        warp_order=int(representation["basis_order"]),
        warp_t_harmonics=int(representation["harmonics"]),
        flow_scale=float(representation["flow_scale"]),
        pe_xy=int(representation["position_encoding_space"]),
        pe_t=int(representation["position_encoding_time"]),
        pe_anneal_frac=float(solver["anneal_fraction"]),
        anchor_tau=float(solver["anchor_tau"]),
        warm_epochs=int(solver["warm_steps"]),
        flow_only_epochs=int(solver["flow_steps"]),
        epochs=int(solver["joint_steps"]),
        lr0=float(solver["lr_start"]), lr1=float(solver["lr_end"]),
        lam_flow_t=float(solver["lam_flow_t"]),
        lam_flow_xy=float(solver["lam_flow_xy"]),
        lam_l1=float(solver["lam_l1"]), tv_xy=float(solver["tv_xy"]),
        lam_tv_canon=float(solver["lam_tv_canon"]),
        lam_ttv=float(solver["lam_ttv"]),
    )
    config.K = round(1.7 * blind_acquisition.T)
    network = recinr.ReCINR(config)
    expected = sum(parameter.numel() for parameter in {
        id(parameter): parameter for parameter in network.parameters()
        if parameter.requires_grad
    }.values())
    result = run_baseline_method(
        method, blind_acquisition,
        algorithm_seed=derive_for(method, blind_acquisition), device="cpu",
    )
    assert result.info["parameter_count"] == expected
