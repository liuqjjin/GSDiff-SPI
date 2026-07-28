from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from gsdiff.data._artifact_identity import array_descriptor
from gsdiff.data._artifact_models import SPIAcquisitionData
from gsdiff.experiments import child_outputs
from gsdiff.experiments.methods import (
    AlgorithmSeed,
    derive_algorithm_seed,
    resolve_method_semantics,
)


CHECKPOINT = Path("checkpoints/diffusion_prior.pt")
CHECKPOINT_SHA256 = (
    "667948800911acb9f9a7271e20af5692b0f007007d0fc32a15ac169eba32c5dd"
)
GSDIFF_IDS = ("siren", "recinr_se2", "gsdiff_tv", "gsdiff_diffusion")
EXPECTED_PARAMETER_COUNTS = {
    "siren": 33_540,
    "recinr_se2": 16_004,
    "gsdiff_tv": 6_003,
}


@pytest.fixture
def blind_acquisition() -> SPIAcquisitionData:
    rng = np.random.default_rng(19)
    T, H, W, rows = 3, 8, 8, 3
    patterns = rng.random((T * rows, H, W), dtype=np.float32)
    frame_indices = np.repeat(np.arange(T, dtype=np.int64), rows)
    source = rng.random((T, H, W), dtype=np.float32)
    measurements = np.einsum(
        "khw,khw->k", patterns, source[frame_indices]
    ).astype(np.float32)
    holdout_patterns = rng.random((T, H, W), dtype=np.float32)
    holdout_frame_indices = np.arange(T, dtype=np.int64)
    holdout_measurements = np.einsum(
        "khw,khw->k",
        holdout_patterns,
        source[holdout_frame_indices],
    ).astype(np.float32)
    arrays = {
        "patterns": patterns,
        "measurements": measurements,
        "frame_indices": frame_indices,
        "time_grid": np.linspace(0, 1, T, dtype=np.float64),
        "holdout_patterns": holdout_patterns,
        "holdout_measurements": holdout_measurements,
        "holdout_frame_indices": holdout_frame_indices,
    }
    return SPIAcquisitionData(
        dataset_identity_sha256="b" * 64,
        **arrays,
        H=H,
        W=W,
        T=T,
        K=len(patterns),
        holdout_K=len(holdout_patterns),
        acquisition={
            "pattern_family": "bernoulli",
            "pattern_values": [0, 1],
            "pattern_order": "sequential",
            "time_assignment": "uniform",
            "holdout_pattern_family": "bernoulli",
            "noise_convention": "absolute-gaussian-sigma",
            "noise_sigma_absolute": 0.0,
        },
        array_descriptors={
            name: array_descriptor(array)
            for name, array in arrays.items()
        },
    )


def _resolve(
    method_id: str,
    acquisition: SPIAcquisitionData,
    *,
    profile: str = "controller-cpu-smoke-v1",
):
    return resolve_method_semantics(
        method_id,
        method_config_id=(
            "default" if profile == "publication-v1" else "smoke-default-v1"
        ),
        base_config=(
            {"gaussian_count": 1000}
            if method_id.startswith("gsdiff_")
            else {}
        ),
        measurements_metadata={
            "H": acquisition.H,
            "W": acquisition.W,
            "T": acquisition.T,
            "K": acquisition.K,
            "holdout_K": acquisition.holdout_K,
        },
        execution_profile=profile,
    )


def _seed(method, acquisition: SPIAcquisitionData) -> AlgorithmSeed:
    return derive_algorithm_seed(
        cell_seed=23,
        dataset_identity_sha256=acquisition.dataset_identity_sha256,
        method_id=method.method_id,
        method_config_sha256=method.method_config_sha256,
    )


def _checkpoint_paths(method_id: str) -> dict[str, Path]:
    if method_id == "gsdiff_diffusion":
        return {"gsdiff-diffusion-prior-v1": CHECKPOINT}
    return {}


def _numpy_state_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_gsdiff_ids_are_exact_and_strict_signature_has_no_truth_capability():
    from gsdiff.experiments.gsdiff_adapter import (
        GSDIFF_METHOD_IDS,
        run_gsdiff_method,
    )

    assert GSDIFF_METHOD_IDS == GSDIFF_IDS
    signature = inspect.signature(run_gsdiff_method)
    assert not any(
        "truth" in parameter.name.lower()
        or "gt" in parameter.name.lower()
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def test_canonical_dispatch_keeps_gsdiff_adapter_lazy_until_selected():
    probe = (
        "import sys; "
        "import gsdiff.experiments.adapters; "
        "assert 'gsdiff.experiments.gsdiff_adapter' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_strict_gsdiff_import_closure_excludes_evaluator_and_truth_sources(
    tmp_path: Path,
):
    source_root = Path(__file__).resolve().parents[2]
    snapshot = tmp_path / "snapshot"
    shutil.copytree(
        source_root / "gsdiff",
        snapshot / "gsdiff",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (snapshot / "gsdiff" / "baselines" / "_evaluation.py").unlink()
    (snapshot / "gsdiff" / "data" / "_artifact_truth.py").unlink()
    shutil.rmtree(snapshot / "gsdiff" / "evaluation")
    probe = (
        "import sys; "
        "import gsdiff.experiments.gsdiff_adapter; "
        "assert 'gsdiff.evaluation' not in sys.modules; "
        "assert 'gsdiff.baselines._evaluation' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=snapshot,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_train_legacy_evaluation_flow_has_named_compatibility_boundary():
    import train

    assert callable(train._run_legacy_compatibility)
    main_source = inspect.getsource(train.main)
    assert "_run_legacy_compatibility" in main_source
    assert "evaluate(" not in main_source


@pytest.mark.parametrize("method_id", ("siren", "recinr_se2", "gsdiff_tv"))
def test_smoke_runtime_uses_exact_registry_model_binding(
    method_id: str,
    blind_acquisition: SPIAcquisitionData,
):
    from gsdiff.experiments.gsdiff_adapter import _construct_gsdiff_runtime

    method = _resolve(method_id, blind_acquisition)
    runtime = _construct_gsdiff_runtime(
        method,
        blind_acquisition,
        checkpoint_paths={},
        device="cpu",
    )
    assert runtime.motion.enable_rotation is True
    assert runtime.motion.poly_degree == 1
    assert runtime.motion.enable_affine is False

    if method_id == "siren":
        sine_layers = list(runtime.scene.net)
        assert runtime.scene.__class__.__name__ == "SIREN"
        assert len(sine_layers) == 3
        assert all(layer.w0 == 8 for layer in sine_layers)
        assert sine_layers[0].lin.out_features == 128
        assert runtime.solver.__class__.__name__ == "SGDSolver"
    elif method_id == "recinr_se2":
        linear_layers = [
            layer
            for layer in runtime.scene.renderer
            if isinstance(layer, torch.nn.Linear)
        ]
        assert runtime.scene.__class__.__name__ == "ReCINRCanonicalScene"
        assert runtime.scene.C == 32
        assert (runtime.scene.gh, runtime.scene.gw) == (20, 20)
        assert len(linear_layers) == 4
        assert runtime.solver.__class__.__name__ == "SGDSolver"
    else:
        assert runtime.scene.__class__.__name__ == "GaussianScene2D"
        assert runtime.scene.M == 1000
        assert runtime.scene.min_scale == 0.0
        torch.testing.assert_close(
            runtime.scene.get_scales(),
            torch.full_like(runtime.scene.get_scales(), 1.5),
        )
        assert runtime.prior.__class__.__name__ == "TVPrior3D"
        assert runtime.prior.max_iter == 50
        assert runtime.prior.temporal_weight == 0.1
        assert runtime.solver.__class__.__name__ == "ADMMSolver"


def test_publication_tv_binds_independent_rounded_motion_warmup(
    blind_acquisition: SPIAcquisitionData,
):
    from gsdiff.experiments.gsdiff_adapter import _construct_gsdiff_runtime

    method = _resolve(
        "gsdiff_tv",
        blind_acquisition,
        profile="publication-v1",
    )
    solver_config = method.semantic_config["solver"]
    assert solver_config["outer_iterations"] == 80
    assert solver_config["splitting_warmup_outer"] == 20
    assert solver_config["motion_warmup_fraction"] == 0.2
    assert solver_config["motion_warmup_outer"] == 16

    runtime = _construct_gsdiff_runtime(
        method,
        blind_acquisition,
        checkpoint_paths={},
        device="cpu",
    )
    assert runtime.solver.splitting_warmup_outer == 20
    assert runtime.solver.motion_warmup_outer == 16


@pytest.mark.parametrize("method_id", ("siren", "recinr_se2", "gsdiff_tv"))
def test_smoke_run_returns_canonical_truth_free_result(
    method_id: str,
    blind_acquisition: SPIAcquisitionData,
):
    from gsdiff.experiments.gsdiff_adapter import run_gsdiff_method

    method = _resolve(method_id, blind_acquisition)
    imported_before = set(sys.modules)
    result = run_gsdiff_method(
        method,
        blind_acquisition,
        algorithm_seed=_seed(method, blind_acquisition),
        checkpoint_paths={},
        device="cpu",
    )

    assert result.method_id == method_id
    assert result.method_id != "gsdiff-admm"
    assert result.reconstruction.shape == (
        blind_acquisition.T,
        blind_acquisition.H,
        blind_acquisition.W,
    )
    assert np.isfinite(result.reconstruction).all()
    assert result.estimated_motion_trajectory is not None
    assert result.estimated_motion_trajectory.shape == (
        blind_acquisition.T,
        3,
    )
    assert len(result.history) == 1
    assert (
        result.info["parameter_count"]
        == EXPECTED_PARAMETER_COUNTS[method_id]
    )
    forbidden = json.dumps(
        {"info": result.info, "history": result.history},
        sort_keys=True,
    ).lower()
    for token in ("psnr", "ssim", "ground_truth", "gt_"):
        assert token not in forbidden
    child_outputs._validate_result(result, blind_acquisition, method)
    imported_by_run = set(sys.modules) - imported_before
    assert "gsdiff.evaluation" not in imported_by_run
    assert "train" not in imported_by_run


def test_diffusion_construction_loads_exact_checkpoint_and_schedule(
    blind_acquisition: SPIAcquisitionData,
):
    from gsdiff.baselines.common import unique_optimizer_parameter_count
    from gsdiff.experiments.gsdiff_adapter import _construct_gsdiff_runtime

    method = _resolve("gsdiff_diffusion", blind_acquisition)
    assert len(method.checkpoint_requirements) == 1
    requirement = method.checkpoint_requirements[0]
    assert requirement.logical_id == "gsdiff-diffusion-prior-v1"
    assert requirement.sha256 == CHECKPOINT_SHA256
    runtime = _construct_gsdiff_runtime(
        method,
        blind_acquisition,
        checkpoint_paths=_checkpoint_paths(method.method_id),
        device="cpu",
    )

    assert runtime.scene.M == 1000
    assert runtime.prior.__class__.__name__ == "DiffusionPrior"
    assert runtime.prior.denoise_steps == 1
    assert runtime.prior.clamp_range == (0.0, 1.0)
    assert runtime.prior.sigma_start == 0.3
    assert runtime.prior.sigma_end == 0.05
    assert runtime.prior.renoise is False
    assert runtime.prior.ddim_spacing == "linear"
    assert runtime.prior._n_steps == 1
    assert len(runtime.checkpoint_snapshots) == 1
    assert runtime.checkpoint_snapshots[0].sha256 == CHECKPOINT_SHA256
    assert unique_optimizer_parameter_count(runtime.solver.optimizer) == 6_003


@pytest.mark.parametrize(
    ("mapping_factory", "message"),
    [
        (lambda tmp: {}, "missing"),
        (
            lambda tmp: {
                "gsdiff-diffusion-prior-v1": CHECKPOINT,
                "extra": CHECKPOINT,
            },
            "extra",
        ),
        (
            lambda tmp: {
                "gsdiff-diffusion-prior-v1": tmp / "missing.pt"
            },
            "checkpoint",
        ),
        (
            lambda tmp: {
                "gsdiff-diffusion-prior-v1": tmp
            },
            "regular file",
        ),
        (
            lambda tmp: {
                "gsdiff-diffusion-prior-v1": tmp / "wrong.pt"
            },
            "sha256",
        ),
    ],
)
def test_checkpoint_contract_fails_before_runtime_construction(
    mapping_factory,
    message: str,
    tmp_path: Path,
    blind_acquisition: SPIAcquisitionData,
    monkeypatch: pytest.MonkeyPatch,
):
    from gsdiff.experiments import gsdiff_adapter

    (tmp_path / "wrong.pt").write_bytes(b"not the declared checkpoint")
    method = _resolve("gsdiff_diffusion", blind_acquisition)

    def forbidden_construction(*args, **kwargs):
        raise AssertionError("runtime construction happened before validation")

    monkeypatch.setattr(
        gsdiff_adapter,
        "_construct_validated_runtime",
        forbidden_construction,
    )
    with pytest.raises(ValueError, match=message):
        gsdiff_adapter._construct_gsdiff_runtime(
            method,
            blind_acquisition,
            checkpoint_paths=mapping_factory(tmp_path),
            device="cpu",
        )


def test_checkpoint_symlink_is_rejected(
    tmp_path: Path,
    blind_acquisition: SPIAcquisitionData,
):
    from gsdiff.experiments.gsdiff_adapter import _construct_gsdiff_runtime

    link = tmp_path / "linked.pt"
    try:
        link.symlink_to(CHECKPOINT.resolve())
    except OSError as error:
        pytest.skip(f"file symlinks unavailable: {error}")
    method = _resolve("gsdiff_diffusion", blind_acquisition)
    with pytest.raises(ValueError, match="linked|reparse"):
        _construct_gsdiff_runtime(
            method,
            blind_acquisition,
            checkpoint_paths={"gsdiff-diffusion-prior-v1": link},
            device="cpu",
        )


def test_publication_diffusion_stays_blocked_with_correct_local_checkpoint(
    blind_acquisition: SPIAcquisitionData,
    monkeypatch: pytest.MonkeyPatch,
):
    from gsdiff.experiments import gsdiff_adapter

    method = _resolve(
        "gsdiff_diffusion",
        blind_acquisition,
        profile="publication-v1",
    )

    def forbidden_construction(*args, **kwargs):
        raise AssertionError("blocked method constructed a runtime")

    monkeypatch.setattr(
        gsdiff_adapter,
        "_construct_validated_runtime",
        forbidden_construction,
    )
    with pytest.raises(
        ValueError,
        match=(
            "missing-reproducible-checkpoint-locator.*"
            "missing-checkpoint-training-provenance"
        ),
    ):
        gsdiff_adapter._construct_gsdiff_runtime(
            method,
            blind_acquisition,
            checkpoint_paths=_checkpoint_paths(method.method_id),
            device="cpu",
        )


def test_gsdiff_run_is_seeded_and_restores_caller_rng(
    blind_acquisition: SPIAcquisitionData,
):
    from gsdiff.experiments.gsdiff_adapter import run_gsdiff_method

    method = _resolve("gsdiff_tv", blind_acquisition)
    algorithm_seed = _seed(method, blind_acquisition)
    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )

    first = run_gsdiff_method(
        method,
        blind_acquisition,
        algorithm_seed=algorithm_seed,
        checkpoint_paths={},
        device="cpu",
    )
    second = run_gsdiff_method(
        method,
        blind_acquisition,
        algorithm_seed=algorithm_seed,
        checkpoint_paths={},
        device="cpu",
    )

    assert np.array_equal(first.reconstruction, second.reconstruction)
    assert random.getstate() == python_state
    assert _numpy_state_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    if cuda_states is not None:
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(
                torch.cuda.get_rng_state_all(), cuda_states, strict=True
            )
        )


def test_gsdiff_physical_calibration_is_train_only_and_scale_equivariant(
    blind_acquisition: SPIAcquisitionData,
):
    from gsdiff.experiments.gsdiff_adapter import run_gsdiff_method

    method = _resolve("gsdiff_tv", blind_acquisition)
    algorithm_seed = _seed(method, blind_acquisition)
    assert blind_acquisition.holdout_measurements is not None
    scaled = replace(
        blind_acquisition,
        measurements=np.asarray(
            blind_acquisition.measurements * 5.0, dtype=np.float32
        ),
        holdout_measurements=np.asarray(
            blind_acquisition.holdout_measurements * 5.0,
            dtype=np.float32,
        ),
    )
    changed_holdout = replace(
        blind_acquisition,
        holdout_measurements=np.asarray(
            blind_acquisition.holdout_measurements * 17.0,
            dtype=np.float32,
        ),
    )

    original = run_gsdiff_method(
        method,
        blind_acquisition,
        algorithm_seed=algorithm_seed,
        checkpoint_paths={},
        device="cpu",
    )
    amplified = run_gsdiff_method(
        method,
        scaled,
        algorithm_seed=algorithm_seed,
        checkpoint_paths={},
        device="cpu",
    )
    holdout_changed = run_gsdiff_method(
        method,
        changed_holdout,
        algorithm_seed=algorithm_seed,
        checkpoint_paths={},
        device="cpu",
    )

    assert np.allclose(
        amplified.reconstruction,
        5.0 * original.reconstruction,
        rtol=2e-3,
        atol=2e-3,
    )
    assert np.array_equal(
        holdout_changed.reconstruction, original.reconstruction
    )


def test_canonical_dispatch_keeps_baseline_and_gsdiff_families_distinct(
    blind_acquisition: SPIAcquisitionData,
):
    from gsdiff.experiments.adapters import run_canonical_method

    gsdiff = _resolve("gsdiff_tv", blind_acquisition)
    result = run_canonical_method(
        gsdiff,
        blind_acquisition,
        algorithm_seed=_seed(gsdiff, blind_acquisition),
        checkpoint_paths={},
        device="cpu",
    )
    assert result.method_id == "gsdiff_tv"

    baseline = _resolve("dgi", blind_acquisition)
    baseline_result = run_canonical_method(
        baseline,
        blind_acquisition,
        algorithm_seed=_seed(baseline, blind_acquisition),
        checkpoint_paths={},
        device="cpu",
    )
    assert baseline_result.method_id == "dgi"

    with pytest.raises(ValueError, match="checkpoint"):
        run_canonical_method(
            baseline,
            blind_acquisition,
            algorithm_seed=_seed(baseline, blind_acquisition),
            checkpoint_paths={"unexpected": CHECKPOINT},
            device="cpu",
        )
