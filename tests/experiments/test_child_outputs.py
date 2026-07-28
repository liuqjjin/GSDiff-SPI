from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gsdiff.data._artifact_identity import array_descriptor
from gsdiff.data._artifact_models import SPIAcquisitionData
from gsdiff.experiments.child_outputs import (
    MethodChildResult,
    load_reconstruction_v2,
    validate_method_child_outputs_v2,
    write_method_child_outputs_v2,
)
from gsdiff.experiments.methods import derive_algorithm_seed, resolve_method_semantics


def acquisition() -> SPIAcquisitionData:
    arrays = {
        "patterns": np.ones((4, 32, 32), dtype=np.float32),
        "measurements": np.ones(4, dtype=np.float32),
        "frame_indices": np.arange(4, dtype=np.int64),
        "time_grid": np.arange(4, dtype=np.float64),
    }
    return SPIAcquisitionData(
        dataset_identity_sha256="b" * 64, **arrays, holdout_patterns=None,
        holdout_measurements=None, holdout_frame_indices=None, H=32, W=32,
        T=4, K=4, holdout_K=0,
        acquisition={"pattern_family": "bernoulli", "pattern_values": [0, 1], "pattern_order": "sequential", "time_assignment": "uniform", "holdout_pattern_family": "bernoulli", "noise_convention": "absolute-gaussian-sigma", "noise_sigma_absolute": 1.0},
        array_descriptors={name: array_descriptor(value) for name, value in arrays.items()},
    )


def resolved_dgi():
    return resolve_method_semantics("dgi", method_config_id="default", base_config={}, measurements_metadata={}, execution_profile="publication-v1")


def result() -> MethodChildResult:
    return MethodChildResult(
        method_id="dgi", reconstruction=np.ones((4, 32, 32), dtype=np.float32),
        estimated_motion_trajectory=None, dgi=np.ones((32, 32), dtype=np.float32),
        info={"parameter_count": 0, "native_iteration_unit": "pass", "native_iteration_budget": 1, "convergence_status": "not-applicable", "selected_hyperparameters": None, "selection": None, "checkpoint_hashes": []}, history=(),
    )


def test_v2_writer_owns_exactly_two_files_and_optional_arrays(tmp_path: Path) -> None:
    data = acquisition()
    method = resolved_dgi()
    seed = derive_algorithm_seed(cell_seed=1, dataset_identity_sha256=data.dataset_identity_sha256, method_id="dgi", method_config_sha256=method.method_config_sha256)
    hashes = write_method_child_outputs_v2(tmp_path, method=method, acquisition=data, measurements_file_sha256="a" * 64, algorithm_seed=seed, result=result(), child_started_at_utc="2026-07-28T00:00:00Z", child_finished_at_utc="2026-07-28T00:00:01Z")
    assert set(hashes) == {"reconstruction.npz", "method-info.json"}
    assert {path.name for path in tmp_path.iterdir()} == set(hashes)
    assert load_reconstruction_v2(tmp_path / "reconstruction.npz").estimated_motion_trajectory is None
    assert validate_method_child_outputs_v2(tmp_path, expected_method=method, expected_dataset_identity_sha256=data.dataset_identity_sha256, expected_measurements_file_sha256="a" * 64, expected_algorithm_seed=seed) == hashes


def test_v2_writer_rejects_precreated_parent_owned_files(tmp_path: Path) -> None:
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")
    data, method = acquisition(), resolved_dgi()
    seed = derive_algorithm_seed(cell_seed=1, dataset_identity_sha256=data.dataset_identity_sha256, method_id="dgi", method_config_sha256=method.method_config_sha256)
    with pytest.raises(ValueError, match="isolated"):
        write_method_child_outputs_v2(tmp_path, method=method, acquisition=data, measurements_file_sha256="a" * 64, algorithm_seed=seed, result=result(), child_started_at_utc="2026-07-28T00:00:00Z", child_finished_at_utc="2026-07-28T00:00:01Z")
