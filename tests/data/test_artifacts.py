import dataclasses
import io
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from copy import deepcopy

import numpy as np
import pytest
import yaml

from gsdiff.data.artifacts import (
    ArtifactValidationError,
    EvaluationTruth,
    MethodExecutionPolicy,
    ReconstructionOutput,
    SPIAcquisitionData,
    artifact_sha256,
    load_acquisition_data,
    load_evaluation_truth,
    load_reconstruction_output,
    method_execution_policy,
    require_promotion_eligible,
    save_acquisition_data,
    save_evaluation_truth,
    split_spi_data,
    validate_evaluation_inputs,
    write_method_child_outputs,
)
from gsdiff.data.simulation import generate_spi_data


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTHORITATIVE_PYTHON = Path(r"D:\conda\envs\spi\python.exe")


def _tiny_pair(*, seed=7, shape="L", holdout_count=6):
    generation_config = {
        "H": 8,
        "W": 8,
        "T": 3,
        "K": 12,
        "seed": seed,
        "target": {"kind": "builtin", "descriptor": shape},
        "pattern": {"family": "bernoulli", "order": "stratified"},
        "time_assignment": {"mode": "uniform"},
        "noise": {
            "convention": "detector-absolute",
            "parameters": {"sigma_abs": 0.01, "snr_db": 25.0},
        },
        "motion": {
            "model": "custom_se2",
            "parameters": {
                "velocity": [1.0, -0.5],
                "acceleration": [0.2, 0.1],
                "omega": 0.15,
                "beta": 0.03,
                "speed_factor": 1.0,
                "motion_mode": 2,
            },
        },
        "holdout": {
            "count": holdout_count,
            "pattern_family": "uniform-random",
            "seed_offset": 9999,
        },
    }
    data = generate_spi_data(
        H=8,
        W=8,
        T=3,
        K=12,
        pattern_type="bernoulli",
        motion_type="custom_se2",
        snr_db=25.0,
        seed=seed,
        shape=shape,
        gt_velocity=[1.0, -0.5],
        gt_omega=0.15,
        gt_accel=[0.2, 0.1],
        gt_beta=0.03,
        noise_sigma_abs=0.01,
        holdout_extra=holdout_count,
        time_assignment_mode="uniform",
        pattern_order="stratified",
    )
    target_asset_sha256 = hashlib.sha256(
        np.ascontiguousarray(data.canonical).tobytes()
    ).hexdigest()
    return split_spi_data(
        data,
        resolved_generation_config=generation_config,
        generator_code_version="gsdiff-simulation-test-v1",
        target_asset_sha256=target_asset_sha256,
    )


def _assert_dataclass_equal(expected, actual):
    assert type(actual) is type(expected)
    for field in dataclasses.fields(expected):
        expected_value = getattr(expected, field.name)
        actual_value = getattr(actual, field.name)
        if isinstance(expected_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value, field.name


def _direct_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_bytes(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _read_metadata(path):
    with zipfile.ZipFile(path, "r") as archive:
        raw = archive.read("__metadata_json__.npy")
    array = np.load(io.BytesIO(raw), allow_pickle=False)
    assert array.dtype == np.uint8
    return json.loads(array.tobytes().decode("utf-8"))


def _npy_bytes(array):
    destination = io.BytesIO()
    np.save(destination, array, allow_pickle=True)
    return destination.getvalue()


def _rewrite_npz(source, destination, transform):
    with zipfile.ZipFile(source, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    transformed = transform(members)
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(transformed):
            archive.writestr(name, transformed[name])


def _replace_metadata(members, mutator):
    metadata_array = np.load(
        io.BytesIO(members["__metadata_json__.npy"]), allow_pickle=False
    )
    metadata = json.loads(metadata_array.tobytes().decode("utf-8"))
    mutator(metadata)
    result = dict(members)
    result["__metadata_json__.npy"] = _npy_bytes(
        np.frombuffer(_canonical_json_bytes(metadata), dtype=np.uint8)
    )
    return result


def _raw_output(acquisition):
    return ReconstructionOutput(
        dataset_identity_sha256=acquisition.dataset_identity_sha256,
        reconstruction=np.zeros(
            (acquisition.T, acquisition.H, acquisition.W), dtype=np.float32
        ),
        dgi=np.ones((acquisition.H, acquisition.W), dtype=np.float32),
        estimated_motion_trajectory=np.zeros(
            (acquisition.T, 3), dtype=np.float32
        ),
        frame_indices=np.arange(acquisition.T, dtype=np.int64),
        time_grid=acquisition.time_grid.copy(),
        method_name="dgi",
        method_metadata={"solver": "direct-dgi"},
        execution_policy=method_execution_policy(truth_path=None),
    )


def _write_entry_config(path, acquisition, output_dir):
    config = {
        "seed": 7,
        "dataset_spec": acquisition.resolved_generation_config,
        "scene": {
            "type": "gaussian",
            "num_gaussians": 4,
            "init_scale": 1.0,
            "init_mode": "random",
        },
        "motion": {"enable_rotation": True},
        "data": {
            "image_size": [acquisition.H, acquisition.W],
            "num_frames": acquisition.T,
            "num_patterns": acquisition.K,
            "pattern_type": acquisition.pattern_family,
            "pattern_order": acquisition.pattern_order,
            "time_assignment_mode": acquisition.time_assignment_mode,
            "shape": "must-not-be-generated",
            "motion_type": acquisition.motion_model,
            "holdout_mod": 0,
        },
        "solver": {
            "type": "sgd",
            "loss_norm": "zscore",
            "lr_scene": 0.001,
            "lr_motion": 0.001,
            "sgd_steps": 1,
            "tv_weight": 0.0,
            "use_3dtv": False,
            "temporal_tv_weight": 0.0,
        },
        "output_dir": str(output_dir),
    }
    path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")


def _assert_blind_output_is_measurement_only(output_dir, expected_identity):
    assert set(entry.name for entry in output_dir.iterdir()) == {
        "reconstruction.npz",
        "iteration-history.jsonl",
        "method-info.json",
    }
    reconstruction = load_reconstruction_output(
        output_dir / "reconstruction.npz"
    )
    assert reconstruction.dataset_identity_sha256 == expected_identity
    assert reconstruction.execution_policy == MethodExecutionPolicy(
        execution_class="method_child_blind",
        truth_access="unavailable",
        promotion_eligible=True,
    )
    serialized_metadata = json.dumps(
        {
            "reconstruction": _read_metadata(
                output_dir / "reconstruction.npz"
            ),
            "method_info": json.loads(
                (output_dir / "method-info.json").read_text(encoding="utf-8")
            ),
            "history": [
                json.loads(line)
                for line in (
                    output_dir / "iteration-history.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ],
        },
        sort_keys=True,
    ).lower()
    for forbidden in (
        "canonical",
        "ground_truth",
        "gt_",
        "gt-",
        "psnr",
        "ssim",
        "nrmse",
        "metric",
        "evaluator",
        "display",
        "normalized",
    ):
        assert forbidden not in serialized_metadata


def test_round_trip_restores_every_field_and_returns_direct_file_hash(tmp_path):
    acquisition, truth = _tiny_pair()
    acquisition_path = tmp_path / "measurements.npz"
    truth_path = tmp_path / "evaluation-truth.npz"

    acquisition_sha256 = save_acquisition_data(acquisition, acquisition_path)
    truth_sha256 = save_evaluation_truth(truth, truth_path)

    loaded_acquisition = load_acquisition_data(
        acquisition_path,
        expected_spec=acquisition.resolved_generation_config,
    )
    loaded_truth = load_evaluation_truth(
        truth_path,
        expected_dataset_identity_sha256=acquisition.dataset_identity_sha256,
    )
    _assert_dataclass_equal(acquisition, loaded_acquisition)
    _assert_dataclass_equal(truth, loaded_truth)
    assert acquisition_sha256 == _direct_sha256(acquisition_path)
    assert truth_sha256 == _direct_sha256(truth_path)
    assert len(acquisition_sha256) == len(truth_sha256) == 64
    assert acquisition_sha256 == acquisition_sha256.lower()
    assert truth_sha256 == truth_sha256.lower()


def test_repeated_saves_are_byte_identical_for_both_artifacts(tmp_path):
    acquisition, truth = _tiny_pair()
    acquisition_paths = [
        tmp_path / "measurements-a.npz",
        tmp_path / "measurements-b.npz",
    ]
    truth_paths = [
        tmp_path / "evaluation-truth-a.npz",
        tmp_path / "evaluation-truth-b.npz",
    ]

    acquisition_hashes = [
        save_acquisition_data(acquisition, path) for path in acquisition_paths
    ]
    truth_hashes = [save_evaluation_truth(truth, path) for path in truth_paths]

    assert acquisition_hashes[0] == acquisition_hashes[1]
    assert truth_hashes[0] == truth_hashes[1]
    assert acquisition_paths[0].read_bytes() == acquisition_paths[1].read_bytes()
    assert truth_paths[0].read_bytes() == truth_paths[1].read_bytes()


def test_dataset_identity_is_canonical_spec_hash_shared_by_pair():
    acquisition, truth = _tiny_pair()
    expected = hashlib.sha256(
        _canonical_json_bytes(acquisition.dataset_identity_spec)
    ).hexdigest()

    assert acquisition.dataset_identity_sha256 == expected
    assert truth.dataset_identity_sha256 == expected
    assert acquisition.dataset_identity_spec == truth.dataset_identity_spec
    assert set(acquisition.dataset_identity_spec) == {
        "arrays",
        "dimensions",
        "generator_code_version",
        "motion",
        "noise",
        "pattern_family",
        "pattern_order",
        "resolved_generation_config",
        "schema",
        "seed",
        "target_asset_sha256",
        "time_assignment_mode",
    }
    assert acquisition.dataset_identity_spec["schema"] == "measurements-v1"
    assert acquisition.dataset_identity_spec["motion"]["parameters"]["acceleration"] == [
        0.2,
        0.1,
    ]
    assert acquisition.dataset_identity_spec["motion"]["parameters"]["beta"] == 0.03


def test_acquisition_archive_has_only_measurement_side_members_and_metadata(tmp_path):
    acquisition, _ = _tiny_pair()
    path = tmp_path / "measurements.npz"
    save_acquisition_data(acquisition, path)

    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
    assert names == sorted(names)
    assert set(names) == {
        "__metadata_json__.npy",
        "frame_indices.npy",
        "holdout_frame_indices.npy",
        "holdout_measurements.npy",
        "holdout_patterns.npy",
        "measurements.npy",
        "patterns.npy",
        "time_grid.npy",
    }
    serialized = json.dumps(
        _read_metadata(path), ensure_ascii=False, sort_keys=True
    ).lower()
    for forbidden in (
        "canonical",
        "ground_truth",
        "gt_",
        "gt-",
        "trajectory",
        "evaluator",
        "evaluation",
        "metric",
        "display",
        "normalized",
    ):
        assert forbidden not in serialized


def test_absent_holdout_is_explicit_and_restored_as_none(tmp_path):
    acquisition, _ = _tiny_pair(holdout_count=0)
    path = tmp_path / "measurements.npz"
    save_acquisition_data(acquisition, path)

    loaded = load_acquisition_data(
        path, expected_spec=acquisition.resolved_generation_config
    )

    assert loaded.holdout_patterns is None
    assert loaded.holdout_measurements is None
    assert loaded.holdout_frame_indices is None
    assert _read_metadata(path)["optional_arrays"] == {
        "holdout_frame_indices": False,
        "holdout_measurements": False,
        "holdout_patterns": False,
    }


def test_artifact_hash_is_direct_lowercase_sha256(tmp_path):
    acquisition, _ = _tiny_pair()
    path = tmp_path / "measurements.npz"
    returned = save_acquisition_data(acquisition, path)

    assert artifact_sha256(path) == returned == _direct_sha256(path)
    assert len(returned) == 64
    assert returned == returned.lower()


@pytest.mark.parametrize(
    ("path", "changed"),
    [
        (("time_assignment", "mode"), "interpolation"),
        (("pattern", "family"), "gaussian"),
        (("pattern", "order"), "random"),
        (("noise", "convention"), "ac-variance-snr"),
        (("motion", "model"), "translation"),
    ],
    ids=[
        "time-assignment",
        "pattern-family",
        "pattern-order",
        "noise-convention",
        "motion-model",
    ],
)
def test_expected_spec_rejects_every_physics_mismatch(tmp_path, path, changed):
    acquisition, _ = _tiny_pair()
    artifact_path = tmp_path / "measurements.npz"
    save_acquisition_data(acquisition, artifact_path)
    mismatched = deepcopy(acquisition.resolved_generation_config)
    cursor = mismatched
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = changed

    with pytest.raises(ArtifactValidationError, match="expected spec"):
        load_acquisition_data(artifact_path, expected_spec=mismatched)


def test_expected_spec_is_fail_closed_for_missing_or_extra_fields(tmp_path):
    acquisition, _ = _tiny_pair()
    artifact_path = tmp_path / "measurements.npz"
    save_acquisition_data(acquisition, artifact_path)
    missing = deepcopy(acquisition.resolved_generation_config)
    del missing["noise"]["parameters"]["snr_db"]
    extra = deepcopy(acquisition.resolved_generation_config)
    extra["motion"]["parameters"]["untrusted_yaml_only"] = 1

    with pytest.raises(ArtifactValidationError, match="expected spec"):
        load_acquisition_data(artifact_path, expected_spec=missing)
    with pytest.raises(ArtifactValidationError, match="expected spec"):
        load_acquisition_data(artifact_path, expected_spec=extra)


def test_truth_loader_rejects_seed_and_target_swaps_before_evaluation(tmp_path):
    acquisition_seed7, truth_seed7 = _tiny_pair(seed=7, shape="L")
    acquisition_seed11, truth_seed11 = _tiny_pair(seed=11, shape="L")
    acquisition_target_t, truth_target_t = _tiny_pair(seed=7, shape="T")
    paths = {
        "seed7": tmp_path / "truth-seed7.npz",
        "seed11": tmp_path / "truth-seed11.npz",
        "target_t": tmp_path / "truth-target-t.npz",
    }
    save_evaluation_truth(truth_seed7, paths["seed7"])
    save_evaluation_truth(truth_seed11, paths["seed11"])
    save_evaluation_truth(truth_target_t, paths["target_t"])

    assert acquisition_seed7.dataset_identity_sha256 != (
        acquisition_seed11.dataset_identity_sha256
    )
    assert acquisition_seed7.dataset_identity_sha256 != (
        acquisition_target_t.dataset_identity_sha256
    )
    with pytest.raises(ArtifactValidationError, match="dataset identity"):
        load_evaluation_truth(
            paths["seed11"],
            expected_dataset_identity_sha256=(
                acquisition_seed7.dataset_identity_sha256
            ),
        )
    with pytest.raises(ArtifactValidationError, match="dataset identity"):
        load_evaluation_truth(
            paths["target_t"],
            expected_dataset_identity_sha256=(
                acquisition_seed7.dataset_identity_sha256
            ),
        )


def test_evaluator_requires_reconstruction_acquisition_truth_identity_equality():
    acquisition, truth = _tiny_pair()
    output = _raw_output(acquisition)

    validate_evaluation_inputs(output, acquisition, truth)

    with pytest.raises(ArtifactValidationError, match="dataset identity"):
        validate_evaluation_inputs(
            dataclasses.replace(
                output,
                dataset_identity_sha256="0" * 64,
            ),
            acquisition,
            truth,
        )
    with pytest.raises(ArtifactValidationError, match="dataset identity"):
        validate_evaluation_inputs(
            output,
            acquisition,
            dataclasses.replace(
                truth,
                dataset_identity_sha256="f" * 64,
            ),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "malformed-zip",
        "malformed-metadata",
        "missing-member",
        "extra-member",
        "object-array",
        "schema-mismatch",
        "content-hash-mismatch",
        "identity-mismatch",
    ],
)
def test_loader_rejects_malformed_or_corrupted_acquisition(tmp_path, mutation):
    acquisition, _ = _tiny_pair()
    source = tmp_path / "source.npz"
    damaged = tmp_path / f"{mutation}.npz"
    save_acquisition_data(acquisition, source)

    if mutation == "malformed-zip":
        damaged.write_bytes(b"not-a-zip")
    elif mutation == "malformed-metadata":
        _rewrite_npz(
            source,
            damaged,
            lambda members: {
                **members,
                "__metadata_json__.npy": _npy_bytes(
                    np.frombuffer(b"{", dtype=np.uint8)
                ),
            },
        )
    elif mutation == "missing-member":
        _rewrite_npz(
            source,
            damaged,
            lambda members: {
                name: value
                for name, value in members.items()
                if name != "measurements.npy"
            },
        )
    elif mutation == "extra-member":
        _rewrite_npz(
            source,
            damaged,
            lambda members: {**members, "surprise.npy": _npy_bytes(np.zeros(1))},
        )
    elif mutation == "object-array":
        _rewrite_npz(
            source,
            damaged,
            lambda members: {
                **members,
                "measurements.npy": _npy_bytes(
                    np.array([{"forbidden": "pickle"}], dtype=object)
                ),
            },
        )
    elif mutation == "schema-mismatch":
        _rewrite_npz(
            source,
            damaged,
            lambda members: _replace_metadata(
                members, lambda metadata: metadata.update(schema="measurements-v2")
            ),
        )
    elif mutation == "content-hash-mismatch":
        changed = acquisition.measurements.copy()
        changed[0] += 1.0
        _rewrite_npz(
            source,
            damaged,
            lambda members: {
                **members,
                "measurements.npy": _npy_bytes(changed),
            },
        )
    else:
        _rewrite_npz(
            source,
            damaged,
            lambda members: _replace_metadata(
                members,
                lambda metadata: metadata.update(
                    dataset_identity_sha256="0" * 64
                ),
            ),
        )

    with pytest.raises(ArtifactValidationError):
        load_acquisition_data(damaged)


def test_loader_does_not_infer_or_open_a_sibling_truth_path(tmp_path):
    acquisition, _ = _tiny_pair()
    path = tmp_path / "measurements.npz"
    save_acquisition_data(acquisition, path)
    assert [entry.name for entry in tmp_path.iterdir()] == ["measurements.npz"]

    loaded = load_acquisition_data(path)

    assert loaded.dataset_identity_sha256 == acquisition.dataset_identity_sha256
    assert [entry.name for entry in tmp_path.iterdir()] == ["measurements.npz"]


def test_dataclasses_are_frozen_and_reject_object_arrays():
    acquisition, truth = _tiny_pair()
    with pytest.raises(dataclasses.FrozenInstanceError):
        acquisition.K = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        truth.T = 1
    with pytest.raises((TypeError, ArtifactValidationError), match="object"):
        dataclasses.replace(
            acquisition,
            measurements=np.array([object()], dtype=object),
        )


def test_method_execution_policy_separates_blind_and_compatibility():
    blind = method_execution_policy(truth_path=None)
    compatibility = method_execution_policy(
        truth_path=Path("explicit-evaluation-truth.npz")
    )

    assert blind == MethodExecutionPolicy(
        execution_class="method_child_blind",
        truth_access="unavailable",
        promotion_eligible=True,
    )
    assert compatibility == MethodExecutionPolicy(
        execution_class="compatibility_unblinded",
        truth_access="child_visible",
        promotion_eligible=False,
    )
    require_promotion_eligible(blind)
    with pytest.raises(ArtifactValidationError, match="promotion"):
        require_promotion_eligible(compatibility)


def test_raw_child_outputs_have_only_native_reconstruction_members(tmp_path):
    acquisition, _ = _tiny_pair()
    output = _raw_output(acquisition)
    output_dir = tmp_path / "outputs"

    hashes = write_method_child_outputs(
        output_dir,
        output,
        history=[{"iteration": 1, "loss_data": np.float32(0.5)}],
    )

    assert set(entry.name for entry in output_dir.iterdir()) == {
        "reconstruction.npz",
        "iteration-history.jsonl",
        "method-info.json",
    }
    assert hashes["reconstruction.npz"] == _direct_sha256(
        output_dir / "reconstruction.npz"
    )
    with zipfile.ZipFile(output_dir / "reconstruction.npz", "r") as archive:
        assert set(archive.namelist()) == {
            "__metadata_json__.npy",
            "dgi.npy",
            "estimated_motion_trajectory.npy",
            "frame_indices.npy",
            "reconstruction.npy",
            "time_grid.npy",
        }
    loaded = load_reconstruction_output(output_dir / "reconstruction.npz")
    _assert_dataclass_equal(output, loaded)
    history_line = json.loads(
        (output_dir / "iteration-history.jsonl").read_text(encoding="utf-8")
    )
    assert history_line == {"iteration": 1, "loss_data": 0.5}
    method_info = json.loads(
        (output_dir / "method-info.json").read_text(encoding="utf-8")
    )
    assert method_info == {
        "dataset_identity_sha256": acquisition.dataset_identity_sha256,
        "execution_class": "method_child_blind",
        "method_metadata": {"solver": "direct-dgi"},
        "method_name": "dgi",
        "promotion_eligible": True,
        "schema": "method-info-v1",
        "truth_access": "unavailable",
    }
    serialized = json.dumps(
        {
            "reconstruction": _read_metadata(output_dir / "reconstruction.npz"),
            "method_info": method_info,
            "history": history_line,
        },
        sort_keys=True,
    ).lower()
    for forbidden in (
        "canonical",
        "ground_truth",
        "gt_",
        "gt-",
        "psnr",
        "ssim",
        "nrmse",
        "metric",
        "evaluator",
        "display",
        "normalized",
    ):
        assert forbidden not in serialized


def test_both_real_entrypoints_load_measurements_without_generation(
    tmp_path, monkeypatch
):
    import train
    import scripts.run_baselines as run_baselines

    acquisition, _ = _tiny_pair()
    measurements_path = tmp_path / "measurements.npz"
    save_acquisition_data(acquisition, measurements_path)

    def forbidden_generation(*args, **kwargs):
        raise AssertionError("generate_spi_data must not run in method-child mode")

    monkeypatch.setattr(train, "generate_spi_data", forbidden_generation)
    train_output = tmp_path / "train-output"
    train_config = tmp_path / "train.yaml"
    _write_entry_config(train_config, acquisition, train_output)
    train.main(
        [
            "--config",
            str(train_config),
            "--measurements-path",
            str(measurements_path),
            "--output-dir",
            str(train_output),
            "--device",
            "cpu",
        ]
    )
    _assert_blind_output_is_measurement_only(
        train_output, acquisition.dataset_identity_sha256
    )

    monkeypatch.setattr(
        run_baselines, "generate_spi_data", forbidden_generation
    )
    baseline_output = tmp_path / "baseline-output"
    run_baselines.main(
        [
            "--config",
            str(train_config),
            "--name",
            "blind-test",
            "--baselines",
            "dgi",
            "--device",
            "cpu",
            "--measurements-path",
            str(measurements_path),
            "--output-dir",
            str(baseline_output),
        ]
    )
    _assert_blind_output_is_measurement_only(
        baseline_output, acquisition.dataset_identity_sha256
    )


def test_explicit_truth_compatibility_paths_are_unblinded_and_not_promotable(
    tmp_path, monkeypatch
):
    import train
    import scripts.run_baselines as run_baselines

    acquisition, truth = _tiny_pair()
    measurements_path = tmp_path / "measurements.npz"
    truth_path = tmp_path / "evaluation-truth.npz"
    save_acquisition_data(acquisition, measurements_path)
    save_evaluation_truth(truth, truth_path)

    def forbidden_generation(*args, **kwargs):
        raise AssertionError("explicit artifacts must not regenerate SPI data")

    train_output = tmp_path / "train-compatibility"
    config_path = tmp_path / "compatibility.yaml"
    _write_entry_config(config_path, acquisition, train_output)
    monkeypatch.setattr(train, "generate_spi_data", forbidden_generation)
    train.main(
        [
            "--config",
            str(config_path),
            "--measurements-path",
            str(measurements_path),
            "--truth-path",
            str(truth_path),
            "--output-dir",
            str(train_output),
            "--device",
            "cpu",
        ]
    )
    train_results = json.loads(
        (train_output / "results.json").read_text(encoding="utf-8")
    )
    assert {
        key: train_results[key]
        for key in ("execution_class", "truth_access", "promotion_eligible")
    } == {
        "execution_class": "compatibility_unblinded",
        "truth_access": "child_visible",
        "promotion_eligible": False,
    }
    assert not (train_output / "method-info.json").exists()

    baseline_output = tmp_path / "baseline-compatibility"
    monkeypatch.setattr(
        run_baselines, "generate_spi_data", forbidden_generation
    )
    run_baselines.main(
        [
            "--config",
            str(config_path),
            "--name",
            "compatibility-test",
            "--baselines",
            "dgi",
            "--device",
            "cpu",
            "--measurements-path",
            str(measurements_path),
            "--truth-path",
            str(truth_path),
            "--output-dir",
            str(baseline_output),
        ]
    )
    baseline_results = json.loads(
        (baseline_output / "baselines.json").read_text(encoding="utf-8")
    )
    assert {
        key: baseline_results[key]
        for key in ("execution_class", "truth_access", "promotion_eligible")
    } == {
        "execution_class": "compatibility_unblinded",
        "truth_access": "child_visible",
        "promotion_eligible": False,
    }
    assert not (baseline_output / "method-info.json").exists()


def test_real_method_child_subprocess_has_no_truth_capability(tmp_path):
    acquisition, _ = _tiny_pair()
    child_cwd = tmp_path / "method-child"
    child_cwd.mkdir()
    save_acquisition_data(acquisition, child_cwd / "measurements.npz")
    output_dir = tmp_path / "method-output"
    config_path = tmp_path / "child.yaml"
    _write_entry_config(config_path, acquisition, output_dir)
    assert [entry.name for entry in child_cwd.iterdir()] == ["measurements.npz"]
    command = [
        str(AUTHORITATIVE_PYTHON),
        str(REPO_ROOT / "scripts" / "run_baselines.py"),
        "--config",
        str(config_path),
        "--name",
        "blind-subprocess",
        "--baselines",
        "dgi",
        "--device",
        "cpu",
        "--measurements-path",
        "measurements.npz",
        "--output-dir",
        str(output_dir),
    ]
    child_env = {
        "PATH": os.environ["PATH"],
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": str(REPO_ROOT),
        "SYSTEMROOT": os.environ["SYSTEMROOT"],
    }
    assert all("truth" not in argument.lower() for argument in command)
    assert "truth" not in json.dumps(child_env).lower()
    assert "truth" not in str(child_cwd).lower()

    completed = subprocess.run(
        command,
        cwd=child_cwd,
        env=child_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    reconstructed = load_reconstruction_output(output_dir / "reconstruction.npz")
    assert reconstructed.method_name == "dgi"
    assert reconstructed.reconstruction.shape == (
        acquisition.T,
        acquisition.H,
        acquisition.W,
    )
    truth_seeker = subprocess.run(
        [
            str(AUTHORITATIVE_PYTHON),
            "-c",
            (
                "from pathlib import Path; "
                "Path('evaluation-truth.npz').read_bytes()"
            ),
        ],
        cwd=child_cwd,
        env=child_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert truth_seeker.returncode != 0
    assert "FileNotFoundError" in truth_seeker.stderr
