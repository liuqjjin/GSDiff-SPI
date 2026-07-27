import dataclasses
import io
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import yaml
from PIL import Image

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


def _tiny_source(*, seed=7, shape="L", holdout_count=6, T=3):
    generation_config = {
        "schema": "measurements-v1",
        "H": 8,
        "W": 8,
        "T": T,
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
            "present": holdout_count > 0,
            "count": holdout_count,
            "pattern_family": "uniform-random",
            "seed_offset": 9999,
        },
    }
    data = generate_spi_data(
        H=8,
        W=8,
        T=T,
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
    return data, generation_config, target_asset_sha256


def _tiny_pair(*, seed=7, shape="L", holdout_count=6, T=3):
    data, generation_config, target_asset_sha256 = _tiny_source(
        seed=seed, shape=shape, holdout_count=holdout_count, T=T
    )
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


def _json_native_for_test(value):
    if isinstance(value, Mapping):
        return {
            str(key): _json_native_for_test(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_native_for_test(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_native_for_test(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canonical_json_bytes(value):
    return json.dumps(
        _json_native_for_test(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _mutable_json(value):
    return json.loads(_canonical_json_bytes(value))


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


def _replace_array_and_descriptor(members, name, array):
    array = np.ascontiguousarray(array)
    result = dict(members)
    result[f"{name}.npy"] = _npy_bytes(array)

    def update_descriptor(metadata):
        metadata["array_descriptors"][name] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }

    return _replace_metadata(result, update_descriptor)


def _refresh_metadata_identity(metadata):
    metadata["dataset_identity_sha256"] = hashlib.sha256(
        _canonical_json_bytes(metadata["dataset_identity_spec"])
    ).hexdigest()


def _set_nested(mapping, path, value):
    cursor = mapping
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


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


def _output_with_index_dtype(acquisition, *, T, dtype):
    return dataclasses.replace(
        _raw_output(acquisition),
        reconstruction=np.zeros(
            (T, acquisition.H, acquisition.W), dtype=np.float32
        ),
        estimated_motion_trajectory=np.zeros((T, 3), dtype=np.float32),
        frame_indices=np.arange(T, dtype=dtype),
        time_grid=np.linspace(0.0, 1.0, T, dtype=np.float32),
    )


@pytest.fixture(scope="module")
def t300_pair():
    return _tiny_pair(T=300)


def _write_entry_config(path, acquisition, output_dir):
    config = {
        "seed": 7,
        "dataset_spec": _mutable_json(acquisition.resolved_generation_config),
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
        execution_class="blind_method_child",
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
    identity_motion = acquisition.dataset_identity_spec["motion"]["parameters"]
    assert identity_motion["acceleration"] == (0.2, 0.1)
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
    mismatched = _mutable_json(acquisition.resolved_generation_config)
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
    missing = _mutable_json(acquisition.resolved_generation_config)
    del missing["noise"]["parameters"]["snr_db"]
    extra = _mutable_json(acquisition.resolved_generation_config)
    extra["motion"]["parameters"]["untrusted_yaml_only"] = 1

    with pytest.raises(ArtifactValidationError, match="expected spec"):
        load_acquisition_data(artifact_path, expected_spec=missing)
    with pytest.raises(ArtifactValidationError, match="expected spec"):
        load_acquisition_data(artifact_path, expected_spec=extra)


@pytest.mark.parametrize(
    ("path", "changed"),
    [
        (("seed",), False),
        (("H",), 8.0),
        (("W",), "8"),
        (("motion", "parameters", "motion_mode"), 3),
        (("motion", "parameters", "omega"), float("inf")),
        (("noise", "parameters", "sigma_abs"), -0.01),
        (("pattern", "family"), ""),
        (("pattern", "order"), "unsupported"),
        (("time_assignment", "mode"), "unsupported"),
        (("holdout", "seed_offset"), "9999"),
    ],
    ids=[
        "bool-seed",
        "float-dimension",
        "string-dimension",
        "motion-mode-enum",
        "nonfinite-motion",
        "negative-sigma",
        "empty-string",
        "pattern-order-enum",
        "time-mode-enum",
        "string-seed-offset",
    ],
)
def test_split_rejects_non_exact_generation_config_types_and_values(
    path, changed
):
    data, config, target_hash = _tiny_source()
    _set_nested(config, path, changed)

    with pytest.raises(ArtifactValidationError):
        split_spi_data(
            data,
            resolved_generation_config=config,
            generator_code_version="gsdiff-simulation-test-v1",
            target_asset_sha256=target_hash,
        )


@pytest.mark.parametrize(
    "mutation",
    ["missing-schema", "wrong-schema", "extra-top", "extra-nested"],
)
def test_split_requires_exact_recursive_versioned_config_schema(mutation):
    data, config, target_hash = _tiny_source()
    if mutation == "missing-schema":
        del config["schema"]
    elif mutation == "wrong-schema":
        config["schema"] = "measurements-v2"
    elif mutation == "extra-top":
        config["truth_path"] = "evaluation-truth.npz"
    else:
        config["motion"]["parameters"]["trajectory"] = [[0.0, 0.0, 0.0]]

    with pytest.raises(ArtifactValidationError):
        split_spi_data(
            data,
            resolved_generation_config=config,
            generator_code_version="gsdiff-simulation-test-v1",
            target_asset_sha256=target_hash,
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("truth_path",), "evaluation-truth.npz"),
        (("canonical",), [[0.0]]),
        (("gt_frames",), [[[0.0]]]),
        (("target", "path"), "../private/target.png"),
        (("motion", "parameters", "trajectory"), [[0.0, 0.0, 0.0]]),
        (("noise", "parameters", "evaluator"), {"metric": "psnr"}),
    ],
    ids=[
        "truth-path",
        "canonical",
        "gt-frames",
        "target-path",
        "trajectory",
        "evaluator-metric",
    ],
)
def test_generation_config_rejects_capability_smuggling(path, value):
    data, config, target_hash = _tiny_source()
    _set_nested(config, path, value)

    with pytest.raises(ArtifactValidationError):
        split_spi_data(
            data,
            resolved_generation_config=config,
            generator_code_version="gsdiff-simulation-test-v1",
            target_asset_sha256=target_hash,
        )


@pytest.mark.parametrize(
    "descriptor",
    [
        r"C:\private\evaluation-truth.npz",
        "/private/evaluation-truth.npz",
        r"\\server\share\target.png",
        "../target.png",
        "https://example.invalid/target.png",
        "evaluation-truth.npz",
        "gt.npz",
        "metrics.json",
        "trajectory-v1",
    ],
    ids=[
        "windows-absolute",
        "posix-absolute",
        "unc",
        "parent-relative",
        "uri",
        "reserved-capability-token",
        "short-gt-token",
        "metrics-token",
        "trajectory-token",
    ],
)
def test_target_descriptor_rejects_paths_uris_and_capability_tokens(
    descriptor,
):
    data, config, target_hash = _tiny_source()
    config["target"]["descriptor"] = descriptor

    with pytest.raises(ArtifactValidationError):
        split_spi_data(
            data,
            resolved_generation_config=config,
            generator_code_version="gsdiff-simulation-test-v1",
            target_asset_sha256=target_hash,
        )


@pytest.mark.parametrize("descriptor", ["L", "tank.png", "asset-01.v2"])
def test_target_descriptor_accepts_opaque_logical_ids(descriptor):
    data, config, target_hash = _tiny_source()
    config["target"]["descriptor"] = descriptor

    acquisition, _ = split_spi_data(
        data,
        resolved_generation_config=config,
        generator_code_version="gsdiff-simulation-test-v1",
        target_asset_sha256=target_hash,
    )

    assert (
        acquisition.resolved_generation_config["target"]["descriptor"]
        == descriptor
    )


@pytest.mark.parametrize("descriptor", ["char:A", "char:5"])
def test_builtin_char_target_round_trips_complete_artifact_pair(
    tmp_path, descriptor
):
    data, config, target_hash = _tiny_source(shape=descriptor)
    acquisition, truth = split_spi_data(
        data,
        resolved_generation_config=config,
        generator_code_version="gsdiff-simulation-test-v1",
        target_asset_sha256=target_hash,
    )
    acquisition_path = tmp_path / "measurements.npz"
    truth_path = tmp_path / "evaluation-truth.npz"
    save_acquisition_data(acquisition, acquisition_path)
    save_evaluation_truth(truth, truth_path)

    loaded_acquisition = load_acquisition_data(
        acquisition_path, expected_spec=config
    )
    loaded_truth = load_evaluation_truth(
        truth_path,
        expected_dataset_identity_sha256=(
            loaded_acquisition.dataset_identity_sha256
        ),
    )

    assert loaded_acquisition.resolved_generation_config["target"] == {
        "kind": "builtin",
        "descriptor": descriptor,
    }
    assert loaded_truth.dataset_identity_sha256 == (
        loaded_acquisition.dataset_identity_sha256
    )


@pytest.mark.parametrize(
    ("kind", "descriptor"),
    [
        ("builtin", "char:../truth"),
        ("builtin", r"char:C:\private\target.png"),
        ("builtin", "char:https://example.invalid/target.png"),
        ("builtin", "char:evaluation"),
        ("asset", "char:A"),
        ("external", "L"),
    ],
    ids=[
        "char-traversal",
        "char-windows-path",
        "char-uri",
        "char-capability",
        "char-only-for-builtin",
        "unsupported-kind",
    ],
)
def test_target_schema_rejects_malicious_char_ids_and_unknown_kinds(
    kind, descriptor
):
    data, config, target_hash = _tiny_source()
    config["target"] = {"kind": kind, "descriptor": descriptor}

    with pytest.raises(ArtifactValidationError, match="target"):
        split_spi_data(
            data,
            resolved_generation_config=config,
            generator_code_version="gsdiff-simulation-test-v1",
            target_asset_sha256=target_hash,
        )


@pytest.mark.parametrize(
    ("metadata_field", "identity_field", "changed"),
    [
        ("seed", "seed", 11),
        ("pattern_family", "pattern_family", "gaussian"),
        ("pattern_order", "pattern_order", "random"),
        ("time_assignment_mode", "time_assignment_mode", "interpolation"),
        ("noise_convention", "noise", "ac-variance-snr"),
        ("motion_model", "motion", "translation"),
    ],
)
def test_loader_rejects_split_brain_redundant_physics(
    tmp_path, metadata_field, identity_field, changed
):
    acquisition, _ = _tiny_pair()
    source = tmp_path / "source.npz"
    forged = tmp_path / "forged.npz"
    save_acquisition_data(acquisition, source)

    def mutate(metadata):
        metadata[metadata_field] = changed
        if identity_field == "noise":
            metadata["dataset_identity_spec"]["noise"]["convention"] = changed
        elif identity_field == "motion":
            metadata["dataset_identity_spec"]["motion"]["model"] = changed
        else:
            metadata["dataset_identity_spec"][identity_field] = changed
        _refresh_metadata_identity(metadata)

    _rewrite_npz(
        source,
        forged,
        lambda members: _replace_metadata(members, mutate),
    )

    with pytest.raises(ArtifactValidationError):
        load_acquisition_data(forged)


def test_loader_rejects_coercible_redundant_seed_type(tmp_path):
    acquisition, _ = _tiny_pair()
    source = tmp_path / "source.npz"
    forged = tmp_path / "forged.npz"
    save_acquisition_data(acquisition, source)
    _rewrite_npz(
        source,
        forged,
        lambda members: _replace_metadata(
            members, lambda metadata: metadata.update(seed=7.0)
        ),
    )

    with pytest.raises(ArtifactValidationError):
        load_acquisition_data(forged)


def test_loader_rejects_capability_injected_inside_identity_bound_config(
    tmp_path,
):
    acquisition, _ = _tiny_pair()
    source = tmp_path / "source.npz"
    forged = tmp_path / "forged.npz"
    save_acquisition_data(acquisition, source)

    def mutate(metadata):
        injected = {"metric": "psnr", "path": "evaluation-truth.npz"}
        metadata["resolved_generation_config"]["noise"]["parameters"][
            "evaluator"
        ] = injected
        metadata["dataset_identity_spec"]["resolved_generation_config"][
            "noise"
        ]["parameters"]["evaluator"] = injected
        _refresh_metadata_identity(metadata)

    _rewrite_npz(
        source,
        forged,
        lambda members: _replace_metadata(members, mutate),
    )

    with pytest.raises(ArtifactValidationError):
        load_acquisition_data(forged)


@pytest.mark.parametrize(
    ("present", "count", "holdout_count"),
    [
        (False, 6, 6),
        (True, 7, 6),
        (True, 1, 0),
    ],
)
def test_split_requires_holdout_metadata_to_match_array_presence_and_count(
    present, count, holdout_count
):
    data, config, target_hash = _tiny_source(holdout_count=holdout_count)
    config["holdout"]["present"] = present
    config["holdout"]["count"] = count

    with pytest.raises(ArtifactValidationError):
        split_spi_data(
            data,
            resolved_generation_config=config,
            generator_code_version="gsdiff-simulation-test-v1",
            target_asset_sha256=target_hash,
        )


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("H", 8.0),
        ("W", "8"),
        ("T", np.int64(3)),
        ("K", np.int32(12)),
        ("H", False),
        ("W", 0),
    ],
    ids=[
        "float",
        "string",
        "numpy-int64",
        "numpy-int32",
        "bool",
        "nonpositive",
    ],
)
def test_split_rejects_non_exact_source_dimension_types(field, changed):
    data, config, target_hash = _tiny_source()
    setattr(data, field, changed)

    with pytest.raises(
        ArtifactValidationError, match="generated SPIData.*exact positive integer"
    ):
        split_spi_data(
            data,
            resolved_generation_config=config,
            generator_code_version="gsdiff-simulation-test-v1",
            target_asset_sha256=target_hash,
        )


@pytest.mark.parametrize(
    "changed",
    [
        np.array([1.0], dtype=np.float32),
        np.array([1.0 + 0.0j, -0.5 + 0.0j]),
        np.array([1.0, -0.5], dtype=object),
        np.array([True, False]),
        np.array([np.nan, -0.5]),
        np.array([np.inf, -0.5]),
    ],
    ids=["shape", "complex", "object", "bool", "nan", "inf"],
)
def test_split_rejects_invalid_source_gt_velocity(changed):
    data, config, target_hash = _tiny_source()
    data.gt_velocity = changed

    with pytest.raises(ArtifactValidationError, match="gt_velocity"):
        split_spi_data(
            data,
            resolved_generation_config=config,
            generator_code_version="gsdiff-simulation-test-v1",
            target_asset_sha256=target_hash,
        )


@pytest.mark.parametrize(
    "changed",
    [True, "0.15", np.array([0.15]), 0.15 + 0.0j, np.nan, np.inf],
    ids=["bool", "string", "array", "complex", "nan", "inf"],
)
def test_split_rejects_invalid_source_gt_omega(changed):
    data, config, target_hash = _tiny_source()
    data.gt_omega = changed

    with pytest.raises(ArtifactValidationError, match="gt_omega"):
        split_spi_data(
            data,
            resolved_generation_config=config,
            generator_code_version="gsdiff-simulation-test-v1",
            target_asset_sha256=target_hash,
        )


def test_split_compares_motion_at_source_float_precision():
    data, config, target_hash = _tiny_source()
    config["motion"]["parameters"]["velocity"] = [0.2, -0.1]
    config["motion"]["parameters"]["omega"] = 0.2
    data.gt_velocity = np.array([0.2, -0.1], dtype=np.float32)
    data.gt_omega = np.float32(0.2)

    acquisition, truth = split_spi_data(
        data,
        resolved_generation_config=config,
        generator_code_version="gsdiff-simulation-test-v1",
        target_asset_sha256=target_hash,
    )

    np.testing.assert_array_equal(
        truth.gt_velocity, np.array([0.2, -0.1], dtype=np.float32)
    )
    assert acquisition.motion_parameters["omega"] == 0.2


@pytest.mark.parametrize("field", ["gt_velocity", "gt_omega"])
def test_split_rejects_real_source_motion_mismatch(field):
    data, config, target_hash = _tiny_source()
    config["motion"]["parameters"]["velocity"] = [0.2, -0.1]
    config["motion"]["parameters"]["omega"] = 0.2
    data.gt_velocity = np.array([0.2, -0.1], dtype=np.float32)
    data.gt_omega = np.float32(0.2)
    if field == "gt_velocity":
        data.gt_velocity[1] = np.float32(-0.11)
    else:
        data.gt_omega = np.float32(0.21)

    with pytest.raises(ArtifactValidationError, match=field):
        split_spi_data(
            data,
            resolved_generation_config=config,
            generator_code_version="gsdiff-simulation-test-v1",
            target_asset_sha256=target_hash,
        )


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("patterns", "bool"),
        ("measurements", "complex"),
        ("time_grid", "nan"),
        ("holdout_patterns", "inf"),
        ("holdout_measurements", "bool"),
    ],
)
def test_acquisition_rejects_nonreal_or_nonfinite_scientific_arrays(
    tmp_path, field, mutation
):
    acquisition, _ = _tiny_pair()
    changed = getattr(acquisition, field).copy()
    if mutation == "bool":
        changed = changed.astype(bool)
    elif mutation == "complex":
        changed = changed.astype(np.complex64)
    elif mutation == "nan":
        changed.flat[0] = np.nan
    else:
        changed.flat[0] = np.inf
    tampered = dataclasses.replace(acquisition, **{field: changed})

    with pytest.raises(
        ArtifactValidationError, match=f"{field}.*real numeric finite"
    ):
        save_acquisition_data(tampered, tmp_path / "measurements.npz")


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("frame_indices", "float"),
        ("frame_indices", "bool"),
        ("frame_indices", "negative"),
        ("frame_indices", "out-of-range"),
        ("holdout_frame_indices", "float"),
        ("holdout_frame_indices", "negative"),
        ("holdout_frame_indices", "out-of-range"),
    ],
)
def test_acquisition_rejects_invalid_index_arrays(
    tmp_path, field, mutation
):
    acquisition, _ = _tiny_pair()
    changed = getattr(acquisition, field).copy()
    if mutation == "float":
        changed = changed.astype(np.float64)
    elif mutation == "bool":
        changed = changed.astype(bool)
    elif mutation == "negative":
        changed[0] = -1
    else:
        changed[0] = acquisition.T
    tampered = dataclasses.replace(acquisition, **{field: changed})

    with pytest.raises(ArtifactValidationError, match=field):
        save_acquisition_data(tampered, tmp_path / "measurements.npz")


@pytest.mark.parametrize(
    "field", ["frame_indices", "holdout_frame_indices"]
)
def test_acquisition_index_dtype_must_represent_full_frame_domain(
    tmp_path, t300_pair, field
):
    acquisition, _ = t300_pair
    wrapped = getattr(acquisition, field).astype(np.uint8)
    tampered = dataclasses.replace(acquisition, **{field: wrapped})

    with pytest.raises(ArtifactValidationError, match="dtype.*represent"):
        save_acquisition_data(tampered, tmp_path / "measurements.npz")


@pytest.mark.parametrize(
    "time_grid",
    [
        np.array([0.0, 0.75, 0.5], dtype=np.float32),
        np.array([0.0, 0.5, 0.5], dtype=np.float32),
        np.array([1, 0, 2], dtype=np.uint8),
    ],
    ids=["decreasing", "duplicate", "unsigned-decreasing"],
)
def test_acquisition_time_grid_must_be_strictly_increasing(
    tmp_path, time_grid
):
    acquisition, _ = _tiny_pair()
    tampered = dataclasses.replace(acquisition, time_grid=time_grid)

    with pytest.raises(ArtifactValidationError, match="strictly increasing"):
        save_acquisition_data(tampered, tmp_path / "measurements.npz")


def test_single_frame_acquisition_accepts_one_finite_time_value():
    data, config, target_hash = _tiny_source(T=1)

    acquisition, truth = split_spi_data(
        data,
        resolved_generation_config=config,
        generator_code_version="gsdiff-simulation-test-v1",
        target_asset_sha256=target_hash,
    )

    np.testing.assert_array_equal(
        acquisition.time_grid, np.array([0.0], dtype=np.float32)
    )
    assert truth.T == 1


def test_loader_rejects_nan_even_with_matching_local_array_descriptor(
    tmp_path,
):
    acquisition, _ = _tiny_pair()
    source = tmp_path / "source.npz"
    forged = tmp_path / "forged.npz"
    save_acquisition_data(acquisition, source)
    changed = acquisition.patterns.copy()
    changed.flat[0] = np.nan
    _rewrite_npz(
        source,
        forged,
        lambda members: _replace_array_and_descriptor(
            members, "patterns", changed
        ),
    )

    with pytest.raises(
        ArtifactValidationError, match="patterns.*real numeric finite"
    ):
        load_acquisition_data(forged)


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("canonical_image", "bool"),
        ("gt_frames", "complex"),
        ("translation_trajectory", "complex"),
        ("rotation_trajectory", "nan"),
    ],
)
def test_truth_rejects_nonreal_or_nonfinite_scientific_arrays(
    tmp_path, field, mutation
):
    _, truth = _tiny_pair()
    changed = getattr(truth, field).copy()
    if mutation == "bool":
        changed = changed.astype(bool)
    elif mutation == "complex":
        changed = changed.astype(np.complex64)
    else:
        changed.flat[0] = np.nan
    tampered = dataclasses.replace(truth, **{field: changed})

    with pytest.raises(
        ArtifactValidationError, match=f"{field}.*real numeric finite"
    ):
        save_evaluation_truth(tampered, tmp_path / "truth.npz")


@pytest.mark.parametrize(
    ("field", "mutation"),
    [
        ("reconstruction", "nan"),
        ("dgi", "inf"),
        ("estimated_motion_trajectory", "complex"),
        ("time_grid", "bool"),
    ],
)
def test_reconstruction_rejects_nonreal_or_nonfinite_scientific_arrays(
    tmp_path, field, mutation
):
    acquisition, _ = _tiny_pair()
    output = _raw_output(acquisition)
    changed = getattr(output, field).copy()
    if mutation == "complex":
        changed = changed.astype(np.complex64)
    elif mutation == "bool":
        changed = changed.astype(bool)
    elif mutation == "nan":
        changed.flat[0] = np.nan
    else:
        changed.flat[0] = np.inf
    tampered = dataclasses.replace(output, **{field: changed})

    with pytest.raises(
        ArtifactValidationError, match=f"{field}.*real numeric finite"
    ):
        write_method_child_outputs(
            tmp_path / "outputs", tampered, history=[]
        )


@pytest.mark.parametrize(
    "mutation", ["float", "bool", "negative", "out-of-range"]
)
def test_reconstruction_rejects_invalid_frame_indices(tmp_path, mutation):
    acquisition, _ = _tiny_pair()
    output = _raw_output(acquisition)
    changed = output.frame_indices.copy()
    if mutation == "float":
        changed = changed.astype(np.float64)
    elif mutation == "bool":
        changed = changed.astype(bool)
    elif mutation == "negative":
        changed[0] = -1
    else:
        changed[0] = acquisition.T
    tampered = dataclasses.replace(output, frame_indices=changed)

    with pytest.raises(ArtifactValidationError, match="frame_indices"):
        write_method_child_outputs(
            tmp_path / "outputs", tampered, history=[]
        )


@pytest.mark.parametrize(
    ("T", "dtype", "accepted"),
    [
        (256, np.uint8, True),
        (257, np.uint8, False),
        (300, np.uint8, False),
        (128, np.int8, True),
        (129, np.int8, False),
        (300, np.int16, True),
    ],
    ids=[
        "uint8-t256-boundary",
        "uint8-t257-overflow",
        "reviewer-t300-uint8-wrap",
        "int8-t128-boundary",
        "int8-t129-overflow",
        "int16-t300-wide-enough",
    ],
)
def test_reconstruction_index_dtype_capacity_boundaries(
    tmp_path, T, dtype, accepted
):
    acquisition, _ = _tiny_pair()
    output = _output_with_index_dtype(acquisition, T=T, dtype=dtype)

    if accepted:
        write_method_child_outputs(
            tmp_path / f"outputs-{T}-{np.dtype(dtype).name}",
            output,
            history=[],
        )
    else:
        with pytest.raises(
            ArtifactValidationError, match="dtype.*represent"
        ):
            write_method_child_outputs(
                tmp_path / f"outputs-{T}-{np.dtype(dtype).name}",
                output,
                history=[],
            )


def test_reconstruction_time_grid_must_be_strictly_increasing(tmp_path):
    acquisition, _ = _tiny_pair()
    output = dataclasses.replace(
        _raw_output(acquisition),
        time_grid=np.array([0.0, 0.75, 0.5], dtype=np.float32),
    )

    with pytest.raises(ArtifactValidationError, match="strictly increasing"):
        write_method_child_outputs(tmp_path / "outputs", output, history=[])


@pytest.mark.parametrize("mutation", ["value", "dtype"])
def test_evaluator_requires_exact_acquisition_time_grid(mutation):
    acquisition, truth = _tiny_pair()
    changed = acquisition.time_grid.copy()
    if mutation == "value":
        changed[1] = np.float32(0.4)
    else:
        changed = changed.astype(np.float64)
    output = dataclasses.replace(_raw_output(acquisition), time_grid=changed)

    with pytest.raises(ArtifactValidationError, match="time_grid"):
        validate_evaluation_inputs(output, acquisition, truth)


def test_evaluator_requires_canonical_output_frame_indices():
    acquisition, truth = _tiny_pair()
    output = dataclasses.replace(
        _raw_output(acquisition),
        frame_indices=np.array([1, 0, 2], dtype=np.int64),
    )

    with pytest.raises(ArtifactValidationError, match="frame_indices"):
        validate_evaluation_inputs(output, acquisition, truth)


def test_evaluator_rejects_wrapped_uint8_arange_for_t300(t300_pair):
    acquisition, truth = t300_pair
    validate_evaluation_inputs(_raw_output(acquisition), acquisition, truth)
    output = dataclasses.replace(
        _raw_output(acquisition),
        frame_indices=np.arange(300, dtype=np.uint8),
    )

    with pytest.raises(ArtifactValidationError, match="dtype.*represent"):
        validate_evaluation_inputs(output, acquisition, truth)


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


@pytest.mark.parametrize(
    ("dimension", "changed"),
    [
        ("H", 8.0),
        ("W", "8"),
        ("T", True),
    ],
)
def test_truth_loader_rejects_coercible_dimension_metadata(
    tmp_path, dimension, changed
):
    acquisition, truth = _tiny_pair()
    source = tmp_path / "truth-source.npz"
    forged = tmp_path / "truth-forged.npz"
    save_evaluation_truth(truth, source)
    _rewrite_npz(
        source,
        forged,
        lambda members: _replace_metadata(
            members,
            lambda metadata: metadata["dimensions"].update(
                {dimension: changed}
            ),
        ),
    )

    with pytest.raises(ArtifactValidationError):
        load_evaluation_truth(
            forged,
            expected_dataset_identity_sha256=(
                acquisition.dataset_identity_sha256
            ),
        )


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("gt_omega", 0.25),
        ("gt_beta", 0.13),
        ("motion_model", "translation"),
    ],
)
def test_truth_loader_binds_scalar_motion_metadata_to_identity(
    tmp_path, field, changed
):
    acquisition, truth = _tiny_pair()
    source = tmp_path / "truth-source.npz"
    forged = tmp_path / "truth-forged.npz"
    save_evaluation_truth(truth, source)
    _rewrite_npz(
        source,
        forged,
        lambda members: _replace_metadata(
            members, lambda metadata: metadata.update({field: changed})
        ),
    )

    with pytest.raises(ArtifactValidationError):
        load_evaluation_truth(
            forged,
            expected_dataset_identity_sha256=(
                acquisition.dataset_identity_sha256
            ),
        )


@pytest.mark.parametrize(
    "field",
    [
        "gt_velocity",
        "gt_acceleration",
        "translation_trajectory",
        "rotation_trajectory",
    ],
)
def test_truth_loader_rejects_identity_inconsistent_array_with_forged_digest(
    tmp_path, field
):
    acquisition, truth = _tiny_pair()
    source = tmp_path / "truth-source.npz"
    forged = tmp_path / "truth-forged.npz"
    save_evaluation_truth(truth, source)
    changed = getattr(truth, field).copy()
    changed.flat[0] += np.asarray(0.25, dtype=changed.dtype)
    _rewrite_npz(
        source,
        forged,
        lambda members: _replace_array_and_descriptor(
            members, field, changed
        ),
    )

    with pytest.raises(ArtifactValidationError):
        load_evaluation_truth(
            forged,
            expected_dataset_identity_sha256=(
                acquisition.dataset_identity_sha256
            ),
        )


def test_raw_target_asset_hash_does_not_equal_decoded_canonical_hash(
    tmp_path,
):
    target_path = tmp_path / "target.png"
    pixels = np.arange(64, dtype=np.uint8).reshape(8, 8) * 4
    Image.fromarray(pixels).save(target_path)
    raw_target_hash = hashlib.sha256(target_path.read_bytes()).hexdigest()
    data = generate_spi_data(
        H=8,
        W=8,
        T=3,
        K=12,
        pattern_type="bernoulli",
        motion_type="custom_se2",
        snr_db=25.0,
        seed=7,
        shape=str(target_path),
        gt_velocity=[1.0, -0.5],
        gt_omega=0.15,
        gt_accel=[0.2, 0.1],
        gt_beta=0.03,
        noise_sigma_abs=0.01,
        holdout_extra=6,
        time_assignment_mode="uniform",
        pattern_order="stratified",
    )
    _, config, _ = _tiny_source()
    config["target"] = {"kind": "asset", "descriptor": "target.png"}
    decoded_canonical_hash = hashlib.sha256(
        np.ascontiguousarray(data.canonical).tobytes()
    ).hexdigest()
    assert raw_target_hash != decoded_canonical_hash

    acquisition, truth = split_spi_data(
        data,
        resolved_generation_config=config,
        generator_code_version="gsdiff-simulation-test-v1",
        target_asset_sha256=raw_target_hash,
    )
    acquisition_path = tmp_path / "measurements.npz"
    truth_path = tmp_path / "truth.npz"
    save_acquisition_data(acquisition, acquisition_path)
    save_evaluation_truth(truth, truth_path)
    loaded_acquisition = load_acquisition_data(
        acquisition_path, expected_spec=config
    )
    loaded_truth = load_evaluation_truth(
        truth_path,
        expected_dataset_identity_sha256=(
            loaded_acquisition.dataset_identity_sha256
        ),
    )

    assert loaded_acquisition.target_asset_sha256 == raw_target_hash
    np.testing.assert_array_equal(loaded_truth.canonical_image, data.canonical)


def test_artifact_metadata_is_recursively_immutable():
    acquisition, truth = _tiny_pair()
    output = dataclasses.replace(
        _raw_output(acquisition),
        method_metadata={"solver": {"schedule": [1, 2]}},
    )
    truth = dataclasses.replace(
        truth,
        evaluator_metadata={"report": {"metrics": ["psnr"]}},
    )

    with pytest.raises(TypeError):
        acquisition.resolved_generation_config["motion"]["parameters"][
            "velocity"
        ][0] = 999.0
    with pytest.raises((TypeError, AttributeError)):
        acquisition.dataset_identity_spec["arrays"]["patterns"]["shape"].append(
            999
        )
    with pytest.raises(TypeError):
        truth.evaluator_metadata["report"]["metrics"][0] = "ssim"
    with pytest.raises(TypeError):
        output.method_metadata["solver"]["schedule"][0] = 999


@pytest.mark.parametrize(
    "artifact_array",
    ["acquisition", "truth", "reconstruction"],
)
def test_artifact_arrays_cannot_reenable_writeability(artifact_array):
    acquisition, truth = _tiny_pair()
    arrays = {
        "acquisition": acquisition.patterns,
        "truth": truth.gt_frames,
        "reconstruction": _raw_output(acquisition).reconstruction,
    }

    with pytest.raises(ValueError):
        arrays[artifact_array].flags.writeable = True


def test_evaluator_revalidates_all_artifact_contracts_not_only_identity():
    acquisition, truth = _tiny_pair()
    output = _raw_output(acquisition)

    tampered_acquisition = dataclasses.replace(acquisition)
    changed_measurements = tampered_acquisition.measurements.copy()
    changed_measurements[0] += 1.0
    object.__setattr__(
        tampered_acquisition, "measurements", changed_measurements
    )
    with pytest.raises(ArtifactValidationError):
        validate_evaluation_inputs(output, tampered_acquisition, truth)

    tampered_truth = dataclasses.replace(truth)
    object.__setattr__(tampered_truth, "gt_omega", truth.gt_omega + 0.1)
    with pytest.raises(ArtifactValidationError):
        validate_evaluation_inputs(output, acquisition, tampered_truth)

    tampered_output = dataclasses.replace(output)
    object.__setattr__(
        tampered_output,
        "execution_policy",
        MethodExecutionPolicy(
            execution_class="blind_method_child",
            truth_access="child_visible",
            promotion_eligible=True,
        ),
    )
    with pytest.raises(ArtifactValidationError):
        validate_evaluation_inputs(tampered_output, acquisition, truth)

    wrong_shape_output = dataclasses.replace(output)
    object.__setattr__(
        wrong_shape_output,
        "reconstruction",
        np.zeros((acquisition.T, acquisition.H, acquisition.W - 1)),
    )
    with pytest.raises(ArtifactValidationError):
        validate_evaluation_inputs(wrong_shape_output, acquisition, truth)


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


def test_loader_rejects_duplicate_zip_members(tmp_path):
    acquisition, _ = _tiny_pair()
    path = tmp_path / "measurements.npz"
    save_acquisition_data(acquisition, path)

    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("patterns.npy", _npy_bytes(acquisition.patterns))

    with pytest.raises(ArtifactValidationError, match="duplicate"):
        load_acquisition_data(path)


def test_deterministic_zip_uses_fixed_member_metadata(tmp_path):
    acquisition, _ = _tiny_pair()
    path = tmp_path / "measurements.npz"
    save_acquisition_data(acquisition, path)

    with zipfile.ZipFile(path, "r") as archive:
        infos = archive.infolist()

    assert [info.filename for info in infos] == sorted(
        info.filename for info in infos
    )
    assert {info.date_time for info in infos} == {(1980, 1, 1, 0, 0, 0)}
    assert {info.create_system for info in infos} == {3}
    assert {info.external_attr >> 16 for info in infos} == {0o100600}
    assert {info.compress_type for info in infos} == {zipfile.ZIP_DEFLATED}


def test_atomic_replace_failure_preserves_target_and_cleans_temp(
    tmp_path, monkeypatch
):
    import gsdiff.data._artifact_io as artifact_io

    acquisition, _ = _tiny_pair()
    path = tmp_path / "measurements.npz"
    original = b"preexisting-target"
    path.write_bytes(original)

    def fail_replace(source, destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(artifact_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        save_acquisition_data(acquisition, path)

    assert path.read_bytes() == original
    assert list(tmp_path.glob(".measurements.npz.*.tmp")) == []


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
        execution_class="blind_method_child",
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
        "execution_class": "blind_method_child",
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
