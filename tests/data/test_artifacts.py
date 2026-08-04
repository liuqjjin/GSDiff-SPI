import dataclasses
import io
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
import subprocess
import sys
import time
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
    blind_acquisition_spec,
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
AUTHORITATIVE_PYTHON = Path(sys.executable)
SEMANTIC_STRING_FIELDS = (
    "pattern_family",
    "noise_convention",
    "motion_model",
    "holdout_pattern_family",
    "generator_code_version",
)
BLIND_SEMANTIC_STRING_FIELDS = (
    "pattern_family",
    "pattern_order",
    "time_assignment",
    "holdout_pattern_family",
    "noise_convention",
)
FORBIDDEN_SEMANTIC_STRINGS = (
    r"C:\private\payload.npz",
    r"\\server\share\payload.npz",
    "/private/payload.npz",
    "../private/payload.npz",
    "file:///private/payload.npz",
    "http://example.invalid/payload",
    "https://example.invalid/payload",
    "evaluator-v1",
    "truth-v1",
    "gt-v1",
)
FORBIDDEN_SEMANTIC_IDS = (
    "windows-drive",
    "unc",
    "posix-absolute",
    "relative-traversal",
    "file-uri",
    "http-uri",
    "https-uri",
    "evaluator-token",
    "truth-token",
    "gt-segment",
)


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
    import gsdiff.data._artifact_io as artifact_io

    with zipfile.ZipFile(source, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    transformed = transform(members)
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name in sorted(transformed):
            archive.writestr(
                artifact_io._zip_info(name),
                transformed[name],
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


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


def _set_config_semantic(config, field, value):
    if field == "pattern_family":
        config["pattern"]["family"] = value
    elif field == "noise_convention":
        config["noise"]["convention"] = value
    elif field == "motion_model":
        config["motion"]["model"] = value
    elif field == "holdout_pattern_family":
        config["holdout"]["pattern_family"] = value
    elif field != "generator_code_version":
        raise AssertionError(f"unknown semantic field: {field}")


def _set_identity_semantic(spec, field, value):
    if field == "generator_code_version":
        spec["generator_code_version"] = value
    elif field == "pattern_family":
        spec["pattern_family"] = value
    elif field == "noise_convention":
        spec["noise"]["convention"] = value
    elif field == "motion_model":
        spec["motion"]["model"] = value


def _coherent_semantic_acquisition(acquisition, field, value):
    acquisition_spec = _mutable_json(acquisition.acquisition)
    acquisition_spec[field] = value
    return dataclasses.replace(acquisition, acquisition=acquisition_spec)


def _set_metadata_semantic(metadata, field, value):
    metadata["acquisition"][field] = value


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


@pytest.fixture
def _preserve_evaluation_import_state():
    prefixes = (
        "gsdiff.evaluation",
        "gsdiff.baselines._evaluation",
    )
    before = {
        name: module
        for name, module in sys.modules.items()
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in prefixes
        )
    }
    parent_attributes = (
        ("gsdiff", "evaluation"),
        ("gsdiff.baselines", "_evaluation"),
    )
    attributes_before = {}
    for parent_name, attribute_name in parent_attributes:
        parent = sys.modules.get(parent_name)
        namespace = vars(parent) if parent is not None else {}
        attributes_before[(parent_name, attribute_name)] = (
            attribute_name in namespace,
            namespace.get(attribute_name),
        )
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in prefixes
            ):
                sys.modules.pop(name, None)
        sys.modules.update(before)
        for parent_name, attribute_name in parent_attributes:
            parent = sys.modules.get(parent_name)
            if parent is None:
                continue
            had_attribute, attribute = attributes_before[
                (parent_name, attribute_name)
            ]
            if had_attribute:
                setattr(parent, attribute_name, attribute)
            else:
                vars(parent).pop(attribute_name, None)


def _write_entry_config(path, acquisition, output_dir):
    config = {
        "seed": 7,
        "acquisition_spec": _mutable_json(
            blind_acquisition_spec(acquisition)
        ),
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
            "motion_type": "blind",
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


def test_round_trip_restores_every_field_and_returns_direct_file_hash(tmp_path):
    acquisition, truth = _tiny_pair()
    acquisition_path = tmp_path / "measurements.npz"
    truth_path = tmp_path / "evaluation-truth.npz"

    acquisition_sha256 = save_acquisition_data(acquisition, acquisition_path)
    truth_sha256 = save_evaluation_truth(truth, truth_path)

    loaded_acquisition = load_acquisition_data(
        acquisition_path,
        expected_dataset_identity_sha256=acquisition.dataset_identity_sha256,
        expected_acquisition_spec=blind_acquisition_spec(acquisition),
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
        _canonical_json_bytes(truth.dataset_identity_spec)
    ).hexdigest()

    assert acquisition.dataset_identity_sha256 == expected
    assert truth.dataset_identity_sha256 == expected
    assert set(truth.dataset_identity_spec) == {
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
    assert truth.dataset_identity_spec["schema"] == "measurements-v1"
    identity_motion = truth.dataset_identity_spec["motion"]["parameters"]
    assert identity_motion["acceleration"] == (0.2, 0.1)
    assert truth.dataset_identity_spec["motion"]["parameters"]["beta"] == 0.03


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
        path,
        expected_dataset_identity_sha256=acquisition.dataset_identity_sha256,
        expected_acquisition_spec=blind_acquisition_spec(acquisition),
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
        (("acquisition", "time_assignment"), "interpolation"),
        (("acquisition", "pattern_family"), "gaussian"),
        (("acquisition", "pattern_order"), "random"),
        (("acquisition", "noise_convention"), "ac-variance-snr"),
        (("acquisition", "holdout_pattern_family"), "other"),
    ],
    ids=[
        "time-assignment",
        "pattern-family",
        "pattern-order",
        "noise-convention",
        "holdout-pattern-family",
    ],
)
def test_expected_spec_rejects_every_physics_mismatch(tmp_path, path, changed):
    acquisition, _ = _tiny_pair()
    artifact_path = tmp_path / "measurements.npz"
    save_acquisition_data(acquisition, artifact_path)
    mismatched = _mutable_json(blind_acquisition_spec(acquisition))
    cursor = mismatched
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = changed

    with pytest.raises(
        ArtifactValidationError, match="expected acquisition spec"
    ):
        load_acquisition_data(
            artifact_path,
            expected_dataset_identity_sha256=(
                acquisition.dataset_identity_sha256
            ),
            expected_acquisition_spec=mismatched,
        )


def test_expected_spec_is_fail_closed_for_missing_or_extra_fields(tmp_path):
    acquisition, _ = _tiny_pair()
    artifact_path = tmp_path / "measurements.npz"
    save_acquisition_data(acquisition, artifact_path)
    missing = _mutable_json(blind_acquisition_spec(acquisition))
    del missing["acquisition"]["noise_sigma_absolute"]
    extra = _mutable_json(blind_acquisition_spec(acquisition))
    extra["untrusted_yaml_only"] = 1

    with pytest.raises(
        ArtifactValidationError, match="expected acquisition spec"
    ):
        load_acquisition_data(
            artifact_path,
            expected_dataset_identity_sha256=(
                acquisition.dataset_identity_sha256
            ),
            expected_acquisition_spec=missing,
        )
    with pytest.raises(
        ArtifactValidationError, match="expected acquisition spec"
    ):
        load_acquisition_data(
            artifact_path,
            expected_dataset_identity_sha256=(
                acquisition.dataset_identity_sha256
            ),
            expected_acquisition_spec=extra,
        )


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

    _, truth = split_spi_data(
        data,
        resolved_generation_config=config,
        generator_code_version="gsdiff-simulation-test-v1",
        target_asset_sha256=target_hash,
    )

    assert (
        truth.dataset_identity_spec["resolved_generation_config"]["target"][
            "descriptor"
        ]
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
        acquisition_path,
        expected_dataset_identity_sha256=acquisition.dataset_identity_sha256,
        expected_acquisition_spec=blind_acquisition_spec(acquisition),
    )
    loaded_truth = load_evaluation_truth(
        truth_path,
        expected_dataset_identity_sha256=(
            loaded_acquisition.dataset_identity_sha256
        ),
    )

    assert loaded_truth.dataset_identity_spec["resolved_generation_config"][
        "target"
    ] == {
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
    "field",
    SEMANTIC_STRING_FIELDS,
)
@pytest.mark.parametrize(
    "value",
    FORBIDDEN_SEMANTIC_STRINGS,
    ids=FORBIDDEN_SEMANTIC_IDS,
)
def test_split_rejects_path_or_capability_semantic_strings(field, value):
    data, config, target_hash = _tiny_source()
    generator_code_version = "gsdiff-simulation-test-v1"
    if field == "generator_code_version":
        generator_code_version = value
    else:
        _set_config_semantic(config, field, value)

    with pytest.raises(ArtifactValidationError):
        split_spi_data(
            data,
            resolved_generation_config=config,
            generator_code_version=generator_code_version,
            target_asset_sha256=target_hash,
        )


@pytest.mark.parametrize(
    ("field", "valid_value"),
    [
        ("pattern_family", "bernoulli"),
        ("noise_convention", "detector-absolute"),
        ("motion_model", "custom_se2"),
        ("holdout_pattern_family", "uniform-random"),
        ("generator_code_version", "gsdiff-simulation-test-v1"),
    ],
)
def test_semantic_strings_reject_coercible_non_exact_string_types(
    field, valid_value
):
    data, config, target_hash = _tiny_source()
    changed = np.str_(valid_value)
    generator_code_version = "gsdiff-simulation-test-v1"
    if field == "generator_code_version":
        generator_code_version = changed
    else:
        _set_config_semantic(config, field, changed)

    with pytest.raises(ArtifactValidationError):
        split_spi_data(
            data,
            resolved_generation_config=config,
            generator_code_version=generator_code_version,
            target_asset_sha256=target_hash,
        )


@pytest.mark.parametrize(
    "field",
    BLIND_SEMANTIC_STRING_FIELDS,
)
@pytest.mark.parametrize(
    "forbidden",
    FORBIDDEN_SEMANTIC_STRINGS,
    ids=FORBIDDEN_SEMANTIC_IDS,
)
def test_save_rejects_coherent_forbidden_semantics_before_writing(
    tmp_path, field, forbidden
):
    acquisition, _ = _tiny_pair()
    tampered = _coherent_semantic_acquisition(
        acquisition, field, forbidden
    )
    destination = tmp_path / "measurements.npz"

    with pytest.raises(ArtifactValidationError):
        save_acquisition_data(tampered, destination)

    assert not destination.exists()


@pytest.mark.parametrize(
    "field",
    BLIND_SEMANTIC_STRING_FIELDS,
)
@pytest.mark.parametrize(
    "forbidden",
    FORBIDDEN_SEMANTIC_STRINGS,
    ids=FORBIDDEN_SEMANTIC_IDS,
)
def test_loader_rejects_identity_coherent_forbidden_semantics(
    tmp_path, field, forbidden
):
    acquisition, _ = _tiny_pair()
    source = tmp_path / "source.npz"
    forged = tmp_path / "forged.npz"
    save_acquisition_data(acquisition, source)
    _rewrite_npz(
        source,
        forged,
        lambda members: _replace_metadata(
            members,
                lambda metadata: _set_metadata_semantic(
                    metadata,
                    field,
                    forbidden,
                ),
        ),
    )

    with pytest.raises(ArtifactValidationError):
        load_acquisition_data(
            forged,
            expected_dataset_identity_sha256=(
                acquisition.dataset_identity_sha256
            ),
        )


@pytest.mark.parametrize(
    ("field", "allowed"),
    [
        *[
            ("pattern_family", value)
            for value in (
                "bernoulli",
                "gaussian",
                "random",
                "hadamard",
                "hadamard_cc",
                "hadamard_walsh",
                "hadamard_natural",
                "fourier",
                "s_matrix",
                "s_matrix_m",
            )
        ],
        ("noise_convention", "ac-variance-snr"),
        ("noise_convention", "detector-absolute"),
        *[
            ("motion_model", value)
            for value in (
                "translation",
                "rotation",
                "shear",
                "swirl",
                "translation_and_rotation",
                "custom_se2",
            )
        ],
        ("holdout_pattern_family", "uniform-random"),
    ],
)
def test_generation_config_accepts_supported_semantic_ids(field, allowed):
    from gsdiff.data._artifact_identity import validate_generation_config

    _, config, _ = _tiny_source()
    _set_config_semantic(config, field, allowed)

    validated = validate_generation_config(config)

    if field == "holdout_pattern_family":
        assert validated["holdout"]["pattern_family"] == allowed
    elif field == "pattern_family":
        assert validated["pattern"]["family"] == allowed
    elif field == "noise_convention":
        assert validated["noise"]["convention"] == allowed
    else:
        assert validated["motion"]["model"] == allowed


@pytest.mark.parametrize(
    "generator_code_version",
    [
        "gsdiff-simulation-test-v1",
        "gsdiff-simulation-v1",
        "gsdiff-sim-v1",
    ],
)
def test_generator_code_version_accepts_path_free_opaque_ids(
    generator_code_version,
):
    data, config, target_hash = _tiny_source()

    _, truth = split_spi_data(
        data,
        resolved_generation_config=config,
        generator_code_version=generator_code_version,
        target_asset_sha256=target_hash,
    )

    assert (
        truth.dataset_identity_spec["generator_code_version"]
        == generator_code_version
    )


@pytest.mark.parametrize(
    "generator_code_version",
    [
        "a",
        "a" * 128,
        "0123456789abcdef0123456789abcdef01234567",
    ],
    ids=["minimum-length", "maximum-length", "git-commit"],
)
def test_generator_code_version_accepts_opaque_id_boundaries(
    generator_code_version,
):
    data, config, target_hash = _tiny_source()

    _, truth = split_spi_data(
        data,
        resolved_generation_config=config,
        generator_code_version=generator_code_version,
        target_asset_sha256=target_hash,
    )

    assert (
        truth.dataset_identity_spec["generator_code_version"]
        == generator_code_version
    )


@pytest.mark.parametrize(
    "generator_code_version",
    [
        "éclair",
        ".version",
        "_version",
        "-version",
        "version with space",
        "a" * 129,
        "safe..id",
    ],
    ids=[
        "non-ascii",
        "leading-dot",
        "leading-underscore",
        "leading-hyphen",
        "whitespace",
        "too-long",
        "traversal-token",
    ],
)
def test_generator_code_version_rejects_non_opaque_ids(
    generator_code_version,
):
    data, config, target_hash = _tiny_source()

    with pytest.raises(ArtifactValidationError):
        split_spi_data(
            data,
            resolved_generation_config=config,
            generator_code_version=generator_code_version,
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

    _rewrite_npz(
        source,
        forged,
        lambda members: _replace_metadata(members, mutate),
    )

    with pytest.raises(ArtifactValidationError):
        load_acquisition_data(
            forged,
            expected_dataset_identity_sha256=(
                acquisition.dataset_identity_sha256
            ),
        )


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
        load_acquisition_data(
            forged,
            expected_dataset_identity_sha256=(
                acquisition.dataset_identity_sha256
            ),
        )


def test_loader_rejects_capability_injected_inside_identity_bound_config(
    tmp_path,
):
    acquisition, _ = _tiny_pair()
    source = tmp_path / "source.npz"
    forged = tmp_path / "forged.npz"
    save_acquisition_data(acquisition, source)

    def mutate(metadata):
        injected = {"metric": "psnr", "path": "evaluation-truth.npz"}
        metadata["acquisition"]["evaluator"] = injected

    _rewrite_npz(
        source,
        forged,
        lambda members: _replace_metadata(members, mutate),
    )

    with pytest.raises(ArtifactValidationError):
        load_acquisition_data(
            forged,
            expected_dataset_identity_sha256=(
                acquisition.dataset_identity_sha256
            ),
        )


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
    assert truth.dataset_identity_spec["motion"]["parameters"]["omega"] == 0.2


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
        load_acquisition_data(
            forged,
            expected_dataset_identity_sha256=(
                acquisition.dataset_identity_sha256
            ),
        )


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
        acquisition_path,
        expected_dataset_identity_sha256=acquisition.dataset_identity_sha256,
        expected_acquisition_spec=blind_acquisition_spec(acquisition),
    )
    loaded_truth = load_evaluation_truth(
        truth_path,
        expected_dataset_identity_sha256=(
            loaded_acquisition.dataset_identity_sha256
        ),
    )

    assert (
        loaded_truth.dataset_identity_spec["target_asset_sha256"]
        == raw_target_hash
    )
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
        acquisition.acquisition["pattern_values"][0] = 999.0
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
        load_acquisition_data(
            path,
            expected_dataset_identity_sha256=(
                acquisition.dataset_identity_sha256
            ),
        )


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
                members,
                lambda metadata: metadata.update(
                    schema_version="measurements-blind-v2"
                ),
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
        load_acquisition_data(
            damaged,
            expected_dataset_identity_sha256=(
                acquisition.dataset_identity_sha256
            ),
        )


def test_loader_does_not_infer_or_open_a_sibling_truth_path(tmp_path):
    acquisition, _ = _tiny_pair()
    path = tmp_path / "measurements.npz"
    save_acquisition_data(acquisition, path)
    assert [entry.name for entry in tmp_path.iterdir()] == ["measurements.npz"]

    loaded = load_acquisition_data(
        path,
        expected_dataset_identity_sha256=(
            acquisition.dataset_identity_sha256
        ),
    )

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


def test_truthless_legacy_entrypoints_fail_before_generation_or_outputs(
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
    blind_config = yaml.safe_load(train_config.read_text(encoding="utf-8"))
    assert "dataset_spec" not in blind_config
    assert blind_config["acquisition_spec"] == _mutable_json(
        blind_acquisition_spec(acquisition)
    )
    with pytest.raises(SystemExit):
        train.main(
            [
                "--legacy-compatibility",
                "--config",
                str(train_config),
                "--measurements-path",
                str(measurements_path),
                "--dataset-identity-sha256",
                acquisition.dataset_identity_sha256,
                "--output-dir",
                str(train_output),
                "--device",
                "cpu",
            ]
        )
    assert not train_output.exists()

    monkeypatch.setattr(
        run_baselines, "generate_spi_data", forbidden_generation
    )
    baseline_output = tmp_path / "baseline-output"
    with pytest.raises(SystemExit):
        run_baselines.main(
            [
                "--legacy-compatibility",
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
                "--dataset-identity-sha256",
                acquisition.dataset_identity_sha256,
                "--output-dir",
                str(baseline_output),
            ]
        )
    assert not baseline_output.exists()


def test_explicit_truth_compatibility_paths_are_unblinded_and_not_promotable(
    tmp_path, monkeypatch, _preserve_evaluation_import_state
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
            "--legacy-compatibility",
            "--config",
            str(config_path),
            "--measurements-path",
            str(measurements_path),
            "--dataset-identity-sha256",
            acquisition.dataset_identity_sha256,
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
            "--legacy-compatibility",
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
            "--dataset-identity-sha256",
            acquisition.dataset_identity_sha256,
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

    legacy_policy = method_execution_policy(truth_path=truth_path)
    with pytest.raises(ArtifactValidationError, match="not eligible"):
        require_promotion_eligible(legacy_policy)

    from gsdiff.experiments.child_outputs import (
        validate_method_child_outputs_v2,
    )
    from gsdiff.experiments.methods import (
        derive_algorithm_seed,
        resolve_method_semantics,
    )

    expected_method = resolve_method_semantics(
        "dgi",
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
    expected_seed = derive_algorithm_seed(
        cell_seed=7,
        dataset_identity_sha256=acquisition.dataset_identity_sha256,
        method_id=expected_method.method_id,
        method_config_sha256=expected_method.method_config_sha256,
    )
    for legacy_output in (train_output, baseline_output):
        with pytest.raises(
            ArtifactValidationError,
            match="exactly two files|inventory must be flat",
        ):
            validate_method_child_outputs_v2(
                legacy_output,
                expected_method=expected_method,
                expected_acquisition=acquisition,
                expected_dataset_identity_sha256=(
                    acquisition.dataset_identity_sha256
                ),
                expected_measurements_file_sha256=artifact_sha256(
                    measurements_path
                ),
                expected_algorithm_seed=expected_seed,
            )


def test_method_child_entry_modules_have_no_top_level_evaluator_capabilities():
    import train
    import scripts.run_baselines as run_baselines

    forbidden_names = {
        "evaluate_video",
        "evaluate_video_global_affine",
        "evaluate_video_legacy_per_frame",
        "load_evaluation_truth",
    }
    for module in (train, run_baselines):
        assert forbidden_names.isdisjoint(module.__dict__)
        metric_callables = {
            name
            for name, value in module.__dict__.items()
            if callable(value)
            and getattr(value, "__module__", None)
            == "gsdiff.evaluation.metrics"
        }
        assert metric_callables == set()


def test_fresh_blind_entry_processes_keep_evaluator_import_graph_closed(
    tmp_path,
):
    acquisition, _ = _tiny_pair()
    child_cwd = tmp_path / "blind-child"
    child_cwd.mkdir()
    measurements_path = child_cwd / "measurements.npz"
    save_acquisition_data(acquisition, measurements_path)

    probe_path = tmp_path / "blind_import_probe.py"
    probe_path.write_text(
        "\n".join(
            [
                "import json",
                "from pathlib import Path",
                "import runpy",
                "import sys",
                "",
                "entry_path = Path(sys.argv[1])",
                "audit_path = Path(sys.argv[2])",
                "entry_args = sys.argv[3:]",
                "sys.argv = [str(entry_path), *entry_args]",
                "namespace = {}",
                "entry_exit_code = None",
                "try:",
                "    namespace = runpy.run_path(",
                "        str(entry_path), run_name='__main__'",
                "    )",
                "except SystemExit as error:",
                "    entry_exit_code = error.code",
                "forbidden = {",
                "    'evaluate_video',",
                "    'evaluate_video_global_affine',",
                "    'evaluate_video_legacy_per_frame',",
                "    'load_evaluation_truth',",
                "}",
                "entry_bound = sorted(forbidden.intersection(namespace))",
                "package_bound = {}",
                "process_bound = {}",
                "for module_name in (",
                "    'gsdiff.data',",
                "    'gsdiff.data.artifacts',",
                "    'gsdiff.baselines',",
                "    'gsdiff.baselines.common',",
                "):",
                "    module = sys.modules.get(module_name)",
                "    if module is not None:",
                "        names = sorted(forbidden.intersection(module.__dict__))",
                "        if names:",
                "            package_bound[module_name] = names",
                "for module_name, module in tuple(sys.modules.items()):",
                "    if module is None or not (",
                "        module_name == 'gsdiff'",
                "        or module_name.startswith('gsdiff.')",
                "    ):",
                "        continue",
                "    names = sorted(forbidden.intersection(module.__dict__))",
                "    if names:",
                "        process_bound[module_name] = names",
                "audit = {",
                "    'entry_exit_code': entry_exit_code,",
                "    'entry_bound': entry_bound,",
                "    'package_bound': package_bound,",
                "    'process_bound': process_bound,",
                "    'metrics_module_loaded': (",
                "        'gsdiff.evaluation.metrics' in sys.modules",
                "    ),",
                "}",
                "audit_path.write_text(",
                "    json.dumps(audit, sort_keys=True), encoding='utf-8'",
                ")",
                "if (",
                "    entry_bound",
                "    or package_bound",
                "    or process_bound",
                "    or audit['metrics_module_loaded']",
                "):",
                "    raise SystemExit(86)",
            ]
        ),
        encoding="utf-8",
    )

    child_env = {
        "PATH": os.environ["PATH"],
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": str(REPO_ROOT),
        "SYSTEMROOT": os.environ["SYSTEMROOT"],
    }
    cases = []
    for scene_type in ("gaussian", "grid", "recinr_se2"):
        output_dir = tmp_path / f"train-{scene_type}-output"
        config_path = tmp_path / f"train-{scene_type}.yaml"
        _write_entry_config(config_path, acquisition, output_dir)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["scene"]["type"] = scene_type
        config_path.write_text(
            yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
        )
        cases.append(
            (
                f"train-{scene_type}",
                REPO_ROOT / "train.py",
                [
                    "--legacy-compatibility",
                    "--config",
                    str(config_path),
                    "--measurements-path",
                    "measurements.npz",
                    "--dataset-identity-sha256",
                    acquisition.dataset_identity_sha256,
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cpu",
                ],
                output_dir,
            )
        )

    baseline_output = tmp_path / "baseline-dgi-output"
    baseline_config = tmp_path / "baseline-dgi.yaml"
    _write_entry_config(baseline_config, acquisition, baseline_output)
    cases.append(
        (
            "baseline-dgi",
            REPO_ROOT / "scripts" / "run_baselines.py",
            [
                "--legacy-compatibility",
                "--config",
                str(baseline_config),
                "--name",
                "blind-import-probe",
                "--baselines",
                "dgi",
                "--device",
                "cpu",
                "--measurements-path",
                "measurements.npz",
                "--dataset-identity-sha256",
                acquisition.dataset_identity_sha256,
                "--output-dir",
                str(baseline_output),
            ],
            baseline_output,
        )
    )

    for case_name, entry_path, entry_args, output_dir in cases:
        audit_path = tmp_path / f"{case_name}-audit.json"
        command = [
            str(AUTHORITATIVE_PYTHON),
            str(probe_path),
            str(entry_path),
            str(audit_path),
            *entry_args,
        ]
        completed = subprocess.run(
            command,
            cwd=child_cwd,
            env=child_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )

        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        assert completed.returncode == 0, (
            case_name,
            audit,
            completed.stdout,
            completed.stderr,
        )
        assert audit == {
            "entry_exit_code": 2,
            "entry_bound": [],
            "metrics_module_loaded": False,
            "package_bound": {},
            "process_bound": {},
        }
        assert not output_dir.exists()


def test_real_legacy_without_evaluator_input_fails_before_artifacts(
    tmp_path,
):
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
        "--legacy-compatibility",
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
        "--dataset-identity-sha256",
        acquisition.dataset_identity_sha256,
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

    assert completed.returncode == 2
    assert "truthless legacy" in completed.stderr.lower()
    assert "strict method interface" in completed.stderr.lower()
    assert not output_dir.exists()
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


# Task 3 publication datasets deliberately use a new incompatible blind
# schema. These tests bind the child-visible capability boundary.
def test_task3_blind_measurement_schema_uses_exact_member_and_metadata_allowlists(
    tmp_path,
):
    acquisition, _ = _tiny_pair(seed=314159, shape="char:5")
    destination = tmp_path / "measurements.npz"
    save_acquisition_data(acquisition, destination)

    with zipfile.ZipFile(destination) as archive:
        assert set(archive.namelist()) == {
            "__metadata_json__.npy",
            "patterns.npy",
            "measurements.npy",
            "frame_indices.npy",
            "time_grid.npy",
            "holdout_patterns.npy",
            "holdout_measurements.npy",
            "holdout_frame_indices.npy",
        }
    metadata = _read_metadata(destination)
    assert set(metadata) == {
        "schema_version",
        "dataset_identity_sha256",
        "dimensions",
        "acquisition",
        "optional_arrays",
        "array_descriptors",
    }
    assert set(metadata["dimensions"]) == {"H", "W", "T", "K", "holdout_K"}
    assert set(metadata["acquisition"]) == {
        "pattern_family",
        "pattern_values",
        "pattern_order",
        "time_assignment",
        "holdout_pattern_family",
        "noise_convention",
        "noise_sigma_absolute",
    }
    assert metadata["schema_version"] == "measurements-blind-v1"
    serialized = json.dumps(metadata, sort_keys=True)
    for sentinel in (
        "char:5",
        "314159",
        "-0.5",
        '"velocity"',
        '"acceleration"',
        '"omega"',
        '"beta"',
        "gsdiff-simulation-test-v1",
    ):
        assert sentinel not in serialized


def test_task3_blind_object_has_no_truth_or_identity_spec_attributes():
    acquisition, _ = _tiny_pair(seed=314159, shape="char:5")
    forbidden = {
        "dataset_identity_spec",
        "resolved_generation_config",
        "generator_code_version",
        "target_asset_sha256",
        "target_id",
        "target",
        "assets_sha256",
        "seed",
        "acquisition_seed",
        "motion_id",
        "motion_model",
        "motion_parameters",
        "velocity",
        "acceleration",
        "omega",
        "beta",
        "scientific_contract",
        "campaign_id",
        "method",
        "truth_path",
        "canonical_image",
        "gt_frames",
        "translation_trajectory",
        "rotation_trajectory",
    }
    assert forbidden.isdisjoint(vars(acquisition))
    assert "314159" not in json.dumps(
        _read_metadata_from_acquisition(acquisition), sort_keys=True
    )


def _read_metadata_from_acquisition(acquisition):
    destination = io.BytesIO()
    from gsdiff.data._artifact_dataset import acquisition_npz_bytes

    destination.write(acquisition_npz_bytes(acquisition))
    destination.seek(0)
    with zipfile.ZipFile(destination) as archive:
        array = np.load(
            io.BytesIO(archive.read("__metadata_json__.npy")),
            allow_pickle=False,
        )
    return json.loads(array.tobytes().decode("utf-8"))


def test_task3_blind_loader_requires_opaque_identity_and_blind_spec(tmp_path):
    from gsdiff.data.artifacts import blind_acquisition_spec

    acquisition, _ = _tiny_pair(seed=314159, shape="char:5")
    path = tmp_path / "measurements.npz"
    save_acquisition_data(acquisition, path)
    with pytest.raises(TypeError):
        load_acquisition_data(path)
    loaded = load_acquisition_data(
        path,
        expected_dataset_identity_sha256=acquisition.dataset_identity_sha256,
        expected_acquisition_spec=blind_acquisition_spec(acquisition),
    )
    assert loaded.dataset_identity_sha256 == acquisition.dataset_identity_sha256
    with pytest.raises(ArtifactValidationError, match="identity"):
        load_acquisition_data(
            path,
            expected_dataset_identity_sha256="f" * 64,
            expected_acquisition_spec=blind_acquisition_spec(acquisition),
        )


def test_task3_dummy_child_receives_only_copied_blind_measurements(tmp_path):
    acquisition, _ = _tiny_pair(seed=8675309, shape="char:5")
    source = tmp_path / "source.npz"
    save_acquisition_data(acquisition, source)
    child_cwd = tmp_path / "isolated"
    child_cwd.mkdir()
    copied = child_cwd / "measurements.npz"
    copied.write_bytes(source.read_bytes())
    probe = (
        "import json,os,sys,zipfile;"
        "p=sys.argv[1];"
        "z=zipfile.ZipFile(p);"
        "raw=' '.join(z.namelist());"
        "print(json.dumps({'argv':sys.argv[1:],'cwd':os.listdir('.'),'members':raw},sort_keys=True))"
    )
    env = {
        "PATH": os.environ["PATH"],
        "PYTHONIOENCODING": "utf-8",
        "SYSTEMROOT": os.environ["SYSTEMROOT"],
    }
    completed = subprocess.run(
        [str(AUTHORITATIVE_PYTHON), "-c", probe, "measurements.npz"],
        cwd=child_cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    observed = json.loads(completed.stdout)
    assert observed["argv"] == ["measurements.npz"]
    assert observed["cwd"] == ["measurements.npz"]
    assert "truth" not in json.dumps(observed).lower()


def _phase3b_generation_inputs(*, holdout_K=5, pattern_family="bernoulli"):
    from gsdiff.data.artifacts import TargetSnapshot

    canonical = np.arange(64, dtype=np.float32).reshape(8, 8) / 63.0
    target = TargetSnapshot(
        target_id="digit5",
        descriptor="char:5",
        assets_sha256={
            "descriptor": hashlib.sha256(b"char:5").hexdigest(),
            "font": "1" * 64,
            "renderer": "2" * 64,
        },
        canonical_image=canonical,
        renderer={
            "font_family": "DejaVu Sans",
            "fill_fraction": 0.8,
            "resample": "lanczos",
            "supersample": 4,
        },
    )
    return {
        "scientific_contract": {
            "id": "gsdiff-test-v1",
            "sha256": "3" * 64,
        },
        "target_snapshot": target,
        "motion": {
            "id": "transrot",
            "velocity": [1.0, -0.5],
            "acceleration": [0.2, 0.1],
            "omega": 0.15,
            "beta": 0.03,
        },
        "seed": 7,
        "acquisition_config": {
            "image_size": [8, 8],
            "num_frames": 3,
            "train_measurements": 12,
            "holdout_measurements": holdout_K,
            "pattern_family": pattern_family,
            "pattern_values": (
                [0, 1] if pattern_family != "gaussian" else ["real"]
            ),
            "pattern_order": "stratified",
            "time_assignment": "uniform",
            "holdout_pattern_family": "uniform-random",
            "snr_db": 25.0,
            "noise_calibration_id": "detector-absolute-v1",
        },
        "noise_calibration_entry": {
            "id": "detector-absolute-v1",
            "mode": "detector-absolute",
            "reference": "corresponding-bernoulli-reference-cell",
            "variance_ddof": 0,
            "sigma_formula": (
                "sqrt(var(y_reference,ddof=0))*10**(-snr_db/20)"
            ),
            "reuse": ["train", "holdout", "alternate-pattern"],
        },
        "generator": {
            "id": "gsdiff-corrected-sim",
            "version": "generator-v1",
            "git_commit": "a" * 40,
        },
        "runtime": {
            "dependencies_sha256": "4" * 64,
            "environment_lock_sha256": "5" * 64,
        },
    }


def _phase3b_generate(**overrides):
    from gsdiff.data.artifacts import generate_corrected_dataset

    inputs = _phase3b_generation_inputs()
    inputs.update(overrides)
    return generate_corrected_dataset(**inputs)


def test_task3_c3a_request_resolver_is_rng_free_and_matches_generation(
    monkeypatch,
):
    import gsdiff.data._corrected_generation as corrected
    from gsdiff.data.artifacts import resolve_corrected_dataset_request

    inputs = _phase3b_generation_inputs()

    def forbidden_rng(*args, **kwargs):
        raise AssertionError("request resolution entered RNG generation")

    monkeypatch.setattr(corrected, "acquisition_rng", forbidden_rng)
    request = resolve_corrected_dataset_request(**inputs)
    monkeypatch.undo()
    generated = _phase3b_generate()

    assert set(request) == {
        "schema_version",
        "scientific_contract",
        "target",
        "motion",
        "seed",
        "acquisition_config",
        "noise_calibration",
        "generator",
        "runtime",
        "resolved_generator_config",
    }
    assert request["schema_version"] == "corrected-dataset-request-v1"
    assert request["resolved_generator_config"] == (
        generated.resolved_generator_config
    )
    assert request["scientific_contract"] == {
        "id": "gsdiff-test-v1",
        "sha256": "3" * 64,
    }
    assert request["target"] == {
        "id": "digit5",
        "descriptor": "char:5",
        "assets_sha256": {
            "descriptor": hashlib.sha256(b"char:5").hexdigest(),
            "font": "1" * 64,
            "renderer": "2" * 64,
        },
        "renderer": {
            "font_family": "DejaVu Sans",
            "fill_fraction": 0.8,
            "resample": "lanczos",
            "supersample": 4,
        },
    }
    assert request["noise_calibration"] == {
        "id": "detector-absolute-v1",
        "registry_entry_sha256": hashlib.sha256(
            _canonical_json_bytes(inputs["noise_calibration_entry"])
        ).hexdigest(),
        "entry": inputs["noise_calibration_entry"],
    }


def test_task3_dataset_identity_v1_is_exact_semantic_payload_without_outputs():
    generated = _phase3b_generate()
    spec = generated.dataset_identity_spec
    assert set(spec) == {
        "schema_version",
        "scientific_contract",
        "target",
        "motion",
        "seed",
        "generator_config_sha256",
        "noise_calibration",
        "generator",
        "runtime",
    }
    assert spec["schema_version"] == "dataset-identity-v1"
    assert generated.dataset_identity_sha256 == hashlib.sha256(
        _canonical_json_bytes(spec)
    ).hexdigest()
    serialized = json.dumps(spec, sort_keys=True)
    for forbidden in (
        "arrays",
        "measurements_file",
        "evaluation_truth_file",
        "preview_file",
        "manifest",
        "path",
        "timestamp",
        "pid",
        "hostname",
        "campaign",
        "method",
        "optimizer",
        "acquisition_config_id",
        "method_config_id",
    ):
        assert forbidden not in serialized


def test_task3_corrected_dataset_metadata_views_are_defensive_native_copies():
    generated = _phase3b_generate()

    identity = generated.dataset_identity_spec
    config = generated.resolved_generator_config
    calibration = generated.noise_calibration_record
    assert type(identity) is dict
    assert type(identity["target"]) is dict
    assert type(config) is dict
    assert type(calibration) is dict

    identity["seed"] = 999
    config["dimensions"]["K"] = 999
    calibration["sigma_absolute"] = 999.0
    assert generated.dataset_identity_spec["seed"] == 7
    assert generated.resolved_generator_config["dimensions"]["K"] == 12
    assert (
        generated.noise_calibration_record["sigma_absolute"]
        != 999.0
    )
    assert generated.truth.dataset_identity_spec["seed"] == 7


@pytest.mark.parametrize(
    "mutation",
    [
        "contract-id",
        "contract-sha",
        "target-id",
        "asset-bytes",
        "motion-id",
        "seed",
        "train-k",
        "holdout-k",
        "snr",
        "pattern-family",
        "calibration-entry",
        "generator-id",
        "generator-version",
        "generator-commit",
        "dependencies",
        "environment",
    ],
)
def test_task3_dataset_identity_changes_for_every_semantic_input(mutation):
    import dataclasses as _dataclasses

    baseline_inputs = _phase3b_generation_inputs()
    changed = _phase3b_generation_inputs()
    if mutation == "contract-id":
        changed["scientific_contract"]["id"] = "other-contract"
    elif mutation == "contract-sha":
        changed["scientific_contract"]["sha256"] = "6" * 64
    elif mutation == "target-id":
        changed["target_snapshot"] = _dataclasses.replace(
            changed["target_snapshot"], target_id="digit6"
        )
    elif mutation == "asset-bytes":
        changed["target_snapshot"] = _dataclasses.replace(
            changed["target_snapshot"],
            assets_sha256={
                **changed["target_snapshot"].assets_sha256,
                "font": "6" * 64,
            },
        )
    elif mutation == "motion-id":
        changed["motion"]["id"] = "rot"
    elif mutation == "seed":
        changed["seed"] = 8
    elif mutation == "train-k":
        changed["acquisition_config"]["train_measurements"] = 13
    elif mutation == "holdout-k":
        changed["acquisition_config"]["holdout_measurements"] = 4
    elif mutation == "snr":
        changed["acquisition_config"]["snr_db"] = 20.0
    elif mutation == "pattern-family":
        changed["acquisition_config"]["pattern_family"] = "random"
    elif mutation == "calibration-entry":
        changed["noise_calibration_entry"]["sigma_formula"] += "+0"
    elif mutation == "generator-id":
        changed["generator"]["id"] = "other-generator"
    elif mutation == "generator-version":
        changed["generator"]["version"] = "generator-v2"
    elif mutation == "generator-commit":
        changed["generator"]["git_commit"] = "b" * 40
    elif mutation == "dependencies":
        changed["runtime"]["dependencies_sha256"] = "6" * 64
    elif mutation == "environment":
        changed["runtime"]["environment_lock_sha256"] = "6" * 64
    baseline = _phase3b_generate(**baseline_inputs)
    mutated = _phase3b_generate(**changed)
    assert mutated.dataset_identity_sha256 != baseline.dataset_identity_sha256


@pytest.mark.parametrize(
    ("stream_id", "expected"),
    [
        (0, [1, 1, 0, 0, 1, 0, 0, 1]),
        (1, [1, 1, 0, 1, 0, 0, 0, 0]),
        (2, [0, 1, 1, 0, 1, 1, 1, 1]),
        (3, [0, 1, 1, 0, 1, 1, 0, 1]),
    ],
)
def test_task3_pcg64_seedsequence_stream_known_answers(stream_id, expected):
    from gsdiff.data.artifacts import acquisition_rng

    actual = acquisition_rng(7, stream_id).integers(0, 2, 8, dtype=np.int8)
    assert actual.tolist() == expected


def test_task3_holdout_presence_cannot_change_any_training_array():
    without = _phase3b_generate(
        **_phase3b_generation_inputs(holdout_K=0)
    )
    with_holdout = _phase3b_generate(
        **_phase3b_generation_inputs(holdout_K=5)
    )
    for name in ("patterns", "measurements", "frame_indices", "time_grid"):
        np.testing.assert_array_equal(
            getattr(without.acquisition, name),
            getattr(with_holdout.acquisition, name),
        )


def test_task3_detector_sigma_uses_ddof_zero_and_is_reused_everywhere():
    generated = _phase3b_generate()
    record = generated.noise_calibration_record
    expected = (
        np.sqrt(record["reference_variance"])
        * 10 ** (-record["requested_snr_db"] / 20.0)
    )
    assert record["ddof"] == 0
    assert record["sigma_absolute"] == pytest.approx(expected, abs=1e-15)
    assert (
        generated.acquisition.acquisition["noise_sigma_absolute"]
        == record["sigma_absolute"]
    )


def test_task3_cross_pattern_cell_reuses_bernoulli_reference_sigma():
    bernoulli = _phase3b_generate()
    random_inputs = _phase3b_generation_inputs(pattern_family="random")
    random_cell = _phase3b_generate(**random_inputs)
    assert (
        bernoulli.noise_calibration_record["reference_cell_sha256"]
        == random_cell.noise_calibration_record["reference_cell_sha256"]
    )
    assert (
        bernoulli.noise_calibration_record["sigma_absolute"]
        == random_cell.noise_calibration_record["sigma_absolute"]
    )


def test_task3_zero_reference_variance_means_exactly_zero_added_noise():
    inputs = _phase3b_generation_inputs()
    inputs["target_snapshot"] = dataclasses.replace(
        inputs["target_snapshot"],
        canonical_image=np.zeros((8, 8), dtype=np.float32),
    )
    generated = _phase3b_generate(**inputs)
    assert generated.noise_calibration_record["sigma_absolute"] == 0.0
    assert np.count_nonzero(generated.acquisition.measurements) == 0
    assert np.count_nonzero(generated.acquisition.holdout_measurements) == 0


def test_task3_motion_uses_normalized_time_without_hidden_half_factor():
    inputs = _phase3b_generation_inputs()
    inputs["motion"] = {
        "id": "accel",
        "velocity": [6, 6],
        "acceleration": [3, 3],
        "omega": 0.2,
        "beta": 0.1,
    }
    generated = _phase3b_generate(**inputs)
    np.testing.assert_array_equal(
        generated.acquisition.time_grid,
        np.array([0.0, 0.5, 1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        generated.truth.translation_trajectory[-1],
        np.array([9.0, 9.0], dtype=np.float32),
        rtol=0,
        atol=1e-6,
    )
    assert generated.truth.rotation_trajectory[-1] == pytest.approx(0.3)


def test_task3_corrected_generation_is_bit_repeatable():
    first = _phase3b_generate()
    second = _phase3b_generate()
    assert first.dataset_identity_sha256 == second.dataset_identity_sha256
    assert first.noise_calibration_sha256 == second.noise_calibration_sha256
    for name in (
        "patterns",
        "measurements",
        "frame_indices",
        "time_grid",
        "holdout_patterns",
        "holdout_measurements",
        "holdout_frame_indices",
    ):
        np.testing.assert_array_equal(
            getattr(first.acquisition, name),
            getattr(second.acquisition, name),
        )
    np.testing.assert_array_equal(first.truth.gt_frames, second.truth.gt_frames)


def test_task3_noise_calibration_sha_hashes_complete_per_cell_record():
    generated = _phase3b_generate()
    record = generated.noise_calibration_record
    assert set(record) == {
        "schema_version",
        "calibration",
        "scientific_contract",
        "target_id",
        "motion_id",
        "seed",
        "reference_cell_sha256",
        "reference_measurements",
        "requested_snr_db",
        "ddof",
        "reference_variance",
        "sigma_absolute",
        "realized_snr_db",
        "generator",
        "generator_config_sha256",
        "runtime",
    }
    assert record["schema_version"] == "noise-calibration-record-v1"
    assert generated.noise_calibration_sha256 == hashlib.sha256(
        _canonical_json_bytes(record)
    ).hexdigest()
    assert (
        generated.noise_calibration_sha256
        != record["calibration"]["registry_entry_sha256"]
    )
    assert set(record["reference_measurements"]) == {
        "dtype",
        "shape",
        "sha256",
    }
    assert set(record["realized_snr_db"]) == {"train", "holdout"}


def test_task3_file_target_hashes_and_decodes_one_raw_snapshot(tmp_path):
    from gsdiff.data.artifacts import resolve_target_snapshot

    repo = tmp_path / "repo"
    asset = repo / "assets" / "target.png"
    asset.parent.mkdir(parents=True)
    pixels = np.arange(64, dtype=np.uint8).reshape(8, 8)
    Image.fromarray(pixels).save(asset, compress_level=1)
    first_raw = asset.read_bytes()
    first = resolve_target_snapshot(
        repo_root=repo,
        target_id="tank",
        descriptor="assets/target.png",
        H=8,
        W=8,
    )
    Image.fromarray(pixels).save(asset, compress_level=9)
    second_raw = asset.read_bytes()
    second = resolve_target_snapshot(
        repo_root=repo,
        target_id="tank",
        descriptor="assets/target.png",
        H=8,
        W=8,
    )
    assert first_raw != second_raw
    np.testing.assert_array_equal(first.canonical_image, second.canonical_image)
    assert first.assets_sha256 != second.assets_sha256
    assert first.assets_sha256 == {
        "assets/target.png": hashlib.sha256(first_raw).hexdigest()
    }


def test_task3_file_target_rejects_symlinked_repository_root(tmp_path):
    from gsdiff.data.artifacts import resolve_target_snapshot

    real_repo = tmp_path / "real-repo"
    asset = real_repo / "assets" / "target.png"
    asset.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(asset)
    linked_repo = tmp_path / "linked-repo"
    try:
        linked_repo.symlink_to(real_repo, target_is_directory=True)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    with pytest.raises(ArtifactValidationError, match="linked|reparse"):
        resolve_target_snapshot(
            repo_root=linked_repo,
            target_id="tank",
            descriptor="assets/target.png",
            H=8,
            W=8,
        )


def test_task3_file_target_rejects_broken_symlink_leaf(tmp_path):
    from gsdiff.data.artifacts import resolve_target_snapshot

    repo = tmp_path / "repo"
    asset = repo / "assets" / "target.png"
    asset.parent.mkdir(parents=True)
    try:
        asset.symlink_to(repo / "missing.png")
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"file symlinks unavailable: {error}")

    with pytest.raises(ArtifactValidationError, match="linked|reparse"):
        resolve_target_snapshot(
            repo_root=repo,
            target_id="tank",
            descriptor="assets/target.png",
            H=8,
            W=8,
        )


def test_task3_file_target_guard_detects_lexists_only_link(
    tmp_path, monkeypatch
):
    import stat as _stat
    from types import SimpleNamespace
    import gsdiff.data._corrected_generation as corrected

    missing_link = (tmp_path / "broken-link").absolute()
    real_lexists = corrected.os.path.lexists
    real_lstat = corrected.os.lstat

    def fake_lexists(path):
        return Path(path) == missing_link or real_lexists(path)

    def fake_lstat(path):
        if Path(path) == missing_link:
            return SimpleNamespace(
                st_mode=_stat.S_IFLNK,
                st_file_attributes=0,
            )
        return real_lstat(path)

    monkeypatch.setattr(corrected.os.path, "lexists", fake_lexists)
    monkeypatch.setattr(corrected.os, "lstat", fake_lstat)
    with pytest.raises(ArtifactValidationError, match="linked|reparse"):
        corrected._validate_existing_ancestors(missing_link)


def test_task3_builtin_glyph_binds_actual_font_bytes_without_cache_writes(
    tmp_path, monkeypatch
):
    import importlib.util
    from gsdiff.data.artifacts import resolve_target_snapshot

    cache = tmp_path / "matplotlib-cache"
    monkeypatch.setenv("MPLCONFIGDIR", str(cache))
    snapshot = resolve_target_snapshot(
        repo_root=REPO_ROOT,
        target_id="digit5",
        descriptor="char:5",
        H=8,
        W=8,
    )
    spec = importlib.util.find_spec("matplotlib")
    assert spec is not None and spec.origin is not None
    font = (
        Path(spec.origin).parent
        / "mpl-data"
        / "fonts"
        / "ttf"
        / "DejaVuSans.ttf"
    )
    assert snapshot.assets_sha256["font"] == hashlib.sha256(
        font.read_bytes()
    ).hexdigest()
    assert set(snapshot.assets_sha256) == {
        "descriptor",
        "font",
        "renderer",
    }
    assert not cache.exists()


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("scientific_contract", type("DictSubclass", (dict,), {})({"id": "x", "sha256": "3" * 64})),
        ("motion", type("DictSubclass", (dict,), {})({"id": "x"})),
        ("seed", True),
        ("seed", 7.0),
        ("generator-id", type("StringSubclass", (str,), {})("gsdiff-corrected-sim")),
        ("image-size", type("ListSubclass", (list,), {})([8, 8])),
        ("snr", float("nan")),
        ("snr", True),
    ],
)
def test_task3_public_generation_rejects_type_spoofs_and_nonfinite(
    field, invalid
):
    inputs = _phase3b_generation_inputs()
    if field in {"scientific_contract", "motion", "seed"}:
        inputs[field] = invalid
    elif field == "generator-id":
        inputs["generator"]["id"] = invalid
    elif field == "image-size":
        inputs["acquisition_config"]["image_size"] = invalid
    else:
        inputs["acquisition_config"]["snr_db"] = invalid
    with pytest.raises((TypeError, ArtifactValidationError, ValueError)):
        _phase3b_generate(**inputs)


def test_task3_blind_model_rejects_direct_mapping_and_string_subclasses():
    acquisition, _ = _tiny_pair()
    mapping_subclass = type("DictSubclass", (dict,), {})
    string_subclass = type("StringSubclass", (str,), {})
    with pytest.raises((TypeError, ArtifactValidationError)):
        dataclasses.replace(
            acquisition,
            acquisition=mapping_subclass(acquisition.acquisition),
        )
    spoofed = _mutable_json(acquisition.acquisition)
    spoofed["pattern_family"] = string_subclass("bernoulli")
    with pytest.raises((TypeError, ArtifactValidationError)):
        dataclasses.replace(acquisition, acquisition=spoofed)


def _phase3c_build_bundle():
    from gsdiff.data.artifacts import (
        build_dataset_manifest,
        build_dataset_payloads,
        dataset_manifest_bytes,
    )

    generated = _phase3b_generate()
    payloads = build_dataset_payloads(generated)
    manifest = build_dataset_manifest(generated, payloads)
    return (
        generated,
        payloads,
        manifest,
        dataset_manifest_bytes(manifest),
    )


def _phase3c_zip_metadata(payload):
    with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
        return archive.comment, archive.infolist()


def test_task3_c1a_three_payloads_are_repeatable_and_round_trip_exact():
    from gsdiff.data.artifacts import verify_dataset_payload_bytes

    generated, first, manifest, _ = _phase3c_build_bundle()
    _, second, _, _ = _phase3c_build_bundle()
    assert set(first) == {
        "measurements.npz",
        "evaluation-truth.npz",
        "preview.png",
    }
    assert all(type(payload) is bytes for payload in first.values())
    assert first == second

    acquisition, truth, preview = verify_dataset_payload_bytes(
        first, manifest
    )
    for name in (
        "patterns",
        "measurements",
        "frame_indices",
        "time_grid",
        "holdout_patterns",
        "holdout_measurements",
        "holdout_frame_indices",
    ):
        expected = getattr(generated.acquisition, name)
        actual = getattr(acquisition, name)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.flags.c_contiguous
        assert actual.tobytes(order="C") == expected.tobytes(order="C")
    for name in (
        "canonical_image",
        "gt_frames",
        "translation_trajectory",
        "rotation_trajectory",
        "gt_velocity",
        "gt_acceleration",
    ):
        expected = getattr(generated.truth, name)
        actual = getattr(truth, name)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.flags.c_contiguous
        assert actual.tobytes(order="C") == expected.tobytes(order="C")

    expected_preview = np.rint(
        np.clip(generated.truth.canonical_image, 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    assert preview.dtype == np.uint8
    assert preview.shape == (generated.truth.H, generated.truth.W)
    assert preview.flags.c_contiguous
    np.testing.assert_array_equal(preview, expected_preview)


def test_task3_c1a_npz_and_png_codecs_bind_canonical_container_metadata():
    _, payloads, _, _ = _phase3c_build_bundle()
    for name in ("measurements.npz", "evaluation-truth.npz"):
        comment, infos = _phase3c_zip_metadata(payloads[name])
        assert comment == b""
        assert [info.filename for info in infos] == sorted(
            info.filename for info in infos
        )
        for info in infos:
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.compress_type == zipfile.ZIP_DEFLATED
            assert info.create_system == 3
            assert info.external_attr == 0x81800000
            assert info.comment == b""
            assert info.extra == b""
            assert info.flag_bits == 0
            assert info.create_version == 20
            assert info.extract_version == 20
            assert info.volume == 0
            assert info.internal_attr == 0

    with Image.open(io.BytesIO(payloads["preview.png"])) as preview:
        assert preview.format == "PNG"
        assert preview.mode == "L"
        assert preview.size == (8, 8)
        assert not preview.info

    with zipfile.ZipFile(
        io.BytesIO(payloads["evaluation-truth.npz"]), "r"
    ) as archive:
        metadata_array = np.load(
            io.BytesIO(archive.read("__metadata_json__.npy")),
            allow_pickle=False,
        )
    metadata = json.loads(metadata_array.tobytes().decode("utf-8"))
    assert metadata["schema"] == "evaluation-truth-v2"


def test_task3_c1a_manifest_is_exact_canonical_semantic_payload():
    generated, payloads, manifest, manifest_bytes = _phase3c_build_bundle()
    assert set(manifest) == {
        "schema_version",
        "status",
        "dataset_identity_sha256",
        "dataset_identity_spec",
        "resolved_generator_config",
        "noise_calibration_record",
        "files",
    }
    assert manifest["schema_version"] == "dataset-manifest-v1"
    assert manifest["status"] == "complete"
    assert (
        manifest["dataset_identity_sha256"]
        == generated.dataset_identity_sha256
    )
    assert manifest["dataset_identity_spec"] == (
        generated.dataset_identity_spec
    )
    assert manifest["resolved_generator_config"] == (
        generated.resolved_generator_config
    )
    assert manifest["noise_calibration_record"] == (
        generated.noise_calibration_record
    )
    assert set(manifest["files"]) == set(payloads)
    expected_descriptors = {
        "measurements.npz": (
            "blind-measurements",
            "measurements-blind-v1",
        ),
        "evaluation-truth.npz": (
            "evaluation-truth",
            "evaluation-truth-v2",
        ),
        "preview.png": ("preview", "dataset-preview-v1"),
    }
    for name, (role, schema) in expected_descriptors.items():
        entry = manifest["files"][name]
        assert set(entry) == {
            "role",
            "schema_version",
            "sha256",
            "size_bytes",
        }
        assert entry == {
            "role": role,
            "schema_version": schema,
            "sha256": hashlib.sha256(payloads[name]).hexdigest(),
            "size_bytes": len(payloads[name]),
        }
    assert manifest_bytes == _canonical_json_bytes(manifest)
    serialized = manifest_bytes.decode("utf-8")
    for forbidden in (
        "dataset_manifest_sha256",
        "timestamp",
        "created_at",
        "path",
        "pid",
        "hostname",
        "staging",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "mutation",
    ["duplicate-key", "nan", "whitespace", "invalid-utf8"],
)
def test_task3_c1a_manifest_parser_rejects_noncanonical_json(mutation):
    from gsdiff.data.artifacts import parse_dataset_manifest_bytes

    _, _, _, canonical = _phase3c_build_bundle()
    if mutation == "duplicate-key":
        payload = canonical.replace(
            b"{",
            b'{"status":"complete",',
            1,
        )
    elif mutation == "nan":
        payload = canonical.replace(
            b'"status":"complete"',
            b'"status":NaN',
            1,
        )
    elif mutation == "whitespace":
        payload = b" " + canonical
    else:
        payload = b"\xff" + canonical
    with pytest.raises((TypeError, ValueError, ArtifactValidationError)):
        parse_dataset_manifest_bytes(payload)


def test_task3_c1a_manifest_public_boundary_rejects_type_spoofs():
    from gsdiff.data.artifacts import (
        build_dataset_manifest,
        dataset_manifest_bytes,
    )

    generated, payloads, manifest, _ = _phase3c_build_bundle()
    dict_subclass = type("DictSubclass", (dict,), {})
    with pytest.raises((TypeError, ValueError, ArtifactValidationError)):
        build_dataset_manifest(generated, dict_subclass(payloads))
    with pytest.raises((TypeError, ValueError, ArtifactValidationError)):
        dataset_manifest_bytes(dict_subclass(manifest))


def test_task3_c1a_payloads_are_fresh_process_and_directory_invariant(
    tmp_path,
):
    script = "\n".join(
        [
            "import runpy",
            "from pathlib import Path",
            "from gsdiff.data.artifacts import (",
            "    build_dataset_manifest,",
            "    build_dataset_payloads,",
            "    dataset_manifest_bytes,",
            ")",
            f"ns = runpy.run_path({str(Path(__file__).resolve())!r})",
            "generated = ns['_phase3b_generate']()",
            "payloads = build_dataset_payloads(generated)",
            "manifest = build_dataset_manifest(generated, payloads)",
            "payloads['dataset-manifest.json'] = (",
            "    dataset_manifest_bytes(manifest)",
            ")",
            "for name, payload in payloads.items():",
            "    Path(name).write_bytes(payload)",
        ]
    )
    directories = [tmp_path / "first", tmp_path / "second"]
    for directory in directories:
        directory.mkdir()
        subprocess.run(
            [str(AUTHORITATIVE_PYTHON), "-c", script],
            cwd=directory,
            env={
                "PATH": os.environ["PATH"],
                "PYTHONIOENCODING": "utf-8",
                "PYTHONPATH": str(REPO_ROOT),
                "SYSTEMROOT": os.environ["SYSTEMROOT"],
            },
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
        )
    assert {
        path.name: path.read_bytes() for path in directories[0].iterdir()
    } == {
        path.name: path.read_bytes() for path in directories[1].iterdir()
    }


def test_task3_c1a_blind_payload_must_match_manifest_generator_config():
    from gsdiff.data._artifact_dataset import acquisition_npz_bytes
    from gsdiff.data.artifacts import verify_dataset_payload_bytes

    generated, payloads, manifest, _ = _phase3c_build_bundle()
    changed_acquisition = {
        **generated.acquisition.acquisition,
        "pattern_family": "random",
    }
    changed = dataclasses.replace(
        generated.acquisition,
        acquisition=changed_acquisition,
    )
    changed_payload = acquisition_npz_bytes(changed)
    payloads["measurements.npz"] = changed_payload
    manifest["files"]["measurements.npz"]["sha256"] = hashlib.sha256(
        changed_payload
    ).hexdigest()
    manifest["files"]["measurements.npz"]["size_bytes"] = len(
        changed_payload
    )

    with pytest.raises(ArtifactValidationError, match="acquisition|config"):
        verify_dataset_payload_bytes(payloads, manifest)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("reference_cell_sha256", "not-a-sha"),
        ("reference_dtype", "<f4"),
        ("reference_shape", [11]),
    ],
)
def test_task3_c1a_calibration_reference_descriptor_is_semantically_bound(
    field, invalid
):
    from gsdiff.data.artifacts import dataset_manifest_bytes

    _, _, manifest, _ = _phase3c_build_bundle()
    record = manifest["noise_calibration_record"]
    if field == "reference_cell_sha256":
        record[field] = invalid
    elif field == "reference_dtype":
        record["reference_measurements"]["dtype"] = invalid
    else:
        record["reference_measurements"]["shape"] = invalid
    record_sha = hashlib.sha256(
        _canonical_json_bytes(record)
    ).hexdigest()
    manifest["dataset_identity_spec"]["noise_calibration"]["sha256"] = (
        record_sha
    )
    manifest["dataset_identity_sha256"] = hashlib.sha256(
        _canonical_json_bytes(manifest["dataset_identity_spec"])
    ).hexdigest()

    with pytest.raises(ArtifactValidationError):
        dataset_manifest_bytes(manifest)


def _phase3c_rezip(
    payload,
    transform,
    *,
    canonical,
    compresslevel=9,
):
    import gsdiff.data._artifact_io as artifact_io

    with zipfile.ZipFile(io.BytesIO(payload), "r") as source:
        members = {
            name: source.read(name) for name in source.namelist()
        }
    members = transform(members)
    destination = io.BytesIO()
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=compresslevel,
        allowZip64=False,
    ) as archive:
        for name in sorted(members):
            if canonical:
                archive.writestr(
                    artifact_io._zip_info(name),
                    members[name],
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=compresslevel,
                )
            else:
                archive.writestr(name, members[name])
    return destination.getvalue()


def test_task3_c1b_safe_snapshot_detects_same_size_path_replacement(
    tmp_path,
):
    from gsdiff.data._artifact_io import (
        read_safe_file_snapshot,
        verify_safe_file_snapshot,
    )

    path = tmp_path / "payload.bin"
    path.write_bytes(b"original")
    snapshot = read_safe_file_snapshot(path, max_bytes=8)
    assert snapshot.raw == b"original"
    assert snapshot.sha256 == hashlib.sha256(b"original").hexdigest()
    assert snapshot.size_bytes == 8

    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"modified")
    os.replace(replacement, path)
    with pytest.raises(ArtifactValidationError, match="changed|snapshot"):
        verify_safe_file_snapshot(snapshot)


@pytest.mark.parametrize("substitution", ["directory", "hardlink", "oversize"])
def test_task3_c1b_safe_snapshot_rejects_unsafe_file_types_and_bounds(
    tmp_path, substitution
):
    from gsdiff.data._artifact_io import read_safe_file_snapshot

    path = tmp_path / "payload.bin"
    max_bytes = 8
    if substitution == "directory":
        path.mkdir()
    elif substitution == "hardlink":
        source = tmp_path / "source.bin"
        source.write_bytes(b"payload")
        try:
            os.link(source, path)
        except OSError as error:
            pytest.skip(f"hardlinks unavailable: {error}")
    else:
        path.write_bytes(b"123456789")
    with pytest.raises(ArtifactValidationError):
        read_safe_file_snapshot(path, max_bytes=max_bytes)


def test_task3_c1b_safe_snapshot_rejects_fifo(tmp_path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable on this platform")
    from gsdiff.data._artifact_io import read_safe_file_snapshot

    path = tmp_path / "payload.fifo"
    os.mkfifo(path)
    with pytest.raises(ArtifactValidationError):
        read_safe_file_snapshot(path, max_bytes=8)


def test_task3_c1b_directory_inventory_detects_post_snapshot_race(tmp_path):
    from gsdiff.data._artifact_io import (
        capture_directory_inventory,
        verify_directory_inventory,
    )

    (tmp_path / "a.bin").write_bytes(b"a")
    inventory = capture_directory_inventory(tmp_path)
    (tmp_path / "b.bin").write_bytes(b"b")
    with pytest.raises(ArtifactValidationError, match="inventory|changed"):
        verify_directory_inventory(inventory)


@pytest.mark.parametrize("mutation", ["unknown", "oversize", "noncanonical"])
def test_task3_c1b_bounded_zip_rejects_before_bulk_decode(mutation):
    from gsdiff.data._artifact_io import read_npz_members_bytes

    _, payloads, _, _ = _phase3c_build_bundle()
    payload = payloads["measurements.npz"]
    with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
        allowed = set(archive.namelist())
        largest = max(info.file_size for info in archive.infolist())
    if mutation == "unknown":
        payload = _phase3c_rezip(
            payload,
            lambda members: {**members, "unknown.npy": b"not-an-array"},
            canonical=True,
        )
        with pytest.raises(ArtifactValidationError, match="unknown"):
            read_npz_members_bytes(
                payload,
                allowed_members=allowed,
            )
    elif mutation == "oversize":
        with pytest.raises(ArtifactValidationError, match="size|large|bound"):
            read_npz_members_bytes(
                payload,
                allowed_members=allowed,
                max_member_bytes=largest - 1,
            )
    else:
        payload = _phase3c_rezip(
            payload, lambda members: members, canonical=False
        )
        with pytest.raises(ArtifactValidationError, match="canonical"):
            read_npz_members_bytes(
                payload,
                allowed_members=allowed,
            )


@pytest.mark.parametrize("mutation", ["trailing-junk", "compresslevel"])
def test_task3_c1b_zip_requires_exact_fixed_codec_bytes(mutation):
    from gsdiff.data._artifact_io import read_npz_members_bytes
    from gsdiff.data.artifacts import verify_dataset_payload_bytes

    _, payloads, manifest, _ = _phase3c_build_bundle()
    payload = payloads["measurements.npz"]
    if mutation == "trailing-junk":
        changed = payload + b"TRAILING-JUNK"
    else:
        changed = _phase3c_rezip(
            payload,
            lambda members: members,
            canonical=True,
            compresslevel=1,
        )
        assert changed != payload
    payloads["measurements.npz"] = changed
    manifest["files"]["measurements.npz"]["sha256"] = hashlib.sha256(
        changed
    ).hexdigest()
    manifest["files"]["measurements.npz"]["size_bytes"] = len(changed)

    with pytest.raises(ArtifactValidationError, match="canonical"):
        read_npz_members_bytes(changed)
    with pytest.raises(ArtifactValidationError, match="canonical"):
        verify_dataset_payload_bytes(payloads, manifest)


@pytest.mark.parametrize("mutation", ["whitespace", "duplicate-key"])
def test_task3_c1b_npz_metadata_json_must_be_canonical_and_unique(mutation):
    from gsdiff.data._artifact_dataset import load_acquisition_data_bytes

    generated, payloads, _, _ = _phase3c_build_bundle()
    payload = payloads["measurements.npz"]

    def mutate(members):
        metadata = np.load(
            io.BytesIO(members["__metadata_json__.npy"]),
            allow_pickle=False,
        ).tobytes()
        if mutation == "whitespace":
            metadata = b" " + metadata
        else:
            metadata = metadata.replace(
                b"{",
                b'{"schema_version":"measurements-blind-v1",',
                1,
            )
        changed = dict(members)
        changed["__metadata_json__.npy"] = _npy_bytes(
            np.frombuffer(metadata, dtype=np.uint8)
        )
        return changed

    payload = _phase3c_rezip(payload, mutate, canonical=True)
    with pytest.raises(ArtifactValidationError, match="metadata|canonical"):
        load_acquisition_data_bytes(
            payload,
            expected_dataset_identity_sha256=(
                generated.dataset_identity_sha256
            ),
        )


@pytest.mark.parametrize("mutation", ["array-subclass", "nonfinite", "object"])
def test_task3_c1b_npz_writer_rejects_unsafe_array_values(mutation):
    from gsdiff.data._artifact_io import npz_bytes

    if mutation == "array-subclass":
        subclass = type("ArraySubclass", (np.ndarray,), {})
        array = np.array([1.0], dtype=np.float32).view(subclass)
    elif mutation == "nonfinite":
        array = np.array([np.nan], dtype=np.float32)
    else:
        array = np.array([object()], dtype=object)
    with pytest.raises((TypeError, ValueError, ArtifactValidationError)):
        npz_bytes(arrays={"unsafe": array}, metadata={"schema": "test"})


def test_task3_c1qa_manifest_bytes_bound_precedes_json_parse(monkeypatch):
    import gsdiff.data._artifact_bundle as bundle

    _, _, _, manifest_bytes = _phase3c_build_bundle()
    monkeypatch.setattr(
        bundle,
        "MAX_DATASET_MANIFEST_BYTES",
        len(manifest_bytes) - 1,
    )

    def forbidden_parse(*args, **kwargs):
        raise AssertionError("oversize manifest reached JSON parser")

    monkeypatch.setattr(bundle.json, "loads", forbidden_parse)
    with pytest.raises(ArtifactValidationError, match="manifest.*bound"):
        bundle.parse_dataset_manifest_bytes(manifest_bytes)


def test_task3_c1qa_preview_bytes_bound_precedes_png_decode(monkeypatch):
    import gsdiff.data._artifact_bundle as bundle

    _, payloads, manifest, _ = _phase3c_build_bundle()
    monkeypatch.setattr(
        bundle,
        "MAX_DATASET_PREVIEW_BYTES",
        len(payloads["preview.png"]) - 1,
    )

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("oversize preview reached PNG decoder")

    monkeypatch.setattr(bundle, "_decode_preview", forbidden_decode)
    with pytest.raises(ArtifactValidationError, match="preview.*bound"):
        bundle.verify_dataset_payload_bytes(payloads, manifest)


def test_task3_c1qa_manifest_declared_role_size_is_bounded():
    from gsdiff.data.artifacts import dataset_manifest_bytes
    import gsdiff.data._artifact_bundle as bundle

    _, _, manifest, _ = _phase3c_build_bundle()
    manifest["files"]["preview.png"]["size_bytes"] = (
        bundle.MAX_DATASET_PREVIEW_BYTES + 1
    )
    with pytest.raises(ArtifactValidationError, match="preview.*bound"):
        dataset_manifest_bytes(manifest)


@pytest.mark.parametrize(
    "mutation",
    [
        "config-nested-extra",
        "renderer-extra",
        "reference-sha",
        "reference-dtype",
        "reference-shape",
        "calibration-nested-extra",
    ],
)
def test_task3_c1qa_schema_and_runtime_reject_same_nested_mutations(
    mutation,
):
    from jsonschema import Draft202012Validator
    from gsdiff.data.artifacts import dataset_manifest_bytes

    schema = json.loads(
        (
            REPO_ROOT / "schemas" / "dataset-manifest-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(schema)
    _, _, manifest, _ = _phase3c_build_bundle()
    if mutation == "config-nested-extra":
        manifest["resolved_generator_config"]["rng"]["extra"] = 1
    elif mutation == "renderer-extra":
        manifest["resolved_generator_config"]["target"]["renderer"][
            "extra"
        ] = 1
    elif mutation == "reference-sha":
        manifest["noise_calibration_record"][
            "reference_cell_sha256"
        ] = "not-a-sha"
    elif mutation == "reference-dtype":
        manifest["noise_calibration_record"]["reference_measurements"][
            "dtype"
        ] = "<f4"
    elif mutation == "reference-shape":
        manifest["noise_calibration_record"]["reference_measurements"][
            "shape"
        ] = [12, 1]
    else:
        manifest["noise_calibration_record"]["calibration"]["extra"] = 1

    assert list(validator.iter_errors(manifest)), mutation
    with pytest.raises((TypeError, ValueError, ArtifactValidationError)):
        dataset_manifest_bytes(manifest)


def test_task3_c1qa_schema_accepts_runtime_valid_manifest():
    from jsonschema import Draft202012Validator

    schema = json.loads(
        (
            REPO_ROOT / "schemas" / "dataset-manifest-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    _, _, manifest, _ = _phase3c_build_bundle()
    Draft202012Validator(schema).validate(manifest)


def _phase3c_write_dataset_directory(tmp_path):
    generated, payloads, manifest, manifest_bytes = _phase3c_build_bundle()
    dataset_dir = tmp_path / generated.dataset_identity_sha256
    dataset_dir.mkdir()
    for name, payload in payloads.items():
        (dataset_dir / name).write_bytes(payload)
    (dataset_dir / "dataset-manifest.json").write_bytes(manifest_bytes)
    return generated, payloads, manifest, manifest_bytes, dataset_dir


def test_task3_c2a1_final_directory_round_trip_returns_isolated_evidence(
    tmp_path,
):
    from gsdiff.data import (
        VerifiedDatasetDirectory as DataVerifiedDirectory,
        verify_dataset_directory as verify_from_data,
    )
    from gsdiff.data.artifacts import (
        VerifiedDatasetDirectory,
        verify_dataset_directory,
    )

    generated, payloads, manifest, manifest_bytes, dataset_dir = (
        _phase3c_write_dataset_directory(tmp_path)
    )
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    verified = verify_dataset_directory(
        dataset_dir,
        expected_dataset_identity_sha256=(
            generated.dataset_identity_sha256
        ),
        expected_dataset_manifest_sha256=manifest_sha256,
    )

    assert DataVerifiedDirectory is VerifiedDatasetDirectory
    assert verify_from_data is verify_dataset_directory
    assert type(verified) is VerifiedDatasetDirectory
    assert verified.dataset_dir == dataset_dir.absolute()
    assert (
        verified.dataset_identity_sha256
        == generated.dataset_identity_sha256
    )
    assert verified.dataset_manifest_sha256 == manifest_sha256
    assert verified.manifest_externally_anchored is True
    assert verified.manifest == manifest
    assert set(verified.payload_evidence) == set(payloads)
    for name, payload in payloads.items():
        evidence = verified.payload_evidence[name]
        assert evidence.sha256 == hashlib.sha256(payload).hexdigest()
        assert evidence.size_bytes == len(payload)
    assert (
        verified.acquisition.dataset_identity_sha256
        == generated.dataset_identity_sha256
    )
    assert (
        verified.truth.dataset_identity_sha256
        == generated.dataset_identity_sha256
    )
    assert verified.preview.flags.c_contiguous
    assert not verified.preview.flags.writeable

    returned_manifest = verified.manifest
    returned_manifest["status"] = "tampered"
    assert verified.manifest["status"] == "complete"
    with pytest.raises(TypeError):
        verified.payload_evidence["preview.png"] = object()
    with pytest.raises(ValueError):
        verified.preview[0, 0] = 0

    self_consistent = verify_dataset_directory(dataset_dir)
    assert self_consistent.manifest_externally_anchored is False


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong-name",
        "extra-file",
        "nested-directory",
        "leaf-directory",
        "hardlink",
        "root-file",
    ],
)
def test_task3_c2a1_final_directory_requires_exact_safe_inventory(
    tmp_path, mutation
):
    from gsdiff.data.artifacts import verify_dataset_directory

    generated, _, _, _, dataset_dir = _phase3c_write_dataset_directory(
        tmp_path
    )
    if mutation == "wrong-name":
        changed = tmp_path / ("0" * 64)
        dataset_dir.rename(changed)
        dataset_dir = changed
    elif mutation == "extra-file":
        (dataset_dir / "extra.bin").write_bytes(b"extra")
    elif mutation == "nested-directory":
        nested = dataset_dir / "nested"
        nested.mkdir()
        (nested / "payload.bin").write_bytes(b"nested")
    elif mutation == "leaf-directory":
        leaf = dataset_dir / "preview.png"
        leaf.unlink()
        leaf.mkdir()
    elif mutation == "hardlink":
        leaf = dataset_dir / "preview.png"
        source = tmp_path / "preview-source.png"
        source.write_bytes(leaf.read_bytes())
        leaf.unlink()
        try:
            os.link(source, leaf)
        except OSError as error:
            pytest.skip(f"hardlinks unavailable: {error}")
    else:
        dataset_dir = tmp_path / ("f" * 64)
        dataset_dir.write_bytes(b"not-a-directory")

    with pytest.raises(ArtifactValidationError):
        verify_dataset_directory(
            dataset_dir,
            expected_dataset_identity_sha256=(
                generated.dataset_identity_sha256
            ),
        )


@pytest.mark.parametrize(
    ("anchor", "value"),
    [
        ("identity", "0" * 64),
        ("manifest", "0" * 64),
        ("identity", "not-a-sha"),
        ("manifest", "not-a-sha"),
    ],
)
def test_task3_c2a1_external_anchors_are_strict(tmp_path, anchor, value):
    from gsdiff.data.artifacts import verify_dataset_directory

    generated, _, _, _, dataset_dir = _phase3c_write_dataset_directory(
        tmp_path
    )
    kwargs = {
        "expected_dataset_identity_sha256": (
            generated.dataset_identity_sha256
        )
    }
    if anchor == "identity":
        kwargs["expected_dataset_identity_sha256"] = value
    else:
        kwargs["expected_dataset_manifest_sha256"] = value
    with pytest.raises(ArtifactValidationError, match=anchor):
        verify_dataset_directory(dataset_dir, **kwargs)


@pytest.mark.parametrize(
    ("name", "constant"),
    [
        ("dataset-manifest.json", "MAX_DATASET_MANIFEST_BYTES"),
        ("preview.png", "MAX_DATASET_PREVIEW_BYTES"),
        ("measurements.npz", "MAX_DATASET_NPZ_BYTES"),
        ("evaluation-truth.npz", "MAX_DATASET_NPZ_BYTES"),
    ],
)
def test_task3_c2a1_role_bounds_precede_snapshot_hash_and_decode(
    tmp_path, monkeypatch, name, constant
):
    import gsdiff.data._artifact_persistence as persistence

    _, _, _, _, dataset_dir = _phase3c_write_dataset_directory(tmp_path)
    monkeypatch.setattr(
        persistence,
        constant,
        (dataset_dir / name).stat().st_size - 1,
    )

    def forbidden_snapshot(*args, **kwargs):
        raise AssertionError("oversize role reached snapshot hashing")

    monkeypatch.setattr(
        persistence, "read_safe_file_snapshot", forbidden_snapshot
    )
    with pytest.raises(ArtifactValidationError, match="bound"):
        persistence.verify_dataset_directory(dataset_dir)


@pytest.mark.parametrize("mutation", ["payload", "manifest-entry"])
def test_task3_c2a1_manifest_file_evidence_must_match_safe_snapshots(
    tmp_path, mutation
):
    from gsdiff.data.artifacts import (
        dataset_manifest_bytes,
        verify_dataset_directory,
    )

    _, _, manifest, _, dataset_dir = _phase3c_write_dataset_directory(
        tmp_path
    )
    if mutation == "payload":
        with (dataset_dir / "preview.png").open("ab") as stream:
            stream.write(b"x")
    else:
        manifest["files"]["preview.png"]["sha256"] = "0" * 64
        (dataset_dir / "dataset-manifest.json").write_bytes(
            dataset_manifest_bytes(manifest)
        )

    with pytest.raises(ArtifactValidationError, match="hash|size|snapshot"):
        verify_dataset_directory(dataset_dir)


def test_task3_c2a2_expected_generated_accepts_bit_exact_directory(
    tmp_path,
):
    from gsdiff.data.artifacts import verify_dataset_directory

    generated, _, _, _, dataset_dir = _phase3c_write_dataset_directory(
        tmp_path
    )
    verified = verify_dataset_directory(
        dataset_dir, expected_generated=generated
    )
    assert verified.expected_generated_verified is True


def test_task3_c2a2_expected_generated_rejects_coordinated_array_rewrite(
    tmp_path,
):
    from gsdiff.data._artifact_dataset import acquisition_npz_bytes
    from gsdiff.data._artifact_identity import array_descriptor
    from gsdiff.data.artifacts import (
        dataset_manifest_bytes,
        verify_dataset_directory,
    )

    generated, payloads, manifest, _, dataset_dir = (
        _phase3c_write_dataset_directory(tmp_path)
    )
    changed_measurements = generated.acquisition.measurements.copy()
    changed_measurements[0] = np.nextafter(
        changed_measurements[0], np.float32(np.inf)
    )
    changed_descriptors = _mutable_json(
        generated.acquisition.array_descriptors
    )
    changed_descriptors["measurements"] = array_descriptor(
        changed_measurements
    )
    changed_acquisition = dataclasses.replace(
        generated.acquisition,
        measurements=changed_measurements,
        array_descriptors=changed_descriptors,
    )
    changed_payload = acquisition_npz_bytes(changed_acquisition)
    payloads["measurements.npz"] = changed_payload
    manifest["files"]["measurements.npz"]["sha256"] = hashlib.sha256(
        changed_payload
    ).hexdigest()
    manifest["files"]["measurements.npz"]["size_bytes"] = len(
        changed_payload
    )
    (dataset_dir / "measurements.npz").write_bytes(changed_payload)
    (dataset_dir / "dataset-manifest.json").write_bytes(
        dataset_manifest_bytes(manifest)
    )

    self_consistent = verify_dataset_directory(dataset_dir)
    assert self_consistent.manifest_externally_anchored is False
    assert (
        self_consistent.acquisition.measurements.tobytes(order="C")
        == changed_measurements.tobytes(order="C")
    )
    with pytest.raises(
        ArtifactValidationError, match="expected.generated|measurements"
    ):
        verify_dataset_directory(
            dataset_dir, expected_generated=generated
        )


@pytest.mark.parametrize(
    "mutation",
    ["same-size-replacement", "inventory-addition", "leaf-substitution"],
)
def test_task3_c2a2_deterministic_post_snapshot_races_are_rejected(
    tmp_path, monkeypatch, mutation
):
    import gsdiff.data._artifact_persistence as persistence

    _, _, _, _, dataset_dir = _phase3c_write_dataset_directory(tmp_path)
    if mutation == "leaf-substitution":
        real_capture = persistence.capture_directory_inventory

        def capture_then_substitute(root):
            inventory = real_capture(root)
            leaf = dataset_dir / "preview.png"
            leaf.unlink()
            leaf.mkdir()
            return inventory

        monkeypatch.setattr(
            persistence,
            "capture_directory_inventory",
            capture_then_substitute,
        )
    else:
        real_verify = persistence.verify_dataset_payload_bytes

        def decode_then_mutate(payloads, manifest):
            decoded = real_verify(payloads, manifest)
            if mutation == "inventory-addition":
                (dataset_dir / "late.bin").write_bytes(b"late")
            else:
                leaf = dataset_dir / "measurements.npz"
                replacement = tmp_path / "replacement.npz"
                replacement.write_bytes(leaf.read_bytes())
                os.replace(replacement, leaf)
            return decoded

        monkeypatch.setattr(
            persistence,
            "verify_dataset_payload_bytes",
            decode_then_mutate,
        )

    with pytest.raises(
        ArtifactValidationError,
        match="changed|snapshot|inventory|regular|directory",
    ):
        persistence.verify_dataset_directory(dataset_dir)


def test_task3_c2a2_partial_snapshot_failure_reverifies_prior_reads(
    tmp_path, monkeypatch
):
    import gsdiff.data._artifact_persistence as persistence

    _, _, _, _, dataset_dir = _phase3c_write_dataset_directory(tmp_path)
    real_read = persistence.read_safe_file_snapshot
    real_postverify = persistence.verify_safe_file_snapshot
    postverified = []

    def read_then_fail(path, **kwargs):
        if path.name == "measurements.npz":
            raise ArtifactValidationError("injected second-file failure")
        snapshot = real_read(path, **kwargs)
        if path.name == "dataset-manifest.json":
            replacement = tmp_path / "replacement-manifest.json"
            replacement.write_bytes(path.read_bytes())
            os.replace(replacement, path)
        return snapshot

    def record_postverify(snapshot):
        postverified.append(snapshot.path.name)
        return real_postverify(snapshot)

    monkeypatch.setattr(
        persistence, "read_safe_file_snapshot", read_then_fail
    )
    monkeypatch.setattr(
        persistence, "verify_safe_file_snapshot", record_postverify
    )
    with pytest.raises(
        ArtifactValidationError, match="injected second-file failure"
    ) as error:
        persistence.verify_dataset_directory(dataset_dir)
    assert "dataset-manifest.json" in postverified
    assert any(
        "snapshot path changed" in note
        for note in getattr(error.value, "__notes__", ())
    )


@pytest.mark.parametrize("stage", ["capture", "postverify"])
def test_task3_c2a2_native_os_errors_are_normalized(
    tmp_path, monkeypatch, stage
):
    import gsdiff.data._artifact_persistence as persistence

    _, _, _, _, dataset_dir = _phase3c_write_dataset_directory(tmp_path)

    def vanished(*args, **kwargs):
        raise FileNotFoundError("injected disappearance")

    monkeypatch.setattr(
        persistence,
        (
            "capture_directory_inventory"
            if stage == "capture"
            else "verify_directory_inventory"
        ),
        vanished,
    )
    with pytest.raises(
        ArtifactValidationError, match="operating-system"
    ) as error:
        persistence.verify_dataset_directory(dataset_dir)
    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_task3_c2a2_inventory_root_disappearance_is_normalized(
    tmp_path, monkeypatch
):
    import gsdiff.data._artifact_io as artifact_io

    root = tmp_path / "inventory"
    root.mkdir()
    (root / "payload.bin").write_bytes(b"payload")
    real_capture = artifact_io._capture_directory_entries
    real_lstat = artifact_io.os.lstat
    scan_complete = False

    def capture_then_arm(path):
        nonlocal scan_complete
        entries = real_capture(path)
        scan_complete = True
        return entries

    def disappear_after_scan(path):
        if scan_complete and Path(path) == root.absolute():
            raise FileNotFoundError("injected root disappearance")
        return real_lstat(path)

    monkeypatch.setattr(
        artifact_io, "_capture_directory_entries", capture_then_arm
    )
    monkeypatch.setattr(artifact_io.os, "lstat", disappear_after_scan)
    with pytest.raises(ArtifactValidationError, match="inventory root"):
        artifact_io.capture_directory_inventory(root)


def test_task3_c2b1a_created_publication_is_manifest_last_and_atomic(
    tmp_path, monkeypatch
):
    import gsdiff.data._artifact_persistence as persistence
    from gsdiff.data import (
        DatasetPublication as DataDatasetPublication,
        publish_dataset as publish_from_data,
    )
    from gsdiff.data.artifacts import (
        DatasetPublication,
        publish_dataset,
    )

    generated = _phase3b_generate()
    artifact_root = tmp_path / "artifacts"
    final_dir = (
        artifact_root
        / "datasets"
        / generated.dataset_identity_sha256
    )
    events = []
    stages = []
    real_fsync = persistence.os.fsync
    file_fsync_count = 0
    real_fsync_directory = persistence._fsync_directory
    directory_fsyncs = []

    def record_fsync(descriptor):
        nonlocal file_fsync_count
        file_fsync_count += 1
        return real_fsync(descriptor)

    def record_directory_fsync(path):
        directory_fsyncs.append(Path(path))
        return real_fsync_directory(path)

    def observe(name, *, staging_dir, final_dir):
        events.append(name)
        stages.append(staging_dir)
        current = {path.name for path in staging_dir.iterdir()}
        if name == "physical-roundtrip":
            assert current == {
                "measurements.npz",
                "evaluation-truth.npz",
                "preview.png",
            }
            assert "dataset-manifest.json" not in current
        elif name in {"manifest", "stage-fsync", "before-promotion"}:
            assert current == {
                "dataset-manifest.json",
                "measurements.npz",
                "evaluation-truth.npz",
                "preview.png",
            }
        if name == "before-promotion":
            assert not final_dir.exists()

    monkeypatch.setattr(persistence.os, "fsync", record_fsync)
    monkeypatch.setattr(
        persistence, "_fsync_directory", record_directory_fsync
    )
    monkeypatch.setattr(persistence, "_publication_barrier", observe)
    publication = publish_dataset(artifact_root, generated)

    assert DataDatasetPublication is DatasetPublication
    assert publish_from_data is publish_dataset
    assert type(publication) is DatasetPublication
    assert publication.status == "created"
    assert publication.dataset_dir == final_dir.absolute()
    assert publication.verified.dataset_manifest_sha256 == (
        publication.dataset_manifest_sha256
    )
    assert publication.verified.expected_generated_verified is True
    assert set(path.name for path in final_dir.iterdir()) == {
        "dataset-manifest.json",
        "measurements.npz",
        "evaluation-truth.npz",
        "preview.png",
    }
    assert events.index("physical-roundtrip") < events.index("manifest")
    assert events.index("manifest") < events.index("stage-fsync")
    assert events.index("stage-fsync") < events.index("before-promotion")
    assert file_fsync_count >= 4
    assert any(path in stages for path in directory_fsyncs)
    assert all(not stage.exists() for stage in stages)


@pytest.mark.parametrize(
    "barrier",
    [
        "measurements",
        "truth",
        "preview",
        "physical-roundtrip",
        "manifest",
        "file-fsync",
        "stage-fsync",
        "before-promotion",
    ],
)
def test_task3_c2b1a_pre_promotion_barrier_failure_cleans_owned_stage(
    tmp_path, monkeypatch, barrier
):
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = tmp_path / "artifacts"
    final_dir = (
        artifact_root
        / "datasets"
        / generated.dataset_identity_sha256
    )

    def fail_at(name, **kwargs):
        if name == barrier:
            raise RuntimeError(f"injected {barrier} failure")

    monkeypatch.setattr(persistence, "_publication_barrier", fail_at)
    with pytest.raises(RuntimeError, match=f"injected {barrier}"):
        persistence.publish_dataset(artifact_root, generated)

    assert not final_dir.exists()
    datasets_dir = artifact_root / "datasets"
    assert not datasets_dir.exists() or not list(datasets_dir.iterdir())


def test_task3_c2b1b_valid_existing_reuses_before_serialization_or_write(
    tmp_path, monkeypatch
):
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = tmp_path / "artifacts"
    created = persistence.publish_dataset(artifact_root, generated)
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in created.dataset_dir.iterdir()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("valid reuse reached serialization or staging")

    monkeypatch.setattr(
        persistence, "build_dataset_payloads", forbidden
    )
    monkeypatch.setattr(persistence, "_create_owned_stage", forbidden)
    reused = persistence.publish_dataset(artifact_root, generated)
    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in reused.dataset_dir.iterdir()
    }

    assert reused.status == "reused"
    assert reused.dataset_dir == created.dataset_dir
    assert reused.dataset_manifest_sha256 == (
        created.dataset_manifest_sha256
    )
    assert before == after


@pytest.mark.parametrize("mutation", ["missing", "extra", "corrupt"])
def test_task3_c2b1b_invalid_existing_fails_closed_without_repair(
    tmp_path, monkeypatch, mutation
):
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = tmp_path / "artifacts"
    created = persistence.publish_dataset(artifact_root, generated)
    if mutation == "missing":
        preview = created.dataset_dir / "preview.png"
        preview.chmod(0o600)
        preview.unlink()
    elif mutation == "extra":
        (created.dataset_dir / "extra.bin").write_bytes(b"extra")
    else:
        path = created.dataset_dir / "preview.png"
        path.chmod(0o600)
        path.write_bytes(b"x" * path.stat().st_size)
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in created.dataset_dir.iterdir()
        if path.is_file()
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid existing reached serialization")

    monkeypatch.setattr(
        persistence, "build_dataset_payloads", forbidden
    )
    with pytest.raises(ArtifactValidationError):
        persistence.publish_dataset(artifact_root, generated)
    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in created.dataset_dir.iterdir()
        if path.is_file()
    }
    assert before == after


@pytest.mark.parametrize(
    "injection", ["extra-file", "directory", "hardlink", "symlink"]
)
def test_task3_c2b1b_hostile_stage_cleanup_refuses_recursive_deletion(
    tmp_path, monkeypatch, injection
):
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = tmp_path / "artifacts"
    outside = tmp_path / "outside-sentinel.bin"
    outside.write_bytes(b"outside-sentinel")
    if injection == "symlink":
        probe = tmp_path / "symlink-probe"
        try:
            probe.symlink_to(outside)
        except OSError as error:
            pytest.skip(f"symlinks unavailable: {error}")
        probe.unlink()
    captured_stage = None

    def inject_and_fail(name, *, staging_dir, final_dir):
        nonlocal captured_stage
        captured_stage = staging_dir
        if name != "before-promotion":
            return
        injected = staging_dir / "injected"
        if injection == "extra-file":
            injected.write_bytes(b"unowned")
        elif injection == "directory":
            injected.mkdir()
        elif injection == "hardlink":
            try:
                os.link(outside, injected)
            except OSError as error:
                pytest.skip(f"hardlinks unavailable: {error}")
        else:
            injected.symlink_to(outside)
        raise RuntimeError("injected hostile cleanup failure")

    monkeypatch.setattr(
        persistence, "_publication_barrier", inject_and_fail
    )
    with pytest.raises(
        RuntimeError, match="hostile cleanup"
    ) as error:
        persistence.publish_dataset(artifact_root, generated)

    assert outside.read_bytes() == b"outside-sentinel"
    assert captured_stage is not None and captured_stage.exists()
    assert any(
        "cleanup refused" in note.lower()
        for note in getattr(error.value, "__notes__", ())
    )
    final_dir = (
        artifact_root
        / "datasets"
        / generated.dataset_identity_sha256
    )
    assert not final_dir.exists()


def test_task3_c2b1b_exclusive_leaf_collision_is_never_overwritten(
    tmp_path, monkeypatch
):
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = tmp_path / "artifacts"
    real_create = persistence._create_owned_stage
    captured_stage = None

    def create_with_collision(datasets_dir, identity):
        nonlocal captured_stage
        stage = real_create(datasets_dir, identity)
        captured_stage = stage.path
        (stage.path / "measurements.npz").write_bytes(b"unowned")
        return stage

    monkeypatch.setattr(
        persistence, "_create_owned_stage", create_with_collision
    )
    with pytest.raises(ArtifactValidationError):
        persistence.publish_dataset(artifact_root, generated)
    assert captured_stage is not None and captured_stage.exists()
    assert (captured_stage / "measurements.npz").read_bytes() == b"unowned"


@pytest.mark.parametrize("mutation", ["same-size-rewrite", "extra-entry"])
def test_task3_c2b1b_stage_mutation_at_final_barrier_is_not_promoted(
    tmp_path, monkeypatch, mutation
):
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = tmp_path / "artifacts"
    captured_stage = None

    def mutate_without_raising(name, *, staging_dir, final_dir):
        nonlocal captured_stage
        captured_stage = staging_dir
        if name != "before-promotion":
            return
        if mutation == "same-size-rewrite":
            leaf = staging_dir / "preview.png"
            leaf.write_bytes(b"x" * leaf.stat().st_size)
        else:
            (staging_dir / "late.bin").write_bytes(b"late")

    monkeypatch.setattr(
        persistence, "_publication_barrier", mutate_without_raising
    )
    with pytest.raises(ArtifactValidationError):
        persistence.publish_dataset(artifact_root, generated)

    final_dir = (
        artifact_root
        / "datasets"
        / generated.dataset_identity_sha256
    )
    assert not final_dir.exists()
    assert captured_stage is not None and captured_stage.exists()


def test_task3_c2b1b_stage_replacement_inside_promotion_is_not_promoted(
    tmp_path, monkeypatch
):
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = tmp_path / "artifacts"
    real_promote = persistence._promote_directory_no_clobber
    captured_stage = None
    displaced_stage = None

    def replace_source_then_promote(stage, final_dir, **kwargs):
        nonlocal captured_stage, displaced_stage
        staging_dir = stage.path if hasattr(stage, "path") else Path(stage)
        captured_stage = staging_dir
        displaced_stage = staging_dir.with_name(
            f"{staging_dir.name}.displaced"
        )
        staging_dir.rename(displaced_stage)
        staging_dir.mkdir()
        (staging_dir / "attacker.bin").write_bytes(b"attacker")
        return real_promote(stage, final_dir, **kwargs)

    monkeypatch.setattr(
        persistence,
        "_promote_directory_no_clobber",
        replace_source_then_promote,
    )
    with pytest.raises(ArtifactValidationError):
        persistence.publish_dataset(artifact_root, generated)

    final_dir = (
        artifact_root
        / "datasets"
        / generated.dataset_identity_sha256
    )
    assert not final_dir.exists()
    assert captured_stage is not None and captured_stage.exists()
    assert (captured_stage / "attacker.bin").read_bytes() == b"attacker"
    assert displaced_stage is not None and displaced_stage.exists()
    assert {
        path.name for path in displaced_stage.iterdir()
    } == {
        "dataset-manifest.json",
        "measurements.npz",
        "evaluation-truth.npz",
        "preview.png",
    }


def test_task3_c2b1b_post_promotion_identity_mismatch_is_quarantined(
    tmp_path, monkeypatch
):
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = tmp_path / "artifacts"
    final_dir = (
        artifact_root
        / "datasets"
        / generated.dataset_identity_sha256
    ).absolute()
    real_lstat = persistence.os.lstat
    captured_stage = None
    displaced_final = final_dir.with_name(f".{final_dir.name}.displaced")
    swapped = False

    def capture_stage(name, *, staging_dir, final_dir):
        nonlocal captured_stage
        if name == "before-promotion":
            captured_stage = staging_dir

    def replace_final_before_identity_check(path, *args, **kwargs):
        nonlocal swapped
        candidate = Path(path).absolute()
        if candidate == final_dir and captured_stage is not None and not swapped:
            try:
                real_lstat(captured_stage)
            except FileNotFoundError:
                try:
                    real_lstat(final_dir)
                except FileNotFoundError:
                    pass
                else:
                    final_dir.rename(displaced_final)
                    final_dir.mkdir()
                    (final_dir / "attacker.bin").write_bytes(b"attacker")
                    swapped = True
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(
        persistence, "_publication_barrier", capture_stage
    )
    monkeypatch.setattr(
        persistence.os, "lstat", replace_final_before_identity_check
    )
    with pytest.raises(
        ArtifactValidationError, match="identity mismatch"
    ):
        persistence.publish_dataset(artifact_root, generated)

    assert swapped is True
    assert not final_dir.exists()
    assert displaced_final.exists()
    assert {
        path.name for path in displaced_final.iterdir()
    } == {
        "dataset-manifest.json",
        "measurements.npz",
        "evaluation-truth.npz",
        "preview.png",
    }
    diagnostics = list(
        final_dir.parent.glob(f".{final_dir.name}.rejected-*")
    )
    assert len(diagnostics) == 1
    assert (diagnostics[0] / "attacker.bin").read_bytes() == b"attacker"


def test_task3_c2b1b_parent_fsync_failure_keeps_complete_final_for_reuse(
    tmp_path, monkeypatch
):
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = tmp_path / "artifacts"
    real_sync_parent = persistence._sync_publication_parent
    calls = 0

    def fail_after_promotion(path):
        nonlocal calls
        calls += 1
        raise ArtifactValidationError(
            "injected parent durability failure"
        )

    monkeypatch.setattr(
        persistence, "_sync_publication_parent", fail_after_promotion
    )
    with pytest.raises(
        ArtifactValidationError, match="durability"
    ):
        persistence.publish_dataset(artifact_root, generated)
    final_dir = (
        artifact_root
        / "datasets"
        / generated.dataset_identity_sha256
    )
    assert calls == 1
    assert final_dir.exists()
    persistence.verify_dataset_directory(
        final_dir, expected_generated=generated
    )

    monkeypatch.setattr(
        persistence, "_sync_publication_parent", real_sync_parent
    )
    reused = persistence.publish_dataset(artifact_root, generated)
    assert reused.status == "reused"


def test_task3_c2b1b_promotion_uses_only_platform_no_clobber_primitives():
    import inspect
    import gsdiff.data._artifact_persistence as persistence

    promotion_source = inspect.getsource(
        persistence._promote_directory_no_clobber
    )
    path_rename_source = inspect.getsource(
        persistence._rename_directory_path_no_clobber
    )
    deletion_source = inspect.getsource(
        persistence._delete_windows_owned_path
    )
    assert "os.replace" not in promotion_source
    assert "os.replace" not in path_rename_source
    assert "SetFileInformationByHandle" in promotion_source
    assert "MoveFileExW" not in promotion_source
    assert "MoveFileExW" in path_rename_source
    assert "renameat2" in path_rename_source
    assert "SetFileInformationByHandle" in deletion_source
    assert "os.unlink" not in deletion_source
    assert persistence._FILE_BASIC_INFO_CLASS == 0
    assert persistence._FILE_RENAME_INFO_CLASS == 3
    assert persistence._FILE_ID_INFO_CLASS == 18
    assert persistence._FILE_DISPOSITION_INFO_EX_CLASS == 21
    assert persistence._FILE_DISPOSITION_FLAG_DELETE == 0x1
    assert (
        persistence._FILE_DISPOSITION_FLAG_IGNORE_READONLY_ATTRIBUTE
        == 0x10
    )
    assert persistence._MOVEFILE_WRITE_THROUGH == 0x8
    assert persistence._RENAME_NOREPLACE == 1


@pytest.mark.parametrize(
    "mutation", ["artifact-root-file", "datasets-file"]
)
def test_task3_c2b1b_publication_roots_must_be_real_directories(
    tmp_path, monkeypatch, mutation
):
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = tmp_path / "artifacts"
    if mutation == "artifact-root-file":
        artifact_root.write_bytes(b"not-a-directory")
    else:
        artifact_root.mkdir()
        (artifact_root / "datasets").write_bytes(b"not-a-directory")

    def forbidden(*args, **kwargs):
        raise AssertionError("unsafe root reached serialization")

    monkeypatch.setattr(
        persistence, "build_dataset_payloads", forbidden
    )
    with pytest.raises(ArtifactValidationError, match="directory"):
        persistence.publish_dataset(artifact_root, generated)


@pytest.mark.parametrize(
    "raced_component", ["artifact-root", "datasets"]
)
def test_task3_c2b2a_concurrent_real_directory_mkdir_winner_is_accepted(
    tmp_path, monkeypatch, raced_component
):
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = (tmp_path / "artifacts").absolute()
    target = (
        artifact_root
        if raced_component == "artifact-root"
        else artifact_root / "datasets"
    )
    if raced_component == "datasets":
        artifact_root.mkdir()
    real_mkdir = persistence.os.mkdir
    injected = False

    def race_mkdir(path, mode=0o777, *, dir_fd=None):
        nonlocal injected
        candidate = Path(path).absolute()
        if candidate == target and not injected:
            injected = True
            real_mkdir(path, mode, dir_fd=dir_fd)
            raise FileExistsError("injected concurrent directory winner")
        return real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(persistence.os, "mkdir", race_mkdir)
    publication = persistence.publish_dataset(artifact_root, generated)

    assert injected is True
    assert publication.status == "created"
    assert publication.dataset_dir.parent == artifact_root / "datasets"
    persistence.verify_dataset_directory(
        publication.dataset_dir,
        expected_dataset_manifest_sha256=(
            publication.dataset_manifest_sha256
        ),
        expected_generated=generated,
    )


@pytest.mark.parametrize(
    "raced_component", ["artifact-root", "datasets"]
)
def test_task3_c2b2a_concurrent_mkdir_file_winner_fails_closed(
    tmp_path, monkeypatch, raced_component
):
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = (tmp_path / "artifacts").absolute()
    target = (
        artifact_root
        if raced_component == "artifact-root"
        else artifact_root / "datasets"
    )
    if raced_component == "datasets":
        artifact_root.mkdir()
    real_mkdir = persistence.os.mkdir
    injected = False

    def race_mkdir(path, mode=0o777, *, dir_fd=None):
        nonlocal injected
        candidate = Path(path).absolute()
        if candidate == target and not injected:
            injected = True
            Path(path).write_bytes(b"not-a-directory")
            raise FileExistsError("injected concurrent file winner")
        return real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(persistence.os, "mkdir", race_mkdir)
    with pytest.raises(ArtifactValidationError, match="real directory"):
        persistence.publish_dataset(artifact_root, generated)

    assert injected is True
    assert target.read_bytes() == b"not-a-directory"


@pytest.mark.parametrize(
    "appearance", ["precheck-window", "promotion-primitive"]
)
@pytest.mark.parametrize("winner_kind", ["identical", "different-payload"])
def test_task3_c2b2a_target_appears_is_verified_without_clobber(
    tmp_path, monkeypatch, appearance, winner_kind
):
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = tmp_path / "artifacts"
    captured_stage = None
    winner_snapshot = None
    installed = False

    def install_winner(staging_dir, final_dir):
        nonlocal captured_stage, winner_snapshot, installed
        if installed:
            raise AssertionError("winner installed more than once")
        installed = True
        captured_stage = staging_dir
        final_dir.mkdir()
        for source in staging_dir.iterdir():
            (final_dir / source.name).write_bytes(source.read_bytes())
        if winner_kind == "different-payload":
            preview = final_dir / "preview.png"
            changed = bytearray(preview.read_bytes())
            changed[-1] ^= 1
            preview.write_bytes(bytes(changed))
        winner_snapshot = {
            path.name: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in final_dir.iterdir()
        }

    if appearance == "precheck-window":
        def inject_at_barrier(name, *, staging_dir, final_dir):
            if name == "before-promotion":
                install_winner(staging_dir, final_dir)

        monkeypatch.setattr(
            persistence, "_publication_barrier", inject_at_barrier
        )
    else:
        def lose_atomic_promotion(stage, final_dir, **kwargs):
            install_winner(stage.path, final_dir)
            raise FileExistsError("injected atomic promotion loser")

        monkeypatch.setattr(
            persistence,
            "_promote_directory_no_clobber",
            lose_atomic_promotion,
        )

    if winner_kind == "identical":
        publication = persistence.publish_dataset(
            artifact_root, generated
        )
        assert publication.status == "reused"
        assert publication.verified.expected_generated_verified is True
    else:
        with pytest.raises(
            ArtifactValidationError,
            match="nondeterministic dataset collision",
        ):
            persistence.publish_dataset(artifact_root, generated)

    final_dir = (
        artifact_root
        / "datasets"
        / generated.dataset_identity_sha256
    )
    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in final_dir.iterdir()
    }
    assert installed is True
    assert winner_snapshot == after
    assert captured_stage is not None
    assert not captured_stage.exists()


def test_task3_c2b2a_collision_handler_uses_physical_staging_evidence(
    tmp_path
):
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    created = persistence.publish_dataset(
        tmp_path / "artifacts", generated
    )
    manifest_payload = (
        created.dataset_dir / "dataset-manifest.json"
    ).read_bytes()
    physical_payloads = {
        name: (created.dataset_dir / name).read_bytes()
        for name in (
            "measurements.npz",
            "evaluation-truth.npz",
            "preview.png",
        )
    }
    changed_preview = bytearray(physical_payloads["preview.png"])
    changed_preview[-1] ^= 1
    physical_payloads["preview.png"] = bytes(changed_preview)

    with pytest.raises(
        ArtifactValidationError,
        match="nondeterministic dataset collision",
    ):
        persistence._verify_concurrent_publication_winner(
            created.dataset_dir,
            identity=generated.dataset_identity_sha256,
            generated=generated,
            staged_manifest_payload=manifest_payload,
            staged_physical_payloads=physical_payloads,
        )


def test_task3_c2b2b_two_fresh_processes_publish_one_created_one_reused(
    tmp_path
):
    import gsdiff.data._artifact_persistence as persistence

    artifact_root = (tmp_path / "artifacts").absolute()
    barrier_dir = tmp_path / "publication-barrier"
    barrier_dir.mkdir()
    release_path = barrier_dir / "release"
    script = "\n".join(
        [
            "import json",
            "from pathlib import Path",
            "import runpy",
            "import sys",
            "import time",
            "import gsdiff.data._artifact_persistence as persistence",
            f"ns = runpy.run_path({str(Path(__file__).resolve())!r})",
            "generated = ns['_phase3b_generate']()",
            "artifact_root = Path(sys.argv[1])",
            "barrier_dir = Path(sys.argv[2])",
            "participant = sys.argv[3]",
            "ready = barrier_dir / f'{participant}.ready'",
            "release = barrier_dir / 'release'",
            "real_promote = persistence._promote_directory_no_clobber",
            "def synchronized_promote(stage, final_dir, **kwargs):",
            "    ready.write_text('ready', encoding='utf-8')",
            "    deadline = time.monotonic() + 30.0",
            "    while not release.exists():",
            "        if time.monotonic() >= deadline:",
            "            raise TimeoutError('publication barrier timed out')",
            "        time.sleep(0.01)",
            "    return real_promote(stage, final_dir, **kwargs)",
            "persistence._promote_directory_no_clobber = (",
            "    synchronized_promote",
            ")",
            "publication = persistence.publish_dataset(",
            "    artifact_root, generated",
            ")",
            "print(json.dumps({",
            "    'status': publication.status,",
            "    'manifest_sha256': publication.dataset_manifest_sha256,",
            "    'dataset_dir': str(publication.dataset_dir),",
            "}, sort_keys=True))",
        ]
    )
    child_env = {
        "PATH": os.environ["PATH"],
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": str(REPO_ROOT),
        "SYSTEMROOT": os.environ["SYSTEMROOT"],
    }
    processes = [
        subprocess.Popen(
            [
                str(AUTHORITATIVE_PYTHON),
                "-c",
                script,
                str(artifact_root),
                str(barrier_dir),
                participant,
            ],
            cwd=REPO_ROOT,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        for participant in ("first", "second")
    ]
    completed = []
    try:
        deadline = time.monotonic() + 30.0
        while len(list(barrier_dir.glob("*.ready"))) != 2:
            if any(process.poll() is not None for process in processes):
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        assert {
            path.name for path in barrier_dir.glob("*.ready")
        } == {"first.ready", "second.ready"}, [
            process.poll() for process in processes
        ]
        release_path.write_text("release", encoding="utf-8")
        for process in processes:
            stdout, stderr = process.communicate(timeout=60)
            completed.append(
                (process.returncode, stdout, stderr)
            )
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)

    assert all(returncode == 0 for returncode, _, _ in completed), completed
    results = [
        json.loads(stdout)
        for _, stdout, _ in completed
    ]
    assert sorted(result["status"] for result in results) == [
        "created",
        "reused",
    ]
    assert len(
        {result["manifest_sha256"] for result in results}
    ) == 1
    generated = _phase3b_generate()
    final_dir = (
        artifact_root
        / "datasets"
        / generated.dataset_identity_sha256
    )
    verified = persistence.verify_dataset_directory(
        final_dir,
        expected_dataset_manifest_sha256=(
            results[0]["manifest_sha256"]
        ),
        expected_generated=generated,
    )
    assert verified.expected_generated_verified is True
    assert {
        path.name for path in final_dir.iterdir()
    } == {
        "dataset-manifest.json",
        "measurements.npz",
        "evaluation-truth.npz",
        "preview.png",
    }
    assert {
        path.name for path in final_dir.parent.iterdir()
    } == {generated.dataset_identity_sha256}
    if os.name != "nt":
        assert os.lstat(final_dir).st_mode & 0o222 == 0
        assert all(
            os.lstat(path).st_mode & 0o222 == 0
            for path in final_dir.iterdir()
        )


def test_task3_c2b2b_crash_left_staging_is_preserved_and_ignored(
    tmp_path
):
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = tmp_path / "artifacts"
    datasets_dir = artifact_root / "datasets"
    datasets_dir.mkdir(parents=True)
    stale = datasets_dir / (
        f".{generated.dataset_identity_sha256}.staging-crashed"
    )
    stale.mkdir()
    (stale / "partial.bin").write_bytes(b"crash-evidence")
    nested = stale / "nested"
    nested.mkdir()
    (nested / "sentinel.bin").write_bytes(b"nested-evidence")
    before = {
        str(path.relative_to(stale)): (
            path.read_bytes() if path.is_file() else None,
            path.stat().st_mtime_ns,
        )
        for path in (stale, *stale.rglob("*"))
    }

    created = persistence.publish_dataset(artifact_root, generated)
    reused = persistence.publish_dataset(artifact_root, generated)

    after = {
        str(path.relative_to(stale)): (
            path.read_bytes() if path.is_file() else None,
            path.stat().st_mtime_ns,
        )
        for path in (stale, *stale.rglob("*"))
    }
    assert created.status == "created"
    assert reused.status == "reused"
    assert before == after
    assert {
        path.name for path in datasets_dir.iterdir()
    } == {
        generated.dataset_identity_sha256,
        stale.name,
    }


def test_task3_c2b2b_final_reparse_branch_fails_without_link_privilege(
    tmp_path, monkeypatch
):
    import stat as _stat
    from types import SimpleNamespace
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = (tmp_path / "artifacts").absolute()
    datasets_dir = artifact_root / "datasets"
    datasets_dir.mkdir(parents=True)
    final_dir = (
        datasets_dir / generated.dataset_identity_sha256
    ).absolute()
    sentinel = tmp_path / "outside-sentinel.bin"
    sentinel.write_bytes(b"outside")
    real_lexists = persistence.os.path.lexists
    real_lstat = persistence.os.lstat

    def fake_lexists(path):
        return Path(path).absolute() == final_dir or real_lexists(path)

    def fake_lstat(path):
        if Path(path).absolute() == final_dir:
            return SimpleNamespace(
                st_mode=_stat.S_IFDIR,
                st_file_attributes=0x400,
            )
        return real_lstat(path)

    monkeypatch.setattr(
        persistence.os.path, "lexists", fake_lexists
    )
    monkeypatch.setattr(persistence.os, "lstat", fake_lstat)
    with pytest.raises(ArtifactValidationError, match="linked|reparse"):
        persistence.publish_dataset(artifact_root, generated)

    assert sentinel.read_bytes() == b"outside"
    assert not list(datasets_dir.iterdir())


def test_task3_c2b2b_concurrent_mkdir_reparse_winner_fails_closed(
    tmp_path, monkeypatch
):
    import stat as _stat
    from types import SimpleNamespace
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = (tmp_path / "artifacts").absolute()
    artifact_root.mkdir()
    datasets_dir = (artifact_root / "datasets").absolute()
    real_mkdir = persistence.os.mkdir
    real_lstat = persistence.os.lstat
    injected = False

    def race_mkdir(path, mode=0o777, *, dir_fd=None):
        nonlocal injected
        if Path(path).absolute() == datasets_dir and not injected:
            injected = True
            real_mkdir(path, mode, dir_fd=dir_fd)
            raise FileExistsError("injected reparse mkdir winner")
        return real_mkdir(path, mode, dir_fd=dir_fd)

    def fake_lstat(path):
        if Path(path).absolute() == datasets_dir and injected:
            return SimpleNamespace(
                st_mode=_stat.S_IFDIR,
                st_file_attributes=0x400,
            )
        return real_lstat(path)

    monkeypatch.setattr(persistence.os, "mkdir", race_mkdir)
    monkeypatch.setattr(persistence.os, "lstat", fake_lstat)
    with pytest.raises(ArtifactValidationError, match="linked|reparse"):
        persistence.publish_dataset(artifact_root, generated)

    assert injected is True
    assert list(datasets_dir.iterdir()) == []


def test_task3_c2b2b_leaf_mutation_inside_promotion_is_rejected_before_rename(
    tmp_path, monkeypatch
):
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = tmp_path / "artifacts"
    final_dir = (
        artifact_root
        / "datasets"
        / generated.dataset_identity_sha256
    )
    real_verify_owned_stage = persistence._verify_owned_stage
    verify_calls = 0
    captured_stage = None

    def verify_then_mutate(stage):
        nonlocal verify_calls, captured_stage
        result = real_verify_owned_stage(stage)
        verify_calls += 1
        captured_stage = stage.path
        if verify_calls == 2:
            preview = stage.path / "preview.png"
            preview.write_bytes(b"x" * preview.stat().st_size)
        return result

    monkeypatch.setattr(
        persistence, "_verify_owned_stage", verify_then_mutate
    )
    with pytest.raises(ArtifactValidationError):
        persistence.publish_dataset(artifact_root, generated)

    assert verify_calls >= 2
    assert not final_dir.exists()
    diagnostics = list(
        final_dir.parent.glob(f".{final_dir.name}.rejected-*")
    )
    assert diagnostics == []
    assert captured_stage is not None and captured_stage.exists()


def test_task3_c2b2b_raw_bytes_detect_restored_mtime_before_rename(
    tmp_path, monkeypatch
):
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = tmp_path / "artifacts"
    final_dir = (
        artifact_root
        / "datasets"
        / generated.dataset_identity_sha256
    )
    captured_stage = None
    stat_signature_restored = None

    def mutate_and_restore_mtime(
        name, *, staging_dir, final_dir
    ):
        nonlocal captured_stage, stat_signature_restored
        if name != "before-promotion":
            return
        captured_stage = staging_dir
        preview = staging_dir / "preview.png"
        before = os.lstat(preview)
        preview.write_bytes(b"x" * before.st_size)
        os.utime(
            preview,
            ns=(before.st_atime_ns, before.st_mtime_ns),
        )
        stat_signature_restored = (
            persistence._stat_signature(os.lstat(preview))
            == persistence._stat_signature(before)
        )

    monkeypatch.setattr(
        persistence,
        "_publication_barrier",
        mutate_and_restore_mtime,
    )
    with pytest.raises(
        ArtifactValidationError, match="staging bytes"
    ):
        persistence.publish_dataset(artifact_root, generated)

    if os.name == "nt":
        assert stat_signature_restored is True
    assert not final_dir.exists()
    assert list(
        final_dir.parent.glob(f".{final_dir.name}.rejected-*")
    ) == []
    assert captured_stage is not None


def test_task3_c2b2b_stage_is_readonly_before_final_raw_gate(
    tmp_path, monkeypatch
):
    import stat as _stat

    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = tmp_path / "artifacts"
    final_dir = (
        artifact_root
        / "datasets"
        / generated.dataset_identity_sha256
    )
    real_verify_bytes = persistence._verify_owned_stage_bytes
    attempted_write = False
    write_denied = False
    expected_files = None

    def verify_then_attempt_ordinary_write(stage, expected):
        nonlocal attempted_write, write_denied, expected_files
        real_verify_bytes(stage, expected)
        expected_files = dict(expected)
        preview = stage.path / "preview.png"
        info = os.lstat(preview)
        if os.name == "nt":
            assert info.st_file_attributes & _stat.FILE_ATTRIBUTE_READONLY
        else:
            assert info.st_mode & 0o222 == 0
        if os.name != "nt" and hasattr(os, "geteuid") and os.geteuid() == 0:
            return
        attempted_write = True
        try:
            preview.write_bytes(b"x" * info.st_size)
        except PermissionError:
            write_denied = True

    monkeypatch.setattr(
        persistence,
        "_verify_owned_stage_bytes",
        verify_then_attempt_ordinary_write,
    )
    publication = persistence.publish_dataset(artifact_root, generated)

    assert publication.status == "created"
    if os.name != "nt" and hasattr(os, "geteuid") and os.geteuid() == 0:
        assert attempted_write is False
    else:
        assert attempted_write is True
        assert write_denied is True
    assert expected_files is not None
    for name, expected in expected_files.items():
        final_file = final_dir / name
        assert final_file.read_bytes() == expected
        info = os.lstat(final_file)
        if os.name == "nt":
            assert info.st_file_attributes & _stat.FILE_ATTRIBUTE_READONLY
        else:
            assert info.st_mode & 0o222 == 0


def test_task3_c2b2b_final_verify_precedes_sync_and_quarantines_failure(
    tmp_path, monkeypatch
):
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = tmp_path / "artifacts"
    final_dir = (
        artifact_root
        / "datasets"
        / generated.dataset_identity_sha256
    )
    real_verify = persistence.verify_dataset_directory
    mutated = False

    def mutate_before_final_verify(dataset_dir, **kwargs):
        nonlocal mutated
        if Path(dataset_dir).absolute() == final_dir.absolute():
            preview = Path(dataset_dir) / "preview.png"
            preview.chmod(0o600)
            preview.write_bytes(b"x" * preview.stat().st_size)
            mutated = True
        return real_verify(dataset_dir, **kwargs)

    def forbidden_parent_sync(path):
        raise AssertionError("invalid final reached parent sync")

    monkeypatch.setattr(
        persistence,
        "verify_dataset_directory",
        mutate_before_final_verify,
    )
    monkeypatch.setattr(
        persistence, "_sync_publication_parent", forbidden_parent_sync
    )
    with pytest.raises(ArtifactValidationError):
        persistence.publish_dataset(artifact_root, generated)

    assert mutated is True
    assert not final_dir.exists()
    diagnostics = list(
        final_dir.parent.glob(f".{final_dir.name}.rejected-*")
    )
    assert len(diagnostics) == 1
    assert {
        path.name for path in diagnostics[0].iterdir()
    } == {
        "dataset-manifest.json",
        "measurements.npz",
        "evaluation-truth.npz",
        "preview.png",
    }


@pytest.mark.skipif(
    os.name != "nt",
    reason=(
        "portable POSIX unlinkat cannot bind deletion to an opened leaf; "
        "hostile same-UID directory writers are outside the threat model"
    ),
)
def test_task3_c2b2b_windows_cleanup_never_deletes_swapped_external_leaf(
    tmp_path, monkeypatch
):
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = tmp_path / "artifacts"
    outside = tmp_path / "outside-sentinel.bin"
    outside.write_bytes(b"outside-sentinel")
    captured_stage = None
    displaced_owned = None
    injected_path = None
    swap_triggered = False

    def fail_before_promotion(name, *, staging_dir, final_dir):
        nonlocal captured_stage
        if name == "before-promotion":
            captured_stage = staging_dir
            raise RuntimeError("injected cleanup trigger")

    def swap_into_leaf(path):
        nonlocal displaced_owned, injected_path, swap_triggered
        candidate = Path(path)
        if not candidate.is_absolute():
            assert captured_stage is not None
            candidate = captured_stage / candidate.name
        if candidate.name != "measurements.npz" or swap_triggered:
            return
        displaced_owned = candidate.with_name("owned-displaced.bin")
        candidate.rename(displaced_owned)
        outside.rename(candidate)
        injected_path = candidate
        swap_triggered = True

    monkeypatch.setattr(
        persistence, "_publication_barrier", fail_before_promotion
    )
    real_delete = persistence._delete_windows_owned_path

    def swap_then_handle_delete(path, *args, **kwargs):
        swap_into_leaf(path)
        return real_delete(path, *args, **kwargs)

    monkeypatch.setattr(
        persistence,
        "_delete_windows_owned_path",
        swap_then_handle_delete,
    )

    with pytest.raises(RuntimeError, match="cleanup trigger"):
        persistence.publish_dataset(artifact_root, generated)

    assert swap_triggered is True
    assert captured_stage is not None and captured_stage.exists()
    assert displaced_owned is not None and displaced_owned.exists()
    assert displaced_owned.read_bytes() != b"outside-sentinel"
    assert injected_path is not None
    preserved_paths = [
        path
        for path in (outside, injected_path)
        if path.exists()
    ]
    assert len(preserved_paths) == 1
    assert preserved_paths[0].read_bytes() == b"outside-sentinel"


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX descriptor-bound chmod/unlinkat semantics",
)
def test_task3_c2b2b_posix_collision_thaws_readonly_stage_for_cleanup(
    tmp_path, monkeypatch
):
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = tmp_path / "artifacts"
    captured_stage = None
    frozen_mode = None

    def install_winner_after_freeze(source, final_dir):
        nonlocal captured_stage, frozen_mode
        captured_stage = Path(source)
        frozen_mode = os.lstat(captured_stage).st_mode
        final_dir.mkdir()
        for source_file in captured_stage.iterdir():
            destination = final_dir / source_file.name
            destination.write_bytes(source_file.read_bytes())
            destination.chmod(0o400)
        final_dir.chmod(0o500)
        raise FileExistsError("injected POSIX promotion loser")

    monkeypatch.setattr(
        persistence,
        "_rename_directory_path_no_clobber",
        install_winner_after_freeze,
    )
    publication = persistence.publish_dataset(artifact_root, generated)

    assert publication.status == "reused"
    assert captured_stage is not None
    assert frozen_mode is not None and frozen_mode & 0o222 == 0
    assert not captured_stage.exists()
    assert {
        path.name
        for path in publication.dataset_dir.parent.iterdir()
    } == {generated.dataset_identity_sha256}


@pytest.mark.parametrize(
    "mutation",
    ["final-file", "extra-file", "leaf-directory", "leaf-hardlink"],
)
def test_task3_c2b2b_unsafe_existing_final_is_never_repaired(
    tmp_path, mutation
):
    import gsdiff.data._artifact_persistence as persistence

    generated = _phase3b_generate()
    artifact_root = tmp_path / "artifacts"
    final_dir = (
        artifact_root
        / "datasets"
        / generated.dataset_identity_sha256
    )
    outside = tmp_path / "outside-sentinel.bin"
    outside.write_bytes(b"outside")
    if mutation == "final-file":
        final_dir.parent.mkdir(parents=True)
        final_dir.write_bytes(b"not-a-directory")
        before = final_dir.read_bytes()
    else:
        persistence.publish_dataset(artifact_root, generated)
        if mutation == "extra-file":
            (final_dir / "extra.bin").write_bytes(b"extra")
        elif mutation == "leaf-directory":
            preview = final_dir / "preview.png"
            preview.chmod(0o600)
            preview.unlink()
            preview.mkdir()
        else:
            preview = final_dir / "preview.png"
            preview.chmod(0o600)
            preview.unlink()
            try:
                os.link(outside, preview)
            except OSError as error:
                pytest.skip(f"hardlinks unavailable: {error}")
        before = {
            path.name: (
                "directory" if path.is_dir() else path.read_bytes()
            )
            for path in final_dir.iterdir()
        }

    with pytest.raises(ArtifactValidationError):
        persistence.publish_dataset(artifact_root, generated)

    if mutation == "final-file":
        assert final_dir.read_bytes() == before
    else:
        after = {
            path.name: (
                "directory" if path.is_dir() else path.read_bytes()
            )
            for path in final_dir.iterdir()
        }
        assert after == before
    assert outside.read_bytes() == b"outside"


def test_task3_c3a_dataset_discovery_is_read_only_sorted_and_recheckable(
    tmp_path,
):
    from gsdiff.data.artifacts import (
        ArtifactValidationError,
        discover_dataset_directories,
        verify_dataset_directory_discovery,
    )

    artifact_root = tmp_path / "artifacts"
    missing = discover_dataset_directories(artifact_root)

    assert missing.datasets_dir_exists is False
    assert missing.canonical_directories == ()
    assert missing.stale_staging_directories == ()
    assert missing.rejected_directories == ()
    assert not artifact_root.exists()
    verify_dataset_directory_discovery(missing)

    datasets_dir = artifact_root / "datasets"
    datasets_dir.mkdir(parents=True)
    first_identity = "a" * 64
    second_identity = "b" * 64
    names = (
        second_identity,
        f".{first_identity}.staging-crashleft",
        f".{first_identity}.rejected-{'1' * 24}",
        first_identity,
    )
    for name in names:
        (datasets_dir / name).mkdir()

    discovery = discover_dataset_directories(artifact_root)

    assert tuple(path.name for path in discovery.canonical_directories) == (
        first_identity,
        second_identity,
    )
    assert tuple(
        path.name for path in discovery.stale_staging_directories
    ) == (f".{first_identity}.staging-crashleft",)
    assert tuple(
        path.name for path in discovery.rejected_directories
    ) == (f".{first_identity}.rejected-{'1' * 24}",)
    verify_dataset_directory_discovery(discovery)

    displaced = datasets_dir / f".{first_identity}.displaced"
    (datasets_dir / first_identity).rename(displaced)
    (datasets_dir / first_identity).write_bytes(b"replacement")

    with pytest.raises(ArtifactValidationError, match="discovery changed"):
        verify_dataset_directory_discovery(discovery)


def test_task3_round1_canonical_recheck_ignores_diagnostic_churn(tmp_path):
    from gsdiff.data.artifacts import (
        discover_dataset_directories,
        verify_canonical_dataset_directory_discovery,
    )

    datasets_dir = tmp_path / "artifacts" / "datasets"
    datasets_dir.mkdir(parents=True)
    identity = "a" * 64
    (datasets_dir / identity).mkdir()
    stale = datasets_dir / f".{identity}.staging-crashleft"
    rejected = datasets_dir / f".{identity}.rejected-{'1' * 24}"
    stale.mkdir()
    rejected.mkdir()
    discovery = discover_dataset_directories(tmp_path / "artifacts")

    stale.rmdir()
    rejected.rmdir()
    (datasets_dir / f".{identity}.staging-new").mkdir()
    (datasets_dir / f".{identity}.rejected-{'2' * 24}").mkdir()

    assert (
        verify_canonical_dataset_directory_discovery(discovery) == ()
    )


def test_task3_round1_canonical_recheck_reports_stable_addition(tmp_path):
    from gsdiff.data.artifacts import (
        discover_dataset_directories,
        verify_canonical_dataset_directory_discovery,
    )

    artifact_root = tmp_path / "artifacts"
    datasets_dir = artifact_root / "datasets"
    datasets_dir.mkdir(parents=True)
    (datasets_dir / ("a" * 64)).mkdir()
    discovery = discover_dataset_directories(artifact_root)
    added = datasets_dir / ("b" * 64)
    added.mkdir()

    assert verify_canonical_dataset_directory_discovery(discovery) == (
        added.absolute(),
    )


@pytest.mark.parametrize("mutation", ["removed", "replaced"])
def test_task3_round1_canonical_recheck_rejects_existing_candidate_change(
    tmp_path,
    mutation,
):
    from gsdiff.data.artifacts import (
        ArtifactValidationError,
        discover_dataset_directories,
        verify_canonical_dataset_directory_discovery,
    )

    artifact_root = tmp_path / "artifacts"
    datasets_dir = artifact_root / "datasets"
    datasets_dir.mkdir(parents=True)
    candidate = datasets_dir / ("a" * 64)
    candidate.mkdir()
    discovery = discover_dataset_directories(artifact_root)
    displaced = tmp_path / "displaced"
    candidate.rename(displaced)
    if mutation == "replaced":
        candidate.mkdir()

    with pytest.raises(
        ArtifactValidationError,
        match="canonical dataset directory changed",
    ):
        verify_canonical_dataset_directory_discovery(discovery)


def test_task3_round1_canonical_recheck_closes_post_signature_addition_race(
    tmp_path,
    monkeypatch,
):
    import gsdiff.data._artifact_persistence as persistence

    artifact_root = tmp_path / "artifacts"
    datasets_dir = artifact_root / "datasets"
    datasets_dir.mkdir(parents=True)
    existing = datasets_dir / ("a" * 64)
    existing.mkdir()
    discovery = persistence.discover_dataset_directories(artifact_root)
    addition = datasets_dir / ("b" * 64)
    real_signature = persistence._discovery_directory_signature
    candidate_signature_calls = 0

    def signature_then_add(path, noun):
        nonlocal candidate_signature_calls
        signature = real_signature(path, noun)
        if Path(path) == existing:
            candidate_signature_calls += 1
            if candidate_signature_calls == 2:
                addition.mkdir()
        return signature

    monkeypatch.setattr(
        persistence,
        "_discovery_directory_signature",
        signature_then_add,
    )

    with pytest.raises(
        ArtifactValidationError,
        match="canonical dataset directory discovery changed",
    ):
        persistence.verify_canonical_dataset_directory_discovery(discovery)
    assert addition.is_dir()


def test_task3_round1_canonical_recheck_closes_same_name_replacement_race(
    tmp_path,
    monkeypatch,
):
    import gsdiff.data._artifact_persistence as persistence

    artifact_root = tmp_path / "artifacts"
    datasets_dir = artifact_root / "datasets"
    datasets_dir.mkdir(parents=True)
    candidate = datasets_dir / ("a" * 64)
    candidate.mkdir()
    discovery = persistence.discover_dataset_directories(artifact_root)
    displaced = tmp_path / "displaced-candidate"
    real_signature = persistence._discovery_directory_signature
    root_signature_calls = 0

    def signature_then_replace(path, noun):
        nonlocal root_signature_calls
        signature = real_signature(path, noun)
        if Path(path) == artifact_root:
            root_signature_calls += 1
            if root_signature_calls == 2:
                candidate.rename(displaced)
                candidate.mkdir()
        return signature

    monkeypatch.setattr(
        persistence,
        "_discovery_directory_signature",
        signature_then_replace,
    )

    with pytest.raises(
        ArtifactValidationError,
        match="canonical dataset directory discovery changed",
    ):
        persistence.verify_canonical_dataset_directory_discovery(discovery)
    assert candidate.is_dir()
    assert displaced.is_dir()


def test_task3_c3a_dataset_discovery_rejects_unknown_or_non_directory_entries(
    tmp_path,
):
    from gsdiff.data.artifacts import (
        ArtifactValidationError,
        discover_dataset_directories,
    )

    datasets_dir = tmp_path / "artifacts" / "datasets"
    datasets_dir.mkdir(parents=True)
    (datasets_dir / "unexpected.txt").write_bytes(b"unexpected")

    with pytest.raises(
        ArtifactValidationError,
        match="unexpected dataset directory entry",
    ):
        discover_dataset_directories(tmp_path / "artifacts")


def test_task3_c3a_dataset_discovery_rejects_cross_device_candidate(
    tmp_path, monkeypatch
):
    import gsdiff.data._artifact_persistence as persistence

    datasets_dir = tmp_path / "artifacts" / "datasets"
    datasets_dir.mkdir(parents=True)
    identity = "a" * 64
    (datasets_dir / identity).mkdir()
    real_signature = persistence._discovery_directory_signature

    def cross_device_signature(path, noun):
        signature = real_signature(path, noun)
        if path.name == identity:
            return (signature[0] + 1, *signature[1:])
        return signature

    monkeypatch.setattr(
        persistence,
        "_discovery_directory_signature",
        cross_device_signature,
    )

    with pytest.raises(
        ArtifactValidationError,
        match="crosses a filesystem boundary",
    ):
        persistence.discover_dataset_directories(tmp_path / "artifacts")


def test_task3_c3a_dataset_discovery_rejects_non_directory_ancestor(
    tmp_path,
):
    from gsdiff.data.artifacts import discover_dataset_directories

    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"not-a-directory")
    artifact_root = blocker / "artifacts"

    with pytest.raises(
        ArtifactValidationError,
        match="ancestor must be a real directory",
    ):
        discover_dataset_directories(artifact_root)

    assert blocker.read_bytes() == b"not-a-directory"


def test_task3_c3a_missing_discovery_recheck_rejects_ancestor_file(
    tmp_path,
):
    from gsdiff.data.artifacts import (
        discover_dataset_directories,
        verify_dataset_directory_discovery,
    )

    blocker = tmp_path / "missing-parent"
    discovery = discover_dataset_directories(blocker / "artifacts")
    blocker.write_bytes(b"replacement")

    with pytest.raises(ArtifactValidationError, match="discovery changed"):
        verify_dataset_directory_discovery(discovery)


def test_task3_c3d_discovery_fails_closed_when_datasets_appears_after_root_scan(
    tmp_path, monkeypatch
):
    import gsdiff.data._artifact_persistence as persistence

    artifact_root = (tmp_path / "artifacts").absolute()
    datasets_dir = artifact_root / "datasets"
    real_lexists = persistence.os.path.lexists
    real_reject_links = persistence._reject_linked_ancestors
    armed = False
    injected = False

    def arm_after_initial_ancestor_scan(path):
        nonlocal armed
        result = real_reject_links(path)
        armed = True
        return result

    def inject_after_missing_root_observation(path):
        nonlocal injected
        result = real_lexists(path)
        if (
            armed
            and not injected
            and Path(path) == artifact_root
            and result is False
        ):
            datasets_dir.mkdir(parents=True)
            injected = True
        return result

    monkeypatch.setattr(
        persistence,
        "_reject_linked_ancestors",
        arm_after_initial_ancestor_scan,
    )
    monkeypatch.setattr(
        persistence.os.path,
        "lexists",
        inject_after_missing_root_observation,
    )

    with pytest.raises(
        ArtifactValidationError,
        match="discovery changed during scan",
    ):
        persistence.discover_dataset_directories(artifact_root)

    assert injected is True
    assert datasets_dir.is_dir()


@pytest.mark.parametrize(
    ("descriptor", "renderer"),
    [
        ("char:5", None),
        ("char:5", {}),
        (
            "char:5",
            {
                "font_family": "DejaVu Sans",
                "fill_fraction": 0.8,
                "resample": "nearest",
                "supersample": 4,
            },
        ),
        (
            "char:5",
            {
                "font_family": "",
                "fill_fraction": 0.8,
                "resample": "lanczos",
                "supersample": 4,
            },
        ),
        (
            "char:5",
            {
                "font_family": "DejaVu Sans",
                "fill_fraction": True,
                "resample": "lanczos",
                "supersample": 4,
            },
        ),
        (
            "char:5",
            {
                "font_family": "DejaVu Sans",
                "fill_fraction": 0.0,
                "resample": "lanczos",
                "supersample": 4,
            },
        ),
        (
            "char:5",
            {
                "font_family": "DejaVu Sans",
                "fill_fraction": 1.01,
                "resample": "lanczos",
                "supersample": 4,
            },
        ),
        (
            "char:5",
            {
                "font_family": "DejaVu Sans",
                "fill_fraction": 0.8,
                "resample": "lanczos",
                "supersample": False,
            },
        ),
        (
            "char:5",
            {
                "font_family": "DejaVu Sans",
                "fill_fraction": 0.8,
                "resample": "lanczos",
                "supersample": 0,
                "extra": 1,
            },
        ),
        ("assets/target.png", None),
        ("assets/target.png", {}),
        (
            "assets/target.png",
            {"color_mode": "rgb", "resample": "lanczos"},
        ),
        (
            "assets/target.png",
            {"color_mode": "grayscale", "resample": "nearest"},
        ),
        (
            "assets/target.png",
            {
                "color_mode": "grayscale",
                "resample": "lanczos",
                "extra": 1,
            },
        ),
    ],
)
def test_task3_round1_target_snapshot_rejects_invalid_renderer(
    descriptor, renderer
):
    from gsdiff.data.artifacts import TargetSnapshot

    with pytest.raises((TypeError, ArtifactValidationError), match="renderer"):
        TargetSnapshot(
            target_id="target",
            descriptor=descriptor,
            assets_sha256={"asset": "a" * 64},
            canonical_image=np.zeros((8, 8), dtype=np.float32),
            renderer=renderer,
        )


@pytest.mark.parametrize(
    "font_family",
    [
        "../DejaVu Sans",
        "DejaVu/Sans",
        "DejaVu\\Sans",
        "DejaVu\x00Sans",
        "DejaVu\x1fSans",
        "x" * 129,
    ],
)
def test_task3_round1_target_snapshot_rejects_pathlike_font_family(
    font_family,
):
    from gsdiff.data.artifacts import TargetSnapshot

    with pytest.raises(ArtifactValidationError, match="font_family"):
        TargetSnapshot(
            target_id="target",
            descriptor="char:5",
            assets_sha256={"asset": "a" * 64},
            canonical_image=np.zeros((8, 8), dtype=np.float32),
            renderer={
                "font_family": font_family,
                "fill_fraction": 0.8,
                "resample": "lanczos",
                "supersample": 4,
            },
        )


def test_task3_round1_request_revalidates_frozen_target_renderer():
    from gsdiff.data.artifacts import resolve_corrected_dataset_request

    inputs = _phase3b_generation_inputs()
    target = inputs["target_snapshot"]
    object.__setattr__(
        target,
        "renderer",
        {
            "font_family": "DejaVu Sans",
            "fill_fraction": 0.8,
            "resample": "nearest",
            "supersample": 4,
        },
    )

    with pytest.raises(ArtifactValidationError, match="renderer"):
        resolve_corrected_dataset_request(**inputs)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("target-id", "target_id"),
        ("descriptor", "descriptor"),
        ("assets-empty", "assets_sha256"),
        ("asset-name-empty", "asset name"),
        ("asset-digest-invalid", "asset hash"),
        ("canonical-integer", "canonical_image"),
        ("canonical-nonfinite", "canonical_image"),
        ("canonical-out-of-range", "canonical_image"),
    ],
)
def test_task3_round1_request_revalidates_all_tampered_target_fields(
    mutation, message
):
    from gsdiff.data.artifacts import resolve_corrected_dataset_request

    inputs = _phase3b_generation_inputs()
    target = inputs["target_snapshot"]
    if mutation == "target-id":
        object.__setattr__(target, "target_id", "../truth")
    elif mutation == "descriptor":
        object.__setattr__(target, "descriptor", None)
    elif mutation == "assets-empty":
        object.__setattr__(target, "assets_sha256", {})
    elif mutation == "asset-name-empty":
        object.__setattr__(
            target, "assets_sha256", {"": "a" * 64}
        )
    elif mutation == "asset-digest-invalid":
        object.__setattr__(
            target, "assets_sha256", {"asset": "not-a-sha"}
        )
    elif mutation == "canonical-integer":
        object.__setattr__(
            target,
            "canonical_image",
            np.zeros((8, 8), dtype=np.int32),
        )
    elif mutation == "canonical-nonfinite":
        image = np.zeros((8, 8), dtype=np.float32)
        image[0, 0] = np.nan
        object.__setattr__(target, "canonical_image", image)
    else:
        image = np.zeros((8, 8), dtype=np.float32)
        image[0, 0] = 1.01
        object.__setattr__(target, "canonical_image", image)

    with pytest.raises(
        (TypeError, ArtifactValidationError), match=message
    ):
        resolve_corrected_dataset_request(**inputs)


def test_task3_round2_resolver_owns_target_after_validation(monkeypatch):
    import gsdiff.data._corrected_generation as corrected

    inputs = _phase3b_generation_inputs()
    target = inputs["target_snapshot"]
    caller_assets = _mutable_json(target.assets_sha256)
    caller_renderer = _mutable_json(target.renderer)
    object.__setattr__(target, "assets_sha256", caller_assets)
    object.__setattr__(target, "renderer", caller_renderer)
    expected_target = {
        "id": target.target_id,
        "descriptor": target.descriptor,
        "assets_sha256": _mutable_json(target.assets_sha256),
        "renderer": _mutable_json(target.renderer),
    }
    real_validator = corrected._validate_generation_inputs
    validation_calls = 0

    def validate_then_mutate(**kwargs):
        nonlocal validation_calls
        validated = real_validator(**kwargs)
        validation_calls += 1
        object.__setattr__(target, "target_id", "mutated-target")
        object.__setattr__(target, "descriptor", "mutated.png")
        caller_assets.clear()
        caller_assets["in-place.png"] = "e" * 64
        caller_renderer.clear()
        caller_renderer.update(
            {"color_mode": "grayscale", "resample": "lanczos"}
        )
        object.__setattr__(
            target, "assets_sha256", {"mutated.png": "f" * 64}
        )
        object.__setattr__(
            target,
            "renderer",
            {"color_mode": "grayscale", "resample": "lanczos"},
        )
        return validated

    monkeypatch.setattr(
        corrected, "_validate_generation_inputs", validate_then_mutate
    )
    request = corrected.resolve_corrected_dataset_request(**inputs)

    assert validation_calls == 1
    assert request["target"] == expected_target
    assert request["resolved_generator_config"]["target"] == expected_target


def test_task3_round2_generator_owns_target_after_validation(monkeypatch):
    import gsdiff.data._corrected_generation as corrected

    baseline = corrected.generate_corrected_dataset(
        **_phase3b_generation_inputs()
    )
    inputs = _phase3b_generation_inputs()
    target = inputs["target_snapshot"]
    caller_image = np.array(target.canonical_image, copy=True)
    object.__setattr__(target, "canonical_image", caller_image)
    real_validator = corrected._validate_generation_inputs
    validation_calls = 0

    def validate_then_mutate(**kwargs):
        nonlocal validation_calls
        validated = real_validator(**kwargs)
        validation_calls += 1
        object.__setattr__(target, "target_id", "mutated-target")
        object.__setattr__(target, "descriptor", "mutated.png")
        object.__setattr__(
            target, "assets_sha256", {"mutated.png": "f" * 64}
        )
        object.__setattr__(
            target,
            "renderer",
            {"color_mode": "grayscale", "resample": "lanczos"},
        )
        caller_image[...] = 1.0
        object.__setattr__(
            target,
            "canonical_image",
            np.ones((8, 8), dtype=np.float32),
        )
        return validated

    monkeypatch.setattr(
        corrected, "_validate_generation_inputs", validate_then_mutate
    )
    generated = corrected.generate_corrected_dataset(**inputs)

    assert validation_calls == 1
    assert (
        generated.dataset_identity_sha256
        == baseline.dataset_identity_sha256
    )
    assert (
        generated.noise_calibration_sha256
        == baseline.noise_calibration_sha256
    )
    assert generated.dataset_identity_spec == baseline.dataset_identity_spec
    assert (
        generated.resolved_generator_config
        == baseline.resolved_generator_config
    )
    assert (
        generated.noise_calibration_record
        == baseline.noise_calibration_record
    )
    _assert_dataclass_equal(baseline.acquisition, generated.acquisition)
    _assert_dataclass_equal(baseline.truth, generated.truth)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contract-id", "../contract"),
        ("contract-id", ""),
        ("contract-sha", "not-a-sha"),
        ("target-assets", {}),
    ],
)
def test_task3_round1_identity_validator_is_authoritative(field, value):
    from gsdiff.data.artifacts import validate_dataset_identity_spec

    identity = _phase3b_generate().dataset_identity_spec
    if field == "contract-id":
        identity["scientific_contract"]["id"] = value
    elif field == "contract-sha":
        identity["scientific_contract"]["sha256"] = value
    else:
        identity["target"]["assets_sha256"] = value

    with pytest.raises((TypeError, ArtifactValidationError)):
        validate_dataset_identity_spec(identity)


def _task3_round1_truth_with_reference_dtype(dtype):
    generated = _phase3b_generate()
    identity = generated.dataset_identity_spec
    metadata = _mutable_json(generated.truth.evaluator_metadata)
    record = metadata["noise_calibration_record"]
    record["reference_measurements"]["dtype"] = dtype
    identity["noise_calibration"]["sha256"] = hashlib.sha256(
        _canonical_json_bytes(record)
    ).hexdigest()
    dataset_identity = hashlib.sha256(
        _canonical_json_bytes(identity)
    ).hexdigest()
    truth = dataclasses.replace(
        generated.truth,
        dataset_identity_sha256=dataset_identity,
        dataset_identity_spec=identity,
        evaluator_metadata=metadata,
    )
    return dataset_identity, truth


def test_task3_round1_truth_serializer_rejects_non_float64_reference_dtype():
    from gsdiff.data._artifact_truth import evaluation_truth_npz_bytes

    _, truth = _task3_round1_truth_with_reference_dtype("<f4")
    with pytest.raises(ArtifactValidationError, match="dtype|float64"):
        evaluation_truth_npz_bytes(truth)


def test_task3_round1_truth_loader_rejects_non_float64_reference_dtype():
    from gsdiff.data._artifact_truth import load_evaluation_truth_bytes

    generated, payloads, _, _ = _phase3c_build_bundle()
    expected_identity = None

    def mutate(members):
        nonlocal expected_identity
        raw = np.load(
            io.BytesIO(members["__metadata_json__.npy"]),
            allow_pickle=False,
        )
        metadata = json.loads(raw.tobytes().decode("utf-8"))
        record = metadata["evaluator_metadata"][
            "noise_calibration_record"
        ]
        record["reference_measurements"]["dtype"] = "<f4"
        identity = metadata["dataset_identity_spec"]
        identity["noise_calibration"]["sha256"] = hashlib.sha256(
            _canonical_json_bytes(record)
        ).hexdigest()
        expected_identity = hashlib.sha256(
            _canonical_json_bytes(identity)
        ).hexdigest()
        metadata["dataset_identity_sha256"] = expected_identity
        encoded = _canonical_json_bytes(metadata)
        changed = dict(members)
        changed["__metadata_json__.npy"] = _npy_bytes(
            np.frombuffer(encoded, dtype=np.uint8)
        )
        return changed

    payload = _phase3c_rezip(
        payloads["evaluation-truth.npz"],
        mutate,
        canonical=True,
    )
    assert expected_identity is not None
    assert expected_identity != generated.dataset_identity_sha256
    with pytest.raises(ArtifactValidationError, match="dtype|float64"):
        load_evaluation_truth_bytes(
            payload,
            expected_dataset_identity_sha256=expected_identity,
        )


def _task3_round1_rehash_manifest(manifest):
    config = manifest["resolved_generator_config"]
    identity = manifest["dataset_identity_spec"]
    record = manifest["noise_calibration_record"]
    config_sha256 = hashlib.sha256(
        _canonical_json_bytes(config)
    ).hexdigest()
    identity["generator_config_sha256"] = config_sha256
    record["generator_config_sha256"] = config_sha256
    identity["noise_calibration"]["sha256"] = hashlib.sha256(
        _canonical_json_bytes(record)
    ).hexdigest()
    manifest["dataset_identity_sha256"] = hashlib.sha256(
        _canonical_json_bytes(identity)
    ).hexdigest()
    return manifest


def _task3_round1_set_nested_value(mapping, path, value):
    current = mapping
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def test_task3_round1_manifest_schema_matches_runtime_path_free_ids():
    from jsonschema import Draft202012Validator
    from gsdiff.data._artifact_identity import (
        validate_path_free_opaque_id,
    )
    from gsdiff.data.artifacts import dataset_manifest_bytes

    schema = json.loads(
        (
            REPO_ROOT / "schemas" / "dataset-manifest-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(schema)
    _, _, base_manifest, _ = _phase3c_build_bundle()
    base_manifest = _mutable_json(base_manifest)
    assert not list(validator.iter_errors(base_manifest))

    path_free_fields = (
        (
            "resolvedTarget.id",
            ("resolved_generator_config", "target", "id"),
        ),
        (
            "resolvedMotion.id",
            ("resolved_generator_config", "motion", "id"),
        ),
        (
            "resolvedAcquisition.noise_calibration_id",
            (
                "resolved_generator_config",
                "acquisition",
                "noise_calibration_id",
            ),
        ),
        (
            "calibrationDescriptor.id",
            ("noise_calibration_record", "calibration", "id"),
        ),
        (
            "noiseCalibrationRecord.scientific_contract.id",
            (
                "noise_calibration_record",
                "scientific_contract",
                "id",
            ),
        ),
        (
            "noiseCalibrationRecord.target_id",
            ("noise_calibration_record", "target_id"),
        ),
        (
            "noiseCalibrationRecord.motion_id",
            ("noise_calibration_record", "motion_id"),
        ),
        (
            "noiseCalibrationRecord.generator.id",
            ("noise_calibration_record", "generator", "id"),
        ),
        (
            "noiseCalibrationRecord.generator.version",
            ("noise_calibration_record", "generator", "version"),
        ),
        (
            "datasetIdentity.scientific_contract.id",
            (
                "dataset_identity_spec",
                "scientific_contract",
                "id",
            ),
        ),
        (
            "datasetIdentity.target.id",
            ("dataset_identity_spec", "target", "id"),
        ),
        (
            "datasetIdentity.motion.id",
            ("dataset_identity_spec", "motion", "id"),
        ),
        (
            "datasetIdentity.noise_calibration.id",
            (
                "dataset_identity_spec",
                "noise_calibration",
                "id",
            ),
        ),
        (
            "datasetIdentity.generator.id",
            ("dataset_identity_spec", "generator", "id"),
        ),
        (
            "datasetIdentity.generator.version",
            ("dataset_identity_spec", "generator", "version"),
        ),
    )
    opaque_id_cases = (
        ("valid-minimum", "A"),
        ("valid-punctuation", "Alpha_1.beta-2"),
        ("valid-maximum", "A" + ("x" * 127)),
        ("valid-embedded-gt", "alphaGTbeta"),
        ("empty", ""),
        ("leading-punctuation", "_alpha"),
        ("forward-slash", "alpha/beta"),
        ("backslash", "alpha\\beta"),
        ("space", "alpha beta"),
        ("non-ascii", "αlpha"),
        ("too-long", "A" + ("x" * 128)),
        ("trailing-newline", "alpha\n"),
        ("double-dot", "alpha..beta"),
        ("reserved-truth", "alphaTruthBeta"),
        ("reserved-evaluation", "alphaEvaluationBeta"),
        ("reserved-evaluator", "alphaEvaluatorBeta"),
        ("reserved-canonical", "alphaCanonicalBeta"),
        ("reserved-trajectory", "alphaTrajectoryBeta"),
        ("reserved-metric", "alphaMetricBeta"),
        ("reserved-display", "alphaDisplayBeta"),
        ("reserved-normalized", "alphaNormalizedBeta"),
        ("reserved-gt-exact", "GT"),
        ("reserved-gt-segment", "alpha-Gt_beta"),
    )
    mismatches = []
    for field_name, path in path_free_fields:
        for case_name, value in opaque_id_cases:
            manifest = _mutable_json(base_manifest)
            _task3_round1_set_nested_value(manifest, path, value)
            schema_accepts = not list(validator.iter_errors(manifest))
            try:
                validate_path_free_opaque_id(value, field_name)
            except (TypeError, ArtifactValidationError):
                runtime_accepts = False
            else:
                runtime_accepts = True
            if schema_accepts != runtime_accepts:
                mismatches.append(
                    {
                        "field": field_name,
                        "case": case_name,
                        "schema_accepts": schema_accepts,
                        "runtime_accepts": runtime_accepts,
                    }
                )

    for field_name, path in (
        (
            "resolvedTarget.assets_sha256",
            ("resolved_generator_config", "target", "assets_sha256"),
        ),
        (
            "datasetIdentity.target.assets_sha256",
            ("dataset_identity_spec", "target", "assets_sha256"),
        ),
    ):
        manifest = _mutable_json(base_manifest)
        named_hashes = manifest
        for key in path:
            named_hashes = named_hashes[key]
        digest = next(iter(named_hashes.values()))
        _task3_round1_set_nested_value(manifest, path, {"": digest})
        schema_accepts = not list(validator.iter_errors(manifest))
        _task3_round1_rehash_manifest(manifest)
        try:
            dataset_manifest_bytes(manifest)
        except (TypeError, ArtifactValidationError):
            runtime_accepts = False
        else:
            runtime_accepts = True
        if schema_accepts != runtime_accepts:
            mismatches.append(
                {
                    "field": field_name,
                    "case": "empty-property-name",
                    "schema_accepts": schema_accepts,
                    "runtime_accepts": runtime_accepts,
                }
            )

    assert not mismatches, json.dumps(mismatches, indent=2)


@pytest.mark.parametrize(
    "mutation",
    [
        "sha256-trailing-newline",
        "identity-git-commit-trailing-newline",
        "record-git-commit-trailing-newline",
        "font-family-trailing-newline",
    ],
)
def test_task3_round1_manifest_schema_matches_runtime_fullmatch_patterns(
    mutation,
):
    from jsonschema import Draft202012Validator
    from gsdiff.data.artifacts import dataset_manifest_bytes

    schema = json.loads(
        (
            REPO_ROOT / "schemas" / "dataset-manifest-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    _, _, manifest, _ = _phase3c_build_bundle()
    manifest = _mutable_json(manifest)
    if mutation == "sha256-trailing-newline":
        manifest["dataset_identity_sha256"] = ("0" * 64) + "\n"
    elif mutation == "identity-git-commit-trailing-newline":
        manifest["dataset_identity_spec"]["generator"]["git_commit"] = (
            ("0" * 40) + "\n"
        )
        _task3_round1_rehash_manifest(manifest)
    elif mutation == "record-git-commit-trailing-newline":
        manifest["noise_calibration_record"]["generator"]["git_commit"] = (
            ("0" * 40) + "\n"
        )
        _task3_round1_rehash_manifest(manifest)
    else:
        manifest["resolved_generator_config"]["target"]["renderer"][
            "font_family"
        ] = "DejaVu Sans\n"
        _task3_round1_rehash_manifest(manifest)

    schema_accepts = not list(
        Draft202012Validator(schema).iter_errors(manifest)
    )
    try:
        dataset_manifest_bytes(manifest)
    except (TypeError, ArtifactValidationError):
        runtime_accepts = False
    else:
        runtime_accepts = True
    assert runtime_accepts is False, mutation
    assert schema_accepts is False, mutation
    assert schema_accepts == runtime_accepts, mutation


@pytest.mark.parametrize(
    "mutation",
    [
        "renderer-null",
        "renderer-empty",
        "renderer-wrong-const",
        "renderer-extra",
        "renderer-fill-range",
        "renderer-supersample-bool",
        "renderer-font-family-path",
        "renderer-char-file-family",
        "renderer-file-glyph-family",
        "contract-path",
        "contract-double-dot",
        "contract-reserved-token",
        "contract-bad-sha",
    ],
)
def test_task3_round1_schema_and_runtime_reject_coherent_semantic_holes(
    mutation,
):
    from jsonschema import Draft202012Validator
    from gsdiff.data.artifacts import dataset_manifest_bytes

    schema = json.loads(
        (
            REPO_ROOT / "schemas" / "dataset-manifest-v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    _, _, manifest, _ = _phase3c_build_bundle()
    manifest = _mutable_json(manifest)
    if mutation == "renderer-null":
        manifest["resolved_generator_config"]["target"]["renderer"] = None
    elif mutation == "renderer-empty":
        manifest["resolved_generator_config"]["target"]["renderer"] = {}
    elif mutation == "renderer-wrong-const":
        manifest["resolved_generator_config"]["target"]["renderer"][
            "resample"
        ] = "nearest"
    elif mutation == "renderer-extra":
        manifest["resolved_generator_config"]["target"]["renderer"][
            "extra"
        ] = 1
    elif mutation == "renderer-fill-range":
        manifest["resolved_generator_config"]["target"]["renderer"][
            "fill_fraction"
        ] = 1.01
    elif mutation == "renderer-supersample-bool":
        manifest["resolved_generator_config"]["target"]["renderer"][
            "supersample"
        ] = True
    elif mutation == "renderer-font-family-path":
        manifest["resolved_generator_config"]["target"]["renderer"][
            "font_family"
        ] = "../DejaVu Sans"
    elif mutation == "renderer-char-file-family":
        manifest["resolved_generator_config"]["target"]["renderer"] = {
            "color_mode": "grayscale",
            "resample": "lanczos",
        }
    elif mutation == "renderer-file-glyph-family":
        manifest["resolved_generator_config"]["target"][
            "descriptor"
        ] = "assets/target.png"
    elif mutation == "contract-path":
        contract = {"id": "../contract", "sha256": "3" * 64}
        manifest["dataset_identity_spec"]["scientific_contract"] = contract
        manifest["noise_calibration_record"][
            "scientific_contract"
        ] = contract
    elif mutation == "contract-double-dot":
        contract = {"id": "contract..v1", "sha256": "3" * 64}
        manifest["dataset_identity_spec"]["scientific_contract"] = contract
        manifest["noise_calibration_record"][
            "scientific_contract"
        ] = contract
    elif mutation == "contract-reserved-token":
        contract = {"id": "truth-contract", "sha256": "3" * 64}
        manifest["dataset_identity_spec"]["scientific_contract"] = contract
        manifest["noise_calibration_record"][
            "scientific_contract"
        ] = contract
    else:
        contract = {"id": "contract", "sha256": "not-a-sha"}
        manifest["dataset_identity_spec"]["scientific_contract"] = contract
        manifest["noise_calibration_record"][
            "scientific_contract"
        ] = contract
    _task3_round1_rehash_manifest(manifest)

    assert list(Draft202012Validator(schema).iter_errors(manifest)), mutation
    with pytest.raises(
        (TypeError, ArtifactValidationError), match="renderer|scientific"
    ):
        dataset_manifest_bytes(manifest)


def test_task3_round1_directory_parser_rejects_coherent_invalid_renderer(
    tmp_path,
):
    from gsdiff.data.artifacts import verify_dataset_directory

    _, _, manifest, _, dataset_dir = _phase3c_write_dataset_directory(
        tmp_path
    )
    manifest = _mutable_json(manifest)
    manifest["resolved_generator_config"]["target"]["renderer"] = None
    _task3_round1_rehash_manifest(manifest)
    (dataset_dir / "dataset-manifest.json").write_bytes(
        _canonical_json_bytes(manifest)
    )

    with pytest.raises(ArtifactValidationError, match="renderer"):
        verify_dataset_directory(dataset_dir)


@pytest.mark.parametrize("loader_kind", ["path", "bytes"])
@pytest.mark.parametrize(
    "spoof",
    [
        "identity-string-subclass",
        "spec-dict-subclass",
        "nested-string-subclass",
        "nested-numpy-int",
    ],
)
def test_task3_round1_blind_loaders_reject_nonexact_expected_anchors(
    tmp_path, loader_kind, spoof
):
    from gsdiff.data._artifact_dataset import load_acquisition_data_bytes

    generated, payloads, _, _ = _phase3c_build_bundle()
    identity = generated.dataset_identity_sha256
    spec = _mutable_json(blind_acquisition_spec(generated.acquisition))
    if spoof == "identity-string-subclass":
        identity = type("StringSubclass", (str,), {})(identity)
    elif spoof == "spec-dict-subclass":
        spec = type("DictSubclass", (dict,), {})(spec)
    elif spoof == "nested-string-subclass":
        spec["acquisition"]["pattern_family"] = type(
            "StringSubclass", (str,), {}
        )(spec["acquisition"]["pattern_family"])
    else:
        spec["dimensions"]["H"] = np.int64(spec["dimensions"]["H"])
    payload = payloads["measurements.npz"]
    path = tmp_path / "measurements.npz"
    path.write_bytes(payload)
    loader = (
        load_acquisition_data
        if loader_kind == "path"
        else load_acquisition_data_bytes
    )
    source = path if loader_kind == "path" else payload

    with pytest.raises((TypeError, ArtifactValidationError)):
        loader(
            source,
            expected_dataset_identity_sha256=identity,
            expected_acquisition_spec=spec,
        )


@pytest.mark.parametrize("loader_kind", ["path", "bytes"])
def test_task3_round1_blind_loaders_accept_exact_mapping_proxy_anchor(
    tmp_path, loader_kind
):
    from types import MappingProxyType
    from gsdiff.data._artifact_dataset import load_acquisition_data_bytes

    generated, payloads, _, _ = _phase3c_build_bundle()
    payload = payloads["measurements.npz"]
    path = tmp_path / "measurements.npz"
    path.write_bytes(payload)
    loader = (
        load_acquisition_data
        if loader_kind == "path"
        else load_acquisition_data_bytes
    )
    source = path if loader_kind == "path" else payload
    spec = MappingProxyType(
        _mutable_json(blind_acquisition_spec(generated.acquisition))
    )

    loaded = loader(
        source,
        expected_dataset_identity_sha256=(
            generated.dataset_identity_sha256
        ),
        expected_acquisition_spec=spec,
    )
    assert loaded.dataset_identity_sha256 == (
        generated.dataset_identity_sha256
    )


def _task3_round1_write_target_png(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.arange(64, dtype=np.uint8).reshape(8, 8),
        mode="L",
    ).save(path, format="PNG")
    return path.read_bytes()


@pytest.mark.parametrize("role", ["target", "font"])
def test_task3_round1_target_inputs_enforce_bound_before_bulk_read(
    tmp_path, monkeypatch, role
):
    import gsdiff.data._artifact_io as artifact_io
    import gsdiff.data._corrected_generation as corrected

    repo = tmp_path / "repo"
    repo.mkdir()
    if role == "target":
        leaf = repo / "assets" / "target.png"
        payload = _task3_round1_write_target_png(leaf)
        monkeypatch.setattr(
            corrected, "_MAX_TARGET_ASSET_BYTES", len(payload) - 1,
            raising=False,
        )
        descriptor = "assets/target.png"
    else:
        leaf = tmp_path / "oversize-font.ttf"
        payload = b"font-bytes"
        leaf.write_bytes(payload)
        monkeypatch.setattr(
            corrected, "_bundled_dejavu_font_path", lambda: leaf
        )
        monkeypatch.setattr(
            corrected, "_MAX_TARGET_FONT_BYTES", len(payload) - 1,
            raising=False,
        )
        descriptor = "char:5"

    def forbidden_bulk_read(*args, **kwargs):
        raise AssertionError("oversize target reached bulk read")

    monkeypatch.setattr(artifact_io.os, "read", forbidden_bulk_read)
    with pytest.raises(ArtifactValidationError, match="byte bound"):
        corrected.resolve_target_snapshot(
            repo_root=repo,
            target_id="target",
            descriptor=descriptor,
            H=8,
            W=8,
        )


@pytest.mark.parametrize("role", ["target", "font"])
def test_task3_round1_target_inputs_reject_hardlinked_leaf(
    tmp_path, monkeypatch, role
):
    import gsdiff.data._corrected_generation as corrected

    repo = tmp_path / "repo"
    repo.mkdir()
    source = tmp_path / f"{role}-source"
    leaf = (
        repo / "assets" / "target.png"
        if role == "target"
        else tmp_path / "font-link.ttf"
    )
    leaf.parent.mkdir(parents=True, exist_ok=True)
    if role == "target":
        _task3_round1_write_target_png(source)
        descriptor = "assets/target.png"
    else:
        source.write_bytes(
            corrected._bundled_dejavu_font_path().read_bytes()
        )
        monkeypatch.setattr(
            corrected, "_bundled_dejavu_font_path", lambda: leaf
        )
        descriptor = "char:5"
    try:
        os.link(source, leaf)
    except OSError as error:
        pytest.skip(f"hardlinks unavailable: {error}")

    with pytest.raises(ArtifactValidationError, match="hardlink"):
        corrected.resolve_target_snapshot(
            repo_root=repo,
            target_id="target",
            descriptor=descriptor,
            H=8,
            W=8,
        )


@pytest.mark.parametrize("role", ["target", "font"])
@pytest.mark.parametrize("mutation", ["restored-mtime", "leaf-replacement"])
def test_task3_round1_target_inputs_recheck_physical_snapshot_after_use(
    tmp_path, monkeypatch, role, mutation
):
    import gsdiff.data._corrected_generation as corrected

    repo = tmp_path / "repo"
    repo.mkdir()
    if role == "target":
        leaf = repo / "assets" / "target.png"
        _task3_round1_write_target_png(leaf)
        descriptor = "assets/target.png"
    else:
        leaf = tmp_path / "font.ttf"
        leaf.write_bytes(
            corrected._bundled_dejavu_font_path().read_bytes()
        )
        monkeypatch.setattr(
            corrected, "_bundled_dejavu_font_path", lambda: leaf
        )
        descriptor = "char:5"
    real_snapshot = getattr(
        corrected, "read_safe_file_snapshot", None
    )
    if real_snapshot is None:
        from gsdiff.data._artifact_io import read_safe_file_snapshot

        real_snapshot = read_safe_file_snapshot
    injected = False

    def read_then_mutate(path, *, max_bytes, noun):
        nonlocal injected
        snapshot = real_snapshot(
            path, max_bytes=max_bytes, noun=noun
        )
        if not injected:
            injected = True
            original_mtime = snapshot.path.stat().st_mtime_ns
            if mutation == "restored-mtime":
                changed = bytes([snapshot.raw[0] ^ 1]) + snapshot.raw[1:]
                snapshot.path.write_bytes(changed)
            else:
                replacement = snapshot.path.with_name(
                    snapshot.path.name + ".replacement"
                )
                replacement.write_bytes(snapshot.raw)
                os.replace(replacement, snapshot.path)
            os.utime(
                snapshot.path,
                ns=(original_mtime, original_mtime),
            )
        return snapshot

    monkeypatch.setattr(
        corrected,
        "read_safe_file_snapshot",
        read_then_mutate,
        raising=False,
    )
    with pytest.raises(ArtifactValidationError, match="snapshot.*changed"):
        corrected.resolve_target_snapshot(
            repo_root=repo,
            target_id="target",
            descriptor=descriptor,
            H=8,
            W=8,
        )
    assert injected is True


@pytest.mark.parametrize("role", ["target", "font"])
def test_task3_round1_target_inputs_reject_same_bytes_replacement_at_second_read(
    tmp_path, monkeypatch, role
):
    import gsdiff.data._corrected_generation as corrected

    repo = tmp_path / "repo"
    repo.mkdir()
    if role == "target":
        leaf = repo / "assets" / "target.png"
        _task3_round1_write_target_png(leaf)
        descriptor = "assets/target.png"
    else:
        leaf = tmp_path / "font.ttf"
        leaf.write_bytes(
            corrected._bundled_dejavu_font_path().read_bytes()
        )
        monkeypatch.setattr(
            corrected, "_bundled_dejavu_font_path", lambda: leaf
        )
        descriptor = "char:5"
    real_snapshot = corrected.read_safe_file_snapshot
    calls = 0

    def replace_before_second_read(path, *, max_bytes, noun):
        nonlocal calls
        calls += 1
        if calls == 2:
            raw = Path(path).read_bytes()
            replacement = Path(path).with_name(
                Path(path).name + ".replacement"
            )
            replacement.write_bytes(raw)
            os.replace(replacement, path)
        return real_snapshot(path, max_bytes=max_bytes, noun=noun)

    monkeypatch.setattr(
        corrected,
        "read_safe_file_snapshot",
        replace_before_second_read,
    )
    with pytest.raises(ArtifactValidationError, match="snapshot.*changed"):
        corrected.resolve_target_snapshot(
            repo_root=repo,
            target_id="target",
            descriptor=descriptor,
            H=8,
            W=8,
        )
    assert calls == 2


_METHOD_REGISTRY_DOC = (
    REPO_ROOT / "docs" / "experiments" / "method-registry-v1.md"
)
_METHOD_REGISTRY_MARKER_START = "<!-- method-registry-machine-v1:start -->"
_METHOD_REGISTRY_MARKER_END = "<!-- method-registry-machine-v1:end -->"
_PILOT_READINESS_BLOCKERS = [
    "all-eleven-method-clean-clone-subprocess-evidence-missing",
    "diffusion-reproducible-checkpoint-locator-missing",
    "diffusion-checkpoint-training-provenance-missing",
    "required-cuda-preflight-evidence-missing",
    "required-disk-preflight-evidence-missing",
]


def _load_method_registry_document():
    text = _METHOD_REGISTRY_DOC.read_text(encoding="utf-8")
    assert text.count(_METHOD_REGISTRY_MARKER_START) == 1
    assert text.count(_METHOD_REGISTRY_MARKER_END) == 1
    payload = text.split(_METHOD_REGISTRY_MARKER_START, 1)[1]
    payload = payload.split(_METHOD_REGISTRY_MARKER_END, 1)[0].strip()
    assert payload.startswith("```json\n")
    assert payload.endswith("\n```")
    snapshot = json.loads(payload[len("```json\n") : -len("\n```")])
    return text, snapshot


def _canonical_json_sha256(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_method_registry_document_binds_all_canonical_publication_and_smoke_profiles():
    _, snapshot = _load_method_registry_document()
    registry_path = REPO_ROOT / "configs" / "protocols" / "methods-v1.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    canonical_ids = [method["id"] for method in registry["methods"]]
    expected_profile_hashes = {
        method["id"]: {
            profile_id: _canonical_json_sha256(method["profiles"][profile_id])
            for profile_id in (
                "publication-v1",
                "controller-cpu-smoke-v1",
            )
        }
        for method in registry["methods"]
    }
    expected_profiles = {
        method["id"]: {
            profile_id: method["profiles"][profile_id]
            for profile_id in (
                "publication-v1",
                "controller-cpu-smoke-v1",
            )
        }
        for method in registry["methods"]
    }

    assert canonical_ids == [
        "dgi",
        "static_cs",
        "perframe_cs",
        "tv3d",
        "monin",
        "gidc3dtv",
        "recinr",
        "siren",
        "recinr_se2",
        "gsdiff_tv",
        "gsdiff_diffusion",
    ]
    assert snapshot["registry"] == {
        "source": "configs/protocols/methods-v1.yaml",
        "source_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "canonical_ids": canonical_ids,
        "profile_hash_algorithm": (
            "sha256(canonical-json(raw-profile-object))"
        ),
        "profile_sha256": expected_profile_hashes,
        "profiles": expected_profiles,
    }


def test_method_registry_document_binds_rec_inr_provenance_and_license_status():
    text, snapshot = _load_method_registry_document()
    provenance = snapshot["recinr_provenance"]
    expected_local_hashes = {
        "gsdiff/baselines/recinr_model.py": (
            "7cfa1c7f20809634bc71fc14b143f815"
            "12358198af946f0039a38ad119c94eb7"
        ),
        "gsdiff/baselines/recinr.py": (
            "a8f3db6b7b34d501c077ff8592c11a47"
            "3b884db9dcc7b3e9fc16fba144ac868e"
        ),
        "gsdiff/baselines/inr.py": (
            "600c26aec1545c0f7aab36d0cad5f95d"
            "4f4005ad0d3b9042e6c09f9ccdeabdf4"
        ),
    }
    actual_local_hashes = {
        path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        for path in expected_local_hashes
    }

    assert provenance == {
        "url": "https://github.com/liuqjjin/ReCINR",
        "pinned_commit": (
            "9149d1d228db2e4eb3ae852a004f1d9e95ee0229"
        ),
        "pinned_tree": "61df3a42e83f3145892ca8bba0aadfc88dc38c08",
        "remote_sha256": {
            "README.md": (
                "ce118bfc1514ae9a15d688ab5c4333a"
                "73600b756370bd666593e140dffefdfa2"
            ),
            "pyproject.toml": (
                "30c8d4e440f9e6ba18b345e035cef7b9"
                "74ed858d3109364ab8eae99b235a8521"
            ),
            "src/recinr/model.py": (
                "9c0d85bbc7e634038c9e060e4458f1df"
                "5d9d72f21069d5a924915b570a08660a"
            ),
            "src/recinr/train.py": (
                "a9a066b454e6be1cbbf67cfe30b23117"
                "67560edd5e6d5c96ca842c0b838c399c"
            ),
        },
        "local_sha256": expected_local_hashes,
        "source_mapping": {
            "gsdiff/baselines/recinr_model.py": {
                "relation": (
                    "earlier-upstream-snapshot-after-removing-641-byte-"
                    "local-header-and-retaining-opening-triple-quote"
                ),
                "ancestor_commit": (
                    "847cca7cafded24ffc36522e92bc504090e48ab0"
                ),
                "ancestor_path": "src/recinr/model.py",
                "ancestor_sha256": (
                    "bf80fc0b2573839ef500c45511e6692c"
                    "5a669aef5fd4619974ddbb220b396047"
                ),
                "ancestor_bytes": 18537,
                "pinned_commit_byte_identical": False,
            },
            "gsdiff/baselines/recinr.py": {
                "relation": "local-adapter-no-one-to-one-remote-blob",
            },
            "gsdiff/baselines/inr.py": {
                "relation": "earlier-local-control-not-vendored",
            },
        },
        "github_license_info": None,
        "github_is_archived": False,
        "pinned_tree_entry_count": 285,
        "pinned_tree_truncated": False,
        "license_files_present": [],
        "license_file_names_checked": [
            "LICENSE",
            "LICENSE.md",
            "LICENSE.txt",
            "COPYING",
            "NOTICE",
        ],
        "repository_license_declarations": (
            "README-and-pyproject-declarations-only"
        ),
        "confirmed_redistribution_grant": False,
        "archive_status": "blocked-license-copyright-review",
    }
    assert actual_local_hashes == expected_local_hashes
    local_model = (
        REPO_ROOT / "gsdiff" / "baselines" / "recinr_model.py"
    ).read_bytes()
    ancestor_snapshot = local_model[:3] + local_model[644:]
    assert len(ancestor_snapshot) == 18537
    assert hashlib.sha256(ancestor_snapshot).hexdigest() == (
        provenance["source_mapping"]["gsdiff/baselines/recinr_model.py"][
            "ancestor_sha256"
        ]
    )
    assert "declarations only" in text
    assert "not a confirmed redistribution grant" in text
    assert "blocked-license-copyright-review" in text


def test_method_registry_document_binds_blind_contract_and_current_blockers():
    text, snapshot = _load_method_registry_document()
    registry = yaml.safe_load(
        (
            REPO_ROOT / "configs" / "protocols" / "methods-v1.yaml"
        ).read_text(encoding="utf-8")
    )
    pilot = yaml.safe_load(
        (
            REPO_ROOT / "configs" / "protocols" / "pilot-v1.yaml"
        ).read_text(encoding="utf-8")
    )
    diffusion = next(
        method
        for method in registry["methods"]
        if method["id"] == "gsdiff_diffusion"
    )

    assert snapshot["blind_contract"] == {
        "selection_formula_id": "heldout-normalized-l2-v1",
        "selection_formula": (
            "||pred-y||_2/max(||y||_2,1e-12); "
            "pred[k]=sum(P[k]*reconstruction[frame_indices[k]])"
        ),
        "selection_dtype": "float64",
        "selection_units": (
            "raw physical detector measurement units before ratio"
        ),
        "algorithm_seed_domain": "algorithm-seed-v1",
        "child_output_files": [
            "reconstruction.npz",
            "method-info.json",
        ],
        "final_output_files": [
            "reconstruction.npz",
            "metrics.json",
            "method-info.json",
            "stdout.log",
            "stderr.log",
        ],
        "boundary": "procedural-boundary-for-trusted-research-code",
        "os_sandbox": False,
        "native_extensions_covered": False,
        "direct_syscalls_covered": False,
    }
    assert snapshot["diffusion_checkpoint"] == {
        "logical_id": diffusion["checkpoints"][0]["logical_id"],
        "sha256": diffusion["checkpoints"][0]["sha256"],
        "provenance_status": (
            diffusion["checkpoints"][0]["provenance_status"]
        ),
        "publication_execution_ready": (
            diffusion["profiles"]["publication-v1"]["execution_ready"]
        ),
        "publication_blockers": (
            diffusion["profiles"]["publication-v1"][
                "execution_blockers"
            ]
        ),
    }
    assert snapshot["pilot_readiness"] == {
        "campaign": "configs/protocols/pilot-v1.yaml",
        "execution_ready": False,
        "blockers": _PILOT_READINESS_BLOCKERS,
    }
    assert pilot["execution_ready"] is False
    assert "not an OS sandbox" in text
    assert "native extensions" in text
    assert "direct system calls" in text
    assert "--legacy-compatibility" in text
    assert "never publication evidence" in text
