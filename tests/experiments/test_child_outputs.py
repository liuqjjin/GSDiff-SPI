from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Callable

from jsonschema import Draft202012Validator
import numpy as np
import pytest

from gsdiff.data._artifact_identity import array_descriptor, canonical_json_bytes
from gsdiff.data._artifact_io import (
    METADATA_MEMBER,
    atomic_write_bytes,
    decode_metadata,
    load_array_member,
    read_npz_members,
    write_npz,
)
from gsdiff.data._artifact_models import SPIAcquisitionData
import gsdiff.experiments.child_outputs as child_outputs
from gsdiff.experiments.child_outputs import (
    MethodChildResult,
    load_reconstruction_v2,
    validate_method_child_outputs_v2,
    write_method_child_outputs_v2,
)
from gsdiff.experiments.methods import (
    AlgorithmSeed,
    ResolvedMethod,
    derive_algorithm_seed,
    resolve_method_semantics,
)


MEASUREMENTS_SHA256 = "a" * 64


def acquisition(
    *,
    dataset_identity_sha256: str = "b" * 64,
    time_grid: np.ndarray | None = None,
) -> SPIAcquisitionData:
    arrays = {
        "patterns": np.ones((4, 32, 32), dtype=np.float32),
        "measurements": np.ones(4, dtype=np.float32),
        "frame_indices": np.arange(4, dtype=np.int64),
        "time_grid": (
            np.arange(4, dtype=np.float64)
            if time_grid is None
            else np.asarray(time_grid)
        ),
    }
    return SPIAcquisitionData(
        dataset_identity_sha256=dataset_identity_sha256,
        **arrays,
        holdout_patterns=None,
        holdout_measurements=None,
        holdout_frame_indices=None,
        H=32,
        W=32,
        T=4,
        K=4,
        holdout_K=0,
        acquisition={
            "pattern_family": "bernoulli",
            "pattern_values": [0, 1],
            "pattern_order": "sequential",
            "time_assignment": "uniform",
            "holdout_pattern_family": "bernoulli",
            "noise_convention": "absolute-gaussian-sigma",
            "noise_sigma_absolute": 1.0,
        },
        array_descriptors={
            name: array_descriptor(value) for name, value in arrays.items()
        },
    )


def resolved_method(
    method_id: str = "dgi",
    *,
    profile: str = "publication-v1",
) -> ResolvedMethod:
    config_id = "default" if profile == "publication-v1" else "smoke-default-v1"
    base_config = (
        {"gaussian_count": 1000}
        if method_id in {"gsdiff_tv", "gsdiff_diffusion"}
        else {}
    )
    return resolve_method_semantics(
        method_id,
        method_config_id=config_id,
        base_config=base_config,
        measurements_metadata={},
        execution_profile=profile,
    )


def algorithm_seed(
    method: ResolvedMethod,
    data: SPIAcquisitionData,
) -> AlgorithmSeed:
    return derive_algorithm_seed(
        cell_seed=1,
        dataset_identity_sha256=data.dataset_identity_sha256,
        method_id=method.method_id,
        method_config_sha256=method.method_config_sha256,
    )


def dgi_result(
    *,
    dgi: np.ndarray | None = None,
    motion: np.ndarray | None = None,
    history: tuple[dict[str, object], ...] = (),
    info_mutation: Callable[[dict[str, object]], None] | None = None,
) -> MethodChildResult:
    info: dict[str, object] = {
        "parameter_count": 0,
        "native_iteration_unit": "pass",
        "native_iteration_budget": 1,
        "convergence_status": "not-applicable",
        "selected_hyperparameters": None,
        "selection": None,
        "checkpoint_hashes": [],
    }
    if motion is not None:
        info["native_motion_model"] = "native-trajectory"
    if info_mutation is not None:
        info_mutation(info)
    return MethodChildResult(
        method_id="dgi",
        reconstruction=np.ones((4, 32, 32), dtype=np.float32),
        estimated_motion_trajectory=motion,
        dgi=dgi,
        info=info,
        history=history,
    )


def static_result(
    *,
    history: tuple[dict[str, object], ...],
    selected_lambda: float = 0.001,
) -> MethodChildResult:
    candidate_grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    rows = [
        {
            "candidate": candidate,
            "formula_id": "heldout-normalized-l2-v1",
            "numerator": float(index + 1),
            "denominator": 1.0,
            "value": float(index + 1),
        }
        for index, candidate in enumerate(candidate_grid)
    ]
    if selected_lambda in candidate_grid:
        rows[candidate_grid.index(selected_lambda)]["value"] = 0.0
        rows[candidate_grid.index(selected_lambda)]["numerator"] = 0.0
    return MethodChildResult(
        method_id="static_cs",
        reconstruction=np.ones((4, 32, 32), dtype=np.float32),
        estimated_motion_trajectory=None,
        dgi=None,
        info={
            "parameter_count": 32 * 32,
            "native_iteration_unit": "admm-iteration",
            "native_iteration_budget": 150,
            "convergence_status": "convergence-required",
            "selected_hyperparameters": {"lambda": selected_lambda},
            "selection": {
                "formula_id": "heldout-normalized-l2-v1",
                "candidate_grid": candidate_grid,
                "selected_candidate": selected_lambda,
                "rows": rows,
            },
            "checkpoint_hashes": [],
        },
        history=history,
    )


def history_row(index: int) -> dict[str, object]:
    return {"kind": "iteration", "iteration": index}


def write_valid(
    output_dir: Path,
    *,
    result: MethodChildResult | None = None,
    method: ResolvedMethod | None = None,
    data: SPIAcquisitionData | None = None,
    started: str = "2026-07-28T00:00:00Z",
    finished: str = "2026-07-28T00:00:01Z",
) -> tuple[ResolvedMethod, SPIAcquisitionData, AlgorithmSeed, dict[str, str]]:
    method = resolved_method() if method is None else method
    data = acquisition() if data is None else data
    seed = algorithm_seed(method, data)
    hashes = dict(
        write_method_child_outputs_v2(
            output_dir,
            method=method,
            acquisition=data,
            measurements_file_sha256=MEASUREMENTS_SHA256,
            algorithm_seed=seed,
            result=dgi_result() if result is None else result,
            child_started_at_utc=started,
            child_finished_at_utc=finished,
        )
    )
    return method, data, seed, hashes


def validate_valid(
    output_dir: Path,
    method: ResolvedMethod,
    data: SPIAcquisitionData,
    seed: AlgorithmSeed,
    *,
    identity: str | None = None,
    measurements_sha256: str = MEASUREMENTS_SHA256,
) -> dict[str, str]:
    return dict(
        validate_method_child_outputs_v2(
            output_dir,
            expected_method=method,
            expected_acquisition=data,
            expected_dataset_identity_sha256=(
                data.dataset_identity_sha256 if identity is None else identity
            ),
            expected_measurements_file_sha256=measurements_sha256,
            expected_algorithm_seed=seed,
        )
    )


def load_info(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def rewrite_info(
    path: Path,
    mutation: Callable[[dict[str, object]], None],
) -> None:
    info = load_info(path)
    mutation(info)
    atomic_write_bytes(path, canonical_json_bytes(info))


@pytest.mark.parametrize(("dgi_present", "motion_present"), [
    (False, False),
    (False, True),
    (True, False),
    (True, True),
])
def test_v2_writer_owns_two_files_for_all_optional_array_combinations(
    tmp_path: Path,
    dgi_present: bool,
    motion_present: bool,
) -> None:
    result = dgi_result(
        dgi=np.ones((32, 32), dtype=np.float32) if dgi_present else None,
        motion=np.ones((4, 3), dtype=np.float32) if motion_present else None,
    )
    method, data, seed, hashes = write_valid(tmp_path, result=result)
    assert set(hashes) == {"reconstruction.npz", "method-info.json"}
    assert {path.name for path in tmp_path.iterdir()} == set(hashes)
    loaded = load_reconstruction_v2(tmp_path / "reconstruction.npz")
    assert (loaded.dgi is not None) is dgi_present
    assert (loaded.estimated_motion_trajectory is not None) is motion_present
    assert validate_valid(tmp_path, method, data, seed) == hashes


@pytest.mark.parametrize("name", ["metrics.json", "stdout.log", "stderr.log"])
def test_v2_writer_rejects_precreated_parent_owned_files(
    tmp_path: Path,
    name: str,
) -> None:
    (tmp_path / name).write_text("{}", encoding="utf-8")
    data, method = acquisition(), resolved_method()
    with pytest.raises(ValueError, match="isolated"):
        write_method_child_outputs_v2(
            tmp_path,
            method=method,
            acquisition=data,
            measurements_file_sha256=MEASUREMENTS_SHA256,
            algorithm_seed=algorithm_seed(method, data),
            result=dgi_result(),
            child_started_at_utc="2026-07-28T00:00:00Z",
            child_finished_at_utc="2026-07-28T00:00:01Z",
        )


@pytest.mark.parametrize(
    ("target", "field"),
    [
        ("selected_hyperparameters", "psnr"),
        ("selection", "gt_frames"),
        ("selection", "promotion_eligible"),
        ("history", "ssim"),
        ("history", "truth_path"),
    ],
)
def test_writer_rejects_unapproved_nested_child_fields(
    tmp_path: Path,
    target: str,
    field: str,
) -> None:
    if target == "history":
        result = dgi_result(history=({field: 1.0},))
    else:
        result = dgi_result(
            info_mutation=lambda info: info.__setitem__(
                target, {field: 1.0}
            )
        )
    with pytest.raises(ValueError):
        write_valid(tmp_path, result=result)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("location", "field"),
    [
        ("semantic_config", "truth_path"),
        ("selected_hyperparameters", "metrics"),
        ("selection", "promotion_eligible"),
        ("history", "evaluator_score"),
    ],
)
def test_schema_rejects_nested_extra_fields(
    tmp_path: Path,
    location: str,
    field: str,
) -> None:
    _, _, _, _ = write_valid(tmp_path)
    info = load_info(tmp_path / "method-info.json")
    if location == "history":
        info["convergence"]["history"] = [{field: 1.0}]  # type: ignore[index]
        info["convergence"]["observed_count"] = 1  # type: ignore[index]
        info["convergence"]["serialized_count"] = 1  # type: ignore[index]
    else:
        info[location] = {field: 1.0}
    schema = json.loads(
        Path("schemas/method-info-v2.schema.json").read_text(encoding="utf-8")
    )
    assert list(Draft202012Validator(schema).iter_errors(info))


@pytest.mark.parametrize(
    ("observed_count", "serialized_count", "expected_indices"),
    [
        (0, 0, []),
        (20, 20, list(range(20))),
        (21, 21, list(range(21))),
        (22, 21, list(range(20)) + [21]),
    ],
)
def test_history_sampling_is_locked(
    tmp_path: Path,
    observed_count: int,
    serialized_count: int,
    expected_indices: list[int],
) -> None:
    history = tuple(history_row(index) for index in range(observed_count))
    write_valid(tmp_path, result=dgi_result(history=history))
    convergence = load_info(tmp_path / "method-info.json")["convergence"]
    assert convergence["observed_count"] == observed_count
    assert convergence["serialized_count"] == serialized_count
    assert [row["iteration"] for row in convergence["history"]] == expected_indices
    expected_policy = (
        "all-observations"
        if observed_count < 21
        else "floor(i*(observed_count-1)/20), i=0..20"
    )
    assert convergence["sampling_policy"] == expected_policy


def test_validator_rejects_invented_sampling_policy_and_impossible_counts(
    tmp_path: Path,
) -> None:
    method, data, seed, _ = write_valid(
        tmp_path,
        result=dgi_result(
            history=tuple(history_row(index) for index in range(22))
        ),
    )

    def mutate(info: dict[str, object]) -> None:
        convergence = info["convergence"]
        convergence["sampling_policy"] = "invented"  # type: ignore[index]
        convergence["serialized_count"] = 1  # type: ignore[index]
        convergence["history"] = [history_row(0)]  # type: ignore[index]

    rewrite_info(tmp_path / "method-info.json", mutate)
    with pytest.raises(ValueError, match="convergence|sampling|invented"):
        validate_valid(tmp_path, method, data, seed)


def test_convergence_required_prevalidation_leaves_no_stale_file(
    tmp_path: Path,
) -> None:
    method = resolved_method("static_cs")
    data = acquisition()
    with pytest.raises(ValueError, match="21"):
        write_method_child_outputs_v2(
            tmp_path,
            method=method,
            acquisition=data,
            measurements_file_sha256=MEASUREMENTS_SHA256,
            algorithm_seed=algorithm_seed(method, data),
            result=static_result(history=()),
            child_started_at_utc="2026-07-28T00:00:00Z",
            child_finished_at_utc="2026-07-28T00:00:01Z",
        )
    assert list(tmp_path.iterdir()) == []


def test_convergence_required_selection_positive_control(
    tmp_path: Path,
) -> None:
    method = resolved_method("static_cs")
    data = acquisition()
    history = tuple(history_row(index) for index in range(21))
    _, _, seed, hashes = write_valid(
        tmp_path,
        method=method,
        data=data,
        result=static_result(history=history),
    )
    assert validate_valid(tmp_path, method, data, seed) == hashes


def test_closed_semantic_schema_accepts_every_resolved_method_profile(
    tmp_path: Path,
) -> None:
    write_valid(tmp_path)
    base = load_info(tmp_path / "method-info.json")
    schema = json.loads(
        Path("schemas/method-info-v2.schema.json").read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(schema)
    for method_id in (
        "dgi", "static_cs", "perframe_cs", "tv3d", "monin", "gidc3dtv",
        "recinr", "siren", "recinr_se2", "gsdiff_tv",
        "gsdiff_diffusion",
    ):
        for profile in ("publication-v1", "controller-cpu-smoke-v1"):
            method = resolved_method(method_id, profile=profile)
            candidate = {
                **base,
                "semantic_config": json.loads(
                    canonical_json_bytes(method.semantic_config)
                ),
            }
            assert not list(validator.iter_errors(candidate)), (
                method_id,
                profile,
            )


def test_locked_dgi_native_metadata_is_not_child_controlled(
    tmp_path: Path,
) -> None:
    result = dgi_result(
        info_mutation=lambda info: info.update(
            parameter_count=999,
            native_iteration_budget=0,
        )
    )
    with pytest.raises(ValueError, match="parameter_count|native"):
        write_valid(tmp_path, result=result)
    assert list(tmp_path.iterdir()) == []


def test_selected_hyperparameter_must_come_from_locked_grid(
    tmp_path: Path,
) -> None:
    method = resolved_method("static_cs")
    data = acquisition()
    history = tuple(history_row(index) for index in range(21))
    with pytest.raises(ValueError, match="hyperparameter|lambda"):
        write_method_child_outputs_v2(
            tmp_path,
            method=method,
            acquisition=data,
            measurements_file_sha256=MEASUREMENTS_SHA256,
            algorithm_seed=algorithm_seed(method, data),
            result=static_result(history=history, selected_lambda=99.0),
            child_started_at_utc="2026-07-28T00:00:00Z",
            child_finished_at_utc="2026-07-28T00:00:01Z",
        )


@pytest.mark.parametrize("second_dimension", [1, 2, 4])
def test_motion_trajectory_requires_exact_t_by_three_shape(
    tmp_path: Path,
    second_dimension: int,
) -> None:
    with pytest.raises(ValueError, match="estimated_motion_trajectory"):
        write_valid(
            tmp_path,
            result=dgi_result(
                motion=np.ones((4, second_dimension), dtype=np.float32)
            ),
        )
    assert list(tmp_path.iterdir()) == []


def test_validator_binds_arrays_to_typed_expected_acquisition(
    tmp_path: Path,
) -> None:
    method, data, seed, _ = write_valid(tmp_path)
    mismatched = acquisition(time_grid=np.array([10.0, 20.0, 30.0, 40.0]))
    with pytest.raises(ValueError, match="time_grid|acquisition"):
        validate_valid(tmp_path, method, mismatched, seed)
    with pytest.raises(TypeError, match="acquisition"):
        validate_method_child_outputs_v2(
            tmp_path,
            expected_method=method,
            expected_acquisition=object(),  # type: ignore[arg-type]
            expected_dataset_identity_sha256=data.dataset_identity_sha256,
            expected_measurements_file_sha256=MEASUREMENTS_SHA256,
            expected_algorithm_seed=seed,
        )


@pytest.mark.parametrize("spoof", ["semantic-bool", "seed-float"])
def test_parent_crosslocks_are_type_exact(tmp_path: Path, spoof: str) -> None:
    method, data, seed, _ = write_valid(tmp_path)

    def mutate(info: dict[str, object]) -> None:
        if spoof == "semantic-bool":
            info["semantic_config"]["native_budget"] = True  # type: ignore[index]
        else:
            info["algorithm_seed"]["seed_u32"] = float(seed.seed_u32)  # type: ignore[index]

    rewrite_info(tmp_path / "method-info.json", mutate)
    with pytest.raises(
        ValueError,
        match="semantic_config|algorithm_seed|integer|schema validation",
    ):
        validate_valid(tmp_path, method, data, seed)


@pytest.mark.parametrize(
    ("field", "spoof"),
    [
        ("parameter_count", 0.0),
        ("native_iteration.budget", 1.0),
        ("warmup.splitting", 0.0),
        ("convergence.observed_count", 0.0),
        ("convergence.serialized_count", 0.0),
    ],
)
def test_all_count_fields_require_exact_json_integers(
    tmp_path: Path,
    field: str,
    spoof: float,
) -> None:
    method, data, seed, _ = write_valid(tmp_path)

    def mutate(info: dict[str, object]) -> None:
        current: object = info
        parts = field.split(".")
        for part in parts[:-1]:
            current = current[part]  # type: ignore[index]
        current[parts[-1]] = spoof  # type: ignore[index]

    rewrite_info(tmp_path / "method-info.json", mutate)
    with pytest.raises(
        ValueError,
        match="integer|schema validation|metadata|warmup",
    ):
        validate_valid(tmp_path, method, data, seed)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_duplicate_and_nonfinite_json_are_rejected(
    tmp_path: Path,
    constant: str,
) -> None:
    method, data, seed, _ = write_valid(tmp_path)
    path = tmp_path / "method-info.json"
    raw = path.read_bytes()
    path.write_bytes(raw.replace(b"{", b'{"schema":"method-info-v2",', 1))
    with pytest.raises(ValueError, match="duplicate"):
        validate_valid(tmp_path, method, data, seed)
    atomic_write_bytes(
        path,
        raw.replace(
            b'"elapsed_seconds":1.0',
            f'"elapsed_seconds":{constant}'.encode("ascii"),
        ),
    )
    with pytest.raises(ValueError, match="non-finite|finite"):
        validate_valid(tmp_path, method, data, seed)


def test_type_spoofed_seed_and_acquisition_scalars_are_rejected(
    tmp_path: Path,
) -> None:
    method, data, seed, _ = write_valid(tmp_path)
    with pytest.raises((TypeError, ValueError), match="seed_u32|integer"):
        validate_method_child_outputs_v2(
            tmp_path,
            expected_method=method,
            expected_acquisition=data,
            expected_dataset_identity_sha256=data.dataset_identity_sha256,
            expected_measurements_file_sha256=MEASUREMENTS_SHA256,
            expected_algorithm_seed=AlgorithmSeed(
                derivation_sha256=seed.derivation_sha256,
                seed_u32=True,  # type: ignore[arg-type]
            ),
        )
    with pytest.raises((TypeError, ValueError), match="expected_acquisition.T"):
        validate_method_child_outputs_v2(
            tmp_path,
            expected_method=method,
            expected_acquisition=replace(data, T=True),  # type: ignore[arg-type]
            expected_dataset_identity_sha256=data.dataset_identity_sha256,
            expected_measurements_file_sha256=MEASUREMENTS_SHA256,
            expected_algorithm_seed=seed,
        )


def test_extra_inventory_and_interrupted_output_are_rejected(
    tmp_path: Path,
) -> None:
    method, data, seed, _ = write_valid(tmp_path)
    (tmp_path / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly two"):
        validate_valid(tmp_path, method, data, seed)
    (tmp_path / "extra.txt").unlink()
    (tmp_path / "method-info.json").unlink()
    with pytest.raises(ValueError, match="exactly two"):
        validate_valid(tmp_path, method, data, seed)


def test_directory_inventory_is_reverified_before_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method, data, seed, _ = write_valid(tmp_path)
    original = child_outputs.verify_directory_inventory

    def race(inventory: object) -> None:
        (tmp_path / "late-extra.txt").write_text("late", encoding="utf-8")
        original(inventory)

    monkeypatch.setattr(child_outputs, "verify_directory_inventory", race)
    with pytest.raises(ValueError, match="inventory"):
        validate_valid(tmp_path, method, data, seed)


@pytest.mark.parametrize(
    ("started", "finished", "accepted"),
    [
        ("2026-07-28T00:00:00.123456Z", "2026-07-28T00:00:01.123456Z", True),
        ("2026-07-28T00:00:00.1234567Z", "2026-07-28T00:00:01.1234567Z", False),
    ],
)
def test_rfc3339_fractional_precision_is_explicit(
    tmp_path: Path,
    started: str,
    finished: str,
    accepted: bool,
) -> None:
    if accepted:
        write_valid(tmp_path, started=started, finished=finished)
        assert (
            load_info(tmp_path / "method-info.json")["child_timing"][
                "elapsed_seconds"
            ]
            == 1.0
        )
    else:
        with pytest.raises(ValueError, match="RFC 3339"):
            write_valid(tmp_path, started=started, finished=finished)
        assert list(tmp_path.iterdir()) == []


def test_timing_hash_and_parent_identity_tampering_are_rejected(
    tmp_path: Path,
) -> None:
    method, data, seed, _ = write_valid(tmp_path)
    rewrite_info(
        tmp_path / "method-info.json",
        lambda info: info["child_timing"].__setitem__("elapsed_seconds", 9.0),  # type: ignore[union-attr]
    )
    with pytest.raises(ValueError, match="timing"):
        validate_valid(tmp_path, method, data, seed)
    rewrite_info(
        tmp_path / "method-info.json",
        lambda info: info["child_timing"].__setitem__("elapsed_seconds", 1.0),  # type: ignore[union-attr]
    )
    with pytest.raises(ValueError, match="dataset|identity"):
        validate_valid(tmp_path, method, data, seed, identity="c" * 64)
    with pytest.raises(ValueError, match="measurements"):
        validate_valid(
            tmp_path,
            method,
            data,
            seed,
            measurements_sha256="d" * 64,
        )


@pytest.mark.parametrize("mutation", ["descriptor", "member"])
def test_npz_descriptor_and_member_tampering_are_rejected(
    tmp_path: Path,
    mutation: str,
) -> None:
    method, data, seed, _ = write_valid(tmp_path)
    reconstruction_path = tmp_path / "reconstruction.npz"
    members = read_npz_members(reconstruction_path)
    metadata = dict(decode_metadata(members))
    names = {
        name.removesuffix(".npy")
        for name in members
        if name != METADATA_MEMBER
    }
    arrays = {name: load_array_member(members, name) for name in names}
    if mutation == "descriptor":
        descriptors = {
            name: dict(descriptor)
            for name, descriptor in metadata["array_descriptors"].items()
        }
        descriptors["reconstruction"]["sha256"] = "0" * 64
        metadata["array_descriptors"] = descriptors
    else:
        arrays["extra"] = np.ones(1)
        metadata["array_descriptors"] = {
            **metadata["array_descriptors"],
            "extra": array_descriptor(arrays["extra"]),
        }
    write_npz(reconstruction_path, arrays=arrays, metadata=metadata)
    with pytest.raises(ValueError, match="descriptor|member|ZIP|content-hash"):
        validate_valid(tmp_path, method, data, seed)


def test_method_info_reconstruction_hash_is_crosschecked(tmp_path: Path) -> None:
    method, data, seed, _ = write_valid(tmp_path)
    rewrite_info(
        tmp_path / "method-info.json",
        lambda info: info["reconstruction"].__setitem__("sha256", "0" * 64),  # type: ignore[union-attr]
    )
    with pytest.raises(ValueError, match="hash"):
        validate_valid(tmp_path, method, data, seed)


def test_linked_child_file_is_rejected_when_supported(tmp_path: Path) -> None:
    method, data, seed, _ = write_valid(tmp_path)
    real = tmp_path / "real-info.json"
    info_path = tmp_path / "method-info.json"
    info_path.replace(real)
    try:
        os.symlink(real, info_path)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"file symlinks unavailable: {error}")
    with pytest.raises(ValueError, match="linked|reparse"):
        validate_valid(tmp_path, method, data, seed)
