"""Immutable, content-addressed experiment manifest validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any

from jsonschema import Draft202012Validator

from .identity import (
    RunIdentity,
    build_run_identity,
    canonical_json_bytes,
    sha256_bytes,
    verify_environment_requirements,
)


_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REQUIREMENTS_LOCK = _ROOT / "requirements-lock.txt"
_DEFAULT_ENVIRONMENT_LOCK = _ROOT / "docs" / "reproducibility" / "environment-lock.json"
_MANIFEST_SCHEMA = json.loads(
    (_ROOT / "schemas" / "experiment-manifest-v1.schema.json").read_text("utf-8")
)
_AGGREGATE_SCHEMA = json.loads(
    (_ROOT / "schemas" / "experiment-aggregate-v1.schema.json").read_text("utf-8")
)
_MANIFEST_VALIDATOR = Draft202012Validator(_MANIFEST_SCHEMA)
_AGGREGATE_VALIDATOR = Draft202012Validator(_AGGREGATE_SCHEMA)
_SHA256 = __import__("re").compile(r"^[0-9a-f]{64}$", __import__("re").ASCII)
_RESERVED_WINDOWS_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{number}" for number in range(1, 10)}
    | {f"lpt{number}" for number in range(1, 10)}
)


def build_manifest(
    *,
    status: str,
    identity: RunIdentity,
    config_resolved: Mapping[str, object],
    inputs: Mapping[str, object],
    runtime: Mapping[str, object],
    execution: Mapping[str, object],
    measurement: Mapping[str, object],
    metrics: Mapping[str, object] | None,
    artifacts: Sequence[Mapping[str, object]],
    failure: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build a manifest exclusively from a constructed blind-run identity."""
    if not isinstance(identity, RunIdentity):
        raise TypeError("identity must be a RunIdentity")
    _require_exact_mapping(
        "inputs", inputs,
        {"measurements_file_sha256", "evaluation_truth_file_sha256", "dataset_manifest_sha256"},
    )
    _require_exact_mapping("runtime", runtime, {"python", "pytorch", "cuda", "gpu", "os"})
    _require_exact_mapping(
        "execution", execution,
        {"command", "started_at_utc", "ended_at_utc", "return_code", "runtime_seconds", "peak_vram_bytes"},
    )
    _require_exact_mapping(
        "measurement", measurement,
        {"train_count", "holdout_count", "pattern_family", "requested_snr_db", "noise_calibration_id", "noise_calibration_sha256", "noise_sigma_absolute", "realized_train_snr_db", "realized_holdout_snr_db"},
    )
    if status == "complete":
        if metrics is None:
            raise ValueError("complete manifests require metrics")
        _require_exact_mapping("metrics", metrics, {"version", "path", "sha256"})
    elif metrics is not None:
        _require_exact_mapping("metrics", metrics, {"version", "path", "sha256"})
    if type(config_resolved) is not dict:
        raise TypeError("config_resolved must be an exact dict")
    _validate_exact_json(config_resolved)
    if type(artifacts) is not list:
        raise TypeError("artifacts must be an exact list")
    for artifact in artifacts:
        _require_exact_mapping(
            "artifact", artifact,
            {"role", "path", "sha256", "size_bytes", "schema_version", "required"},
        )
    if status == "failed":
        if failure is None:
            raise ValueError("failed manifests require failure diagnostics")
        _require_exact_mapping("failure", failure, {"stderr_tail", "diagnostic_paths"})
    elif failure is not None:
        raise ValueError("only failed manifests may carry failure diagnostics")
    payload = identity.payload()
    manifest: dict[str, object] = {
        "schema_version": "experiment-manifest-v1",
        "status": status,
        "execution_class": payload["execution_class"],
        "metric_version": payload["metric_version"],
        "run_id": identity.run_id,
        "identity_sha256": identity.identity_sha256,
        "protocol": {
            "scientific_contract_id": payload["scientific_contract_id"],
            "scientific_contract_sha256": payload["scientific_contract_sha256"],
            "target": payload["target_id"],
            "motion": payload["motion_id"],
            "seed": payload["seed"],
            "method": payload["method_id"],
        },
        "code": {
            "git_commit": payload["code_commit"],
            "dirty_worktree": payload["dirty_worktree"],
            "source_tree_sha256": payload["source_tree_hash"],
        },
        "config": {
            "resolved": _copy_exact_json(config_resolved),
            "sha256": payload["config_sha256"],
        },
        "inputs": {
            "dataset_identity_sha256": payload["dataset_identity_sha256"],
            "measurements_file_sha256": inputs.get("measurements_file_sha256"),
            "evaluation_truth_file_sha256": inputs.get("evaluation_truth_file_sha256"),
            "dataset_manifest_sha256": inputs.get("dataset_manifest_sha256"),
            "assets": _plain_json(payload["assets_sha256"]),
            "checkpoints": _plain_json(payload["checkpoints_sha256"]),
        },
        "runtime": {
            "python": runtime.get("python"),
            "pytorch": runtime.get("pytorch"),
            "cuda": runtime.get("cuda"),
            "gpu": runtime.get("gpu"),
            "os": runtime.get("os"),
            "dependencies_sha256": payload["dependencies_sha256"],
            "environment_lock_sha256": payload["environment_lock_sha256"],
        },
        "execution": _copy_exact_json(execution),
        "measurement": _copy_exact_json(measurement),
        "metrics": _copy_exact_json(metrics),
        "artifacts": [_copy_exact_json(item) for item in artifacts],
    }
    if failure is not None:
        manifest["failure"] = _copy_exact_json(failure)
    validate_manifest(manifest, identity=identity)
    return manifest


def validate_manifest(value: Mapping[str, object], *, identity: RunIdentity | None = None) -> None:
    """Reject malformed, mutable, or identity-inconsistent manifest data."""
    _validate_exact_json(value)
    _raise_schema_errors(_MANIFEST_VALIDATOR, value, "manifest")
    manifest = value
    _validate_manifest_semantics(manifest, identity=identity)


def validate_aggregate_index(
    value: Mapping[str, object], *, manifests: Mapping[str, Mapping[str, object]] | None = None
) -> None:
    """Validate a campaign index without putting campaign data in run manifests."""
    _validate_exact_json(value)
    _raise_schema_errors(_AGGREGATE_VALIDATOR, value, "aggregate index")
    expected = value["expected_identity_sha256s"]
    records = value["run_manifests"]
    assert type(expected) is list and type(records) is list
    if expected != sorted(expected) or len(expected) != len(set(expected)):
        raise ValueError("expected identities must be sorted and unique")
    record_ids = [record["identity_sha256"] for record in records]
    if record_ids != sorted(record_ids) or len(record_ids) != len(set(record_ids)):
        raise ValueError("run manifest identities must be sorted and unique")
    if record_ids != expected:
        raise ValueError("campaign index coverage must exactly equal expected identities")
    if manifests is not None:
        for run_identity in record_ids:
            manifest = manifests.get(run_identity)
            if manifest is None:
                raise ValueError("campaign index is missing its referenced manifest")
            validate_manifest(manifest)
            record = next(item for item in records if item["identity_sha256"] == run_identity)
            if manifest["identity_sha256"] != run_identity:
                raise ValueError("campaign index identity does not match manifest identity")
            if record["manifest_sha256"] != sha256_bytes(canonical_json_bytes(manifest)):
                raise ValueError("campaign index manifest hash does not match canonical manifest")
            if manifest["status"] != "complete" or manifest["code"]["dirty_worktree"]:
                raise ValueError("campaign index may reference only clean complete manifests")
            protocol = manifest["protocol"]
            assert type(protocol) is dict
            if (
                protocol["scientific_contract_id"] != value["scientific_contract_id"]
                or protocol["scientific_contract_sha256"] != value["scientific_contract_sha256"]
                or manifest["metric_version"] != value["metric_version"]
            ):
                raise ValueError("campaign index protocol or metric version disagrees with manifest")


def build_aggregate_index(
    *,
    campaign_id: str,
    campaign_sha256: str,
    protocol_sha256: str,
    scientific_contract_id: str,
    scientific_contract_sha256: str,
    metric_version: str,
    expected_identity_sha256s: Sequence[str],
    manifests: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Build a campaign-local reference index for immutable clean manifests."""
    if type(expected_identity_sha256s) is not list:
        raise TypeError("expected identities must be an exact list")
    expected = list(expected_identity_sha256s)
    records: list[dict[str, str]] = []
    for run_identity in expected:
        manifest = manifests.get(run_identity)
        if manifest is None:
            raise ValueError("expected identity has no manifest")
        validate_manifest(manifest)
        if manifest["identity_sha256"] != run_identity:
            raise ValueError("manifest mapping key does not match manifest identity")
        if manifest["status"] != "complete" or manifest["code"]["dirty_worktree"]:
            raise ValueError("campaign index may reference only clean complete manifests")
        records.append(
            {
                "identity_sha256": run_identity,
                "manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
            }
        )
    index: dict[str, object] = {
        "schema_version": "experiment-aggregate-v1",
        "document_kind": "campaign-index",
        "campaign_id": campaign_id,
        "campaign_sha256": campaign_sha256,
        "protocol_sha256": protocol_sha256,
        "scientific_contract_id": scientific_contract_id,
        "scientific_contract_sha256": scientific_contract_sha256,
        "metric_version": metric_version,
        "expected_identity_sha256s": sorted(expected),
        "run_manifests": sorted(records, key=lambda item: item["identity_sha256"]),
    }
    validate_aggregate_index(index, manifests=manifests)
    return index


def load_complete_manifest(
    path: Path,
    *,
    expected_identity_sha256: str | None = None,
    requirements_lock: Path | None = None,
    environment_lock: Path | None = None,
    live_fingerprint: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    """Load a physical clean complete run directory, otherwise fail closed."""
    manifest_path = Path(path)
    value = _load_unique_json(manifest_path)
    if type(value) is not dict:
        raise ValueError("manifest JSON must be an object")
    validate_manifest(value)
    if value["status"] != "complete":
        return None
    if value["code"]["dirty_worktree"]:
        raise ValueError("dirty complete manifests are diagnostic and not reusable")
    if expected_identity_sha256 is not None and (
        type(expected_identity_sha256) is not str
        or _SHA256.fullmatch(expected_identity_sha256) is None
        or value["identity_sha256"] != expected_identity_sha256
    ):
        raise ValueError("manifest identity does not match expected identity")
    if manifest_path.parent.name != value["identity_sha256"]:
        raise ValueError("complete manifest directory must be the full identity SHA-256")
    if manifest_path.read_bytes() != canonical_json_bytes(value):
        raise ValueError("manifest file is not canonical JSON bytes")
    if requirements_lock is None:
        requirements_lock = _DEFAULT_REQUIREMENTS_LOCK
    if environment_lock is None:
        environment_lock = _DEFAULT_ENVIRONMENT_LOCK
    if (requirements_lock is None) != (environment_lock is None):
        raise ValueError("requirements_lock and environment_lock must be provided together")
    if requirements_lock is not None and environment_lock is not None:
        hashes = verify_environment_requirements(
            requirements_lock, environment_lock, live_fingerprint=live_fingerprint
        )
        runtime = value["runtime"]
        assert type(runtime) is dict
        if (
            runtime["dependencies_sha256"] != hashes["dependencies_sha256"]
            or runtime["environment_lock_sha256"] != hashes["environment_lock_sha256"]
        ):
            raise ValueError("manifest runtime lock hashes do not match the strict environment")
    _verify_complete_outputs(manifest_path, value)
    return value


def _validate_manifest_semantics(manifest: Mapping[str, object], *, identity: RunIdentity | None) -> None:
    config = manifest["config"]
    protocol = manifest["protocol"]
    code = manifest["code"]
    inputs = manifest["inputs"]
    runtime = manifest["runtime"]
    execution = manifest["execution"]
    measurement = manifest["measurement"]
    metrics = manifest["metrics"]
    artifacts = manifest["artifacts"]
    assert all(type(item) is dict for item in (config, protocol, code, inputs, runtime, execution, measurement))
    assert type(artifacts) is list
    if sha256_bytes(canonical_json_bytes(config["resolved"])) != config["sha256"]:
        raise ValueError("config.sha256 does not hash config.resolved")
    if measurement["noise_calibration_id"] != "detector-absolute-v1":
        raise ValueError("noise calibration must be detector-absolute-v1")
    if manifest["status"] == "complete":
        if execution["return_code"] != 0 or type(execution["ended_at_utc"]) is not str:
            raise ValueError("complete manifest requires a completed successful execution")
    elif manifest["status"] == "running":
        if execution["return_code"] is not None or execution["ended_at_utc"] is not None:
            raise ValueError("running manifest must not claim an execution result")
        if "failure" in manifest:
            raise ValueError("running manifest must not carry failure diagnostics")
    else:
        if (
            type(execution["return_code"]) is not int
            or execution["return_code"] == 0
            or type(execution["ended_at_utc"]) is not str
        ):
            raise ValueError("failed manifest requires a nonzero return code")
        failure = manifest.get("failure")
        if type(failure) is not dict:
            raise ValueError("failed manifest requires stderr_tail and diagnostic_paths")
        for path in failure["diagnostic_paths"]:
            _validate_relative_path(path)
    if manifest["status"] == "complete" and "failure" in manifest:
        raise ValueError("complete manifest must not carry failure diagnostics")
    if code["dirty_worktree"] and code["source_tree_sha256"] is None:
        raise ValueError("dirty manifest requires a deterministic source-tree hash")
    if not code["dirty_worktree"] and code["source_tree_sha256"] is not None:
        raise ValueError("clean manifest must not carry a source-tree hash")
    if manifest["status"] == "complete" and type(metrics) is not dict:
        raise ValueError("complete manifest requires metrics")
    metric_paths = [] if metrics is None else [metrics["path"]]
    for path in [*metric_paths, *(artifact["path"] for artifact in artifacts)]:
        _validate_relative_path(path)
    paths = [*metric_paths, *(artifact["path"] for artifact in artifacts)]
    folded = [path.casefold() for path in paths]
    if len(folded) != len(set(folded)):
        raise ValueError("metric and artifact paths must not collide case-insensitively")
    for artifact in artifacts:
        if artifact["size_bytes"] < 0:
            raise ValueError("artifact size must be nonnegative")
    expected = identity or _identity_from_manifest(manifest)
    if manifest["identity_sha256"] != expected.identity_sha256 or manifest["run_id"] != expected.run_id:
        raise ValueError("manifest identity fields do not match identity-bearing content")
    if identity is not None and _identity_from_manifest(manifest).identity_sha256 != identity.identity_sha256:
        raise ValueError("manifest fields do not match the supplied identity")
    if manifest["metric_version"] != expected.payload()["metric_version"]:
        raise ValueError("metrics version does not match identity")
    if metrics is not None and metrics["version"] != manifest["metric_version"]:
        raise ValueError("metrics version does not match manifest metric version")


def _identity_from_manifest(manifest: Mapping[str, object]) -> RunIdentity:
    protocol = manifest["protocol"]
    code = manifest["code"]
    config = manifest["config"]
    inputs = manifest["inputs"]
    runtime = manifest["runtime"]
    metrics = manifest["metrics"]
    assert all(type(item) is dict for item in (protocol, code, config, inputs, runtime))
    return build_run_identity(
        execution_class=manifest["execution_class"],
        scientific_contract_id=protocol["scientific_contract_id"],
        scientific_contract_sha256=protocol["scientific_contract_sha256"],
        method_id=protocol["method"],
        target_id=protocol["target"],
        motion_id=protocol["motion"],
        seed=protocol["seed"],
        config_sha256=config["sha256"],
        dataset_identity_sha256=inputs["dataset_identity_sha256"],
        assets_sha256=inputs["assets"],
        checkpoints_sha256=inputs["checkpoints"],
        code_commit=code["git_commit"],
        dirty_worktree=code["dirty_worktree"],
        source_tree_hash=code["source_tree_sha256"],
        dependencies_sha256=runtime["dependencies_sha256"],
        environment_lock_sha256=runtime["environment_lock_sha256"],
        metric_version=manifest["metric_version"],
    )


def _validate_exact_json(value: object, path: str = "$") -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _validate_exact_json(child, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} has a non-string object key")
            _validate_exact_json(child, f"{path}.{key}")
        return
    raise TypeError(f"{path} uses unsupported JSON type {type(value).__name__}")


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain_json(child) for child in value]
    return value


def _copy_exact_json(value: object) -> object:
    _validate_exact_json(value)
    if type(value) is dict:
        return {key: _copy_exact_json(child) for key, child in value.items()}
    if type(value) is list:
        return [_copy_exact_json(child) for child in value]
    return value


def _require_exact_mapping(name: str, value: object, keys: set[str]) -> None:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an exact dict")
    actual = set(value)
    if actual != keys:
        raise ValueError(
            f"{name} keys do not match: missing={sorted(keys - actual)}, unknown={sorted(actual - keys)}"
        )
    _validate_exact_json(value)


def _raise_schema_errors(validator: Draft202012Validator, value: object, noun: str) -> None:
    errors = sorted(validator.iter_errors(value), key=str)
    if errors:
        raise ValueError(f"invalid {noun}: " + "; ".join(error.message for error in errors))


def _validate_relative_path(value: object) -> None:
    if type(value) is not str or not value or "\x00" in value or "\\" in value:
        raise ValueError("output path must be a nonempty POSIX-relative path")
    if value.startswith(("/", "//")) or len(value) >= 2 and value[1] == ":":
        raise ValueError("output path must not be absolute, UNC, or drive-qualified")
    raw_parts = value.split("/")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in raw_parts):
        raise ValueError("output path contains an unsafe component")
    for part in path.parts:
        stem = part.split(".", 1)[0].casefold()
        if ":" in part or part.rstrip(". ") != part or stem in _RESERVED_WINDOWS_NAMES:
            raise ValueError("output path contains a Windows-unsafe component")


def _load_unique_json(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except OSError as error:
        raise ValueError(f"cannot read manifest: {path}") from error
    try:
        return json.loads(text, object_pairs_hook=_unique_object)
    except json.JSONDecodeError as error:
        raise ValueError("manifest is not valid JSON") from error


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _verify_complete_outputs(manifest_path: Path, manifest: Mapping[str, object]) -> None:
    root = manifest_path.parent
    for ancestor in (root, *root.parents):
        _reject_linked_path(ancestor)
    metrics = manifest["metrics"]
    artifacts = manifest["artifacts"]
    assert type(metrics) is dict and type(artifacts) is list
    expected: dict[str, tuple[str, int | None]] = {
        metrics["path"]: (metrics["sha256"], None)
    }
    for artifact in artifacts:
        expected[artifact["path"]] = (artifact["sha256"], artifact["size_bytes"])
    found: set[str] = set()
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        _reject_linked_path(current_path)
        for directory in directories:
            _reject_linked_path(current_path / directory)
        for filename in files:
            path = current_path / filename
            relative = path.relative_to(root).as_posix()
            if relative == manifest_path.name:
                continue
            _reject_linked_path(path)
            found.add(relative)
            if relative not in expected:
                raise ValueError(f"unlisted output file: {relative}")
            expected_hash, expected_size = expected[relative]
            actual_hash, actual_size = _hash_regular_file(path)
            if actual_hash != expected_hash or expected_size is not None and actual_size != expected_size:
                raise ValueError(f"output hash or size mismatch: {relative}")
    if found != set(expected):
        raise ValueError("complete manifest is missing a declared output")


def _reject_linked_path(path: Path) -> None:
    try:
        info = os.lstat(path)
    except OSError as error:
        raise ValueError(f"cannot stat output path: {path}") from error
    attributes = getattr(info, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if stat.S_ISLNK(info.st_mode) or attributes & reparse:
        raise ValueError(f"output path contains a symlink or reparse point: {path}")


def _hash_regular_file(path: Path) -> tuple[str, int]:
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"output is not a regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
        after = os.fstat(stream.fileno())
    if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size) or size != before.st_size:
        raise ValueError(f"output changed while being verified: {path}")
    return digest.hexdigest(), size
