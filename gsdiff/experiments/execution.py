"""Materialize one blind method request into an isolated, audited stage.

The semantic request is deliberately path-free.  Absolute runtime paths live
only in ``MaterializedMethodExecution`` and in the ``runtime`` portion of its
materialization record, so moving a stage cannot change method identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
import ntpath
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import stat
import sys
from types import MappingProxyType

from gsdiff.data._artifact_dataset import _validate_blind_acquisition_spec
from gsdiff.data._artifact_identity import (
    ArtifactValidationError,
    validate_exact_json_native,
)

from .identity import canonical_json_bytes
from .methods import (
    AlgorithmSeed,
    CheckpointRequirement,
    ResolvedMethod,
    canonical_method_id,
    thaw_json,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z", flags=re.ASCII)
_DEVICE = re.compile(r"cuda:([0-9]+)\Z", flags=re.ASCII)
_CHECKPOINT_ID = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z",
    flags=re.ASCII,
)
_CHECKPOINT_TOKEN = re.compile(
    r"\$\{CHECKPOINT:([A-Za-z0-9][A-Za-z0-9._-]{0,127})\}\Z",
    flags=re.ASCII,
)
_CHECKPOINT_ASSIGNMENT = re.compile(
    r"(?P<logical_id>[A-Za-z0-9][A-Za-z0-9._-]{0,127})="
    r"\$\{CHECKPOINT:(?P=logical_id)\}\Z",
    flags=re.ASCII,
)
_EMBEDDED_WINDOWS_DRIVE_PATH = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]",
    flags=re.ASCII,
)
_EMBEDDED_UNC_PATH = re.compile(
    r"(?<![A-Za-z0-9])\\\\"
    r"(?:[?.]\\|[^\\/\s]+\\[^\\/\s]+)",
    flags=re.ASCII,
)
_EMBEDDED_POSIX_PATH = re.compile(
    r"(?<![A-Za-z0-9_/.])/(?!/)(?=\S)",
    flags=re.ASCII,
)
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_SOURCE_EXCLUDED_DIRECTORIES = frozenset(
    {
        "__pycache__",
        ".git",
        ".claude",
        ".superpowers",
        "results",
        "checkpoints",
    }
)
_SOURCE_EXCLUDED_PREFIXES = (("gsdiff", "evaluation"),)
_SOURCE_EXCLUDED_FILES = frozenset(
    {
        ("gsdiff", "baselines", "_evaluation.py"),
        ("gsdiff", "data", "_artifact_truth.py"),
    }
)
_ENTRYPOINT_BY_FAMILY = {
    "baseline": "scripts/run_baselines.py",
    "gsdiff": "train.py",
}
_PLAIN_TOKENS = frozenset(
    {
        "${PYTHON}",
        "${MEASUREMENTS_PATH}",
        "${OUTPUT_DIR}",
        "${METHOD_CONFIG_PATH}",
        "${DATASET_IDENTITY_SHA256}",
        "${ALGORITHM_SEED}",
        "${DEVICE}",
        "${AUDIT_LOG_PATH}",
    }
)


@dataclass(frozen=True)
class MaterializedMethodExecution:
    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str]
    measurements_path: Path
    method_config_path: Path
    child_output_dir: Path
    expected_acquisition_spec: Mapping[str, object]
    audit_log_path: Path
    stdout_path: Path
    stderr_path: Path
    read_allowlist: tuple[Path, ...]
    read_root_allowlist: tuple[Path, ...]
    write_root_allowlist: tuple[Path, ...]
    requested_runtime_device: str
    child_runtime_device: str
    audit_policy_path: Path
    audit_policy_sha256: str
    materialization_record: Mapping[str, object]


@dataclass(frozen=True)
class MaterializedMethodRequest:
    method: ResolvedMethod
    algorithm_seed: AlgorithmSeed
    dataset_identity_sha256: str
    measurements_file_sha256: str
    expected_acquisition_spec: Mapping[str, object]
    measurements_path: Path
    child_output_dir: Path
    checkpoint_paths: Mapping[str, Path]
    requested_runtime_device: str
    child_runtime_device: str


@dataclass(frozen=True)
class _DirectoryIdentity:
    path: Path
    device: int
    inode: int


@dataclass(frozen=True)
class _StageRootCandidate:
    path: Path
    identity: _DirectoryIdentity | None


@dataclass(frozen=True)
class _FileIdentity:
    path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    links: int


def materialize_method_execution(
    method: ResolvedMethod,
    *,
    stage_root: Path,
    measurements_source: Path,
    measurements_file_sha256: str,
    dataset_identity_sha256: str,
    expected_acquisition_spec: Mapping[str, object],
    algorithm_seed: AlgorithmSeed,
    checkpoint_store: Mapping[str, Path],
    python_executable: Path,
    source_root: Path,
    requested_runtime_device: str,
) -> MaterializedMethodExecution:
    """Create a new path-private stage after validating every parent input."""
    requested_device, child_device, physical_cuda = _runtime_device(
        requested_runtime_device
    )
    method = _snapshot_resolved_method(method)
    _validate_resolved_method(method)
    measurement_hash = _require_sha256(
        "measurements_file_sha256", measurements_file_sha256
    )
    dataset_hash = _require_sha256(
        "dataset_identity_sha256", dataset_identity_sha256
    )
    _validate_algorithm_seed(algorithm_seed)
    frozen_acquisition_spec = _validated_frozen_acquisition_spec(
        expected_acquisition_spec
    )

    if not isinstance(stage_root, Path):
        raise TypeError("stage_root must be a Path")
    if not isinstance(measurements_source, Path):
        raise TypeError("measurements_source must be a Path")
    if not isinstance(python_executable, Path):
        raise TypeError("python_executable must be a Path")
    if not isinstance(source_root, Path):
        raise TypeError("source_root must be a Path")
    if not isinstance(checkpoint_store, Mapping):
        raise TypeError("checkpoint_store must be a mapping")

    stage_candidate = _validate_stage_root_candidate(stage_root)
    python_path = _resolved_regular_file(
        python_executable,
        noun="python executable",
        require_single_link=False,
    )
    runtime_root = _resolved_real_directory(
        python_path.parent, noun="Python runtime root"
    )
    system_root = _windows_system_root()
    system32 = (
        _resolved_real_directory(
            system_root / "System32", noun="Windows System32 directory"
        )
        if os.name == "nt"
        else system_root
    )
    selected_entrypoint = _validate_command_template(method, python_path)
    source_path = _resolved_real_directory(source_root, noun="source root")
    source_files = _selected_source_files(source_path)
    measurement_source_path = _resolved_regular_file(
        measurements_source, noun="measurement source"
    )
    measurement_source_hash = _sha256_regular_file(
        measurement_source_path, noun="measurement source"
    )
    if measurement_source_hash != measurement_hash:
        raise ValueError("measurement source hash mismatch")
    checkpoint_sources = _validate_checkpoint_store(
        method, checkpoint_store
    )
    _reject_broad_read_root_overlaps(
        stage_root=stage_root,
        source_root=source_path,
        measurements_source=measurement_source_path,
        checkpoint_sources=checkpoint_sources,
        broad_read_roots=(runtime_root, system32),
    )

    stage_identity = _prepare_stage_root(stage_candidate)
    stage = stage_identity.path
    paths = _stage_paths(stage)
    work_directories = {
        name: paths["work"] / name
        for name in (
            "tmp",
            "home",
            "xdg-cache",
            "torch",
            "matplotlib",
        )
    }
    desired_directories = {
        _lexical_absolute(directory)
        for directory in (
            *paths["directories"],
            *work_directories.values(),
        )
    }
    for relative, _source in source_files:
        parent = paths["code"].joinpath(*relative.parts).parent
        while not _same_lexical_path(parent, stage):
            desired_directories.add(_lexical_absolute(parent))
            parent = parent.parent
    directory_identities = _create_stage_directory_tree(
        stage_identity,
        desired_directories,
    )

    _copy_exact_file(
        measurement_source_path,
        paths["measurements"],
        expected_sha256=measurement_hash,
        noun="measurement source",
        parent_identity=_stage_directory_identity(
            directory_identities,
            paths["measurements"].parent,
        ),
    )
    checkpoint_destinations: dict[str, Path] = {}
    for requirement in method.checkpoint_requirements:
        physical_name = (
            hashlib.sha256(
                requirement.logical_id.encode("utf-8", errors="strict")
            ).hexdigest()
            + ".checkpoint"
        )
        destination = paths["checkpoints"] / physical_name
        _copy_exact_file(
            checkpoint_sources[requirement.logical_id],
            destination,
            expected_sha256=requirement.sha256,
            noun=f"checkpoint {requirement.logical_id!r}",
            parent_identity=_stage_directory_identity(
                directory_identities,
                destination.parent,
            ),
        )
        checkpoint_destinations[requirement.logical_id] = destination

    source_inventory: list[dict[str, str]] = []
    for relative, source in source_files:
        destination = paths["code"].joinpath(*relative.parts)
        digest = _copy_exact_file(
            source,
            destination,
            expected_sha256=None,
            noun=f"source file {relative.as_posix()!r}",
            parent_identity=_stage_directory_identity(
                directory_identities,
                destination.parent,
            ),
        )
        source_inventory.append(
            {"path": relative.as_posix(), "sha256": digest}
        )

    semantic = _method_semantic_document(method)
    semantic_sha256 = hashlib.sha256(
        canonical_json_bytes(semantic)
    ).hexdigest()
    runtime_checkpoints = {
        requirement.logical_id: {
            "path": checkpoint_destinations[
                requirement.logical_id
            ].relative_to(stage).as_posix(),
            "sha256": requirement.sha256,
        }
        for requirement in method.checkpoint_requirements
    }
    config_document = {
        "schema": "materialized-method-config-v1",
        "semantic": semantic,
        "semantic_sha256": semantic_sha256,
        "request": {
            "method_id": method.method_id,
            "method_config_id": method.method_config_id,
            "execution_profile": method.execution_profile,
            "method_config_sha256": method.method_config_sha256,
            "semantic_sha256": semantic_sha256,
            "dataset_identity_sha256": dataset_hash,
            "measurements_file_sha256": measurement_hash,
            "expected_acquisition_spec": thaw_json(
                frozen_acquisition_spec
            ),
            "algorithm_seed": {
                "derivation_sha256": algorithm_seed.derivation_sha256,
                "seed_u32": algorithm_seed.seed_u32,
            },
        },
        "runtime": {
            "measurements_path": "input/measurements.npz",
            "child_output_dir": "child-output",
            "checkpoints": runtime_checkpoints,
            "requested_runtime_device": requested_device,
            "child_runtime_device": child_device,
        },
    }
    config_bytes = canonical_json_bytes(config_document)
    config_sha256 = _exclusive_write_bytes(
        paths["method_config"],
        config_bytes,
        parent_identity=_stage_directory_identity(
            directory_identities,
            paths["method_config"].parent,
        ),
        noun="method config",
    )

    runtime_site_roots = _runtime_site_package_roots(runtime_root)
    read_allowlist = (
        paths["measurements"],
        paths["method_config"],
        paths["audit_policy"],
        *(
            checkpoint_destinations[key]
            for key in sorted(checkpoint_destinations)
        ),
    )
    read_root_allowlist = _unique_paths(
        (
            paths["code"],
            runtime_root,
            system32,
            paths["child_output"],
            paths["work"],
        )
    )
    write_root_allowlist = (
        paths["child_output"],
        paths["work"],
    )
    policy_document = {
        "schema": "method-audit-policy-v1",
        "audit_log_path": str(paths["audit_log"]),
        "exact_read_paths": [str(path) for path in read_allowlist],
        "read_roots": [str(path) for path in read_root_allowlist],
        "write_roots": [
            str(path) for path in write_root_allowlist
        ],
        "chdir_roots": [str(paths["code"]), str(paths["work"])],
        "python_runtime_root": str(runtime_root),
        "windows_system_read_root": str(system32),
        "runtime_site_package_roots": [
            str(path) for path in runtime_site_roots
        ],
        "logged_unrelated_events": [],
    }
    policy_bytes = canonical_json_bytes(policy_document)
    policy_sha256 = _exclusive_write_bytes(
        paths["audit_policy"],
        policy_bytes,
        parent_identity=_stage_directory_identity(
            directory_identities,
            paths["audit_policy"].parent,
        ),
        noun="audit policy",
    )
    _exclusive_write_bytes(
        paths["audit_log"],
        b"",
        parent_identity=_stage_directory_identity(
            directory_identities,
            paths["audit_log"].parent,
        ),
        noun="audit log",
    )
    _exclusive_write_bytes(
        paths["stdout"],
        b"",
        parent_identity=_stage_directory_identity(
            directory_identities,
            paths["stdout"].parent,
        ),
        noun="stdout log",
    )
    _exclusive_write_bytes(
        paths["stderr"],
        b"",
        parent_identity=_stage_directory_identity(
            directory_identities,
            paths["stderr"].parent,
        ),
        noun="stderr log",
    )

    token_values = {
        "${PYTHON}": str(python_path),
        "${MEASUREMENTS_PATH}": str(paths["measurements"]),
        "${OUTPUT_DIR}": str(paths["child_output"]),
        "${METHOD_CONFIG_PATH}": str(paths["method_config"]),
        "${DATASET_IDENTITY_SHA256}": dataset_hash,
        "${ALGORITHM_SEED}": str(algorithm_seed.seed_u32),
        "${DEVICE}": child_device,
        "${AUDIT_LOG_PATH}": str(paths["audit_log"]),
    }
    child_arguments = _materialize_child_arguments(
        method.command_template[2:],
        token_values=token_values,
        checkpoints=checkpoint_destinations,
    )
    original_child_argv = (
        str(python_path),
        str(paths["code"].joinpath(*Path(selected_entrypoint).parts)),
        *child_arguments,
    )
    bootstrap = paths["code"] / "scripts" / "experiments" / (
        "method_child_bootstrap.py"
    )
    wrapped_argv = (
        str(python_path),
        "-I",
        "-S",
        "-B",
        "-X",
        "utf8",
        str(bootstrap),
        "--policy",
        str(paths["audit_policy"]),
        "--code-root",
        str(paths["code"]),
        "--entrypoint",
        selected_entrypoint,
        "--",
        *child_arguments,
    )
    env = _fresh_child_environment(
        runtime_root=runtime_root,
        system_root=system_root,
        system32=system32,
        work_directories=work_directories,
        physical_cuda=physical_cuda,
    )
    expected_stage_files = {
        paths["measurements"]: measurement_hash,
        paths["method_config"]: hashlib.sha256(config_bytes).hexdigest(),
        paths["audit_policy"]: hashlib.sha256(policy_bytes).hexdigest(),
        paths["audit_log"]: hashlib.sha256(b"").hexdigest(),
        paths["stdout"]: hashlib.sha256(b"").hexdigest(),
        paths["stderr"]: hashlib.sha256(b"").hexdigest(),
        **{
            checkpoint_destinations[requirement.logical_id]: (
                requirement.sha256
            )
            for requirement in method.checkpoint_requirements
        },
        **{
            paths["code"].joinpath(*Path(item["path"]).parts): item[
                "sha256"
            ]
            for item in source_inventory
        },
    }
    _verify_exact_stage_inventory(
        stage_identity=stage_identity,
        directory_identities=directory_identities,
        expected_files=expected_stage_files,
    )
    final_config_bytes = _read_stable_regular_bytes(
        paths["method_config"], noun="materialized method config"
    )
    if final_config_bytes != config_bytes:
        raise ValueError("materialized method config bytes changed")
    config_sha256 = hashlib.sha256(final_config_bytes).hexdigest()
    final_config_document = _load_exact_json_bytes(
        final_config_bytes, noun="materialized method config"
    )
    final_semantic = final_config_document.get("semantic")
    final_semantic_sha256 = hashlib.sha256(
        canonical_json_bytes(final_semantic)
    ).hexdigest()
    if final_semantic_sha256 != semantic_sha256:
        raise ValueError("materialized semantic hash changed")
    semantic_sha256 = final_semantic_sha256
    final_policy_bytes = _read_stable_regular_bytes(
        paths["audit_policy"], noun="materialized audit policy"
    )
    if final_policy_bytes != policy_bytes:
        raise ValueError("materialized audit policy bytes changed")
    policy_sha256 = hashlib.sha256(final_policy_bytes).hexdigest()
    source_inventory = _verify_final_source_closure(
        source_path,
        source_inventory,
    )
    _verify_exact_stage_inventory(
        stage_identity=stage_identity,
        directory_identities=directory_identities,
        expected_files=expected_stage_files,
    )

    source_snapshot_sha256 = hashlib.sha256(
        canonical_json_bytes(source_inventory)
    ).hexdigest()
    logical_record = {
        "schema": "materialized-method-execution-v1",
        "method_id": method.method_id,
        "method_config_id": method.method_config_id,
        "execution_profile": method.execution_profile,
        "method_config_sha256": method.method_config_sha256,
        "semantic_sha256": semantic_sha256,
        "materialized_config_sha256": config_sha256,
        "dataset_identity_sha256": dataset_hash,
        "measurements_file_sha256": measurement_hash,
        "expected_acquisition_spec": thaw_json(
            frozen_acquisition_spec
        ),
        "algorithm_seed": {
            "derivation_sha256": algorithm_seed.derivation_sha256,
            "seed_u32": algorithm_seed.seed_u32,
        },
        "checkpoint_sha256": {
            requirement.logical_id: requirement.sha256
            for requirement in method.checkpoint_requirements
        },
        "source_inventory": source_inventory,
        "source_snapshot_sha256": source_snapshot_sha256,
        "requested_runtime_device": requested_device,
        "child_runtime_device": child_device,
        "entrypoint": selected_entrypoint,
        "command_template": list(method.command_template),
    }
    runtime_record = {
        "stage_root": str(stage),
        "cwd": str(paths["code"]),
        "original_child_argv": list(original_child_argv),
        "wrapped_argv": list(wrapped_argv),
        "environment": dict(env),
        "measurements_path": str(paths["measurements"]),
        "method_config_path": str(paths["method_config"]),
        "child_output_dir": str(paths["child_output"]),
        "audit_log_path": str(paths["audit_log"]),
        "audit_policy_path": str(paths["audit_policy"]),
        "audit_policy_sha256": policy_sha256,
        "stdout_path": str(paths["stdout"]),
        "stderr_path": str(paths["stderr"]),
    }
    return MaterializedMethodExecution(
        argv=wrapped_argv,
        cwd=paths["code"],
        env=env,
        measurements_path=paths["measurements"],
        method_config_path=paths["method_config"],
        child_output_dir=paths["child_output"],
        expected_acquisition_spec=frozen_acquisition_spec,
        audit_log_path=paths["audit_log"],
        stdout_path=paths["stdout"],
        stderr_path=paths["stderr"],
        read_allowlist=read_allowlist,
        read_root_allowlist=read_root_allowlist,
        write_root_allowlist=write_root_allowlist,
        requested_runtime_device=requested_device,
        child_runtime_device=child_device,
        audit_policy_path=paths["audit_policy"],
        audit_policy_sha256=policy_sha256,
        materialization_record=_freeze_json(
            {"logical": logical_record, "runtime": runtime_record}
        ),
    )


def load_materialized_method_request(
    path: Path,
) -> MaterializedMethodRequest:
    if not isinstance(path, Path):
        raise TypeError("materialized method config path must be a Path")
    config_path = _resolved_regular_file(
        path,
        noun="materialized method config",
    )
    if (
        config_path.name != "method-config.json"
        or config_path.parent.name != "config"
    ):
        raise ValueError(
            "materialized method config path is outside its exact stage "
            "location"
        )
    stage_identity = _directory_identity(
        config_path.parent.parent,
        noun="materialized stage root",
    )
    config_directory_identity = _directory_identity(
        config_path.parent,
        noun="materialized config directory",
    )
    expected_config_path = (
        stage_identity.path / "config" / "method-config.json"
    )
    if not _same_lexical_path(config_path, expected_config_path):
        raise ValueError(
            "materialized method config path is outside its exact stage "
            "location"
        )
    config_bytes = _read_stable_regular_bytes(
        config_path,
        noun="materialized method config",
    )
    document = _load_exact_json_bytes(
        config_bytes,
        noun="materialized method config",
    )
    _require_exact_keys(
        document,
        {
            "schema",
            "semantic",
            "semantic_sha256",
            "request",
            "runtime",
        },
        noun="materialized method config",
    )
    if document["schema"] != "materialized-method-config-v1":
        raise ValueError("materialized method config schema is unsupported")

    semantic = _require_json_object(
        document["semantic"],
        noun="materialized semantic section",
    )
    _reject_absolute_semantic_paths(
        semantic,
        location="materialized semantic section",
    )
    method = _resolved_method_from_semantic(semantic)
    declared_semantic_sha256 = _require_sha256(
        "semantic_sha256",
        document["semantic_sha256"],
    )
    computed_semantic_sha256 = hashlib.sha256(
        canonical_json_bytes(semantic)
    ).hexdigest()
    if computed_semantic_sha256 != declared_semantic_sha256:
        raise ValueError("materialized semantic hash mismatch")

    request = _require_json_object(
        document["request"],
        noun="materialized request section",
    )
    _require_exact_keys(
        request,
        {
            "method_id",
            "method_config_id",
            "execution_profile",
            "method_config_sha256",
            "semantic_sha256",
            "dataset_identity_sha256",
            "measurements_file_sha256",
            "expected_acquisition_spec",
            "algorithm_seed",
        },
        noun="materialized request section",
    )
    request_method_id = _require_exact_string(
        request["method_id"], noun="request method_id"
    )
    request_config_id = _require_exact_string(
        request["method_config_id"], noun="request method_config_id"
    )
    request_profile = _require_exact_string(
        request["execution_profile"], noun="request execution_profile"
    )
    request_method_sha256 = _require_sha256(
        "request method_config_sha256",
        request["method_config_sha256"],
    )
    request_semantic_sha256 = _require_sha256(
        "request semantic_sha256",
        request["semantic_sha256"],
    )
    crosslocks = (
        ("method_id", request_method_id, method.method_id),
        ("method_config_id", request_config_id, method.method_config_id),
        ("execution_profile", request_profile, method.execution_profile),
        (
            "method_config_sha256",
            request_method_sha256,
            method.method_config_sha256,
        ),
        (
            "semantic_sha256",
            request_semantic_sha256,
            declared_semantic_sha256,
        ),
    )
    for name, supplied, expected in crosslocks:
        if supplied != expected:
            raise ValueError(f"request {name} crosslock mismatch")

    dataset_identity_sha256 = _require_sha256(
        "dataset_identity_sha256",
        request["dataset_identity_sha256"],
    )
    measurements_file_sha256 = _require_sha256(
        "measurements_file_sha256",
        request["measurements_file_sha256"],
    )
    expected_acquisition_spec = _validated_frozen_acquisition_spec(
        request["expected_acquisition_spec"]
    )
    seed_document = _require_json_object(
        request["algorithm_seed"],
        noun="algorithm seed",
    )
    _require_exact_keys(
        seed_document,
        {"derivation_sha256", "seed_u32"},
        noun="algorithm seed",
    )
    algorithm_seed = AlgorithmSeed(
        derivation_sha256=_require_sha256(
            "algorithm_seed.derivation_sha256",
            seed_document["derivation_sha256"],
        ),
        seed_u32=seed_document["seed_u32"],
    )
    _validate_algorithm_seed(algorithm_seed)

    runtime = _require_json_object(
        document["runtime"],
        noun="materialized runtime section",
    )
    _require_exact_keys(
        runtime,
        {
            "measurements_path",
            "child_output_dir",
            "checkpoints",
            "requested_runtime_device",
            "child_runtime_device",
        },
        noun="materialized runtime section",
    )
    requested_runtime_device = _require_exact_string(
        runtime["requested_runtime_device"],
        noun="requested runtime device",
    )
    child_runtime_device = _require_exact_string(
        runtime["child_runtime_device"],
        noun="child runtime device",
    )
    _requested, expected_child_device, _physical = _runtime_device(
        requested_runtime_device
    )
    if child_runtime_device != expected_child_device:
        raise ValueError("runtime device mapping is inconsistent")

    measurements_path = _exact_stage_runtime_path(
        stage_identity.path,
        runtime["measurements_path"],
        expected_relative="input/measurements.npz",
        noun="staged measurement path",
    )
    if (
        _sha256_regular_file(
            measurements_path,
            noun="staged measurement input",
        )
        != measurements_file_sha256
    ):
        raise ValueError("staged measurement hash mismatch")
    child_output_dir = _exact_stage_runtime_path(
        stage_identity.path,
        runtime["child_output_dir"],
        expected_relative="child-output",
        noun="child output path",
    )
    output_identity = _directory_identity(
        child_output_dir,
        noun="child output directory",
    )

    runtime_checkpoints = _require_json_object(
        runtime["checkpoints"],
        noun="runtime checkpoint mapping",
    )
    requirements = {
        requirement.logical_id: requirement
        for requirement in method.checkpoint_requirements
    }
    if set(runtime_checkpoints) != set(requirements):
        raise ValueError(
            "runtime checkpoint logical IDs disagree with resolved method"
        )
    checkpoint_paths: dict[str, Path] = {}
    for logical_id in sorted(requirements):
        requirement = requirements[logical_id]
        record = _require_json_object(
            runtime_checkpoints[logical_id],
            noun=f"runtime checkpoint {logical_id!r}",
        )
        _require_exact_keys(
            record,
            {"path", "sha256"},
            noun=f"runtime checkpoint {logical_id!r}",
        )
        record_sha256 = _require_sha256(
            f"runtime checkpoint {logical_id!r} sha256",
            record["sha256"],
        )
        if record_sha256 != requirement.sha256:
            raise ValueError(
                f"runtime checkpoint {logical_id!r} hash disagrees with "
                "resolved method"
            )
        physical_name = (
            hashlib.sha256(
                logical_id.encode("utf-8", errors="strict")
            ).hexdigest()
            + ".checkpoint"
        )
        checkpoint_path = _exact_stage_runtime_path(
            stage_identity.path,
            record["path"],
            expected_relative=(
                PurePosixPath("checkpoints") / physical_name
            ).as_posix(),
            noun=f"runtime checkpoint {logical_id!r} path",
        )
        if (
            _sha256_regular_file(
                checkpoint_path,
                noun=f"staged checkpoint {logical_id!r}",
            )
            != requirement.sha256
        ):
            raise ValueError(
                f"staged checkpoint {logical_id!r} hash mismatch"
            )
        checkpoint_paths[logical_id] = checkpoint_path

    if (
        _read_stable_regular_bytes(
            config_path,
            noun="materialized method config final check",
        )
        != config_bytes
    ):
        raise ValueError("materialized method config changed during loading")
    if (
        _sha256_regular_file(
            measurements_path,
            noun="staged measurement final check",
        )
        != measurements_file_sha256
    ):
        raise ValueError("staged measurement changed during loading")
    for logical_id, checkpoint_path in checkpoint_paths.items():
        if (
            _sha256_regular_file(
                checkpoint_path,
                noun=f"staged checkpoint {logical_id!r} final check",
            )
            != requirements[logical_id].sha256
        ):
            raise ValueError(
                f"staged checkpoint {logical_id!r} changed during loading"
            )
    _verify_directory_identity(
        stage_identity,
        noun="materialized stage root",
    )
    _verify_directory_identity(
        config_directory_identity,
        noun="materialized config directory",
    )
    _verify_directory_identity(
        output_identity,
        noun="child output directory",
    )
    return MaterializedMethodRequest(
        method=method,
        algorithm_seed=algorithm_seed,
        dataset_identity_sha256=dataset_identity_sha256,
        measurements_file_sha256=measurements_file_sha256,
        expected_acquisition_spec=expected_acquisition_spec,
        measurements_path=measurements_path,
        child_output_dir=child_output_dir,
        checkpoint_paths=MappingProxyType(checkpoint_paths),
        requested_runtime_device=requested_runtime_device,
        child_runtime_device=child_runtime_device,
    )


def _require_json_object(value: object, *, noun: str) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{noun} must be an exact JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    noun: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{noun} keys mismatch; missing={missing!r}, extra={extra!r}"
        )


def _require_exact_string(value: object, *, noun: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{noun} must be a nonempty exact string")
    return value


def _require_exact_bool(value: object, *, noun: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{noun} must be an exact boolean")
    return value


def _require_string_list(value: object, *, noun: str) -> tuple[str, ...]:
    if type(value) is not list or not all(
        type(item) is str for item in value
    ):
        raise ValueError(f"{noun} must be a JSON list of exact strings")
    return tuple(value)


def _resolved_method_from_semantic(
    semantic: dict[str, object],
) -> ResolvedMethod:
    _require_exact_keys(
        semantic,
        {
            "method_id",
            "requested_method_config_id",
            "method_config_id",
            "execution_family",
            "command_template",
            "semantic_config",
            "method_config_sha256",
            "required_child_outputs",
            "checkpoint_requirements",
            "execution_profile",
            "profile_policy",
        },
        noun="materialized semantic section",
    )
    profile = _require_json_object(
        semantic["profile_policy"],
        noun="materialized profile policy",
    )
    _require_exact_keys(
        profile,
        {
            "publication_eligible",
            "selection_eligible",
            "promotion_eligible",
            "convergence_status",
            "execution_ready",
            "execution_blockers",
        },
        noun="materialized profile policy",
    )
    checkpoint_documents = semantic["checkpoint_requirements"]
    if type(checkpoint_documents) is not list:
        raise ValueError(
            "materialized checkpoint requirements must be a JSON list"
        )
    checkpoint_requirements: list[CheckpointRequirement] = []
    for index, raw_requirement in enumerate(checkpoint_documents):
        requirement = _require_json_object(
            raw_requirement,
            noun=f"materialized checkpoint requirement {index}",
        )
        _require_exact_keys(
            requirement,
            {"logical_id", "sha256", "provenance_status"},
            noun=f"materialized checkpoint requirement {index}",
        )
        checkpoint_requirements.append(
            CheckpointRequirement(
                logical_id=_require_exact_string(
                    requirement["logical_id"],
                    noun=f"checkpoint requirement {index} logical_id",
                ),
                sha256=_require_sha256(
                    f"checkpoint requirement {index} sha256",
                    requirement["sha256"],
                ),
                provenance_status=_require_exact_string(
                    requirement["provenance_status"],
                    noun=(
                        f"checkpoint requirement {index} "
                        "provenance_status"
                    ),
                ),
            )
        )
    semantic_config = _require_json_object(
        semantic["semantic_config"],
        noun="materialized semantic_config",
    )
    frozen_semantic_config = _freeze_json(semantic_config)
    if not isinstance(frozen_semantic_config, Mapping):
        raise ValueError("materialized semantic_config must be a mapping")
    method = ResolvedMethod(
        method_id=_require_exact_string(
            semantic["method_id"], noun="semantic method_id"
        ),
        requested_method_config_id=_require_exact_string(
            semantic["requested_method_config_id"],
            noun="semantic requested_method_config_id",
        ),
        method_config_id=_require_exact_string(
            semantic["method_config_id"],
            noun="semantic method_config_id",
        ),
        execution_family=_require_exact_string(
            semantic["execution_family"],
            noun="semantic execution_family",
        ),
        command_template=_require_string_list(
            semantic["command_template"],
            noun="semantic command_template",
        ),
        semantic_config=frozen_semantic_config,
        method_config_sha256=_require_sha256(
            "semantic method_config_sha256",
            semantic["method_config_sha256"],
        ),
        required_child_outputs=_require_string_list(
            semantic["required_child_outputs"],
            noun="semantic required_child_outputs",
        ),
        checkpoint_requirements=tuple(checkpoint_requirements),
        execution_profile=_require_exact_string(
            semantic["execution_profile"],
            noun="semantic execution_profile",
        ),
        publication_eligible=_require_exact_bool(
            profile["publication_eligible"],
            noun="profile publication_eligible",
        ),
        selection_eligible=_require_exact_bool(
            profile["selection_eligible"],
            noun="profile selection_eligible",
        ),
        promotion_eligible=_require_exact_bool(
            profile["promotion_eligible"],
            noun="profile promotion_eligible",
        ),
        convergence_status=_require_exact_string(
            profile["convergence_status"],
            noun="profile convergence_status",
        ),
        execution_ready=_require_exact_bool(
            profile["execution_ready"],
            noun="profile execution_ready",
        ),
        execution_blockers=_require_string_list(
            profile["execution_blockers"],
            noun="profile execution_blockers",
        ),
    )
    _validate_resolved_method(method)
    return method


def _exact_stage_runtime_path(
    stage_root: Path,
    value: object,
    *,
    expected_relative: str,
    noun: str,
) -> Path:
    raw = _require_exact_string(value, noun=noun)
    if _absolute_path_fragment(raw) is not None:
        raise ValueError(f"{noun} must be relative to the materialized stage")
    relative = PurePosixPath(raw)
    if (
        raw != relative.as_posix()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or raw != expected_relative
    ):
        raise ValueError(
            f"{noun} must equal the exact stage-relative path "
            f"{expected_relative!r}"
        )
    candidate = _lexical_absolute(
        stage_root.joinpath(*relative.parts)
    )
    if not _path_is_within(candidate, stage_root):
        raise ValueError(f"{noun} escapes the materialized stage")
    _reject_linked_ancestors(candidate, noun=noun)
    return candidate


def _runtime_device(value: object) -> tuple[str, str, str | None]:
    if value == "cpu" and type(value) is str:
        return "cpu", "cpu", None
    if type(value) is not str:
        raise ValueError("requested runtime device must be cpu or cuda:N")
    matched = _DEVICE.fullmatch(value)
    if matched is None:
        raise ValueError("requested runtime device must be cpu or cuda:N")
    return value, "cuda:0", matched.group(1)


def _require_sha256(name: str, value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(
            f"{name} must be 64 lowercase hexadecimal characters"
        )
    return value


def _validate_algorithm_seed(seed: object) -> None:
    if type(seed) is not AlgorithmSeed:
        raise TypeError("algorithm_seed must be an exact AlgorithmSeed")
    digest = _require_sha256(
        "algorithm_seed.derivation_sha256", seed.derivation_sha256
    )
    if (
        type(seed.seed_u32) is not int
        or seed.seed_u32 < 0
        or seed.seed_u32 > 0xFFFFFFFF
    ):
        raise ValueError("algorithm_seed.seed_u32 must be a uint32 integer")
    derived = int.from_bytes(bytes.fromhex(digest)[:4], "big")
    if seed.seed_u32 != derived:
        raise ValueError(
            "algorithm seed integer disagrees with derivation digest"
        )


def _validated_frozen_acquisition_spec(
    value: object,
) -> Mapping[str, object]:
    try:
        validated = _validate_blind_acquisition_spec(value)
    except (ArtifactValidationError, TypeError, ValueError) as error:
        raise ValueError("expected acquisition spec is invalid") from error
    frozen = _freeze_json(thaw_json(validated))
    if not isinstance(frozen, Mapping):
        raise ValueError("expected acquisition spec is invalid")
    return frozen


def _method_identity_payload(method: ResolvedMethod) -> dict[str, object]:
    return {
        "method_id": method.method_id,
        "method_config_id": method.method_config_id,
        "execution_family": method.execution_family,
        "execution_profile": method.execution_profile,
        "command_template": list(method.command_template),
        "semantic_config": thaw_json(method.semantic_config),
        "checkpoint_requirements": [
            {
                "logical_id": requirement.logical_id,
                "sha256": requirement.sha256,
                "provenance_status": requirement.provenance_status,
            }
            for requirement in method.checkpoint_requirements
        ],
        "required_child_outputs": [
            "reconstruction.npz",
            "method-info.json",
        ],
        "profile_policy": {
            "publication_eligible": method.publication_eligible,
            "selection_eligible": method.selection_eligible,
            "promotion_eligible": method.promotion_eligible,
            "convergence_status": method.convergence_status,
            "execution_ready": method.execution_ready,
            "execution_blockers": list(method.execution_blockers),
        },
    }


def _snapshot_resolved_method(method: object) -> ResolvedMethod:
    if type(method) is not ResolvedMethod:
        raise TypeError("method must be an exact ResolvedMethod")
    try:
        semantic_bytes = canonical_json_bytes(thaw_json(method.semantic_config))
        semantic_native = _load_exact_json_bytes(
            semantic_bytes,
            noun="method semantic_config snapshot",
        )
    except (TypeError, ValueError) as error:
        raise ValueError("method semantic_config cannot be snapshotted") from error
    frozen = _freeze_json(semantic_native)
    if not isinstance(frozen, Mapping):
        raise ValueError("method semantic_config must be a mapping")
    return replace(method, semantic_config=frozen)


def _absolute_path_fragment(value: str) -> str | None:
    candidates = [value]
    if "=" in value:
        candidates.append(value.split("=", 1)[1])
    for candidate in candidates:
        drive, _tail = ntpath.splitdrive(candidate)
        if (
            drive
            or ntpath.isabs(candidate)
            or posixpath.isabs(candidate)
        ):
            return candidate
    for pattern in (
        _EMBEDDED_WINDOWS_DRIVE_PATH,
        _EMBEDDED_UNC_PATH,
        _EMBEDDED_POSIX_PATH,
    ):
        match = pattern.search(value)
        if match is not None:
            return match.group(0)
    return None


def _reject_absolute_semantic_paths(value: object, *, location: str) -> None:
    if type(value) is str:
        if _absolute_path_fragment(value) is not None:
            raise ValueError(
                f"{location} must be path-free and cannot contain "
                "absolute paths"
            )
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if type(key) is str and _absolute_path_fragment(key) is not None:
                raise ValueError(
                    f"{location} must be path-free and cannot contain "
                    "absolute paths"
                )
            _reject_absolute_semantic_paths(
                child,
                location=f"{location}.{key}",
            )
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_absolute_semantic_paths(
                child,
                location=f"{location}[{index}]",
            )


def _validate_resolved_method(method: object) -> None:
    if type(method) is not ResolvedMethod:
        raise TypeError("method must be an exact ResolvedMethod")
    if type(method.execution_ready) is not bool:
        raise ValueError("method execution_ready must be an exact boolean")
    if not method.execution_ready:
        blockers = ", ".join(method.execution_blockers)
        raise ValueError(
            f"method execution is not ready; blockers: {blockers}"
        )
    if method.execution_blockers:
        raise ValueError("execution-ready method cannot retain blockers")
    if canonical_method_id(method.method_id) != method.method_id:
        raise ValueError("method_id is not canonical")
    for name in (
        "requested_method_config_id",
        "method_config_id",
        "execution_family",
        "execution_profile",
        "convergence_status",
    ):
        value = getattr(method, name)
        if type(value) is not str or not value:
            raise ValueError(f"method {name} must be a nonempty string")
    if method.execution_family not in _ENTRYPOINT_BY_FAMILY:
        raise ValueError("method execution_family is unsupported")
    if type(method.command_template) is not tuple or not all(
        type(item) is str for item in method.command_template
    ):
        raise ValueError("method command_template must contain exact strings")
    if type(method.required_child_outputs) is not tuple or (
        method.required_child_outputs
        != ("reconstruction.npz", "method-info.json")
    ):
        raise ValueError("method required child outputs are inconsistent")
    if type(method.checkpoint_requirements) is not tuple:
        raise ValueError("method checkpoint requirements must be a tuple")
    checkpoint_ids: set[str] = set()
    for requirement in method.checkpoint_requirements:
        if type(requirement) is not CheckpointRequirement:
            raise ValueError("checkpoint requirement has an invalid type")
        if (
            _CHECKPOINT_ID.fullmatch(requirement.logical_id) is None
            or requirement.logical_id in checkpoint_ids
        ):
            raise ValueError("checkpoint logical IDs must be unique and safe")
        checkpoint_ids.add(requirement.logical_id)
        _require_sha256("checkpoint sha256", requirement.sha256)
        if (
            type(requirement.provenance_status) is not str
            or not requirement.provenance_status
        ):
            raise ValueError(
                "checkpoint provenance_status must be nonempty"
            )
    validate_exact_json_native(
        thaw_json(method.semantic_config), "method semantic_config"
    )
    _reject_absolute_semantic_paths(
        method.semantic_config,
        location="method semantic_config",
    )
    _reject_absolute_semantic_paths(
        _method_semantic_document(method),
        location="method semantic document",
    )
    expected_hash = hashlib.sha256(
        canonical_json_bytes(_method_identity_payload(method))
    ).hexdigest()
    if method.method_config_sha256 != expected_hash:
        raise ValueError("resolved method_config_sha256 mismatch")
    for field in (
        "publication_eligible",
        "selection_eligible",
        "promotion_eligible",
    ):
        if type(getattr(method, field)) is not bool:
            raise ValueError(f"method {field} must be an exact boolean")


def _validate_command_template(
    method: ResolvedMethod, python_path: Path
) -> str:
    del python_path
    expected_entrypoint = _ENTRYPOINT_BY_FAMILY[method.execution_family]
    if (
        len(method.command_template) < 2
        or method.command_template[0] != "${PYTHON}"
        or method.command_template[1] != expected_entrypoint
    ):
        raise ValueError(
            "method command template must start with the selected Python "
            "and exact family entrypoint"
        )
    declared_checkpoints = {
        requirement.logical_id
        for requirement in method.checkpoint_requirements
    }
    for value in method.command_template:
        if _absolute_path_fragment(value) is not None:
            raise ValueError(
                "method command template must be path-free and cannot "
                "contain absolute paths"
            )
        if "${" not in value:
            continue
        if value in _PLAIN_TOKENS:
            continue
        checkpoint_match = _CHECKPOINT_TOKEN.fullmatch(value)
        if (
            checkpoint_match is not None
            and checkpoint_match.group(1) in declared_checkpoints
        ):
            continue
        assignment_match = _CHECKPOINT_ASSIGNMENT.fullmatch(value)
        if (
            assignment_match is not None
            and assignment_match.group("logical_id")
            in declared_checkpoints
        ):
            continue
        raise ValueError(f"unapproved or embedded command token: {value!r}")
    return expected_entrypoint


def _method_semantic_document(
    method: ResolvedMethod,
) -> dict[str, object]:
    payload = _method_identity_payload(method)
    return {
        "method_id": method.method_id,
        "requested_method_config_id": method.requested_method_config_id,
        "method_config_id": method.method_config_id,
        "execution_family": method.execution_family,
        "command_template": list(method.command_template),
        "semantic_config": thaw_json(method.semantic_config),
        "method_config_sha256": method.method_config_sha256,
        "required_child_outputs": list(method.required_child_outputs),
        "checkpoint_requirements": payload[
            "checkpoint_requirements"
        ],
        "execution_profile": method.execution_profile,
        "profile_policy": payload["profile_policy"],
    }


def _is_link_or_reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_POINT
    )


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_linked_ancestors(path: Path, *, noun: str) -> None:
    current = _lexical_absolute(path)
    while True:
        if os.path.lexists(current):
            try:
                info = os.lstat(current)
            except OSError as error:
                raise ValueError(f"cannot inspect {noun}: {current}") from error
            if _is_link_or_reparse(info):
                raise ValueError(
                    f"{noun} uses a symlink, junction, or reparse point: "
                    f"{current}"
                )
        parent = current.parent
        if parent == current:
            return
        current = parent


def _resolved_regular_file(
    path: Path,
    *,
    noun: str,
    require_single_link: bool = True,
) -> Path:
    lexical = _lexical_absolute(path)
    _reject_linked_ancestors(lexical, noun=noun)
    try:
        info = os.lstat(lexical)
    except OSError as error:
        raise ValueError(f"cannot inspect {noun}: {lexical}") from error
    if (
        _is_link_or_reparse(info)
        or not stat.S_ISREG(info.st_mode)
        or (require_single_link and info.st_nlink != 1)
    ):
        raise ValueError(
            f"{noun} must be an unlinked exact regular file"
        )
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"cannot resolve {noun}: {lexical}") from error
    return resolved


def _resolved_real_directory(path: Path, *, noun: str) -> Path:
    lexical = _lexical_absolute(path)
    _reject_linked_ancestors(lexical, noun=noun)
    try:
        info = os.lstat(lexical)
    except OSError as error:
        raise ValueError(f"cannot inspect {noun}: {lexical}") from error
    if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{noun} must be a real directory")
    try:
        return lexical.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"cannot resolve {noun}: {lexical}") from error


def _directory_identity(path: Path, *, noun: str) -> _DirectoryIdentity:
    resolved = _resolved_real_directory(path, noun=noun)
    try:
        info = os.lstat(resolved)
    except OSError as error:
        raise ValueError(f"cannot inspect {noun}: {resolved}") from error
    return _DirectoryIdentity(
        path=resolved,
        device=info.st_dev,
        inode=info.st_ino,
    )


def _verify_directory_identity(
    identity: _DirectoryIdentity,
    *,
    noun: str,
) -> None:
    _reject_linked_ancestors(identity.path, noun=noun)
    try:
        info = os.lstat(identity.path)
    except OSError as error:
        raise ValueError(f"{noun} disappeared or changed identity") from error
    if (
        _is_link_or_reparse(info)
        or not stat.S_ISDIR(info.st_mode)
        or info.st_dev != identity.device
        or info.st_ino != identity.inode
    ):
        raise ValueError(f"{noun} disappeared or changed identity")


def _same_lexical_path(first: Path, second: Path) -> bool:
    return os.path.normcase(str(_lexical_absolute(first))) == os.path.normcase(
        str(_lexical_absolute(second))
    )


def _path_is_within(path: Path, root: Path) -> bool:
    candidate = os.path.normcase(str(_lexical_absolute(path)))
    boundary = os.path.normcase(str(_lexical_absolute(root)))
    try:
        return os.path.commonpath((candidate, boundary)) == boundary
    except ValueError:
        return False


def _reject_broad_read_root_overlaps(
    *,
    stage_root: Path,
    source_root: Path,
    measurements_source: Path,
    checkpoint_sources: Mapping[str, Path],
    broad_read_roots: tuple[Path, ...],
) -> None:
    stage = _lexical_absolute(stage_root)
    directory_inputs = (
        ("stage root", stage),
        ("source root", source_root),
    )
    file_inputs = (
        ("measurement source", measurements_source),
        *(
            (f"checkpoint source {logical_id!r}", path)
            for logical_id, path in checkpoint_sources.items()
        ),
    )
    for broad_root in broad_read_roots:
        for noun, directory in directory_inputs:
            if _path_is_within(directory, broad_root) or _path_is_within(
                broad_root, directory
            ):
                raise ValueError(
                    f"{noun} overlaps broad runtime read root "
                    f"{broad_root}"
                )
        for noun, path in file_inputs:
            if _path_is_within(path, broad_root):
                raise ValueError(
                    f"{noun} is contained by broad runtime read root "
                    f"{broad_root}"
                )
    if _path_is_within(stage, source_root) or _path_is_within(
        source_root, stage
    ):
        raise ValueError("stage root and source root overlap")


def _sha256_regular_file(path: Path, *, noun: str) -> str:
    resolved = _resolved_regular_file(path, noun=noun)
    before = os.lstat(resolved)
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_nlink,
            ) != (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_nlink,
            ):
                raise ValueError(f"{noun} changed while being opened")
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            after_handle = os.fstat(stream.fileno())
            if (
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                opened.st_nlink,
            ) != (
                after_handle.st_size,
                after_handle.st_mtime_ns,
                after_handle.st_ctime_ns,
                after_handle.st_nlink,
            ):
                raise ValueError(f"{noun} changed while being hashed")
    except OSError as error:
        raise ValueError(f"cannot hash {noun}: {resolved}") from error
    after_path = os.lstat(resolved)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
    ) != (
        after_path.st_dev,
        after_path.st_ino,
        after_path.st_size,
        after_path.st_mtime_ns,
        after_path.st_ctime_ns,
        after_path.st_nlink,
    ):
        raise ValueError(f"{noun} changed while being hashed")
    return digest.hexdigest()


def _copy_exact_file(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str | None,
    noun: str,
    parent_identity: _DirectoryIdentity,
) -> str:
    before = _sha256_regular_file(source, noun=noun)
    if expected_sha256 is not None and before != expected_sha256:
        raise ValueError(f"{noun} hash mismatch before copy")
    if not _same_lexical_path(destination.parent, parent_identity.path):
        raise ValueError(f"{noun} destination parent is inconsistent")
    _verify_directory_identity(
        parent_identity, noun=f"{noun} destination directory"
    )
    source_path = _resolved_regular_file(source, noun=noun)
    source_before = os.lstat(source_path)
    source_fd = -1
    destination_fd = -1
    copied_digest = hashlib.sha256()
    destination_opened: os.stat_result | None = None
    try:
        source_fd = os.open(
            source_path,
            os.O_RDONLY | getattr(os, "O_BINARY", 0),
        )
        source_opened = os.fstat(source_fd)
        if (
            source_opened.st_dev,
            source_opened.st_ino,
            source_opened.st_size,
            source_opened.st_mtime_ns,
            source_opened.st_nlink,
        ) != (
            source_before.st_dev,
            source_before.st_ino,
            source_before.st_size,
            source_before.st_mtime_ns,
            source_before.st_nlink,
        ):
            raise ValueError(f"{noun} changed while being opened")
        destination_fd = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        destination_opened = os.fstat(destination_fd)
        if (
            not stat.S_ISREG(destination_opened.st_mode)
            or destination_opened.st_nlink != 1
        ):
            raise ValueError(
                f"{noun} destination is not an exclusive regular file"
            )
        for block in iter(lambda: os.read(source_fd, 1024 * 1024), b""):
            copied_digest.update(block)
            view = memoryview(block)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise OSError("short destination write")
                view = view[written:]
        os.fsync(destination_fd)
        source_after_handle = os.fstat(source_fd)
        destination_after_handle = os.fstat(destination_fd)
        if (
            source_opened.st_size,
            source_opened.st_mtime_ns,
            source_opened.st_ctime_ns,
            source_opened.st_nlink,
        ) != (
            source_after_handle.st_size,
            source_after_handle.st_mtime_ns,
            source_after_handle.st_ctime_ns,
            source_after_handle.st_nlink,
        ):
            raise ValueError(f"{noun} changed during copy")
        if (
            destination_after_handle.st_dev,
            destination_after_handle.st_ino,
            destination_after_handle.st_nlink,
            destination_after_handle.st_size,
        ) != (
            destination_opened.st_dev,
            destination_opened.st_ino,
            1,
            source_opened.st_size,
        ):
            raise ValueError(f"{noun} destination changed during copy")
    except OSError as error:
        raise ValueError(
            f"cannot exclusively create or copy {noun} destination"
        ) from error
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        if source_fd >= 0:
            os.close(source_fd)
    _verify_directory_identity(
        parent_identity, noun=f"{noun} destination directory"
    )
    try:
        destination_info = os.lstat(destination)
    except OSError as error:
        raise ValueError(f"{noun} destination disappeared") from error
    if (
        destination_opened is None
        or _is_link_or_reparse(destination_info)
        or not stat.S_ISREG(destination_info.st_mode)
        or destination_info.st_nlink != 1
        or destination_info.st_dev != destination_opened.st_dev
        or destination_info.st_ino != destination_opened.st_ino
    ):
        raise ValueError(f"{noun} destination changed identity")
    copied = _sha256_regular_file(destination, noun=f"copied {noun}")
    after = _sha256_regular_file(source, noun=noun)
    if (
        copied_digest.hexdigest() != before
        or copied != before
        or after != before
    ):
        raise ValueError(f"{noun} changed or hash mismatched during copy")
    if expected_sha256 is not None and copied != expected_sha256:
        raise ValueError(f"{noun} copied hash mismatch")
    return copied


def _exclusive_write_bytes(
    destination: Path,
    data: bytes,
    *,
    parent_identity: _DirectoryIdentity,
    noun: str,
) -> str:
    if type(data) is not bytes:
        raise TypeError("exclusive file data must be exact bytes")
    if not _same_lexical_path(destination.parent, parent_identity.path):
        raise ValueError(f"{noun} destination parent is inconsistent")
    _verify_directory_identity(
        parent_identity, noun=f"{noun} destination directory"
    )
    descriptor = -1
    opened: os.stat_result | None = None
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ValueError(
                f"{noun} destination is not an exclusive regular file"
            )
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short destination write")
            view = view[written:]
        os.fsync(descriptor)
        after_handle = os.fstat(descriptor)
        if (
            after_handle.st_dev,
            after_handle.st_ino,
            after_handle.st_nlink,
            after_handle.st_size,
        ) != (
            opened.st_dev,
            opened.st_ino,
            1,
            len(data),
        ):
            raise ValueError(f"{noun} destination changed during write")
    except OSError as error:
        raise ValueError(
            f"cannot exclusively create {noun} destination"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _verify_directory_identity(
        parent_identity, noun=f"{noun} destination directory"
    )
    try:
        info = os.lstat(destination)
    except OSError as error:
        raise ValueError(f"{noun} destination disappeared") from error
    if (
        opened is None
        or _is_link_or_reparse(info)
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_dev != opened.st_dev
        or info.st_ino != opened.st_ino
    ):
        raise ValueError(f"{noun} destination changed identity")
    digest = _sha256_regular_file(destination, noun=noun)
    if digest != hashlib.sha256(data).hexdigest():
        raise ValueError(f"{noun} destination bytes changed")
    return digest


def _read_stable_regular_bytes(path: Path, *, noun: str) -> bytes:
    resolved = _resolved_regular_file(path, noun=noun)
    before = os.lstat(resolved)
    try:
        with resolved.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_nlink,
            ) != (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_nlink,
            ):
                raise ValueError(f"{noun} changed while being opened")
            data = stream.read()
            after_handle = os.fstat(stream.fileno())
            if (
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
                opened.st_nlink,
            ) != (
                after_handle.st_size,
                after_handle.st_mtime_ns,
                after_handle.st_ctime_ns,
                after_handle.st_nlink,
            ):
                raise ValueError(f"{noun} changed while being read")
    except OSError as error:
        raise ValueError(f"cannot read {noun}") from error
    after_path = os.lstat(resolved)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_nlink,
    ) != (
        after_path.st_dev,
        after_path.st_ino,
        after_path.st_size,
        after_path.st_mtime_ns,
        after_path.st_ctime_ns,
        after_path.st_nlink,
    ):
        raise ValueError(f"{noun} changed while being read")
    return data


def _load_exact_json_bytes(data: bytes, *, noun: str) -> dict[str, object]:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{noun} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        decoded = data.decode("utf-8", errors="strict")
        value = json.loads(decoded, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{noun} is not strict UTF-8 JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{noun} must be a JSON object")
    validate_exact_json_native(value, noun)
    return value


def _scan_exact_stage(
    stage_identity: _DirectoryIdentity,
) -> tuple[set[str], dict[str, _FileIdentity]]:
    directories: set[str] = set()
    files: dict[str, _FileIdentity] = {}
    pending = [stage_identity.path]
    while pending:
        directory = pending.pop()
        key = _stage_directory_key(directory)
        if key in directories:
            raise ValueError("stage directory identity or name collision")
        directories.add(key)
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            raise ValueError(
                f"cannot scan staged directory: {directory}"
            ) from error
        for entry in entries:
            path = Path(entry.path)
            try:
                info = os.lstat(path)
            except OSError as error:
                raise ValueError(f"cannot inspect staged path: {path}") from error
            if _is_link_or_reparse(info):
                raise ValueError(
                    f"staged path is linked or reparsed: {path}"
                )
            child_key = _stage_directory_key(path)
            if stat.S_ISDIR(info.st_mode):
                pending.append(path)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                if child_key in files:
                    raise ValueError("staged file name collision")
                files[child_key] = _FileIdentity(
                    path=path,
                    device=info.st_dev,
                    inode=info.st_ino,
                    size=info.st_size,
                    mtime_ns=info.st_mtime_ns,
                    ctime_ns=info.st_ctime_ns,
                    links=info.st_nlink,
                )
            else:
                raise ValueError(
                    f"staged path is not an unlinked regular file: {path}"
                )
    return directories, files


def _verify_exact_stage_inventory(
    *,
    stage_identity: _DirectoryIdentity,
    directory_identities: Mapping[str, _DirectoryIdentity],
    expected_files: Mapping[Path, str],
) -> None:
    _verify_directory_identity(stage_identity, noun="stage root")
    for identity in directory_identities.values():
        _verify_directory_identity(identity, noun="stage directory")
    expected_file_keys = {
        _stage_directory_key(path): (path, digest)
        for path, digest in expected_files.items()
    }
    if len(expected_file_keys) != len(expected_files):
        raise ValueError("expected stage file names collide")
    first_directories, first_files = _scan_exact_stage(stage_identity)
    if first_directories != set(directory_identities):
        raise ValueError("stage directory inventory contains extras or gaps")
    if set(first_files) != set(expected_file_keys):
        raise ValueError("stage file inventory contains extras or gaps")
    for key, (path, expected_digest) in expected_file_keys.items():
        if not _same_lexical_path(path, first_files[key].path):
            raise ValueError("stage file inventory path changed")
        actual = _sha256_regular_file(path, noun=f"staged file {path}")
        if actual != expected_digest:
            raise ValueError(f"staged file hash changed: {path}")
    second_directories, second_files = _scan_exact_stage(stage_identity)
    if (
        second_directories != first_directories
        or second_files != first_files
    ):
        raise ValueError("stage inventory changed during final verification")
    for path, expected_digest in expected_files.items():
        if (
            _sha256_regular_file(
                path,
                noun=f"twice-verified staged file {path}",
            )
            != expected_digest
        ):
            raise ValueError(f"staged file hash changed: {path}")
    for identity in directory_identities.values():
        _verify_directory_identity(identity, noun="stage directory")


def _verify_final_source_closure(
    source_root: Path,
    expected_inventory: list[dict[str, str]],
) -> list[dict[str, str]]:
    expected = {
        item["path"]: item["sha256"] for item in expected_inventory
    }
    if len(expected) != len(expected_inventory):
        raise ValueError("source inventory contains duplicate paths")
    final_files = _selected_source_files(source_root)
    final_paths = tuple(relative.as_posix() for relative, _ in final_files)
    if set(final_paths) != set(expected) or len(final_paths) != len(expected):
        raise ValueError("source closure changed after initial enumeration")
    verified: list[dict[str, str]] = []
    for relative, source in final_files:
        relative_text = relative.as_posix()
        digest = _sha256_regular_file(
            source,
            noun=f"final source file {relative_text!r}",
        )
        if digest != expected[relative_text]:
            raise ValueError(
                f"source file changed after copy: {relative_text}"
            )
        verified.append({"path": relative_text, "sha256": digest})
    second_paths = tuple(
        relative.as_posix()
        for relative, _source in _selected_source_files(source_root)
    )
    if second_paths != final_paths:
        raise ValueError("source closure changed during final verification")
    return verified


def _validate_stage_root_candidate(
    stage_root: Path,
) -> _StageRootCandidate:
    lexical = _lexical_absolute(stage_root)
    _reject_linked_ancestors(lexical, noun="stage root ancestry")
    identity: _DirectoryIdentity | None = None
    if os.path.lexists(lexical):
        try:
            info = os.lstat(lexical)
        except OSError as error:
            raise ValueError("cannot inspect stage root") from error
        if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
            raise ValueError("stage root must be a real directory")
        resolved = lexical.resolve(strict=True)
        identity = _DirectoryIdentity(
            path=resolved,
            device=info.st_dev,
            inode=info.st_ino,
        )
        try:
            if next(os.scandir(lexical), None) is not None:
                raise ValueError("stage root must be new or empty")
        except OSError as error:
            raise ValueError("cannot inspect stage root contents") from error
        _verify_directory_identity(identity, noun="observed stage root")
    return _StageRootCandidate(path=lexical, identity=identity)


def _create_pinned_directory(
    path: Path,
    *,
    parent_identity: _DirectoryIdentity,
    noun: str,
) -> _DirectoryIdentity:
    lexical = _lexical_absolute(path)
    if not _same_lexical_path(lexical.parent, parent_identity.path):
        raise ValueError(f"{noun} parent path is inconsistent")
    _verify_directory_identity(
        parent_identity,
        noun=f"{noun} parent directory",
    )
    try:
        os.mkdir(lexical)
    except OSError as error:
        raise ValueError(f"cannot exclusively create {noun}") from error
    _verify_directory_identity(
        parent_identity,
        noun=f"{noun} parent directory",
    )
    identity = _directory_identity(lexical, noun=noun)
    _verify_directory_identity(identity, noun=noun)
    return identity


def _prepare_stage_root(
    candidate: _StageRootCandidate,
) -> _DirectoryIdentity:
    lexical = candidate.path
    if candidate.identity is None:
        if os.path.lexists(lexical):
            raise ValueError("stage root appeared after initial observation")
        missing: list[Path] = []
        cursor = lexical
        while not os.path.lexists(cursor):
            missing.append(cursor)
            parent = cursor.parent
            if parent == cursor:
                raise ValueError("cannot find an existing stage root ancestor")
            cursor = parent
        parent_identity = _directory_identity(
            cursor, noun="stage root existing ancestor"
        )
        for directory in reversed(missing):
            parent_identity = _create_pinned_directory(
                directory,
                parent_identity=parent_identity,
                noun="stage root directory",
            )
        identity = parent_identity
    else:
        identity = candidate.identity
        _verify_directory_identity(
            identity,
            noun="stage root observed before pin",
        )
    _verify_directory_identity(identity, noun="stage root")
    try:
        if next(os.scandir(identity.path), None) is not None:
            raise ValueError("stage root must remain empty before staging")
    except OSError as error:
        raise ValueError("cannot inspect stage root contents") from error
    return identity


def _stage_directory_key(path: Path) -> str:
    return os.path.normcase(str(_lexical_absolute(path)))


def _stage_directory_identity(
    identities: Mapping[str, _DirectoryIdentity],
    path: Path,
) -> _DirectoryIdentity:
    try:
        identity = identities[_stage_directory_key(path)]
    except KeyError as error:
        raise ValueError(f"unregistered stage directory: {path}") from error
    _verify_directory_identity(identity, noun="stage directory")
    return identity


def _create_stage_directory_tree(
    stage_identity: _DirectoryIdentity,
    desired_directories: object,
) -> dict[str, _DirectoryIdentity]:
    identities = {
        _stage_directory_key(stage_identity.path): stage_identity,
    }
    unique = {
        _stage_directory_key(Path(directory)): _lexical_absolute(
            Path(directory)
        )
        for directory in desired_directories
    }
    ordered = sorted(
        unique.values(),
        key=lambda path: (
            len(path.relative_to(stage_identity.path).parts),
            str(path).encode("utf-8", errors="strict"),
        ),
    )
    for directory in ordered:
        if _same_lexical_path(directory, stage_identity.path):
            continue
        try:
            directory.relative_to(stage_identity.path)
        except ValueError as error:
            raise ValueError(
                "stage directory escapes the pinned stage root"
            ) from error
        parent = _stage_directory_identity(identities, directory.parent)
        identity = _create_pinned_directory(
            directory,
            parent_identity=parent,
            noun="stage directory",
        )
        key = _stage_directory_key(identity.path)
        if key in identities:
            raise ValueError("stage directory identity collision")
        identities[key] = identity
    _verify_directory_identity(stage_identity, noun="stage root")
    return identities


def _stage_paths(stage: Path) -> dict[str, object]:
    input_dir = stage / "input"
    config_dir = stage / "config"
    checkpoint_dir = stage / "checkpoints"
    code_dir = stage / "code"
    work_dir = stage / "work"
    output_dir = stage / "child-output"
    parent_dir = stage / "parent"
    audit_dir = parent_dir / "audit"
    logs_dir = parent_dir / "logs"
    return {
        "directories": (
            input_dir,
            config_dir,
            checkpoint_dir,
            code_dir,
            work_dir,
            output_dir,
            parent_dir,
            audit_dir,
            logs_dir,
        ),
        "measurements": input_dir / "measurements.npz",
        "method_config": config_dir / "method-config.json",
        "checkpoints": checkpoint_dir,
        "code": code_dir,
        "work": work_dir,
        "child_output": output_dir,
        "audit_policy": audit_dir / "policy.json",
        "audit_log": audit_dir / "file-opens.jsonl",
        "stdout": logs_dir / "stdout.log",
        "stderr": logs_dir / "stderr.log",
    }


def _is_excluded_source(relative: Path, *, is_directory: bool) -> bool:
    parts = tuple(
        os.path.normcase(part).rstrip(" .") for part in relative.parts
    )
    excluded_prefixes = tuple(
        tuple(os.path.normcase(part).rstrip(" .") for part in prefix)
        for prefix in _SOURCE_EXCLUDED_PREFIXES
    )
    excluded_files = {
        tuple(os.path.normcase(part).rstrip(" .") for part in entry)
        for entry in _SOURCE_EXCLUDED_FILES
    }
    excluded_directories = {
        os.path.normcase(name).rstrip(" .")
        for name in _SOURCE_EXCLUDED_DIRECTORIES
    }
    if any(
        parts[: len(prefix)] == prefix
        for prefix in excluded_prefixes
    ):
        return True
    if not is_directory and parts in excluded_files:
        return True
    return is_directory and parts[-1] in excluded_directories


def _selected_source_files(
    source_root: Path,
) -> tuple[tuple[Path, Path], ...]:
    selected: list[tuple[Path, Path]] = []
    package_root = source_root / "gsdiff"
    _resolved_real_directory(package_root, noun="gsdiff source root")

    def visit(directory: Path, relative: Path) -> None:
        try:
            entries = sorted(
                os.scandir(directory),
                key=lambda entry: entry.name.encode(
                    "utf-8", errors="strict"
                ),
            )
        except OSError as error:
            raise ValueError(f"cannot scan source directory: {directory}") from error
        for entry in entries:
            source = Path(entry.path)
            child_relative = relative / entry.name
            try:
                info = os.lstat(source)
            except OSError as error:
                raise ValueError(f"cannot inspect source: {source}") from error
            if _is_link_or_reparse(info):
                raise ValueError(
                    f"source symlink or reparse point rejected: {source}"
                )
            if stat.S_ISDIR(info.st_mode):
                if not _is_excluded_source(
                    child_relative, is_directory=True
                ):
                    visit(source, child_relative)
            elif stat.S_ISREG(info.st_mode):
                if info.st_nlink != 1:
                    raise ValueError(
                        f"source hardlink rejected: {source}"
                    )
                if (
                    source.suffix == ".py"
                    and not _is_excluded_source(
                        child_relative, is_directory=False
                    )
                ):
                    selected.append((child_relative, source))
            else:
                raise ValueError(
                    f"non-regular source entry rejected: {source}"
                )

    visit(package_root, Path("gsdiff"))
    explicit = (
        Path("train.py"),
        Path("scripts/run_baselines.py"),
        Path("scripts/experiments/method_child_bootstrap.py"),
        Path("schemas/method-info-v2.schema.json"),
    )
    for relative in explicit:
        source = source_root.joinpath(*relative.parts)
        _resolved_regular_file(
            source, noun=f"required source {relative.as_posix()!r}"
        )
        selected.append((relative, source))
    return tuple(
        sorted(
            selected,
            key=lambda item: item[0].as_posix().encode(
                "utf-8", errors="strict"
            ),
        )
    )


def _validate_checkpoint_store(
    method: ResolvedMethod,
    checkpoint_store: Mapping[str, Path],
) -> dict[str, Path]:
    if any(type(key) is not str for key in checkpoint_store):
        raise ValueError("checkpoint store keys must be exact strings")
    required = {
        requirement.logical_id: requirement
        for requirement in method.checkpoint_requirements
    }
    if set(checkpoint_store) != set(required):
        raise ValueError(
            "checkpoint store logical IDs disagree with resolved method"
        )
    result: dict[str, Path] = {}
    for logical_id, requirement in required.items():
        source = checkpoint_store[logical_id]
        if not isinstance(source, Path):
            raise TypeError("checkpoint store values must be Path values")
        digest = _sha256_regular_file(
            source, noun=f"checkpoint {logical_id!r}"
        )
        if digest != requirement.sha256:
            raise ValueError(
                f"checkpoint {logical_id!r} hash mismatch"
            )
        result[logical_id] = _resolved_regular_file(
            source, noun=f"checkpoint {logical_id!r}"
        )
    return result


def _materialize_child_arguments(
    template: tuple[str, ...],
    *,
    token_values: Mapping[str, str],
    checkpoints: Mapping[str, Path],
) -> tuple[str, ...]:
    result: list[str] = []
    for item in template:
        if item in token_values:
            result.append(token_values[item])
            continue
        checkpoint_match = _CHECKPOINT_TOKEN.fullmatch(item)
        if checkpoint_match is not None:
            logical_id = checkpoint_match.group(1)
            try:
                result.append(str(checkpoints[logical_id]))
            except KeyError as error:
                raise ValueError(
                    f"undeclared checkpoint token: {logical_id!r}"
                ) from error
            continue
        assignment_match = _CHECKPOINT_ASSIGNMENT.fullmatch(item)
        if assignment_match is not None:
            logical_id = assignment_match.group("logical_id")
            try:
                result.append(
                    f"{logical_id}={checkpoints[logical_id]}"
                )
            except KeyError as error:
                raise ValueError(
                    f"undeclared checkpoint token: {logical_id!r}"
                ) from error
            continue
        if "${" in item:
            raise ValueError(
                f"unapproved or residual command token: {item!r}"
            )
        result.append(item)
    if any("${" in item for item in result):
        raise ValueError("residual command token after materialization")
    return tuple(result)


def _windows_system_root() -> Path:
    if os.name != "nt":
        return _resolved_real_directory(
            Path(os.path.abspath(os.sep)),
            noun="platform system root",
        )
    try:
        import ctypes

        buffer = ctypes.create_unicode_buffer(32768)
        length = ctypes.windll.kernel32.GetWindowsDirectoryW(
            buffer, len(buffer)
        )
    except (AttributeError, OSError, ValueError) as error:
        raise ValueError(
            "cannot obtain authoritative Windows directory"
        ) from error
    if length <= 0 or length >= len(buffer) or not buffer.value:
        raise ValueError("cannot obtain authoritative Windows directory")
    authoritative = _directory_identity(
        Path(buffer.value),
        noun="authoritative Windows directory",
    )
    for alias in ("SYSTEMROOT", "WINDIR"):
        value = os.environ.get(alias)
        if value is None:
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            raise ValueError(
                f"{alias} disagrees with authoritative Windows directory"
            )
        supplied = _directory_identity(
            candidate,
            noun=f"{alias} directory",
        )
        if (
            supplied.device != authoritative.device
            or supplied.inode != authoritative.inode
        ):
            raise ValueError(
                f"{alias} disagrees with authoritative Windows directory"
            )
    return authoritative.path


def _runtime_site_package_roots(runtime_root: Path) -> tuple[Path, ...]:
    candidates = (
        runtime_root / "Lib" / "site-packages",
        runtime_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages",
    )
    roots: list[Path] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        roots.append(
            _resolved_real_directory(
                candidate, noun="runtime site-packages root"
            )
        )
    return _unique_paths(roots)


def _unique_paths(paths: object) -> tuple[Path, ...]:
    ordered: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        normalized = os.path.normcase(str(path))
        if normalized not in seen:
            seen.add(normalized)
            ordered.append(path)
    return tuple(ordered)


def _fresh_child_environment(
    *,
    runtime_root: Path,
    system_root: Path,
    system32: Path,
    work_directories: Mapping[str, Path],
    physical_cuda: str | None,
) -> Mapping[str, str]:
    path_entries = _validated_child_path_entries(
        runtime_root=runtime_root,
        system_root=system_root,
        system32=system32,
    )
    environment = {
        "SYSTEMROOT": str(system_root),
        "WINDIR": str(system_root),
        "PATH": os.pathsep.join(str(path) for path in path_entries),
        "TEMP": str(work_directories["tmp"]),
        "TMP": str(work_directories["tmp"]),
        "HOME": str(work_directories["home"]),
        "USERPROFILE": str(work_directories["home"]),
        "XDG_CACHE_HOME": str(work_directories["xdg-cache"]),
        "TORCH_HOME": str(work_directories["torch"]),
        "MPLCONFIGDIR": str(work_directories["matplotlib"]),
    }
    if physical_cuda is not None:
        environment["CUDA_VISIBLE_DEVICES"] = physical_cuda
    return MappingProxyType(environment)


def _validated_child_path_entries(
    *,
    runtime_root: Path,
    system_root: Path,
    system32: Path,
) -> tuple[Path, ...]:
    runtime_identity = _directory_identity(
        runtime_root,
        noun="Python runtime PATH root",
    )
    authoritative_system_root = _windows_system_root()
    supplied_system_root = _directory_identity(
        system_root,
        noun="child SystemRoot",
    )
    authoritative_identity = _directory_identity(
        authoritative_system_root,
        noun="authoritative child SystemRoot",
    )
    if (
        supplied_system_root.device != authoritative_identity.device
        or supplied_system_root.inode != authoritative_identity.inode
    ):
        raise ValueError(
            "child SystemRoot disagrees with authoritative Windows root"
        )
    expected_system32 = (
        authoritative_identity.path / "System32"
        if os.name == "nt"
        else authoritative_identity.path
    )
    expected_system32_identity = _directory_identity(
        expected_system32,
        noun="authoritative System32 PATH entry",
    )
    supplied_system32_identity = _directory_identity(
        system32,
        noun="supplied System32 PATH entry",
    )
    if (
        supplied_system32_identity.device
        != expected_system32_identity.device
        or supplied_system32_identity.inode
        != expected_system32_identity.inode
    ):
        raise ValueError(
            "System32 PATH entry disagrees with authoritative directory"
        )

    entries: list[Path] = [runtime_identity.path]
    for candidate in (
        runtime_identity.path / "Scripts",
        runtime_identity.path / "Library" / "bin",
    ):
        _reject_linked_ancestors(
            candidate,
            noun="optional Python runtime PATH entry",
        )
        if not os.path.lexists(candidate):
            continue
        resolved = _resolved_real_directory(
            candidate,
            noun="optional Python runtime PATH entry",
        )
        if not _path_is_within(resolved, runtime_identity.path):
            raise ValueError(
                "Python runtime PATH entry escapes the runtime root"
            )
        entries.append(resolved)
    entries.append(expected_system32_identity.path)
    _verify_directory_identity(
        runtime_identity,
        noun="Python runtime PATH root",
    )
    return _unique_paths(entries)


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_json(child)
                for key, child in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(child) for child in value)
    return value
