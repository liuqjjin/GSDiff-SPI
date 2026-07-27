from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import sysconfig
from types import MappingProxyType
from typing import Any, Literal

import torch


NUMERICAL_ENV_ALLOWLIST = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_LAUNCH_BLOCKING",
    "MKL_NUM_THREADS",
    "NVIDIA_TF32_OVERRIDE",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
    "VECLIB_MAXIMUM_THREADS",
)

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$", flags=re.ASCII)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$", flags=re.ASCII)
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$", flags=re.ASCII)
_GIT_BASELINE = "c03420784bc92b4e9b9eef8330cbd9571ebebc68"
_SOURCE_TREE_MAGIC = b"source-tree-v1\0"
_SOURCE_TREE_EXCLUDED_COMPONENTS = frozenset(
    {
        ".git",
        "artifacts",
        "results",
        "_trash",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".cache",
        ".tox",
        ".nox",
        ".hypothesis",
        ".ipynb_checkpoints",
        ".venv",
        "venv",
        "env",
        "ENV",
        "__pypackages__",
        ".pixi",
    }
)
_GENERATED_PAPER_PREFIXES = (
    ("paper", "figure_data"),
    ("paper", "figures"),
    ("paper", "tables"),
    ("paper", "build"),
    ("paper", "generated"),
)
_RUN_IDENTITY_FIELDS = frozenset(
    {
        "assets_sha256",
        "checkpoints_sha256",
        "code_commit",
        "config_sha256",
        "dataset_identity_sha256",
        "dependencies_sha256",
        "dirty_worktree",
        "environment_lock_sha256",
        "execution_class",
        "method_id",
        "metric_version",
        "motion_id",
        "scientific_contract_id",
        "scientific_contract_sha256",
        "seed",
        "source_tree_hash",
        "target_id",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _normalize_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_json(value: object) -> object:
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical JSON mapping keys must be strings")
            normalized[key] = _normalize_json(child)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json(child) for child in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def resolved_config_sha256(config: Mapping[str, object]) -> str:
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    return sha256_bytes(canonical_json_bytes(_normalize_json(config)))


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(child) for key, child in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(child) for child in value)
    return value


@dataclass(frozen=True)
class RunIdentity:
    canonical_payload_json: bytes
    identity_sha256: str
    run_id: str

    def __post_init__(self) -> None:
        if type(self.canonical_payload_json) is not bytes:
            raise TypeError("canonical_payload_json must be immutable bytes")
        if type(self.identity_sha256) is not str:
            raise TypeError("identity_sha256 must be an exact string")
        if type(self.run_id) is not str:
            raise TypeError("run_id must be an exact string")
        try:
            decoded = json.loads(self.canonical_payload_json.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("canonical_payload_json must be valid UTF-8 JSON") from error
        if not isinstance(decoded, dict):
            raise ValueError("run identity payload must be a mapping")
        try:
            recanonicalized = canonical_json_bytes(decoded)
        except (TypeError, ValueError) as error:
            raise ValueError("run identity payload must be canonical JSON") from error
        if recanonicalized != self.canonical_payload_json:
            raise ValueError("canonical_payload_json is not canonical")
        _validate_run_identity_payload(decoded)
        expected_sha256 = sha256_bytes(self.canonical_payload_json)
        if self.identity_sha256 != expected_sha256:
            raise ValueError("identity_sha256 must hash canonical_payload_json")
        try:
            expected_run_id = (
                f"{decoded['scientific_contract_id']}--{decoded['method_id']}--"
                f"{decoded['target_id']}--{decoded['motion_id']}--"
                f"s{decoded['seed']}--{expected_sha256[:8]}"
            )
        except KeyError as error:
            raise ValueError("run identity payload is missing display-ID fields") from error
        if self.run_id != expected_run_id:
            raise ValueError("run_id does not match the canonical payload")

    def payload(self) -> Mapping[str, object]:
        """Return a newly decoded read-only view for manifest serialization."""
        decoded = json.loads(self.canonical_payload_json.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("run identity payload must decode to a mapping")
        frozen = _freeze_json(decoded)
        if not isinstance(frozen, Mapping):
            raise ValueError("run identity payload must decode to a mapping")
        return frozen


def _require_id(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if _ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must match {_ID_PATTERN.pattern}")
    return value


def _require_execution_class(value: object) -> str:
    if type(value) is not str or value != "blind_method_child":
        raise ValueError("execution_class must be exactly 'blind_method_child'")
    return value


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _normalize_named_hashes(
    name: str, values: Mapping[str, str]
) -> dict[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized: dict[str, str] = {}
    for key, value in values.items():
        normalized[_require_id(f"{name} key", key)] = _require_sha256(
            f"{name}[{key!r}]", value
        )
    return normalized


def _validate_run_identity_payload(payload: Mapping[str, object]) -> None:
    payload_fields = set(payload)
    if payload_fields != _RUN_IDENTITY_FIELDS:
        missing = sorted(_RUN_IDENTITY_FIELDS - payload_fields)
        unknown = sorted(payload_fields - _RUN_IDENTITY_FIELDS)
        raise ValueError(
            "run identity payload schema fields do not match: "
            f"missing={missing}, unknown={unknown}"
        )
    _require_execution_class(payload["execution_class"])
    for field in (
        "scientific_contract_id",
        "method_id",
        "target_id",
        "motion_id",
        "metric_version",
    ):
        _require_id(field, payload[field])
    for field in (
        "scientific_contract_sha256",
        "config_sha256",
        "dataset_identity_sha256",
        "dependencies_sha256",
        "environment_lock_sha256",
    ):
        _require_sha256(field, payload[field])
    code_commit = payload["code_commit"]
    if not isinstance(code_commit, str):
        raise TypeError("code_commit must be a string")
    if _COMMIT_PATTERN.fullmatch(code_commit) is None:
        raise ValueError(
            "code_commit must be exactly 40 lowercase hexadecimal characters"
        )
    if type(payload["seed"]) is not int:
        raise TypeError("seed must be an exact integer")
    dirty_worktree = payload["dirty_worktree"]
    if type(dirty_worktree) is not bool:
        raise TypeError("dirty_worktree must be an exact boolean")
    source_tree_hash = payload["source_tree_hash"]
    if dirty_worktree:
        if source_tree_hash is None:
            raise ValueError("dirty execution requires source_tree_hash")
        _require_sha256("source_tree_hash", source_tree_hash)
    elif source_tree_hash is not None:
        raise ValueError("clean execution must not provide source_tree_hash")
    _normalize_named_hashes("assets_sha256", payload["assets_sha256"])  # type: ignore[arg-type]
    _normalize_named_hashes(
        "checkpoints_sha256", payload["checkpoints_sha256"]  # type: ignore[arg-type]
    )


def build_run_identity(
    execution_class: Literal["blind_method_child"],
    scientific_contract_id: str,
    scientific_contract_sha256: str,
    method_id: str,
    target_id: str,
    motion_id: str,
    seed: int,
    config_sha256: str,
    dataset_identity_sha256: str,
    assets_sha256: Mapping[str, str],
    checkpoints_sha256: Mapping[str, str],
    code_commit: str,
    dirty_worktree: bool,
    source_tree_hash: str | None,
    dependencies_sha256: str,
    environment_lock_sha256: str,
    metric_version: str,
) -> RunIdentity:
    normalized_execution_class = _require_execution_class(execution_class)
    if type(seed) is not int:
        raise TypeError("seed must be an exact integer")
    if type(dirty_worktree) is not bool:
        raise TypeError("dirty_worktree must be an exact boolean")
    if dirty_worktree:
        if source_tree_hash is None:
            raise ValueError("dirty execution requires source_tree_hash")
        normalized_source_hash = _require_sha256(
            "source_tree_hash", source_tree_hash
        )
    elif source_tree_hash is not None:
        raise ValueError("clean execution must not provide source_tree_hash")
    else:
        normalized_source_hash = None
    if not isinstance(code_commit, str):
        raise TypeError("code_commit must be a string")
    if _COMMIT_PATTERN.fullmatch(code_commit) is None:
        raise ValueError(
            "code_commit must be exactly 40 lowercase hexadecimal characters"
        )

    normalized_scientific_contract_id = _require_id(
        "scientific_contract_id", scientific_contract_id
    )
    normalized_method_id = _require_id("method_id", method_id)
    normalized_target_id = _require_id("target_id", target_id)
    normalized_motion_id = _require_id("motion_id", motion_id)
    payload = {
        "assets_sha256": _normalize_named_hashes(
            "assets_sha256", assets_sha256
        ),
        "checkpoints_sha256": _normalize_named_hashes(
            "checkpoints_sha256", checkpoints_sha256
        ),
        "code_commit": code_commit,
        "config_sha256": _require_sha256("config_sha256", config_sha256),
        "dataset_identity_sha256": _require_sha256(
            "dataset_identity_sha256", dataset_identity_sha256
        ),
        "dependencies_sha256": _require_sha256(
            "dependencies_sha256", dependencies_sha256
        ),
        "dirty_worktree": dirty_worktree,
        "environment_lock_sha256": _require_sha256(
            "environment_lock_sha256", environment_lock_sha256
        ),
        "execution_class": normalized_execution_class,
        "method_id": normalized_method_id,
        "metric_version": _require_id("metric_version", metric_version),
        "motion_id": normalized_motion_id,
        "scientific_contract_id": normalized_scientific_contract_id,
        "scientific_contract_sha256": _require_sha256(
            "scientific_contract_sha256", scientific_contract_sha256
        ),
        "seed": seed,
        "source_tree_hash": normalized_source_hash,
        "target_id": normalized_target_id,
    }
    _validate_run_identity_payload(payload)
    canonical_payload = canonical_json_bytes(payload)
    identity_sha256 = sha256_bytes(canonical_payload)
    run_id = (
        f"{normalized_scientific_contract_id}--{normalized_method_id}--"
        f"{normalized_target_id}--{normalized_motion_id}--s{seed}--"
        f"{identity_sha256[:8]}"
    )
    return RunIdentity(
        canonical_payload_json=canonical_payload,
        identity_sha256=identity_sha256,
        run_id=run_id,
    )


def git_state(
    repo: Path, source_roots: Sequence[Path]
) -> dict[str, object]:
    (
        resolved_repo,
        head,
        head_entries,
        index_entries,
        tracked_paths,
        scanned_files,
    ) = _collect_source_inputs(repo, source_roots)
    commit = head.decode(
        "ascii", errors="strict"
    )
    branch_result = subprocess.run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=resolved_repo,
        check=False,
        capture_output=True,
    )
    if branch_result.returncode not in (0, 1):
        branch_result.check_returncode()
    branch = (
        branch_result.stdout.strip().decode("utf-8", errors="strict")
        if branch_result.returncode == 0
        else None
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=resolved_repo,
        check=True,
        capture_output=True,
    ).stdout
    actual_source_hash = _source_snapshot_sha256(
        resolved_repo,
        head,
        head_entries,
        index_entries,
        tracked_paths,
        scanned_files,
        clean_head=False,
    )
    clean_source_hash = _source_snapshot_sha256(
        resolved_repo,
        head,
        head_entries,
        index_entries,
        tracked_paths,
        scanned_files,
        clean_head=True,
    )
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) or actual_source_hash != clean_source_hash,
        "baseline": _GIT_BASELINE,
    }


def _git_bytes(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout


def _is_within_repo(repo: Path, path: Path) -> bool:
    try:
        common = os.path.commonpath((str(repo), str(path)))
    except ValueError:
        return False
    return os.path.normcase(common) == os.path.normcase(str(repo))


def _repo_relative(repo: Path, path: Path) -> str:
    relative = path.relative_to(repo)
    return "" if relative == Path(".") else relative.as_posix()


def _validate_git_path(raw_path: bytes) -> str:
    try:
        relative = raw_path.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("Git source paths must be valid UTF-8") from error
    path = Path(*relative.split("/"))
    if (
        not relative
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ValueError(f"unsafe Git source path: {relative!r}")
    return relative


def _parse_tree_records(payload: bytes) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    for record in payload.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            raw_mode, raw_object_type, raw_object_id = metadata.split(b" ", 2)
            mode = raw_mode.decode("ascii", errors="strict")
            object_type = raw_object_type.decode("ascii", errors="strict")
            object_id = raw_object_id.decode("ascii", errors="strict")
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("malformed Git tree record") from error
        if object_type not in ("blob", "commit"):
            raise ValueError(f"unsupported Git tree object type: {object_type!r}")
        relative = _validate_git_path(raw_path)
        if relative in entries:
            raise ValueError(f"duplicate Git tree path: {relative!r}")
        entries[relative] = (mode, object_id)
    return entries


def _parse_index_records(payload: bytes) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    for record in payload.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            raw_mode, _object_id, raw_stage = metadata.split(b" ", 2)
            mode = raw_mode.decode("ascii", errors="strict")
            object_id = _object_id.decode("ascii", errors="strict")
            stage = raw_stage.decode("ascii", errors="strict")
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("malformed Git index record") from error
        if stage != "0":
            raise ValueError("source tree cannot be hashed with unmerged index entries")
        relative = _validate_git_path(raw_path)
        if relative in entries:
            raise ValueError(f"duplicate Git index path: {relative!r}")
        entries[relative] = (mode, object_id)
    return entries


def _is_excluded_source_path(relative: str) -> bool:
    parts = tuple(relative.split("/")) if relative else ()
    if any(part in _SOURCE_TREE_EXCLUDED_COMPONENTS for part in parts):
        return True
    return any(parts[: len(prefix)] == prefix for prefix in _GENERATED_PAPER_PREFIXES)


def _path_is_in_source_roots(relative: str, roots: tuple[str, ...]) -> bool:
    return any(
        root == "" or relative == root or relative.startswith(f"{root}/")
        for root in roots
    )


def _tracked_root_prefixes(
    relative: str, tracked_paths: set[str]
) -> set[str]:
    root_parts = tuple(relative.split("/"))
    candidates: set[str] = set()
    for tracked_path in tracked_paths:
        tracked_parts = tuple(tracked_path.split("/"))
        if len(tracked_parts) < len(root_parts):
            continue
        candidates.add("/".join(tracked_parts[: len(root_parts)]))
    return candidates


def _canonical_existing_root(
    repo: Path,
    path: Path,
    relative: str,
    tracked_paths: set[str],
) -> str:
    if os.name != "nt":
        return relative
    same_paths: set[str] = set()
    for candidate in _tracked_root_prefixes(relative, tracked_paths):
        candidate_path = repo.joinpath(*candidate.split("/"))
        try:
            if candidate_path.exists() and os.path.samefile(path, candidate_path):
                same_paths.add(candidate)
        except OSError as error:
            raise ValueError(
                f"source root identity cannot be resolved: {path}"
            ) from error
    if len(same_paths) > 1:
        raise ValueError(
            f"ambiguous case-colliding Git source roots for {relative!r}"
        )
    return same_paths.pop() if same_paths else relative


def _ascii_lower(value: str) -> str | None:
    try:
        value.encode("ascii", errors="strict")
    except UnicodeEncodeError:
        return None
    return value.lower()


def _canonical_missing_root(
    relative: str, tracked_paths: set[str]
) -> str | None:
    candidates = _tracked_root_prefixes(relative, tracked_paths)
    exact = {candidate for candidate in candidates if candidate == relative}
    if len(exact) == 1:
        return exact.pop()
    if os.name != "nt":
        return None
    folded = _ascii_lower(relative)
    if folded is None:
        return None
    matching = {
        candidate
        for candidate in candidates
        if _ascii_lower(candidate) == folded
    }
    return matching.pop() if len(matching) == 1 else None


def _resolve_inside_repo(repo: Path, path: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"source path cannot be resolved: {path}") from error
    if not _is_within_repo(repo, resolved):
        raise ValueError(f"source path escapes the repository: {path}")
    return resolved


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _is_reparse_or_symlink(path_stat: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    file_attributes = getattr(path_stat, "st_file_attributes", 0)
    return stat.S_ISLNK(path_stat.st_mode) or bool(file_attributes & reparse_flag)


def _scan_source_path(
    repo: Path,
    path: Path,
    relative: str,
    files: set[str],
    active_directories: set[str],
) -> None:
    if relative and _is_excluded_source_path(relative):
        return
    path_stat = _lstat(path)
    if path_stat is None:
        raise ValueError(f"source root does not exist: {path}")
    resolved = _resolve_inside_repo(repo, path)
    is_link = _is_reparse_or_symlink(path_stat)
    if is_link:
        raise ValueError(f"source symlink or reparse point is not regular: {path}")

    if resolved.is_file():
        if relative:
            relative.encode("utf-8", errors="strict")
            files.add(relative)
        return
    if not resolved.is_dir():
        raise ValueError(f"source path is not a regular file or directory: {path}")

    directory_key = os.path.normcase(str(resolved))
    if directory_key in active_directories:
        raise ValueError(f"source directory link cycle: {path}")
    active_directories.add(directory_key)
    try:
        entries = sorted(
            os.scandir(path),
            key=lambda entry: entry.name.encode("utf-8", errors="strict"),
        )
        for entry in entries:
            child = Path(entry.path)
            child_relative = (
                f"{relative}/{entry.name}" if relative else entry.name
            )
            if _is_excluded_source_path(child_relative):
                continue
            _scan_source_path(
                repo,
                child,
                child_relative,
                files,
                active_directories,
            )
    finally:
        active_directories.remove(directory_key)


def _normalize_source_roots(
    repo: Path,
    source_roots: Sequence[Path],
    tracked_paths: set[str],
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    if isinstance(source_roots, (str, bytes)) or not isinstance(
        source_roots, Sequence
    ):
        raise TypeError("source_roots must be a sequence of Path values")
    if not source_roots:
        raise ValueError("source_roots must not be empty")
    paths: dict[str, Path] = {}
    for root in source_roots:
        if not isinstance(root, Path):
            raise TypeError("each source root must be a Path")
        combined = root if root.is_absolute() else repo / root
        lexical = Path(os.path.abspath(combined))
        if not _is_within_repo(repo, lexical):
            raise ValueError(f"source root is outside the repository: {root}")
        root_stat = _lstat(lexical)
        if root_stat is None:
            relative = _repo_relative(repo, lexical)
            resolved = lexical.resolve(strict=False)
            if not _is_within_repo(repo, resolved):
                raise ValueError(f"source root escapes the repository: {root}")
            canonical_relative = _canonical_missing_root(
                relative, tracked_paths
            )
            if canonical_relative is None:
                raise ValueError(f"source root does not exist: {root}")
            relative = canonical_relative
            lexical = repo.joinpath(*relative.split("/"))
        else:
            resolved = _resolve_inside_repo(repo, lexical)
            if _is_reparse_or_symlink(root_stat):
                raise ValueError(
                    f"source root symlink or reparse point is not regular: {root}"
                )
            lexical = resolved
            relative = _canonical_existing_root(
                repo,
                lexical,
                _repo_relative(repo, lexical),
                tracked_paths,
            )
        if relative and _is_excluded_source_path(relative):
            raise ValueError(f"source root is excluded by literal policy: {root}")
        paths[relative] = lexical
    ordered_relatives = tuple(
        sorted(paths, key=lambda value: value.encode("utf-8", errors="strict"))
    )
    return tuple(paths[relative] for relative in ordered_relatives), ordered_relatives


def _collect_source_inputs(
    repo: Path,
    source_roots: Sequence[Path],
) -> tuple[
    Path,
    bytes,
    dict[str, tuple[str, str]],
    dict[str, tuple[str, str]],
    set[str],
    set[str],
]:
    resolved_repo = Path(repo).resolve(strict=True)
    if (
        _git_bytes(resolved_repo, "rev-parse", "--is-inside-work-tree").strip()
        != b"true"
    ):
        raise ValueError("repo must be a Git worktree root")
    if _git_bytes(resolved_repo, "rev-parse", "--show-prefix").strip():
        raise ValueError("repo must be the Git worktree root")

    head = _git_bytes(resolved_repo, "rev-parse", "HEAD").strip()
    try:
        head_text = head.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("Git HEAD must be ASCII hexadecimal") from error
    if _COMMIT_PATTERN.fullmatch(head_text) is None:
        raise ValueError("Git HEAD must be a full lowercase commit")

    head_entries = _parse_tree_records(
        _git_bytes(resolved_repo, "ls-tree", "-r", "-z", "HEAD")
    )
    index_entries = _parse_index_records(
        _git_bytes(resolved_repo, "ls-files", "--stage", "-z")
    )
    all_tracked_paths = head_entries.keys() | index_entries.keys()
    roots, root_relatives = _normalize_source_roots(
        resolved_repo, source_roots, set(all_tracked_paths)
    )
    tracked_paths = {
        relative
        for relative in all_tracked_paths
        if _path_is_in_source_roots(relative, root_relatives)
        and not _is_excluded_source_path(relative)
    }
    for relative in tracked_paths:
        head_entry = head_entries.get(relative)
        index_entry = index_entries.get(relative)
        if head_entry is not None and head_entry[0] not in ("100644", "100755"):
            raise ValueError(
                f"non-regular Git mode {head_entry[0]} "
                f"for source path {relative!r}"
            )
        if index_entry is not None and index_entry[0] not in ("100644", "100755"):
            raise ValueError(
                f"non-regular Git mode {index_entry[0]} "
                f"for source path {relative!r}"
            )
    scanned_files: set[str] = set()
    for root, relative in zip(roots, root_relatives, strict=True):
        if _lstat(root) is None:
            continue
        _scan_source_path(
            resolved_repo,
            root,
            relative,
            scanned_files,
            set(),
        )
    return (
        resolved_repo,
        head,
        head_entries,
        index_entries,
        tracked_paths,
        scanned_files,
    )


def _effective_executable_mode(
    path: Path,
    relative: str,
    head_entries: Mapping[str, tuple[str, str]],
    index_entries: Mapping[str, tuple[str, str]],
) -> bool:
    index_entry = index_entries.get(relative)
    if os.name == "nt":
        head_entry = head_entries.get(relative)
        if index_entry is not None:
            effective_mode = index_entry[0]
        elif head_entry is not None:
            effective_mode = head_entry[0]
        else:
            effective_mode = None
        return effective_mode == "100755"
    return bool(path.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))


def _update_source_snapshot_frame(
    digest: Any,
    marker: bytes,
    executable: bool | None,
    content: bytes | None,
) -> None:
    digest.update(marker)
    if content is None:
        digest.update(b"-")
        digest.update((0).to_bytes(8, "big"))
        digest.update(b"\0" * 32)
        return
    digest.update(b"X" if executable else b"R")
    digest.update(len(content).to_bytes(8, "big"))
    digest.update(hashlib.sha256(content).digest())


def _source_snapshot_sha256(
    resolved_repo: Path,
    head: bytes,
    head_entries: Mapping[str, tuple[str, str]],
    index_entries: Mapping[str, tuple[str, str]],
    tracked_paths: set[str],
    scanned_files: set[str],
    *,
    clean_head: bool,
) -> str:
    effective_paths = (
        tracked_paths & head_entries.keys()
        if clean_head
        else tracked_paths | scanned_files
    )
    digest = hashlib.sha256()
    digest.update(_SOURCE_TREE_MAGIC)
    digest.update(head)
    digest.update(b"\0")
    index_blob_cache: dict[str, bytes] = {}
    for relative in sorted(
        effective_paths, key=lambda value: value.encode("utf-8", errors="strict")
    ):
        raw_path = relative.encode("utf-8", errors="strict")
        path = resolved_repo.joinpath(*relative.split("/"))
        digest.update(len(raw_path).to_bytes(8, "big"))
        digest.update(raw_path)
        if clean_head:
            head_mode, object_id = head_entries[relative]
            if object_id not in index_blob_cache:
                index_blob_cache[object_id] = _git_bytes(
                    resolved_repo, "cat-file", "blob", object_id
                )
            content = index_blob_cache[object_id]
            executable = head_mode == "100755"
            _update_source_snapshot_frame(
                digest, b"P", executable, content
            )
            _update_source_snapshot_frame(
                digest, b"P", executable, content
            )
            continue

        path_stat = _lstat(path)
        is_present_file = False
        if path_stat is not None:
            resolved = _resolve_inside_repo(resolved_repo, path)
            is_present_file = resolved.is_file()
        index_entry = index_entries.get(relative)
        if index_entry is not None:
            index_mode, object_id = index_entry
            if object_id not in index_blob_cache:
                index_blob_cache[object_id] = _git_bytes(
                    resolved_repo, "cat-file", "blob", object_id
                )
            _update_source_snapshot_frame(
                digest,
                b"P",
                index_mode == "100755",
                index_blob_cache[object_id],
            )
        elif relative in head_entries:
            _update_source_snapshot_frame(digest, b"D", None, None)
        else:
            _update_source_snapshot_frame(digest, b"U", None, None)
        if is_present_file:
            content = path.read_bytes()
            executable = _effective_executable_mode(
                path, relative, head_entries, index_entries
            )
            _update_source_snapshot_frame(digest, b"P", executable, content)
        else:
            _update_source_snapshot_frame(digest, b"D", None, None)
    return digest.hexdigest()


def source_tree_sha256(repo: Path, source_roots: Sequence[Path]) -> str:
    (
        resolved_repo,
        head,
        head_entries,
        index_entries,
        tracked_paths,
        scanned_files,
    ) = _collect_source_inputs(repo, source_roots)
    return _source_snapshot_sha256(
        resolved_repo,
        head,
        head_entries,
        index_entries,
        tracked_paths,
        scanned_files,
        clean_head=False,
    )


def collect_runtime_metadata() -> dict[str, object]:
    return {
        "python_executable": os.path.realpath(sys.executable),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "os": platform.platform(),
    }


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _installed_distributions() -> list[dict[str, str]]:
    records = []
    for distribution in importlib_metadata.distributions():
        name = distribution.metadata["Name"] or distribution.name
        records.append(
            {
                "name": _normalize_distribution_name(name),
                "version": str(distribution.version),
            }
        )
    return sorted(records, key=lambda item: (item["name"], item["version"]))


def _gpu_driver_version() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    versions = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    return ",".join(versions) if versions else None


def _gpu_fingerprint() -> dict[str, object]:
    available = torch.cuda.is_available()
    devices = []
    if available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "compute_capability": [
                        int(properties.major),
                        int(properties.minor),
                    ],
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    return {
        "available": available,
        "device_count": len(devices),
        "devices": devices,
        "driver_version": _gpu_driver_version() if available else None,
    }


def collect_environment_fingerprint() -> dict[str, object]:
    """Return a deterministic, canonical, secret-free environment-lock payload."""
    implementation_version = sys.implementation.version
    return {
        "installed_distributions": _installed_distributions(),
        "numerical_environment": {
            name: os.environ.get(name) for name in NUMERICAL_ENV_ALLOWLIST
        },
        "platform": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "release": platform.release(),
            "system": platform.system(),
            "version": platform.version(),
        },
        "python": {
            "abi": {
                "abiflags": getattr(sys, "abiflags", ""),
                "cache_tag": sys.implementation.cache_tag,
                "soabi": sysconfig.get_config_var("SOABI"),
            },
            "implementation": platform.python_implementation(),
            "implementation_version": ".".join(
                str(part)
                for part in (
                    implementation_version.major,
                    implementation_version.minor,
                    implementation_version.micro,
                )
            ),
            "version": platform.python_version(),
        },
        "pytorch": {
            "cuda_build": torch.version.cuda,
            "cudnn_version": (
                torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available()
                else None
            ),
            "version": str(torch.__version__),
        },
        "gpu": _gpu_fingerprint(),
    }
