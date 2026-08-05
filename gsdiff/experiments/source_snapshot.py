"""Immutable source snapshots materialized only from claimed Git objects."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import stat
import subprocess
from types import MappingProxyType
from typing import Mapping
import uuid

from gsdiff.data._artifact_persistence import (
    _promote_exact_directory_no_clobber,
)

from .execution import (
    _directory_identity,
    _read_stable_regular_bytes,
    _reject_linked_ancestors,
    _resolved_real_directory,
    _selected_source_files,
)
from .identity import canonical_json_bytes, _git_command, _git_read_environment
from ._owned_tree import cleanup_pinned_owned_tree
from ._windows_paths import windows_component_collision_key


_COMMIT = re.compile(r"[0-9a-f]{40}\Z", re.ASCII)
_BLOB = re.compile(r"[0-9a-f]{40,64}\Z", re.ASCII)
_ALLOWED_MODES = frozenset({"100644", "100755"})
@dataclass(frozen=True)
class SourceSnapshot:
    root: Path
    commit: str
    snapshot_sha256: str
    inventory: tuple[Mapping[str, object], ...]


def selected_source_evidence(
    snapshot: SourceSnapshot,
) -> tuple[tuple[Mapping[str, str], ...], str]:
    """Return the exact inventory/hash consumed by the method materializer."""
    verified = verify_source_snapshot(snapshot)
    inventory: list[Mapping[str, str]] = []
    for relative, path in _selected_source_files(verified.root):
        payload = _read_stable_regular_bytes(
            path,
            noun=f"selected snapshot source {relative.as_posix()!r}",
        )
        inventory.append(
            MappingProxyType(
                {
                    "path": relative.as_posix(),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        )
    frozen = tuple(inventory)
    digest = hashlib.sha256(
        canonical_json_bytes([dict(entry) for entry in frozen])
    ).hexdigest()
    return frozen, digest


def materialize_source_snapshot(
    repo_root: Path,
    artifact_root: Path,
    commit: str,
    source_roots: tuple[Path, ...],
) -> SourceSnapshot:
    """Build or strictly reuse a content-addressed claimed-commit snapshot."""
    repo = _resolved_real_directory(repo_root, noun="Git source repository")
    if type(commit) is not str or _COMMIT.fullmatch(commit) is None:
        raise ValueError("source snapshot commit must be a full lowercase Git commit")
    roots = _validate_source_roots(source_roots)
    inventory, payloads = _git_object_inventory(repo, commit, roots)
    identity = {
        "schema": "source-snapshot-identity-v1",
        "commit": commit,
        "inventory": inventory,
    }
    snapshot_sha256 = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    manifest = {
        "schema": "source-snapshot-v1",
        "commit": commit,
        "snapshot_sha256": snapshot_sha256,
        "inventory": inventory,
    }
    artifact = _ensure_real_directory_tree(artifact_root.absolute())
    snapshots = _ensure_real_directory_tree(artifact / "source-snapshots")
    final = snapshots / snapshot_sha256
    if os.path.lexists(final):
        return verify_source_snapshot(final, expected_manifest=manifest)

    stage = snapshots / f".tmp-{uuid.uuid4().hex}"
    os.mkdir(stage, 0o700)
    stage_identity = _directory_identity(stage, noun="source snapshot staging")
    try:
        for entry in inventory:
            relative = str(entry["path"])
            destination = stage.joinpath(*relative.split("/"))
            destination.parent.mkdir(parents=True, exist_ok=True)
            _reject_linked_ancestors(destination, noun="source snapshot destination")
            payload = payloads[relative]
            with destination.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise ValueError("source snapshot payload hash changed before persistence")
        manifest_path = stage / "source-snapshot.json"
        with manifest_path.open("xb") as stream:
            stream.write(canonical_json_bytes(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        verify_source_snapshot(stage, expected_manifest=manifest)
        try:
            _promote_exact_directory_no_clobber(
                stage,
                final,
                expected_device=stage_identity.device,
                expected_inode=stage_identity.inode,
            )
        except FileExistsError:
            winner = verify_source_snapshot(final, expected_manifest=manifest)
            cleanup_pinned_owned_tree(
                stage_identity,
                noun="source snapshot staging",
            )
            return winner
        return verify_source_snapshot(final, expected_manifest=manifest)
    except BaseException as error:
        if os.path.lexists(stage):
            try:
                cleanup_pinned_owned_tree(
                    stage_identity,
                    noun="source snapshot staging",
                )
            except (OSError, ValueError) as cleanup_error:
                error.add_note(
                    "Pinned source snapshot staging cleanup refused: "
                    f"{cleanup_error}"
                )
        raise


def verify_source_snapshot(
    snapshot: SourceSnapshot | Path,
    *,
    expected_manifest: Mapping[str, object] | None = None,
) -> SourceSnapshot:
    root = snapshot.root if type(snapshot) is SourceSnapshot else Path(snapshot)
    root = _resolved_real_directory(root, noun="source snapshot")
    manifest_bytes = _read_stable_regular_bytes(
        root / "source-snapshot.json",
        noun="source snapshot manifest",
    )
    manifest = _load_canonical_manifest(manifest_bytes)
    staging_name = root.name.startswith(".tmp-")
    if root.name != manifest["snapshot_sha256"] and not (
        expected_manifest is not None and staging_name
    ):
        raise ValueError("source snapshot directory name disagrees with identity")
    if expected_manifest is not None and canonical_json_bytes(manifest) != canonical_json_bytes(
        dict(expected_manifest)
    ):
        raise ValueError("source snapshot manifest disagrees with claimed Git inventory")
    inventory = manifest["inventory"]
    assert type(inventory) is list
    expected_files = {"source-snapshot.json"}
    expected_directories: set[str] = set()
    for entry in inventory:
        assert type(entry) is dict
        relative = str(entry["path"])
        expected_files.add(relative)
        parent = Path(relative).parent
        while parent != Path("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
        payload = _read_stable_regular_bytes(
            root.joinpath(*relative.split("/")),
            noun=f"source snapshot file {relative!r}",
        )
        if len(payload) != entry["size_bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError("source snapshot file bytes changed")
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in directories:
            path = current_path / name
            info = os.lstat(path)
            if (
                not stat.S_ISDIR(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or getattr(info, "st_file_attributes", 0) & 0x400
            ):
                raise ValueError("source snapshot contains a linked directory")
            observed_directories.add(path.relative_to(root).as_posix())
        for name in files:
            path = current_path / name
            info = os.lstat(path)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("source snapshot contains a non-regular file")
            observed_files.add(path.relative_to(root).as_posix())
    if (
        observed_files != expected_files
        or observed_directories != expected_directories
    ):
        raise ValueError("source snapshot inventory has extra or missing paths")
    frozen_inventory = tuple(
        MappingProxyType(dict(entry)) for entry in inventory
    )
    return SourceSnapshot(
        root=root,
        commit=str(manifest["commit"]),
        snapshot_sha256=str(manifest["snapshot_sha256"]),
        inventory=frozen_inventory,
    )


def verify_source_snapshot_against_git(
    snapshot: SourceSnapshot,
    *,
    trusted_repo_root: Path,
    source_roots: tuple[Path, ...],
) -> SourceSnapshot:
    """Reconstruct a snapshot claim from one trusted repository's Git objects."""
    verified = verify_source_snapshot(snapshot)
    repo = _resolved_real_directory(
        trusted_repo_root,
        noun="trusted Git source repository",
    )
    roots = _validate_source_roots(source_roots)
    inventory, _payloads = _git_object_inventory(
        repo,
        verified.commit,
        roots,
    )
    identity = {
        "schema": "source-snapshot-identity-v1",
        "commit": verified.commit,
        "inventory": inventory,
    }
    expected_manifest = {
        "schema": "source-snapshot-v1",
        "commit": verified.commit,
        "snapshot_sha256": hashlib.sha256(
            canonical_json_bytes(identity)
        ).hexdigest(),
        "inventory": inventory,
    }
    return verify_source_snapshot(
        verified,
        expected_manifest=expected_manifest,
    )


def _git_object_inventory(
    repo: Path,
    commit: str,
    roots: tuple[str, ...],
) -> tuple[list[dict[str, object]], dict[str, bytes]]:
    git_environment = _git_read_environment()
    object_type = subprocess.run(
        _git_command("cat-file", "-t", commit),
        cwd=repo,
        capture_output=True,
        check=False,
        env=git_environment,
    )
    if object_type.returncode != 0 or object_type.stdout.strip() != b"commit":
        raise ValueError("source snapshot identity must name a Git commit object")
    command = _git_command(
        "--literal-pathspecs",
        "ls-tree",
        "-rlz",
        "--full-tree",
        commit,
        "--",
        *roots,
    )
    result = subprocess.run(
        command,
        cwd=repo,
        capture_output=True,
        check=False,
        env=git_environment,
    )
    if result.returncode != 0:
        raise ValueError("cannot enumerate claimed Git commit")
    entries: list[dict[str, object]] = []
    payloads: dict[str, bytes] = {}
    collision_keys: set[tuple[str, ...]] = set()
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        header, separator, raw_path = record.partition(b"\t")
        fields = header.split()
        if not separator or len(fields) != 4:
            raise ValueError("Git tree inventory record is malformed")
        mode, object_type, raw_oid, raw_size = fields
        if mode.decode("ascii", errors="strict") not in _ALLOWED_MODES or object_type != b"blob":
            raise ValueError("Git tree contains a symlink, gitlink, or unsupported mode")
        try:
            path = raw_path.decode("utf-8", errors="strict")
            mode_text = mode.decode("ascii", errors="strict")
            oid = raw_oid.decode("ascii", errors="strict")
            size = int(raw_size.decode("ascii", errors="strict"))
        except (UnicodeError, ValueError) as error:
            raise ValueError("Git tree inventory is not canonical text") from error
        _validate_git_path(path, roots)
        collision = _windows_collision_key(path)
        if collision in collision_keys:
            raise ValueError("Git tree has a case or Win32 path collision")
        collision_keys.add(collision)
        if _BLOB.fullmatch(oid) is None or size < 0:
            raise ValueError("Git blob identity or size is malformed")
        payload = subprocess.run(
            _git_command("cat-file", "blob", oid),
            cwd=repo,
            capture_output=True,
            check=False,
            env=git_environment,
        )
        if payload.returncode != 0 or len(payload.stdout) != size:
            raise ValueError("cannot read exact claimed Git blob")
        digest = hashlib.sha256(payload.stdout).hexdigest()
        entries.append(
            {
                "path": path,
                "mode": mode_text,
                "git_blob": oid,
                "sha256": digest,
                "size_bytes": size,
            }
        )
        payloads[path] = payload.stdout
    entries.sort(key=lambda item: str(item["path"]).encode("utf-8"))
    if not entries:
        raise ValueError("claimed Git source inventory is empty")
    return entries, payloads


def _validate_source_roots(source_roots: tuple[Path, ...]) -> tuple[str, ...]:
    if type(source_roots) is not tuple or not source_roots:
        raise TypeError("source roots must be a nonempty exact tuple")
    roots: list[str] = []
    for root in source_roots:
        if not isinstance(root, Path) or root.is_absolute() or not root.parts:
            raise ValueError("source root must be a relative exact Path")
        text = root.as_posix()
        _validate_git_path(text, (text,))
        if text in roots:
            raise ValueError("source roots contain duplicates")
        roots.append(text)
    return tuple(roots)


def _validate_git_path(path: str, roots: tuple[str, ...]) -> None:
    _validate_snapshot_relative_path(path)
    if not any(path == root or path.startswith(root + "/") for root in roots):
        raise ValueError("Git source path escapes approved roots")


def _validate_snapshot_relative_path(path: str) -> None:
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or "\x00" in path
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ValueError("source snapshot path is not canonical relative POSIX text")
    _windows_collision_key(path)


def _windows_collision_key(path: str) -> tuple[str, ...]:
    key: list[str] = []
    for part in path.split("/"):
        key.append(windows_component_collision_key(part))
    return tuple(key)


def _load_canonical_manifest(raw: bytes) -> dict[str, object]:
    import json

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("source snapshot manifest has duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {token}")
            ),
        )
    except (UnicodeError, ValueError) as error:
        raise ValueError("source snapshot manifest is malformed") from error
    if type(value) is not dict or set(value) != {
        "schema",
        "commit",
        "snapshot_sha256",
        "inventory",
    }:
        raise ValueError("source snapshot manifest shape is invalid")
    if value["schema"] != "source-snapshot-v1" or canonical_json_bytes(value) != raw:
        raise ValueError("source snapshot manifest is not canonical")
    commit = value["commit"]
    snapshot_sha256 = value["snapshot_sha256"]
    inventory = value["inventory"]
    if type(commit) is not str or _COMMIT.fullmatch(commit) is None:
        raise ValueError("source snapshot manifest commit is invalid")
    if type(snapshot_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", snapshot_sha256) is None:
        raise ValueError("source snapshot identity is invalid")
    if type(inventory) is not list or not inventory:
        raise ValueError("source snapshot inventory is invalid")
    expected_identity = {
        "schema": "source-snapshot-identity-v1",
        "commit": commit,
        "inventory": inventory,
    }
    if hashlib.sha256(canonical_json_bytes(expected_identity)).hexdigest() != snapshot_sha256:
        raise ValueError("source snapshot identity hash is invalid")
    prior = None
    collisions: set[tuple[str, ...]] = set()
    for entry in inventory:
        if type(entry) is not dict or set(entry) != {
            "path",
            "mode",
            "git_blob",
            "sha256",
            "size_bytes",
        }:
            raise ValueError("source snapshot inventory entry is malformed")
        path = entry["path"]
        if type(path) is not str:
            raise ValueError("source snapshot path is invalid")
        _validate_snapshot_relative_path(path)
        collision = _windows_collision_key(path)
        if collision in collisions:
            raise ValueError("source snapshot inventory has a path collision")
        collisions.add(collision)
        if prior is not None and prior.encode("utf-8") >= path.encode("utf-8"):
            raise ValueError("source snapshot inventory is not strictly sorted")
        prior = path
        if (
            entry["mode"] not in _ALLOWED_MODES
            or type(entry["git_blob"]) is not str
            or _BLOB.fullmatch(entry["git_blob"]) is None
            or type(entry["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
            or type(entry["size_bytes"]) is not int
            or entry["size_bytes"] < 0
        ):
            raise ValueError("source snapshot inventory entry fields are invalid")
    return value


def _ensure_real_directory_tree(path: Path) -> Path:
    _reject_linked_ancestors(path, noun="source snapshot directory")
    missing: list[Path] = []
    current = path
    while not os.path.lexists(current):
        missing.append(current)
        if current.parent == current:
            raise ValueError("source snapshot directory has no real ancestor")
        current = current.parent
    _resolved_real_directory(current, noun="source snapshot ancestor")
    for directory in reversed(missing):
        try:
            os.mkdir(directory, 0o700)
        except FileExistsError:
            _resolved_real_directory(
                directory,
                noun="concurrent source snapshot directory winner",
            )
        _resolved_real_directory(directory, noun="source snapshot directory")
    return _resolved_real_directory(path, noun="source snapshot directory")
