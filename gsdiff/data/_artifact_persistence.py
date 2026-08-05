"""Read-only verification of complete dataset artifact directories."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
import re
import stat
import tempfile
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ._artifact_bundle import (
    MAX_DATASET_MANIFEST_BYTES,
    MAX_DATASET_NPZ_BYTES,
    MAX_DATASET_PREVIEW_BYTES,
    _native_copy,
    _quantized_preview,
    build_dataset_manifest,
    build_dataset_payloads,
    dataset_manifest_bytes,
    parse_dataset_manifest_bytes,
    verify_dataset_payload_bytes,
)
from ._artifact_dataset import _validate_acquisition_identity
from ._artifact_identity import (
    ArtifactValidationError,
    canonical_json_bytes,
    deep_freeze_json,
    readonly_array,
    validate_sha256,
)
from ._artifact_io import (
    DirectoryInventory,
    SafeFileSnapshot,
    _is_link_or_reparse,
    _reject_linked_ancestors,
    _stat_signature,
    _validate_regular_snapshot_stat,
    capture_directory_inventory,
    read_safe_file_snapshot,
    verify_directory_inventory,
    verify_safe_file_snapshot,
)
from ._artifact_models import EvaluationTruth, SPIAcquisitionData
from ._corrected_generation import (
    CorrectedDataset,
    validate_corrected_truth,
)


_MANIFEST_NAME = "dataset-manifest.json"
_PAYLOAD_NAMES = (
    "measurements.npz",
    "evaluation-truth.npz",
    "preview.png",
)
_DIRECTORY_NAMES = frozenset({_MANIFEST_NAME, *_PAYLOAD_NAMES})
_MOVEFILE_WRITE_THROUGH = 0x8
_RENAME_NOREPLACE = 1
_FILE_BASIC_INFO_CLASS = 0
_FILE_RENAME_INFO_CLASS = 3
_FILE_ID_INFO_CLASS = 18
_FILE_DISPOSITION_INFO_EX_CLASS = 21
_FILE_ATTRIBUTE_READONLY = 0x1
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_FILE_DISPOSITION_FLAG_DELETE = 0x1
_FILE_DISPOSITION_FLAG_IGNORE_READONLY_ATTRIBUTE = 0x10
_CANONICAL_DATASET_DIRECTORY = re.compile(r"[0-9a-f]{64}\Z")
_STAGING_DATASET_DIRECTORY = re.compile(
    r"\.[0-9a-f]{64}\.staging-[A-Za-z0-9_-]+\Z"
)
_REJECTED_DATASET_DIRECTORY = re.compile(
    r"\.[0-9a-f]{64}\.rejected-[0-9a-f]{24}\Z"
)
_DirectorySignature = tuple[int, ...]


class _WindowsFileId128(ctypes.Structure):
    _fields_ = [("identifier", ctypes.c_ubyte * 16)]


class _WindowsFileIdInfo(ctypes.Structure):
    _fields_ = [
        ("volume_serial_number", ctypes.c_uint64),
        ("file_id", _WindowsFileId128),
    ]


class _WindowsRenameInfo(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("root_directory", ctypes.c_void_p),
        ("file_name_length", ctypes.c_uint32),
        ("file_name", ctypes.c_wchar * 1),
    ]


class _WindowsBasicInfo(ctypes.Structure):
    _fields_ = [
        ("creation_time", ctypes.c_int64),
        ("last_access_time", ctypes.c_int64),
        ("last_write_time", ctypes.c_int64),
        ("change_time", ctypes.c_int64),
        ("file_attributes", ctypes.c_uint32),
    ]


class _WindowsDispositionInfoEx(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint32)]


@dataclass(frozen=True)
class DatasetPayloadEvidence:
    """Hash and byte length observed from one bounded safe snapshot."""

    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class VerifiedDatasetDirectory:
    """Evidence returned after a complete final-directory verification.

    ``manifest_externally_anchored`` is false when no expected manifest SHA
    was supplied. In that case the result establishes internal consistency
    only; it does not claim resistance to a coordinated manifest rewrite.
    """

    dataset_dir: Path
    dataset_identity_sha256: str
    dataset_manifest_sha256: str
    manifest_externally_anchored: bool
    expected_generated_verified: bool
    payload_evidence: Mapping[str, DatasetPayloadEvidence]
    acquisition: SPIAcquisitionData
    truth: EvaluationTruth
    preview: np.ndarray
    _manifest: Mapping[str, object] = field(repr=False)

    def __post_init__(self) -> None:
        frozen_manifest = deep_freeze_json(self._manifest)
        if not isinstance(frozen_manifest, Mapping):
            raise TypeError("verified manifest must be a mapping")
        evidence = MappingProxyType(
            {
                name: self.payload_evidence[name]
                for name in sorted(self.payload_evidence)
            }
        )
        object.__setattr__(self, "_manifest", frozen_manifest)
        object.__setattr__(self, "payload_evidence", evidence)
        object.__setattr__(
            self, "preview", readonly_array(self.preview, "preview")
        )

    @property
    def manifest(self) -> dict[str, object]:
        """Return a fresh exact-native copy of the canonical manifest."""

        native = _native_copy(self._manifest)
        if type(native) is not dict:
            raise RuntimeError("verified manifest is not an object")
        return native


@dataclass(frozen=True)
class DatasetPublication:
    """Result of creating or safely reusing one dataset directory."""

    status: str
    dataset_dir: Path
    dataset_manifest_sha256: str
    verified: VerifiedDatasetDirectory


@dataclass(frozen=True)
class DatasetDirectoryDiscovery:
    """Read-only, recheckable snapshot of physical dataset directories."""

    artifact_root: Path
    datasets_dir: Path
    datasets_dir_exists: bool
    canonical_directories: tuple[Path, ...]
    stale_staging_directories: tuple[Path, ...]
    rejected_directories: tuple[Path, ...]
    _artifact_root_exists: bool = field(repr=False)
    _artifact_root_signature: _DirectorySignature | None = field(
        repr=False
    )
    _datasets_dir_signature: _DirectorySignature | None = field(
        repr=False
    )
    _entry_signatures: tuple[
        tuple[str, _DirectorySignature], ...
    ] = field(repr=False)


@dataclass
class _OwnedStage:
    path: Path
    device: int
    inode: int
    leaves: dict[str, tuple[int, int, int, int, int, int]] = field(
        default_factory=dict
    )


def _discovery_directory_signature(
    path: Path,
    noun: str,
) -> _DirectorySignature:
    info = _validate_real_directory(path, noun)
    return (
        *_stat_signature(info),
        info.st_mode,
        getattr(info, "st_file_attributes", 0),
    )


def _classify_dataset_directory_name(name: str) -> str:
    if _CANONICAL_DATASET_DIRECTORY.fullmatch(name):
        return "canonical"
    if _STAGING_DATASET_DIRECTORY.fullmatch(name):
        return "staging"
    if _REJECTED_DATASET_DIRECTORY.fullmatch(name):
        return "rejected"
    raise ArtifactValidationError(
        f"unexpected dataset directory entry: {name}"
    )


def _reject_non_directory_discovery_ancestors(path: Path) -> None:
    current = path.absolute()
    while True:
        if os.path.lexists(current):
            try:
                info = os.lstat(current)
            except OSError as error:
                raise ArtifactValidationError(
                    "cannot inspect dataset discovery ancestor"
                ) from error
            if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
                raise ArtifactValidationError(
                    "dataset discovery ancestor must be a real directory"
                )
        parent = current.parent
        if parent == current:
            return
        current = parent


def discover_dataset_directories(
    artifact_root: Path,
) -> DatasetDirectoryDiscovery:
    """Discover physical dataset directories without creating or mutating."""

    if not isinstance(artifact_root, Path):
        raise TypeError("artifact_root must be a Path")
    absolute_root = artifact_root.absolute()
    datasets_dir = absolute_root / "datasets"
    _reject_non_directory_discovery_ancestors(datasets_dir)
    _reject_linked_ancestors(datasets_dir)

    artifact_root_exists = os.path.lexists(absolute_root)
    artifact_root_signature: _DirectorySignature | None = None
    if artifact_root_exists:
        artifact_root_signature = _discovery_directory_signature(
            absolute_root,
            "artifact root",
        )

    if not os.path.lexists(datasets_dir):
        _reject_linked_ancestors(datasets_dir)
        if os.path.lexists(datasets_dir):
            raise ArtifactValidationError(
                "dataset directory discovery changed during scan"
            )
        if artifact_root_exists:
            observed_root = _discovery_directory_signature(
                absolute_root,
                "artifact root",
            )
            if observed_root != artifact_root_signature:
                raise ArtifactValidationError(
                    "dataset directory discovery changed during scan"
                )
        elif os.path.lexists(absolute_root):
            raise ArtifactValidationError(
                "dataset directory discovery changed during scan"
            )
        return DatasetDirectoryDiscovery(
            artifact_root=absolute_root,
            datasets_dir=datasets_dir,
            datasets_dir_exists=False,
            canonical_directories=(),
            stale_staging_directories=(),
            rejected_directories=(),
            _artifact_root_exists=artifact_root_exists,
            _artifact_root_signature=artifact_root_signature,
            _datasets_dir_signature=None,
            _entry_signatures=(),
        )

    datasets_signature = _discovery_directory_signature(
        datasets_dir,
        "datasets directory",
    )
    if artifact_root_signature is None:
        raise ArtifactValidationError(
            "dataset directory discovery changed during scan"
        )
    if datasets_signature[0] != artifact_root_signature[0]:
        raise ArtifactValidationError(
            "datasets directory crosses an artifact filesystem boundary"
        )
    classified: dict[str, list[Path]] = {
        "canonical": [],
        "staging": [],
        "rejected": [],
    }
    entry_signatures: list[tuple[str, _DirectorySignature]] = []
    try:
        with os.scandir(datasets_dir) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
    except OSError as error:
        raise ArtifactValidationError(
            "cannot enumerate datasets directory"
        ) from error
    for entry in entries:
        role = _classify_dataset_directory_name(entry.name)
        path = datasets_dir / entry.name
        signature = _discovery_directory_signature(
            path,
            f"{role} dataset directory",
        )
        if signature[0] != datasets_signature[0]:
            raise ArtifactValidationError(
                f"{role} dataset directory crosses a filesystem boundary"
            )
        classified[role].append(path)
        entry_signatures.append((entry.name, signature))

    if (
        _discovery_directory_signature(
            datasets_dir,
            "datasets directory",
        )
        != datasets_signature
    ):
        raise ArtifactValidationError(
            "dataset directory discovery changed during scan"
        )
    try:
        with os.scandir(datasets_dir) as iterator:
            observed_names = tuple(
                entry.name
                for entry in sorted(
                    iterator,
                    key=lambda entry: entry.name,
                )
            )
    except OSError as error:
        raise ArtifactValidationError(
            "dataset directory discovery changed during scan"
        ) from error
    if observed_names != tuple(name for name, _ in entry_signatures):
        raise ArtifactValidationError(
            "dataset directory discovery changed during scan"
        )
    for name, expected_signature in entry_signatures:
        if (
            _discovery_directory_signature(
                datasets_dir / name,
                "dataset directory",
            )
            != expected_signature
        ):
            raise ArtifactValidationError(
                "dataset directory discovery changed during scan"
            )

    return DatasetDirectoryDiscovery(
        artifact_root=absolute_root,
        datasets_dir=datasets_dir,
        datasets_dir_exists=True,
        canonical_directories=tuple(classified["canonical"]),
        stale_staging_directories=tuple(classified["staging"]),
        rejected_directories=tuple(classified["rejected"]),
        _artifact_root_exists=artifact_root_exists,
        _artifact_root_signature=artifact_root_signature,
        _datasets_dir_signature=datasets_signature,
        _entry_signatures=tuple(entry_signatures),
    )


def verify_dataset_directory_discovery(
    discovery: DatasetDirectoryDiscovery,
) -> None:
    """Reject any physical change since a discovery snapshot was captured."""

    if type(discovery) is not DatasetDirectoryDiscovery:
        raise TypeError(
            "discovery must be an exact DatasetDirectoryDiscovery"
        )
    try:
        _reject_non_directory_discovery_ancestors(
            discovery.datasets_dir
        )
        _reject_linked_ancestors(discovery.datasets_dir)
    except ArtifactValidationError as error:
        raise ArtifactValidationError(
            "dataset directory discovery changed"
        ) from error
    root_exists = os.path.lexists(discovery.artifact_root)
    if root_exists != discovery._artifact_root_exists:
        raise ArtifactValidationError("dataset directory discovery changed")
    if root_exists:
        if (
            _discovery_directory_signature(
                discovery.artifact_root,
                "artifact root",
            )
            != discovery._artifact_root_signature
        ):
            raise ArtifactValidationError(
                "dataset directory discovery changed"
            )
    datasets_exists = os.path.lexists(discovery.datasets_dir)
    if datasets_exists != discovery.datasets_dir_exists:
        raise ArtifactValidationError("dataset directory discovery changed")
    if not datasets_exists:
        return
    if (
        _discovery_directory_signature(
            discovery.datasets_dir,
            "datasets directory",
        )
        != discovery._datasets_dir_signature
    ):
        raise ArtifactValidationError("dataset directory discovery changed")
    try:
        with os.scandir(discovery.datasets_dir) as iterator:
            observed_entries = sorted(
                iterator,
                key=lambda entry: entry.name,
            )
    except OSError as error:
        raise ArtifactValidationError(
            "dataset directory discovery changed"
        ) from error
    expected_names = tuple(
        name for name, _ in discovery._entry_signatures
    )
    if tuple(entry.name for entry in observed_entries) != expected_names:
        raise ArtifactValidationError("dataset directory discovery changed")
    for name, expected_signature in discovery._entry_signatures:
        try:
            observed_signature = _discovery_directory_signature(
                discovery.datasets_dir / name,
                "dataset directory",
            )
        except ArtifactValidationError as error:
            raise ArtifactValidationError(
                "dataset directory discovery changed"
            ) from error
        if observed_signature != expected_signature:
            raise ArtifactValidationError(
                "dataset directory discovery changed"
            )


def _stable_directory_identity(
    signature: _DirectorySignature,
) -> tuple[int, int, int, bool]:
    return (
        signature[0],
        signature[1],
        stat.S_IFMT(signature[6]),
        bool(signature[7] & _FILE_ATTRIBUTE_REPARSE_POINT),
    )


def _canonical_directory_names(datasets_dir: Path) -> tuple[str, ...]:
    try:
        with os.scandir(datasets_dir) as iterator:
            return tuple(
                entry.name
                for entry in sorted(
                    iterator,
                    key=lambda entry: entry.name,
                )
                if _classify_dataset_directory_name(entry.name)
                == "canonical"
            )
    except OSError as error:
        raise ArtifactValidationError(
            "canonical dataset directory discovery changed"
        ) from error


def _canonical_directory_signatures(
    datasets_dir: Path,
    *,
    datasets_device: int,
) -> tuple[tuple[str, _DirectorySignature], ...]:
    names = _canonical_directory_names(datasets_dir)
    signatures: list[tuple[str, _DirectorySignature]] = []
    for name in names:
        signature = _discovery_directory_signature(
            datasets_dir / name,
            "canonical dataset directory",
        )
        if signature[0] != datasets_device:
            raise ArtifactValidationError(
                "canonical dataset directory crosses a filesystem boundary"
            )
        signatures.append((name, signature))
    if _canonical_directory_names(datasets_dir) != names:
        raise ArtifactValidationError(
            "canonical dataset directory discovery changed"
        )
    for name, expected_signature in signatures:
        if (
            _discovery_directory_signature(
                datasets_dir / name,
                "canonical dataset directory",
            )
            != expected_signature
        ):
            raise ArtifactValidationError(
                "canonical dataset directory discovery changed"
            )
    if _canonical_directory_names(datasets_dir) != names:
        raise ArtifactValidationError(
            "canonical dataset directory discovery changed"
        )
    return tuple(signatures)


def verify_canonical_dataset_directory_discovery(
    discovery: DatasetDirectoryDiscovery,
) -> tuple[Path, ...]:
    """Recheck canonical entries while ignoring diagnostic directory churn.

    Existing canonical directories must retain their complete discovery
    signatures. Stable canonical additions are returned for full verification
    by the caller. Staging and rejected names and signatures are intentionally
    outside this recheck contract.
    """

    if type(discovery) is not DatasetDirectoryDiscovery:
        raise TypeError(
            "discovery must be an exact DatasetDirectoryDiscovery"
        )
    try:
        _reject_non_directory_discovery_ancestors(discovery.datasets_dir)
        _reject_linked_ancestors(discovery.datasets_dir)
    except ArtifactValidationError as error:
        raise ArtifactValidationError(
            "canonical dataset directory changed"
        ) from error
    if (
        os.path.lexists(discovery.artifact_root)
        != discovery._artifact_root_exists
        or os.path.lexists(discovery.datasets_dir)
        != discovery.datasets_dir_exists
    ):
        raise ArtifactValidationError(
            "canonical dataset directory changed"
        )
    if not discovery.datasets_dir_exists:
        return ()
    observed_root = _discovery_directory_signature(
        discovery.artifact_root,
        "artifact root",
    )
    observed_datasets = _discovery_directory_signature(
        discovery.datasets_dir,
        "datasets directory",
    )
    if (
        discovery._artifact_root_signature is None
        or discovery._datasets_dir_signature is None
        or _stable_directory_identity(observed_root)
        != _stable_directory_identity(discovery._artifact_root_signature)
        or _stable_directory_identity(observed_datasets)
        != _stable_directory_identity(discovery._datasets_dir_signature)
    ):
        raise ArtifactValidationError(
            "canonical dataset directory changed"
        )
    observed_entries = _canonical_directory_signatures(
        discovery.datasets_dir,
        datasets_device=observed_datasets[0],
    )
    if (
        _stable_directory_identity(
            _discovery_directory_signature(
                discovery.artifact_root,
                "artifact root",
            )
        )
        != _stable_directory_identity(observed_root)
        or _stable_directory_identity(
            _discovery_directory_signature(
                discovery.datasets_dir,
                "datasets directory",
            )
        )
        != _stable_directory_identity(observed_datasets)
    ):
        raise ArtifactValidationError(
            "canonical dataset directory changed"
        )
    if (
        _canonical_directory_signatures(
            discovery.datasets_dir,
            datasets_device=observed_datasets[0],
        )
        != observed_entries
    ):
        raise ArtifactValidationError(
            "canonical dataset directory discovery changed"
        )
    expected_entries = {
        name: signature
        for name, signature in discovery._entry_signatures
        if _CANONICAL_DATASET_DIRECTORY.fullmatch(name)
    }
    current_entries = dict(observed_entries)
    for name, expected_signature in expected_entries.items():
        if current_entries.get(name) != expected_signature:
            raise ArtifactValidationError(
                "canonical dataset directory changed"
            )
    return tuple(
        discovery.datasets_dir / name
        for name in sorted(current_entries.keys() - expected_entries.keys())
    )


def _role_byte_limit(name: str) -> int:
    if name == _MANIFEST_NAME:
        return MAX_DATASET_MANIFEST_BYTES
    if name == "preview.png":
        return MAX_DATASET_PREVIEW_BYTES
    if name in {"measurements.npz", "evaluation-truth.npz"}:
        return MAX_DATASET_NPZ_BYTES
    raise ArtifactValidationError(f"unknown dataset file role: {name}")


def _normalized_os_error(
    stage: str,
    error: OSError,
) -> ArtifactValidationError:
    normalized = ArtifactValidationError(
        f"{stage} failed with an operating-system error"
    )
    normalized.__cause__ = error
    return normalized


def _validate_real_directory(path: Path, noun: str) -> os.stat_result:
    _reject_linked_ancestors(path)
    try:
        info = os.lstat(path)
    except OSError as error:
        raise ArtifactValidationError(
            f"cannot inspect {noun}: {path}"
        ) from error
    if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
        raise ArtifactValidationError(
            f"{noun} must be a real directory"
        )
    return info


def _ensure_real_directory(path: Path, noun: str) -> Path:
    absolute = path.absolute()
    _reject_linked_ancestors(absolute)
    missing: list[Path] = []
    current = absolute
    while not os.path.lexists(current):
        missing.append(current)
        parent = current.parent
        if parent == current:
            raise ArtifactValidationError(
                f"cannot find an existing ancestor for {noun}"
            )
        current = parent
    _validate_real_directory(current, f"{noun} ancestor")
    for directory in reversed(missing):
        try:
            os.mkdir(directory, 0o700)
        except FileExistsError:
            _validate_real_directory(
                directory, f"concurrent {noun} winner"
            )
        except OSError as error:
            raise ArtifactValidationError(
                f"cannot create {noun}: {directory}"
            ) from error
        _validate_real_directory(directory, noun)
    _validate_real_directory(absolute, noun)
    return absolute


def _create_owned_stage(
    datasets_dir: Path,
    identity: str,
) -> _OwnedStage:
    try:
        raw_path = tempfile.mkdtemp(
            prefix=f".{identity}.staging-",
            dir=datasets_dir,
        )
    except OSError as error:
        raise ArtifactValidationError(
            "cannot create private dataset staging directory"
        ) from error
    path = Path(raw_path).absolute()
    info = _validate_real_directory(
        path, "dataset staging directory"
    )
    return _OwnedStage(
        path=path,
        device=info.st_dev,
        inode=info.st_ino,
    )


def _publication_barrier(
    name: str,
    *,
    staging_dir: Path,
    final_dir: Path,
) -> None:
    """Private deterministic test hook for publication barriers."""


def _record_owned_leaf(stage: _OwnedStage, name: str) -> None:
    path = stage.path / name
    _reject_linked_ancestors(path)
    try:
        info = os.lstat(path)
    except OSError as error:
        raise ArtifactValidationError(
            f"cannot record owned staging leaf: {name}"
        ) from error
    if _is_link_or_reparse(info):
        raise ArtifactValidationError(
            f"owned staging leaf became linked or reparse: {name}"
        )
    _validate_regular_snapshot_stat(
        info, noun=f"owned staging leaf {name}"
    )
    stage.leaves[name] = _stat_signature(info)


def _exclusive_write_owned_file(
    stage: _OwnedStage,
    name: str,
    payload: bytes,
) -> None:
    if name not in _DIRECTORY_NAMES or type(payload) is not bytes:
        raise TypeError("owned dataset file name or payload is invalid")
    path = stage.path / name
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    created = False
    primary_error: BaseException | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        created = True
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise ArtifactValidationError(
                    f"short write for owned staging leaf: {name}"
                )
            offset += written
        os.fsync(descriptor)
    except OSError as error:
        primary_error = _normalized_os_error(
            f"owned staging leaf write: {name}",
            error,
        )
    except BaseException as error:
        primary_error = error
    if descriptor is not None:
        try:
            os.close(descriptor)
        except OSError as error:
            close_error = _normalized_os_error(
                f"owned staging leaf close: {name}",
                error,
            )
            if primary_error is None:
                primary_error = close_error
            else:
                primary_error.add_note(str(close_error))
    if created:
        try:
            _record_owned_leaf(stage, name)
        except BaseException as error:
            if primary_error is None:
                primary_error = error
            else:
                primary_error.add_note(
                    "Owned-leaf recording also failed: "
                    f"{type(error).__name__}: {error}"
                )
    if primary_error is not None:
        raise primary_error


def _verify_owned_stage(stage: _OwnedStage) -> tuple[str, ...]:
    info = _validate_real_directory(
        stage.path, "owned staging directory"
    )
    if (info.st_dev, info.st_ino) != (stage.device, stage.inode):
        raise ArtifactValidationError(
            "owned staging directory identity changed; cleanup refused"
        )
    try:
        children = sorted(
            os.scandir(stage.path), key=lambda entry: entry.name
        )
    except OSError as error:
        raise ArtifactValidationError(
            "cannot inspect owned staging directory for cleanup"
        ) from error
    if {child.name for child in children} != set(stage.leaves):
        raise ArtifactValidationError(
            "owned staging inventory changed; recursive cleanup refused"
        )
    for child in children:
        path = stage.path / child.name
        _reject_linked_ancestors(path)
        try:
            leaf_info = os.lstat(path)
        except OSError as error:
            raise ArtifactValidationError(
                "owned staging leaf disappeared; cleanup refused"
            ) from error
        if _is_link_or_reparse(leaf_info):
            raise ArtifactValidationError(
                "linked staging leaf found; cleanup refused"
            )
        _validate_regular_snapshot_stat(
            leaf_info, noun=f"owned staging leaf {child.name}"
        )
        if _stat_signature(leaf_info) != stage.leaves[child.name]:
            raise ArtifactValidationError(
                "owned staging leaf changed; cleanup refused"
            )
    return tuple(child.name for child in children)


def _verify_owned_stage_bytes(
    stage: _OwnedStage,
    expected_files: Mapping[str, bytes],
) -> None:
    if set(expected_files) != _DIRECTORY_NAMES or any(
        type(payload) is not bytes
        for payload in expected_files.values()
    ):
        raise TypeError(
            "expected staging files must be the exact four byte payloads"
        )
    snapshots: dict[str, SafeFileSnapshot] = {}
    primary_error: BaseException | None = None
    try:
        for name in sorted(_DIRECTORY_NAMES):
            snapshot = read_safe_file_snapshot(
                stage.path / name,
                max_bytes=_role_byte_limit(name),
                noun=f"owned staging {name}",
            )
            snapshots[name] = snapshot
            expected = expected_files[name]
            if (
                snapshot.size_bytes != len(expected)
                or snapshot.sha256
                != hashlib.sha256(expected).hexdigest()
                or snapshot.raw != expected
            ):
                raise ArtifactValidationError(
                    "owned staging bytes differ from physically "
                    "verified publication evidence"
                )
    except BaseException as error:
        primary_error = error

    post_error: BaseException | None = None
    try:
        for snapshot in snapshots.values():
            verify_safe_file_snapshot(snapshot)
        _verify_owned_stage(stage)
    except BaseException as error:
        post_error = error
    if primary_error is not None:
        if post_error is not None:
            primary_error.add_note(
                "Owned staging byte post-verification also failed: "
                f"{type(post_error).__name__}: {post_error}"
            )
        raise primary_error
    if post_error is not None:
        raise post_error


def _same_leaf_except_change_time(
    signature: tuple[int, int, int, int, int, int],
    info: os.stat_result,
) -> bool:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_nlink,
    ) == (
        signature[0],
        signature[1],
        signature[2],
        signature[3],
        signature[5],
    )


def _freeze_owned_stage_for_promotion(
    stage: _OwnedStage,
    *,
    directory_descriptor: int | None = None,
) -> None:
    """Make an owned stage resistant to ordinary post-check writes.

    The freeze covers crashes, cooperating publishers and ordinary processes
    that attempt to open a leaf for writing after the final byte gate. It is
    not a security boundary against a same-identity process that deliberately
    restores owner permissions or clears the Windows read-only attribute.
    Windows cannot rename a directory while verified child handles remain
    open, so the handles are closed only after their exact file IDs have been
    frozen. The caller must immediately re-read exact bytes before promotion.
    """

    if os.name == "nt":
        if directory_descriptor is not None:
            raise TypeError("Windows stage freeze does not accept a descriptor")
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        create_file.restype = ctypes.c_void_p
        get_information = kernel32.GetFileInformationByHandleEx
        get_information.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        get_information.restype = ctypes.c_int
        set_information = kernel32.SetFileInformationByHandle
        set_information.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        set_information.restype = ctypes.c_int
        close = kernel32.CloseHandle
        close.argtypes = [ctypes.c_void_p]
        close.restype = ctypes.c_int
        invalid = ctypes.c_void_p(-1).value

        for name in sorted(_DIRECTORY_NAMES):
            signature = stage.leaves[name]
            handle = create_file(
                str(stage.path / name),
                0x00000080 | 0x00000100,
                0x1 | 0x4,
                None,
                3,
                0x00200000,
                None,
            )
            if handle in (None, invalid):
                raise ArtifactValidationError(
                    f"cannot freeze exact owned staging leaf: {name}"
                ) from ctypes.WinError(ctypes.get_last_error())
            try:
                file_id_info = _WindowsFileIdInfo()
                if not get_information(
                    handle,
                    _FILE_ID_INFO_CLASS,
                    ctypes.byref(file_id_info),
                    ctypes.sizeof(file_id_info),
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                handle_identity = (
                    int(file_id_info.volume_serial_number),
                    int.from_bytes(
                        bytes(file_id_info.file_id.identifier),
                        byteorder="little",
                    ),
                )
                if handle_identity != (signature[0], signature[1]):
                    raise ArtifactValidationError(
                        "owned staging leaf changed before read-only freeze"
                    )
                basic_info = _WindowsBasicInfo()
                if not get_information(
                    handle,
                    _FILE_BASIC_INFO_CLASS,
                    ctypes.byref(basic_info),
                    ctypes.sizeof(basic_info),
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                if basic_info.file_attributes & (
                    _FILE_ATTRIBUTE_REPARSE_POINT
                ):
                    raise ArtifactValidationError(
                        "linked staging leaf found during read-only freeze"
                    )
                basic_info.file_attributes |= _FILE_ATTRIBUTE_READONLY
                if not set_information(
                    handle,
                    _FILE_BASIC_INFO_CLASS,
                    ctypes.byref(basic_info),
                    ctypes.sizeof(basic_info),
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                frozen_info = _WindowsBasicInfo()
                if not get_information(
                    handle,
                    _FILE_BASIC_INFO_CLASS,
                    ctypes.byref(frozen_info),
                    ctypes.sizeof(frozen_info),
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
                if not (
                    frozen_info.file_attributes
                    & _FILE_ATTRIBUTE_READONLY
                ):
                    raise ArtifactValidationError(
                        "owned staging leaf did not become read-only"
                    )
            finally:
                close(handle)

            try:
                current = os.lstat(stage.path / name)
            except OSError as error:
                raise ArtifactValidationError(
                    "owned staging leaf disappeared after read-only freeze"
                ) from error
            if (
                _is_link_or_reparse(current)
                or not _same_leaf_except_change_time(signature, current)
                or not (
                    getattr(current, "st_file_attributes", 0)
                    & _FILE_ATTRIBUTE_READONLY
                )
            ):
                raise ArtifactValidationError(
                    "owned staging leaf changed during read-only freeze"
                )
            stage.leaves[name] = _stat_signature(current)
        return

    if directory_descriptor is None:
        raise TypeError("POSIX stage freeze requires its bound descriptor")
    opened_stage = os.fstat(directory_descriptor)
    if (opened_stage.st_dev, opened_stage.st_ino) != (
        stage.device,
        stage.inode,
    ):
        raise ArtifactValidationError(
            "promotion source changed before read-only freeze"
        )
    for name in sorted(_DIRECTORY_NAMES):
        signature = stage.leaves[name]
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
        try:
            before = os.fstat(descriptor)
            if _stat_signature(before) != signature:
                raise ArtifactValidationError(
                    "owned staging leaf changed before read-only freeze"
                )
            os.fchmod(descriptor, stat.S_IRUSR)
            frozen = os.fstat(descriptor)
            if (
                not _same_leaf_except_change_time(signature, frozen)
                or frozen.st_mode & 0o222
            ):
                raise ArtifactValidationError(
                    "owned staging leaf changed during read-only freeze"
                )
            os.fsync(descriptor)
            stage.leaves[name] = _stat_signature(frozen)
        finally:
            os.close(descriptor)
    os.fchmod(
        directory_descriptor,
        stat.S_IRUSR | stat.S_IXUSR,
    )
    os.fsync(directory_descriptor)
    frozen_stage = os.fstat(directory_descriptor)
    if (
        (frozen_stage.st_dev, frozen_stage.st_ino)
        != (stage.device, stage.inode)
        or frozen_stage.st_mode & 0o222
    ):
        raise ArtifactValidationError(
            "owned staging directory changed during read-only freeze"
        )


def _delete_windows_owned_path(
    path: Path,
    *,
    expected_device: int,
    expected_inode: int,
    directory: bool,
) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    create_file.restype = ctypes.c_void_p
    get_information = kernel32.GetFileInformationByHandleEx
    get_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    get_information.restype = ctypes.c_int
    set_information = kernel32.SetFileInformationByHandle
    set_information.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    set_information.restype = ctypes.c_int
    close = kernel32.CloseHandle
    close.argtypes = [ctypes.c_void_p]
    close.restype = ctypes.c_int
    flags = 0x00200000
    if directory:
        flags |= 0x02000000
    handle = create_file(
        str(path),
        0x00010000 | 0x00000080,
        0x1 | 0x2 | 0x4,
        None,
        3,
        flags,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle in (None, invalid):
        raise ArtifactValidationError(
            "cannot open exact owned staging object for deletion"
        ) from ctypes.WinError(ctypes.get_last_error())
    try:
        file_id_info = _WindowsFileIdInfo()
        if not get_information(
            handle,
            _FILE_ID_INFO_CLASS,
            ctypes.byref(file_id_info),
            ctypes.sizeof(file_id_info),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        handle_identity = (
            int(file_id_info.volume_serial_number),
            int.from_bytes(
                bytes(file_id_info.file_id.identifier),
                byteorder="little",
            ),
        )
        if handle_identity != (expected_device, expected_inode):
            raise ArtifactValidationError(
                "owned staging object identity changed; deletion refused"
            )
        disposition = _WindowsDispositionInfoEx(
            flags=(
                _FILE_DISPOSITION_FLAG_DELETE
                | _FILE_DISPOSITION_FLAG_IGNORE_READONLY_ATTRIBUTE
            )
        )
        if not set_information(
            handle,
            _FILE_DISPOSITION_INFO_EX_CLASS,
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        close(handle)


def _cleanup_owned_stage(stage: _OwnedStage) -> None:
    children = _verify_owned_stage(stage)
    if os.name == "nt":
        for name in children:
            signature = stage.leaves[name]
            _delete_windows_owned_path(
                stage.path / name,
                expected_device=signature[0],
                expected_inode=signature[1],
                directory=False,
            )
        _delete_windows_owned_path(
            stage.path,
            expected_device=stage.device,
            expected_inode=stage.inode,
            directory=True,
        )
        if os.path.lexists(stage.path):
            raise ArtifactValidationError(
                "owned staging directory remains after handle deletion"
            )
        return
    # POSIX has no portable unlink-by-handle operation. Keep every lookup
    # confined to opened directory descriptors and recheck the exact inode
    # immediately before unlinkat. A hostile same-UID process with write
    # access to the staging parent remains outside this portable guarantee.
    root_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_descriptor = os.open(stage.path, root_flags)
    try:
        opened_root = os.fstat(root_descriptor)
        if (opened_root.st_dev, opened_root.st_ino) != (
            stage.device,
            stage.inode,
        ):
            raise ArtifactValidationError(
                "owned staging directory changed before cleanup"
            )
        os.fchmod(root_descriptor, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        thawed_root = os.fstat(root_descriptor)
        if (thawed_root.st_dev, thawed_root.st_ino) != (
            stage.device,
            stage.inode,
        ):
            raise ArtifactValidationError(
                "owned staging directory changed while enabling cleanup"
            )
        for name in children:
            leaf_flags = (
                os.O_RDONLY
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            leaf_descriptor = os.open(
                name, leaf_flags, dir_fd=root_descriptor
            )
            try:
                opened_leaf = os.fstat(leaf_descriptor)
                if _stat_signature(opened_leaf) != stage.leaves[name]:
                    raise ArtifactValidationError(
                        "owned staging leaf changed before cleanup"
                    )
                current_leaf = os.stat(
                    name,
                    dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
                if _stat_signature(current_leaf) != stage.leaves[name]:
                    raise ArtifactValidationError(
                        "owned staging leaf changed before cleanup"
                    )
                os.unlink(name, dir_fd=root_descriptor)
            finally:
                os.close(leaf_descriptor)
    finally:
        os.close(root_descriptor)
    parent_descriptor = os.open(stage.path.parent, root_flags)
    try:
        current_root = os.stat(
            stage.path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (current_root.st_dev, current_root.st_ino) != (
            stage.device,
            stage.inode,
        ):
            raise ArtifactValidationError(
                "owned staging directory changed before removal"
            )
        os.rmdir(stage.path.name, dir_fd=parent_descriptor)
    except OSError as error:
        raise ArtifactValidationError(
            "cannot remove verified empty staging directory"
        ) from error
    finally:
        os.close(parent_descriptor)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        create_file.restype = ctypes.c_void_p
        flush = kernel32.FlushFileBuffers
        flush.argtypes = [ctypes.c_void_p]
        flush.restype = ctypes.c_int
        close = kernel32.CloseHandle
        close.argtypes = [ctypes.c_void_p]
        close.restype = ctypes.c_int
        handle = create_file(
            str(path),
            0x40000000,
            0x1 | 0x2 | 0x4,
            None,
            3,
            0x02000000,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle in (None, invalid):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not flush(handle):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            close(handle)
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_path_no_clobber(
    source_dir: Path,
    destination_dir: Path,
) -> None:
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        move = kernel32.MoveFileExW
        move.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_uint32,
        ]
        move.restype = ctypes.c_int
        if move(
            str(source_dir),
            str(destination_dir),
            _MOVEFILE_WRITE_THROUGH,
        ):
            return
        error_code = ctypes.get_last_error()
        if error_code in {80, 183}:
            raise FileExistsError(error_code, "final dataset already exists")
        raise ctypes.WinError(error_code)

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ArtifactValidationError(
            "atomic no-clobber directory promotion is unsupported"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source_dir),
        -100,
        os.fsencode(destination_dir),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_code = ctypes.get_errno()
    if error_code in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_code, "final dataset already exists"
        )
    if error_code in {
        errno.ENOSYS,
        errno.EINVAL,
        getattr(errno, "EOPNOTSUPP", errno.ENOSYS),
    }:
        raise ArtifactValidationError(
            "atomic no-clobber directory promotion is unsupported"
        )
    raise OSError(error_code, os.strerror(error_code))


def _handle_bound_promotion_barrier(
    source_dir: Path,
    destination_dir: Path,
) -> None:
    """Private deterministic hook after pinning the promotion source."""


def _promote_exact_directory_no_clobber(
    source_dir: Path,
    destination_dir: Path,
    *,
    expected_device: int,
    expected_inode: int,
) -> None:
    """No-clobber promote the exact opened directory identity."""
    if type(expected_device) is not int or type(expected_inode) is not int:
        raise TypeError("promotion directory identity must use exact integers")
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        create_file.restype = ctypes.c_void_p
        get_information = kernel32.GetFileInformationByHandleEx
        get_information.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        get_information.restype = ctypes.c_int
        rename_handle = kernel32.SetFileInformationByHandle
        rename_handle.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        rename_handle.restype = ctypes.c_int
        close = kernel32.CloseHandle
        close.argtypes = [ctypes.c_void_p]
        close.restype = ctypes.c_int
        handle = create_file(
            str(source_dir),
            0x00010000 | 0x00000080,
            0x1 | 0x2 | 0x4,
            None,
            3,
            0x02000000 | 0x00200000,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle in (None, invalid):
            raise ArtifactValidationError(
                "cannot open exact directory promotion source"
            ) from ctypes.WinError(ctypes.get_last_error())
        try:
            file_id_info = _WindowsFileIdInfo()
            if not get_information(
                handle,
                _FILE_ID_INFO_CLASS,
                ctypes.byref(file_id_info),
                ctypes.sizeof(file_id_info),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            handle_identity = (
                int(file_id_info.volume_serial_number),
                int.from_bytes(
                    bytes(file_id_info.file_id.identifier),
                    byteorder="little",
                ),
            )
            if handle_identity != (expected_device, expected_inode):
                raise ArtifactValidationError(
                    "promotion source is not the pinned directory identity"
                )
            _handle_bound_promotion_barrier(source_dir, destination_dir)
            encoded_name = str(destination_dir.absolute()).encode("utf-16-le")
            payload_size = _WindowsRenameInfo.file_name.offset + len(encoded_name)
            buffer_size = payload_size + ctypes.sizeof(ctypes.c_wchar)
            buffer = ctypes.create_string_buffer(buffer_size)
            rename_info = _WindowsRenameInfo.from_buffer(buffer)
            rename_info.flags = 0
            rename_info.root_directory = None
            rename_info.file_name_length = len(encoded_name)
            ctypes.memmove(
                ctypes.addressof(buffer) + _WindowsRenameInfo.file_name.offset,
                encoded_name,
                len(encoded_name),
            )
            if not rename_handle(
                handle,
                _FILE_RENAME_INFO_CLASS,
                buffer,
                buffer_size,
            ):
                error_code = ctypes.get_last_error()
                if error_code in {80, 183}:
                    raise FileExistsError(
                        error_code,
                        "promotion destination already exists",
                    )
                raise ctypes.WinError(error_code)
        finally:
            close(handle)
    else:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(source_dir, flags)
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (
                expected_device,
                expected_inode,
            ):
                raise ArtifactValidationError(
                    "promotion source is not the pinned directory identity"
                )
            _handle_bound_promotion_barrier(source_dir, destination_dir)
            observed = os.lstat(source_dir)
            if (observed.st_dev, observed.st_ino) != (
                expected_device,
                expected_inode,
            ):
                raise ArtifactValidationError(
                    "promotion source path changed after pinning"
                )
            _rename_directory_path_no_clobber(source_dir, destination_dir)
        finally:
            os.close(descriptor)
    promoted = os.lstat(destination_dir)
    if (
        _is_link_or_reparse(promoted)
        or not stat.S_ISDIR(promoted.st_mode)
        or (promoted.st_dev, promoted.st_ino)
        != (expected_device, expected_inode)
    ):
        raise ArtifactValidationError(
            "promoted directory identity does not match the pinned source"
        )


def _quarantine_unexpected_final(final_dir: Path) -> Path:
    diagnostic = final_dir.with_name(
        f".{final_dir.name}.rejected-{os.urandom(12).hex()}"
    )
    _rename_directory_path_no_clobber(final_dir, diagnostic)
    if os.path.lexists(final_dir):
        raise ArtifactValidationError(
            "unexpected promoted object remains at canonical final path"
        )
    return diagnostic


def _quarantine_owned_final(
    final_dir: Path,
    stage: _OwnedStage,
) -> Path:
    diagnostic = final_dir.with_name(
        f".{final_dir.name}.rejected-{os.urandom(12).hex()}"
    )
    quarantine_source = _OwnedStage(
        path=final_dir,
        device=stage.device,
        inode=stage.inode,
    )
    _promote_directory_no_clobber(
        quarantine_source,
        diagnostic,
        verify_leaves=False,
    )
    if os.path.lexists(final_dir):
        raise ArtifactValidationError(
            "owned failed publication remains at canonical final path"
        )
    return diagnostic


def _promote_directory_no_clobber(
    stage: _OwnedStage,
    final_dir: Path,
    *,
    verify_leaves: bool = True,
    expected_files: Mapping[str, bytes] | None = None,
) -> None:
    """Promote the exact owned stage without replacing a final directory.

    Windows binds the rename to an opened directory handle. POSIX has no
    renameat2 variant that accepts an empty source path, so it keeps an open
    descriptor for identity evidence, performs the path-based no-replace
    syscall immediately after the last identity check, and verifies the
    promoted inode before accepting it.
    """

    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        create_file.restype = ctypes.c_void_p
        get_information = kernel32.GetFileInformationByHandleEx
        get_information.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        get_information.restype = ctypes.c_int
        rename_handle = kernel32.SetFileInformationByHandle
        rename_handle.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        rename_handle.restype = ctypes.c_int
        close = kernel32.CloseHandle
        close.argtypes = [ctypes.c_void_p]
        close.restype = ctypes.c_int

        handle = create_file(
            str(stage.path),
            0x00010000 | 0x00000080,
            0x1 | 0x2 | 0x4,
            None,
            3,
            0x02000000 | 0x00200000,
            None,
        )
        invalid = ctypes.c_void_p(-1).value
        if handle in (None, invalid):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            file_id_info = _WindowsFileIdInfo()
            if not get_information(
                handle,
                _FILE_ID_INFO_CLASS,
                ctypes.byref(file_id_info),
                ctypes.sizeof(file_id_info),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            handle_identity = (
                int(file_id_info.volume_serial_number),
                int.from_bytes(
                    bytes(file_id_info.file_id.identifier),
                    byteorder="little",
                ),
            )
            if handle_identity != (stage.device, stage.inode):
                raise ArtifactValidationError(
                    "promotion source is not the owned staging directory"
                )
            if verify_leaves:
                _verify_owned_stage(stage)
                if expected_files is None:
                    raise TypeError(
                        "promotion requires exact expected file bytes"
                    )
                _freeze_owned_stage_for_promotion(stage)
                _verify_owned_stage(stage)
                _verify_owned_stage_bytes(stage, expected_files)

            encoded_name = str(final_dir).encode("utf-16-le")
            payload_size = (
                _WindowsRenameInfo.file_name.offset
                + len(encoded_name)
            )
            buffer_size = payload_size + ctypes.sizeof(ctypes.c_wchar)
            buffer = ctypes.create_string_buffer(buffer_size)
            rename_info = _WindowsRenameInfo.from_buffer(buffer)
            rename_info.flags = 0
            rename_info.root_directory = None
            rename_info.file_name_length = len(encoded_name)
            ctypes.memmove(
                ctypes.addressof(buffer)
                + _WindowsRenameInfo.file_name.offset,
                encoded_name,
                len(encoded_name),
            )
            if not rename_handle(
                handle,
                _FILE_RENAME_INFO_CLASS,
                buffer,
                buffer_size,
            ):
                error_code = ctypes.get_last_error()
                if error_code in {80, 183}:
                    raise FileExistsError(
                        error_code, "final dataset already exists"
                    )
                raise ctypes.WinError(error_code)
        finally:
            close(handle)
    else:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(stage.path, flags)
        try:
            source_info = os.fstat(descriptor)
            if (source_info.st_dev, source_info.st_ino) != (
                stage.device,
                stage.inode,
            ):
                raise ArtifactValidationError(
                    "promotion source is not the owned staging directory"
                )
            if verify_leaves:
                _verify_owned_stage(stage)
                if expected_files is None:
                    raise TypeError(
                        "promotion requires exact expected file bytes"
                    )
                _freeze_owned_stage_for_promotion(
                    stage,
                    directory_descriptor=descriptor,
                )
                _verify_owned_stage(stage)
                _verify_owned_stage_bytes(stage, expected_files)
            _rename_directory_path_no_clobber(
                stage.path, final_dir
            )
        finally:
            os.close(descriptor)

    try:
        promoted_info = os.lstat(final_dir)
    except OSError as error:
        raise ArtifactValidationError(
            "cannot verify promoted dataset directory identity"
        ) from error
    if (
        _is_link_or_reparse(promoted_info)
        or not stat.S_ISDIR(promoted_info.st_mode)
        or (promoted_info.st_dev, promoted_info.st_ino)
        != (stage.device, stage.inode)
    ):
        quarantine_error: BaseException | None = None
        diagnostic: Path | None = None
        try:
            diagnostic = _quarantine_unexpected_final(final_dir)
        except BaseException as error:
            quarantine_error = error
        validation_error = ArtifactValidationError(
            "promoted dataset directory identity mismatch"
        )
        if diagnostic is not None:
            validation_error.add_note(
                f"Unexpected promoted object retained at {diagnostic}"
            )
        if quarantine_error is not None:
            validation_error.add_note(
                "Failed to remove the unexpected object from the "
                "canonical final path: "
                f"{type(quarantine_error).__name__}: {quarantine_error}"
            )
        raise validation_error
    if verify_leaves:
        promoted_stage = _OwnedStage(
            path=final_dir,
            device=stage.device,
            inode=stage.inode,
            leaves=dict(stage.leaves),
        )
        try:
            _verify_owned_stage(promoted_stage)
        except BaseException as error:
            try:
                diagnostic = _quarantine_owned_final(
                    final_dir, stage
                )
            except BaseException as quarantine_error:
                error.add_note(
                    "Failed to quarantine the owned invalid final: "
                    f"{type(quarantine_error).__name__}: "
                    f"{quarantine_error}"
                )
            else:
                error.add_note(
                    "Owned invalid final retained for diagnosis at "
                    f"{diagnostic}"
                )
            raise


def _sync_publication_parent(datasets_dir: Path) -> None:
    """Persist the directory entry where POSIX supports that guarantee."""

    if os.name != "nt":
        _fsync_directory(datasets_dir)


def _preflight_inventory(inventory: DirectoryInventory) -> None:
    entries = dict(inventory._entries)
    if len(entries) != len(_DIRECTORY_NAMES) or set(entries) != (
        _DIRECTORY_NAMES
    ):
        raise ArtifactValidationError(
            "final dataset directory must contain exactly four fixed files"
        )
    for name, signature in entries.items():
        if signature[2] > _role_byte_limit(name):
            raise ArtifactValidationError(
                f"{name} exceeds its role byte bound"
            )


def _snapshot_directory_files(
    dataset_dir: Path,
    snapshots: dict[str, SafeFileSnapshot],
) -> None:
    for name in (_MANIFEST_NAME, *_PAYLOAD_NAMES):
        snapshots[name] = read_safe_file_snapshot(
            dataset_dir / name,
            max_bytes=_role_byte_limit(name),
            noun=name,
        )


def _postverify_directory(
    snapshots: Mapping[str, SafeFileSnapshot],
    inventory: DirectoryInventory,
) -> None:
    failures: list[ArtifactValidationError] = []
    for name in (_MANIFEST_NAME, *_PAYLOAD_NAMES):
        snapshot = snapshots.get(name)
        if snapshot is not None:
            try:
                verify_safe_file_snapshot(snapshot)
            except ArtifactValidationError as error:
                failures.append(error)
            except OSError as error:
                failures.append(
                    _normalized_os_error(
                        f"{name} snapshot post-verification",
                        error,
                    )
                )
    try:
        verify_directory_inventory(inventory)
    except ArtifactValidationError as error:
        failures.append(error)
    except OSError as error:
        failures.append(
            _normalized_os_error(
                "dataset inventory post-verification",
                error,
            )
        )
    if failures:
        primary = failures[0]
        for additional in failures[1:]:
            primary.add_note(
                "Additional post-verification failure: "
                f"{type(additional).__name__}: {additional}"
            )
        raise primary


def _compare_json_field(
    field: str,
    actual: object,
    expected: object,
) -> None:
    if canonical_json_bytes(actual) != canonical_json_bytes(expected):
        raise ArtifactValidationError(
            f"expected-generated semantic mismatch: {field}"
        )


def _compare_scalar_fields(
    noun: str,
    actual: object,
    expected: object,
    fields: tuple[str, ...],
) -> None:
    for name in fields:
        actual_value = getattr(actual, name)
        expected_value = getattr(expected, name)
        if (
            type(actual_value) is not type(expected_value)
            or actual_value != expected_value
        ):
            raise ArtifactValidationError(
                f"expected-generated {noun} scalar mismatch: {name}"
            )


def _compare_array(
    noun: str,
    actual: np.ndarray | None,
    expected: np.ndarray | None,
) -> None:
    if actual is None or expected is None:
        if actual is not expected:
            raise ArtifactValidationError(
                f"expected-generated array presence mismatch: {noun}"
            )
        return
    if (
        type(actual) is not np.ndarray
        or type(expected) is not np.ndarray
        or actual.dtype != expected.dtype
        or actual.shape != expected.shape
        or not actual.flags.c_contiguous
        or not expected.flags.c_contiguous
        or actual.tobytes(order="C") != expected.tobytes(order="C")
    ):
        raise ArtifactValidationError(
            f"expected-generated array mismatch: {noun}"
        )


def _compare_expected_generated(
    *,
    manifest: Mapping[str, object],
    acquisition: SPIAcquisitionData,
    truth: EvaluationTruth,
    preview: np.ndarray,
    expected: CorrectedDataset,
) -> None:
    _validate_acquisition_identity(expected.acquisition)
    validate_corrected_truth(expected.truth)
    if (
        expected.dataset_identity_sha256
        != manifest["dataset_identity_sha256"]
        or expected.noise_calibration_sha256
        != manifest["dataset_identity_spec"]["noise_calibration"]["sha256"]
    ):
        raise ArtifactValidationError(
            "expected-generated identity or calibration mismatch"
        )
    for field, expected_value in (
        ("dataset_identity_spec", expected.dataset_identity_spec),
        (
            "resolved_generator_config",
            expected.resolved_generator_config,
        ),
        (
            "noise_calibration_record",
            expected.noise_calibration_record,
        ),
    ):
        _compare_json_field(field, manifest[field], expected_value)

    for name in (
        "patterns",
        "measurements",
        "frame_indices",
        "time_grid",
        "holdout_patterns",
        "holdout_measurements",
        "holdout_frame_indices",
    ):
        _compare_array(
            f"acquisition.{name}",
            getattr(acquisition, name),
            getattr(expected.acquisition, name),
        )
    _compare_scalar_fields(
        "acquisition",
        acquisition,
        expected.acquisition,
        (
            "dataset_identity_sha256",
            "H",
            "W",
            "T",
            "K",
            "holdout_K",
        ),
    )
    for field in ("acquisition", "array_descriptors"):
        _compare_json_field(
            f"acquisition.{field}",
            getattr(acquisition, field),
            getattr(expected.acquisition, field),
        )

    for name in (
        "canonical_image",
        "gt_frames",
        "translation_trajectory",
        "rotation_trajectory",
        "gt_velocity",
        "gt_acceleration",
    ):
        _compare_array(
            f"truth.{name}",
            getattr(truth, name),
            getattr(expected.truth, name),
        )
    _compare_scalar_fields(
        "truth",
        truth,
        expected.truth,
        (
            "dataset_identity_sha256",
            "gt_omega",
            "gt_beta",
            "motion_model",
            "H",
            "W",
            "T",
        ),
    )
    for field in ("dataset_identity_spec", "evaluator_metadata"):
        _compare_json_field(
            f"truth.{field}",
            getattr(truth, field),
            getattr(expected.truth, field),
        )
    _compare_array(
        "preview",
        preview,
        _quantized_preview(expected.truth),
    )


def verify_dataset_directory(
    dataset_dir: Path,
    *,
    expected_dataset_identity_sha256: str | None = None,
    expected_dataset_manifest_sha256: str | None = None,
    expected_generated: CorrectedDataset | None = None,
) -> VerifiedDatasetDirectory:
    """Verify one immutable final dataset directory without writing to it.

    Supplying ``expected_dataset_manifest_sha256`` externally anchors the
    manifest. Without that argument, verification proves only that the four
    observed files and their embedded semantics are self-consistent.
    """

    if not isinstance(dataset_dir, Path):
        raise TypeError("dataset_dir must be a Path")
    if expected_dataset_identity_sha256 is not None:
        validate_sha256(
            expected_dataset_identity_sha256,
            "expected dataset identity",
        )
    if expected_dataset_manifest_sha256 is not None:
        validate_sha256(
            expected_dataset_manifest_sha256,
            "expected dataset manifest",
        )
    if expected_generated is not None:
        if type(expected_generated) is not CorrectedDataset:
            raise TypeError(
                "expected_generated must be an exact CorrectedDataset"
            )

    absolute = dataset_dir.absolute()
    directory_identity = validate_sha256(
        absolute.name, "dataset directory identity"
    )
    if (
        expected_dataset_identity_sha256 is not None
        and directory_identity != expected_dataset_identity_sha256
    ):
        raise ArtifactValidationError(
            "dataset directory identity disagrees with expected identity"
        )

    inventory: DirectoryInventory | None = None
    snapshots: dict[str, SafeFileSnapshot] = {}
    result: VerifiedDatasetDirectory | None = None
    primary_error: BaseException | None = None
    try:
        inventory = capture_directory_inventory(absolute)
        _preflight_inventory(inventory)
        _snapshot_directory_files(absolute, snapshots)
        manifest_snapshot = snapshots[_MANIFEST_NAME]
        if (
            expected_dataset_manifest_sha256 is not None
            and manifest_snapshot.sha256
            != expected_dataset_manifest_sha256
        ):
            raise ArtifactValidationError(
                "dataset manifest hash disagrees with external anchor"
            )
        manifest = parse_dataset_manifest_bytes(manifest_snapshot.raw)
        manifest_identity = manifest["dataset_identity_sha256"]
        if manifest_identity != directory_identity:
            raise ArtifactValidationError(
                "dataset manifest identity disagrees with directory name"
            )
        if (
            expected_dataset_identity_sha256 is not None
            and manifest_identity
            != expected_dataset_identity_sha256
        ):
            raise ArtifactValidationError(
                "dataset manifest identity disagrees with expected identity"
            )

        files = manifest["files"]
        payload_snapshots = {
            name: snapshots[name] for name in _PAYLOAD_NAMES
        }
        for name, snapshot in payload_snapshots.items():
            descriptor = files[name]
            if (
                descriptor["sha256"] != snapshot.sha256
                or descriptor["size_bytes"] != snapshot.size_bytes
            ):
                raise ArtifactValidationError(
                    f"dataset payload snapshot hash or size mismatch: {name}"
                )
        acquisition, truth, preview = verify_dataset_payload_bytes(
            {
                name: snapshot.raw
                for name, snapshot in payload_snapshots.items()
            },
            manifest,
        )
        if expected_generated is not None:
            _compare_expected_generated(
                manifest=manifest,
                acquisition=acquisition,
                truth=truth,
                preview=preview,
                expected=expected_generated,
            )
        result = VerifiedDatasetDirectory(
            dataset_dir=absolute,
            dataset_identity_sha256=manifest_identity,
            dataset_manifest_sha256=manifest_snapshot.sha256,
            manifest_externally_anchored=(
                expected_dataset_manifest_sha256 is not None
            ),
            expected_generated_verified=(
                expected_generated is not None
            ),
            payload_evidence={
                name: DatasetPayloadEvidence(
                    sha256=snapshot.sha256,
                    size_bytes=snapshot.size_bytes,
                )
                for name, snapshot in payload_snapshots.items()
            },
            acquisition=acquisition,
            truth=truth,
            preview=preview,
            _manifest=manifest,
        )
    except OSError as error:
        primary_error = _normalized_os_error(
            "dataset directory verification",
            error,
        )
    except BaseException as error:
        primary_error = error

    post_error: BaseException | None = None
    if inventory is not None:
        try:
            _postverify_directory(snapshots, inventory)
        except OSError as error:
            post_error = _normalized_os_error(
                "dataset directory post-verification",
                error,
            )
        except BaseException as error:
            post_error = error
    if primary_error is not None:
        if post_error is not None:
            primary_error.add_note(
                "Post-verification also failed: "
                f"{type(post_error).__name__}: {post_error}"
            )
            for note in getattr(post_error, "__notes__", ()):
                primary_error.add_note(note)
        raise primary_error
    if post_error is not None:
        raise post_error
    if result is None:
        raise RuntimeError(
            "dataset directory verification produced no result"
        )
    return result


def _verify_staged_payloads(
    stage: _OwnedStage,
    generated: CorrectedDataset,
) -> tuple[dict[str, bytes], dict[str, object]]:
    inventory = capture_directory_inventory(stage.path)
    entries = dict(inventory._entries)
    if len(entries) != len(_PAYLOAD_NAMES) or set(entries) != set(
        _PAYLOAD_NAMES
    ):
        raise ArtifactValidationError(
            "staging payload inventory must contain exactly three files"
        )
    for name, signature in entries.items():
        if signature[2] > _role_byte_limit(name):
            raise ArtifactValidationError(
                f"{name} exceeds its role byte bound"
            )
    snapshots: dict[str, SafeFileSnapshot] = {}
    primary_error: BaseException | None = None
    result: tuple[dict[str, bytes], dict[str, object]] | None = None
    try:
        for name in _PAYLOAD_NAMES:
            snapshots[name] = read_safe_file_snapshot(
                stage.path / name,
                max_bytes=_role_byte_limit(name),
                noun=name,
            )
        physical_payloads = {
            name: snapshots[name].raw for name in _PAYLOAD_NAMES
        }
        manifest = build_dataset_manifest(
            generated, physical_payloads
        )
        acquisition, truth, preview = verify_dataset_payload_bytes(
            physical_payloads, manifest
        )
        _compare_expected_generated(
            manifest=manifest,
            acquisition=acquisition,
            truth=truth,
            preview=preview,
            expected=generated,
        )
        result = physical_payloads, manifest
    except OSError as error:
        primary_error = _normalized_os_error(
            "staging payload round-trip",
            error,
        )
    except BaseException as error:
        primary_error = error
    post_error: BaseException | None = None
    try:
        _postverify_directory(snapshots, inventory)
    except BaseException as error:
        post_error = error
    if primary_error is not None:
        if post_error is not None:
            primary_error.add_note(
                "Staging payload post-verification also failed: "
                f"{type(post_error).__name__}: {post_error}"
            )
        raise primary_error
    if post_error is not None:
        raise post_error
    if result is None:
        raise RuntimeError("staging payload verification produced no result")
    return result


def _publication_from_verified(
    status: str,
    verified: VerifiedDatasetDirectory,
) -> DatasetPublication:
    return DatasetPublication(
        status=status,
        dataset_dir=verified.dataset_dir,
        dataset_manifest_sha256=(
            verified.dataset_manifest_sha256
        ),
        verified=verified,
    )


def _verify_concurrent_publication_winner(
    final_dir: Path,
    *,
    identity: str,
    generated: CorrectedDataset,
    staged_manifest_payload: bytes,
    staged_physical_payloads: Mapping[str, bytes],
) -> DatasetPublication:
    expected_manifest_sha256 = hashlib.sha256(
        staged_manifest_payload
    ).hexdigest()
    expected_payload_evidence = {
        name: DatasetPayloadEvidence(
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )
        for name, payload in staged_physical_payloads.items()
    }
    try:
        verified = verify_dataset_directory(
            final_dir,
            expected_dataset_identity_sha256=identity,
            expected_dataset_manifest_sha256=(
                expected_manifest_sha256
            ),
            expected_generated=generated,
        )
        if dict(verified.payload_evidence) != expected_payload_evidence:
            raise ArtifactValidationError(
                "concurrent winner payload evidence differs from staging"
            )
    except Exception as error:
        collision = ArtifactValidationError(
            "nondeterministic dataset collision: concurrent winner "
            "differs from the physically verified staged publication"
        )
        collision.__cause__ = error
        raise collision
    return _publication_from_verified("reused", verified)


def publish_dataset(
    artifact_root: Path,
    generated: CorrectedDataset,
) -> DatasetPublication:
    """Create or safely reuse one immutable dataset publication."""

    if not isinstance(artifact_root, Path):
        raise TypeError("artifact_root must be a Path")
    if type(generated) is not CorrectedDataset:
        raise TypeError("generated must be an exact CorrectedDataset")
    identity = validate_sha256(
        generated.dataset_identity_sha256,
        "generated dataset identity",
    )
    root = _ensure_real_directory(
        artifact_root, "artifact root"
    )
    datasets_dir = _ensure_real_directory(
        root / "datasets", "dataset publication root"
    )
    datasets_info = _validate_real_directory(
        datasets_dir, "dataset publication root"
    )
    final_dir = (datasets_dir / identity).absolute()
    _reject_linked_ancestors(final_dir)
    if os.path.lexists(final_dir):
        verified = verify_dataset_directory(
            final_dir,
            expected_dataset_identity_sha256=identity,
            expected_generated=generated,
        )
        return _publication_from_verified("reused", verified)

    stage: _OwnedStage | None = None
    moved = False
    promoted = False
    try:
        stage = _create_owned_stage(datasets_dir, identity)
        if stage.device != datasets_info.st_dev:
            raise ArtifactValidationError(
                "staging directory is not on the publication filesystem"
            )
        payloads = build_dataset_payloads(generated)
        for name, barrier in (
            ("measurements.npz", "measurements"),
            ("evaluation-truth.npz", "truth"),
            ("preview.png", "preview"),
        ):
            _exclusive_write_owned_file(
                stage, name, payloads[name]
            )
            _publication_barrier(
                "file-fsync",
                staging_dir=stage.path,
                final_dir=final_dir,
            )
            _publication_barrier(
                barrier,
                staging_dir=stage.path,
                final_dir=final_dir,
            )

        physical_payloads, manifest = _verify_staged_payloads(
            stage, generated
        )
        _publication_barrier(
            "physical-roundtrip",
            staging_dir=stage.path,
            final_dir=final_dir,
        )
        if physical_payloads != payloads:
            raise ArtifactValidationError(
                "physical staging payload bytes mismatch"
            )
        manifest_payload = dataset_manifest_bytes(manifest)
        manifest_sha256 = hashlib.sha256(
            manifest_payload
        ).hexdigest()
        _exclusive_write_owned_file(
            stage, _MANIFEST_NAME, manifest_payload
        )
        _publication_barrier(
            "file-fsync",
            staging_dir=stage.path,
            final_dir=final_dir,
        )
        _publication_barrier(
            "manifest",
            staging_dir=stage.path,
            final_dir=final_dir,
        )

        complete_inventory = capture_directory_inventory(stage.path)
        _preflight_inventory(complete_inventory)
        verify_directory_inventory(complete_inventory)
        _fsync_directory(stage.path)
        _publication_barrier(
            "stage-fsync",
            staging_dir=stage.path,
            final_dir=final_dir,
        )
        _publication_barrier(
            "before-promotion",
            staging_dir=stage.path,
            final_dir=final_dir,
        )
        current_datasets = _validate_real_directory(
            datasets_dir, "dataset publication root"
        )
        if (
            current_datasets.st_dev,
            current_datasets.st_ino,
        ) != (datasets_info.st_dev, datasets_info.st_ino):
            raise ArtifactValidationError(
                "dataset publication root changed before promotion"
            )
        _verify_owned_stage(stage)
        _reject_linked_ancestors(final_dir)
        if os.path.lexists(final_dir):
            publication = _verify_concurrent_publication_winner(
                final_dir,
                identity=identity,
                generated=generated,
                staged_manifest_payload=manifest_payload,
                staged_physical_payloads=physical_payloads,
            )
            _cleanup_owned_stage(stage)
            stage = None
            return publication
        expected_stage_files = {
            **physical_payloads,
            _MANIFEST_NAME: manifest_payload,
        }
        try:
            _promote_directory_no_clobber(
                stage,
                final_dir,
                expected_files=expected_stage_files,
            )
        except FileExistsError:
            publication = _verify_concurrent_publication_winner(
                final_dir,
                identity=identity,
                generated=generated,
                staged_manifest_payload=manifest_payload,
                staged_physical_payloads=physical_payloads,
            )
            _cleanup_owned_stage(stage)
            stage = None
            return publication
        moved = True
        try:
            verified = verify_dataset_directory(
                final_dir,
                expected_dataset_identity_sha256=identity,
                expected_dataset_manifest_sha256=manifest_sha256,
                expected_generated=generated,
            )
        except BaseException as error:
            try:
                diagnostic = _quarantine_owned_final(
                    final_dir, stage
                )
            except BaseException as quarantine_error:
                error.add_note(
                    "Failed to quarantine the owned publication after "
                    "final verification failed: "
                    f"{type(quarantine_error).__name__}: "
                    f"{quarantine_error}"
                )
                try:
                    diagnostic = _quarantine_unexpected_final(
                        final_dir
                    )
                except BaseException as fallback_error:
                    error.add_note(
                        "Fallback canonical-path quarantine also "
                        "failed: "
                        f"{type(fallback_error).__name__}: "
                        f"{fallback_error}"
                    )
                else:
                    error.add_note(
                        "Unexpected final retained for diagnosis at "
                        f"{diagnostic}"
                    )
            else:
                error.add_note(
                    "Owned failed publication retained for diagnosis at "
                    f"{diagnostic}"
                )
            raise
        promoted = True
        _sync_publication_parent(datasets_dir)
        return _publication_from_verified("created", verified)
    except BaseException as error:
        if stage is not None and not moved and not promoted:
            try:
                _cleanup_owned_stage(stage)
            except BaseException as cleanup_error:
                error.add_note(
                    "Owned staging cleanup refused or failed; "
                    f"staging retained at {stage.path}: "
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        raise
