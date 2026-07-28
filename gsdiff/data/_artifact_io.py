"""Deterministic NPY/ZIP and atomic file I/O for SPI artifacts."""

from dataclasses import dataclass, field
import ctypes
from ctypes import wintypes
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Mapping
import zipfile

import numpy as np

from ._artifact_identity import (
    ArtifactValidationError,
    canonical_json_bytes,
    validate_exact_json_native,
)


METADATA_MEMBER = "__metadata_json__.npy"
_READ_BLOCK_BYTES = 1024 * 1024
_MAX_NPZ_BYTES = 1024 * 1024 * 1024
_MAX_NPZ_MEMBER_BYTES = 512 * 1024 * 1024
_MAX_NPZ_TOTAL_BYTES = 1024 * 1024 * 1024
_MAX_NPZ_MEMBERS = 32
_REPARSE_POINT = getattr(
    stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
)
_StatSignature = tuple[int, int, int, int, int, int]
_ERROR_HANDLE_EOF = 38
_INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
_MAX_PATH = 260
_MAX_ALTERNATE = 36


class _WIN32_FIND_STREAM_DATA(ctypes.Structure):
    _fields_ = [
        ("StreamSize", ctypes.c_longlong),
        (
            "cStreamName",
            wintypes.WCHAR * (_MAX_PATH + _MAX_ALTERNATE),
        ),
    ]


@dataclass(frozen=True)
class SafeFileSnapshot:
    path: Path
    raw: bytes
    sha256: str
    size_bytes: int
    _path_signature: _StatSignature = field(repr=False)
    _handle_signature: _StatSignature = field(repr=False)


@dataclass(frozen=True)
class DirectoryInventory:
    root: Path
    _root_signature: _StatSignature = field(repr=False)
    _entries: tuple[tuple[str, _StatSignature], ...] = field(
        repr=False
    )


def _stat_signature(info: os.stat_result) -> _StatSignature:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
        info.st_nlink,
    )


def _cross_stat_signature(
    info: os.stat_result,
) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_nlink,
    )


def _is_link_or_reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & _REPARSE_POINT
    )


def _reject_linked_ancestors(path: Path) -> None:
    current = path.absolute()
    while True:
        if os.path.lexists(current):
            try:
                info = os.lstat(current)
            except OSError as error:
                raise ArtifactValidationError(
                    f"cannot inspect artifact path: {current}"
                ) from error
            if _is_link_or_reparse(info):
                raise ArtifactValidationError(
                    f"linked or reparse artifact path rejected: {current}"
                )
        parent = current.parent
        if parent == current:
            return
        current = parent


def _reject_windows_named_streams(path: Path) -> None:
    if os.name != "nt":
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    find_first = kernel32.FindFirstStreamW
    find_first.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(_WIN32_FIND_STREAM_DATA),
        wintypes.DWORD,
    )
    find_first.restype = wintypes.HANDLE
    find_next = kernel32.FindNextStreamW
    find_next.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(_WIN32_FIND_STREAM_DATA),
    )
    find_next.restype = wintypes.BOOL
    find_close = kernel32.FindClose
    find_close.argtypes = (wintypes.HANDLE,)
    find_close.restype = wintypes.BOOL

    data = _WIN32_FIND_STREAM_DATA()
    ctypes.set_last_error(0)
    handle = find_first(str(path), 0, ctypes.byref(data), 0)
    if handle == _INVALID_HANDLE_VALUE:
        error = ctypes.get_last_error()
        if error == _ERROR_HANDLE_EOF:
            return
        raise ArtifactValidationError(
            f"cannot enumerate Windows data streams: {path}"
        )
    try:
        while True:
            stream_name = str(data.cStreamName)
            if stream_name.casefold() != "::$data":
                raise ArtifactValidationError(
                    f"Windows named stream rejected: {path}"
                )
            ctypes.set_last_error(0)
            if find_next(handle, ctypes.byref(data)):
                continue
            error = ctypes.get_last_error()
            if error == _ERROR_HANDLE_EOF:
                break
            raise ArtifactValidationError(
                f"cannot enumerate Windows data streams: {path}"
            )
    finally:
        if not find_close(handle):
            raise ArtifactValidationError(
                f"cannot close Windows stream enumeration handle: {path}"
            )


def _validate_regular_snapshot_stat(
    info: os.stat_result,
    *,
    noun: str,
) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise ArtifactValidationError(f"{noun} must be a regular file")
    if info.st_nlink != 1:
        raise ArtifactValidationError(f"{noun} hardlink rejected")


def read_safe_file_snapshot(
    path: Path,
    *,
    max_bytes: int,
    noun: str = "artifact",
) -> SafeFileSnapshot:
    if not isinstance(path, Path):
        raise TypeError("snapshot path must be a Path")
    if type(max_bytes) is not int or max_bytes < 1:
        raise TypeError("max_bytes must be a positive exact integer")
    if type(noun) is not str or not noun:
        raise TypeError("snapshot noun must be a nonempty exact string")
    absolute = path.absolute()
    _reject_linked_ancestors(absolute)
    try:
        before = os.lstat(absolute)
    except OSError as error:
        raise ArtifactValidationError(
            f"cannot stat {noun}: {absolute}"
        ) from error
    _validate_regular_snapshot_stat(before, noun=noun)
    if before.st_size > max_bytes:
        raise ArtifactValidationError(f"{noun} exceeds byte bound")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise ArtifactValidationError(
            f"cannot safely open {noun}: {absolute}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        _validate_regular_snapshot_stat(opened, noun=noun)
        if _cross_stat_signature(before) != _cross_stat_signature(
            opened
        ):
            raise ArtifactValidationError(
                f"{noun} changed while being opened"
            )
        chunks: list[bytes] = []
        observed_size = 0
        while True:
            remaining = max_bytes + 1 - observed_size
            if remaining <= 0:
                raise ArtifactValidationError(
                    f"{noun} exceeds byte bound"
                )
            chunk = os.read(
                descriptor, min(_READ_BLOCK_BYTES, remaining)
            )
            if not chunk:
                break
            chunks.append(chunk)
            observed_size += len(chunk)
        if observed_size > max_bytes:
            raise ArtifactValidationError(f"{noun} exceeds byte bound")
        after_handle = os.fstat(descriptor)
        if _stat_signature(opened) != _stat_signature(after_handle):
            raise ArtifactValidationError(
                f"{noun} changed while being read"
            )
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    if len(raw) != opened.st_size:
        raise ArtifactValidationError(f"{noun} changed while being read")
    _reject_linked_ancestors(absolute)
    try:
        after_path = os.lstat(absolute)
    except OSError as error:
        raise ArtifactValidationError(
            f"cannot restat {noun}: {absolute}"
        ) from error
    _validate_regular_snapshot_stat(after_path, noun=noun)
    if _stat_signature(before) != _stat_signature(after_path):
        raise ArtifactValidationError(
            f"{noun} path changed while being read"
        )
    return SafeFileSnapshot(
        path=absolute,
        raw=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        _path_signature=_stat_signature(after_path),
        _handle_signature=_stat_signature(after_handle),
    )


def verify_safe_file_snapshot(snapshot: SafeFileSnapshot) -> None:
    if type(snapshot) is not SafeFileSnapshot:
        raise TypeError("snapshot must be an exact SafeFileSnapshot")
    _reject_linked_ancestors(snapshot.path)
    try:
        current = os.lstat(snapshot.path)
    except OSError as error:
        raise ArtifactValidationError(
            f"cannot verify snapshot path: {snapshot.path}"
        ) from error
    _validate_regular_snapshot_stat(current, noun="snapshot")
    if _stat_signature(current) != snapshot._path_signature:
        raise ArtifactValidationError(
            "snapshot path changed after the safe read"
        )


def _capture_directory_entries(
    root: Path,
) -> tuple[tuple[str, _StatSignature], ...]:
    entries: list[tuple[str, _StatSignature]] = []

    def visit(directory: Path) -> None:
        try:
            children = sorted(
                os.scandir(directory), key=lambda entry: entry.name
            )
        except OSError as error:
            raise ArtifactValidationError(
                f"cannot inventory directory: {directory}"
            ) from error
        for child in children:
            path = directory / child.name
            _reject_linked_ancestors(path)
            try:
                info = os.lstat(path)
            except OSError as error:
                raise ArtifactValidationError(
                    f"cannot inventory path: {path}"
                ) from error
            if _is_link_or_reparse(info):
                raise ArtifactValidationError(
                    f"linked inventory path rejected: {path}"
                )
            _reject_windows_named_streams(path)
            relative = path.relative_to(root).as_posix()
            if stat.S_ISDIR(info.st_mode):
                entries.append((relative, _stat_signature(info)))
                visit(path)
            elif stat.S_ISREG(info.st_mode):
                _validate_regular_snapshot_stat(
                    info, noun=f"inventory file {relative}"
                )
                entries.append((relative, _stat_signature(info)))
            else:
                raise ArtifactValidationError(
                    f"non-regular inventory entry rejected: {relative}"
                )

    visit(root)
    return tuple(entries)


def capture_directory_inventory(root: Path) -> DirectoryInventory:
    if not isinstance(root, Path):
        raise TypeError("inventory root must be a Path")
    absolute = root.absolute()
    _reject_linked_ancestors(absolute)
    _reject_windows_named_streams(absolute)
    try:
        before = os.lstat(absolute)
    except OSError as error:
        raise ArtifactValidationError(
            f"cannot stat inventory root: {absolute}"
        ) from error
    if _is_link_or_reparse(before) or not stat.S_ISDIR(before.st_mode):
        raise ArtifactValidationError(
            "inventory root must be a real directory"
        )
    entries = _capture_directory_entries(absolute)
    _reject_windows_named_streams(absolute)
    try:
        after = os.lstat(absolute)
    except OSError as error:
        raise ArtifactValidationError(
            f"cannot restat inventory root: {absolute}"
        ) from error
    if _stat_signature(before) != _stat_signature(after):
        raise ArtifactValidationError(
            "inventory root changed during snapshot"
        )
    return DirectoryInventory(
        root=absolute,
        _root_signature=_stat_signature(after),
        _entries=entries,
    )


def verify_directory_inventory(inventory: DirectoryInventory) -> None:
    if type(inventory) is not DirectoryInventory:
        raise TypeError("inventory must be an exact DirectoryInventory")
    current = capture_directory_inventory(inventory.root)
    if (
        current._root_signature != inventory._root_signature
        or current._entries != inventory._entries
    ):
        raise ArtifactValidationError(
            "directory inventory changed after snapshot"
        )


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npy_bytes(array: np.ndarray) -> bytes:
    if type(array) is not np.ndarray:
        raise TypeError("array members must be exact ndarrays")
    if (
        array.dtype.hasobject
        or not np.issubdtype(array.dtype, np.number)
        or np.iscomplexobj(array)
        or not np.isfinite(array).all()
    ):
        raise ArtifactValidationError(
            "array members must contain finite real numeric values"
        )
    destination = io.BytesIO()
    np.save(destination, np.ascontiguousarray(array), allow_pickle=False)
    return destination.getvalue()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (0o100600 & 0xFFFF) << 16
    return info


def _zip_payload_bytes(members: Mapping[str, bytes]) -> bytes:
    destination = io.BytesIO()
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=False,
    ) as archive:
        for name in sorted(members):
            archive.writestr(
                _zip_info(name),
                members[name],
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    return destination.getvalue()


def atomic_write_bytes(path: Path, payload: bytes) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return artifact_sha256(path)


def npz_bytes(
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
) -> bytes:
    if type(arrays) is not dict:
        raise TypeError("arrays must be an exact dict")
    if type(metadata) is not dict:
        raise TypeError("metadata must be an exact dict")
    validate_exact_json_native(metadata, "metadata")
    if any(
        type(name) is not str
        or not name
        or "/" in name
        or "\\" in name
        or name == METADATA_MEMBER.removesuffix(".npy")
        for name in arrays
    ):
        raise ArtifactValidationError("unsafe array member name")
    members = {
        f"{name}.npy": _npy_bytes(array) for name, array in arrays.items()
    }
    metadata_bytes = canonical_json_bytes(metadata)
    members[METADATA_MEMBER] = _npy_bytes(
        np.frombuffer(metadata_bytes, dtype=np.uint8)
    )
    return _zip_payload_bytes(members)


def write_npz(
    path: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
) -> str:
    return atomic_write_bytes(
        Path(path),
        npz_bytes(arrays=arrays, metadata=metadata),
    )


def _read_npy(payload: bytes, member: str) -> np.ndarray:
    try:
        array = np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception as exc:
        raise ArtifactValidationError(
            f"malformed or object array member: {member}"
        ) from exc
    if (
        type(array) is not np.ndarray
        or array.dtype.hasobject
        or not np.issubdtype(array.dtype, np.number)
        or np.iscomplexobj(array)
        or not np.isfinite(array).all()
    ):
        raise ArtifactValidationError(
            f"{member} must contain real numeric finite values"
        )
    contiguous = np.ascontiguousarray(array)
    if _npy_bytes(contiguous) != payload:
        raise ArtifactValidationError(
            f"noncanonical NPY member rejected: {member}"
        )
    return contiguous


def _validate_zip_info(info: zipfile.ZipInfo) -> None:
    if (
        info.date_time != (1980, 1, 1, 0, 0, 0)
        or info.compress_type != zipfile.ZIP_DEFLATED
        or info.create_system != 3
        or info.external_attr != 0x81800000
        or info.comment != b""
        or info.extra != b""
        or info.flag_bits != 0
        or info.create_version != 20
        or info.extract_version != 20
        or info.volume != 0
        or info.internal_attr != 0
    ):
        raise ArtifactValidationError(
            f"noncanonical ZIP member metadata: {info.filename}"
        )


def read_npz_members_bytes(
    payload: bytes,
    *,
    allowed_members: set[str] | frozenset[str] | None = None,
    max_member_bytes: int = _MAX_NPZ_MEMBER_BYTES,
    max_total_bytes: int = _MAX_NPZ_TOTAL_BYTES,
) -> Mapping[str, bytes]:
    if type(payload) is not bytes:
        raise TypeError("NPZ payload must be exact bytes")
    if len(payload) > _MAX_NPZ_BYTES:
        raise ArtifactValidationError("NPZ payload exceeds byte bound")
    for value, field_name in (
        (max_member_bytes, "max_member_bytes"),
        (max_total_bytes, "max_total_bytes"),
    ):
        if type(value) is not int or value < 1:
            raise TypeError(
                f"{field_name} must be a positive exact integer"
            )
    if allowed_members is not None:
        if type(allowed_members) not in (set, frozenset) or any(
            type(name) is not str for name in allowed_members
        ):
            raise TypeError(
                "allowed_members must be an exact set of exact strings"
            )
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if archive.comment != b"":
                raise ArtifactValidationError(
                    "noncanonical ZIP archive comment"
                )
            if len(infos) > _MAX_NPZ_MEMBERS:
                raise ArtifactValidationError(
                    "ZIP member count exceeds bound"
                )
            if len(names) != len(set(names)) or any(
                name.endswith("/") for name in names
            ):
                raise ArtifactValidationError("duplicate or directory ZIP member")
            if names != sorted(names):
                raise ArtifactValidationError(
                    "noncanonical ZIP member order"
                )
            if allowed_members is not None and not set(names).issubset(
                allowed_members
            ):
                raise ArtifactValidationError("unknown ZIP member")
            total_size = 0
            for info in infos:
                _validate_zip_info(info)
                if (
                    info.file_size < 0
                    or info.file_size > max_member_bytes
                    or info.compress_size < 0
                    or info.compress_size > len(payload)
                ):
                    raise ArtifactValidationError(
                        "ZIP member size exceeds bound"
                    )
                total_size += info.file_size
                if total_size > max_total_bytes:
                    raise ArtifactValidationError(
                        "ZIP total size exceeds bound"
                    )
            members: dict[str, bytes] = {}
            for info in infos:
                chunks: list[bytes] = []
                observed = 0
                with archive.open(info, "r") as stream:
                    while True:
                        remaining = max_member_bytes + 1 - observed
                        if remaining <= 0:
                            raise ArtifactValidationError(
                                "ZIP member size exceeds bound"
                            )
                        chunk = stream.read(
                            min(_READ_BLOCK_BYTES, remaining)
                        )
                        if not chunk:
                            break
                        chunks.append(chunk)
                        observed += len(chunk)
                if observed != info.file_size:
                    raise ArtifactValidationError(
                        "ZIP member declared size mismatch"
                    )
                members[info.filename] = b"".join(chunks)
            if _zip_payload_bytes(members) != payload:
                raise ArtifactValidationError(
                    "ZIP payload does not use the canonical fixed codec"
                )
            return members
    except ArtifactValidationError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError) as exc:
        raise ArtifactValidationError("malformed or corrupted ZIP artifact") from exc


def read_npz_members(
    path: Path,
    *,
    allowed_members: set[str] | frozenset[str] | None = None,
) -> Mapping[str, bytes]:
    snapshot = read_safe_file_snapshot(
        Path(path), max_bytes=_MAX_NPZ_BYTES, noun="ZIP artifact"
    )
    members = read_npz_members_bytes(
        snapshot.raw, allowed_members=allowed_members
    )
    verify_safe_file_snapshot(snapshot)
    return members


def _unique_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactValidationError(
                f"duplicate metadata JSON key: {key}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ArtifactValidationError(
        f"non-finite metadata JSON constant rejected: {value}"
    )


def decode_metadata(members: Mapping[str, bytes]) -> Mapping[str, object]:
    if METADATA_MEMBER not in members:
        raise ArtifactValidationError("missing metadata member")
    array = _read_npy(members[METADATA_MEMBER], METADATA_MEMBER)
    if array.dtype != np.uint8 or array.ndim != 1:
        raise ArtifactValidationError("metadata member must be a uint8 vector")
    try:
        raw = array.tobytes()
        metadata = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except ArtifactValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError("malformed metadata JSON") from exc
    if type(metadata) is not dict:
        raise ArtifactValidationError("metadata JSON must be an object")
    validate_exact_json_native(metadata, "metadata")
    if canonical_json_bytes(metadata) != raw:
        raise ArtifactValidationError(
            "metadata JSON bytes are not canonical"
        )
    return metadata


def load_array_member(members: Mapping[str, bytes], name: str) -> np.ndarray:
    member = f"{name}.npy"
    return _read_npy(members[member], member)
