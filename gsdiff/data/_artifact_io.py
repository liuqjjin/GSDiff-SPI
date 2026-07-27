"""Deterministic NPY/ZIP and atomic file I/O for SPI artifacts."""

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping
import zipfile

import numpy as np

from ._artifact_identity import (
    ArtifactValidationError,
    canonical_json_bytes,
)


METADATA_MEMBER = "__metadata_json__.npy"


def artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npy_bytes(array: np.ndarray) -> bytes:
    if array.dtype.hasobject:
        raise ArtifactValidationError("object arrays cannot be serialized")
    destination = io.BytesIO()
    np.save(destination, np.ascontiguousarray(array), allow_pickle=False)
    return destination.getvalue()


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (0o100600 & 0xFFFF) << 16
    return info


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


def write_npz(
    path: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
) -> str:
    members = {
        f"{name}.npy": _npy_bytes(array) for name, array in arrays.items()
    }
    metadata_bytes = canonical_json_bytes(metadata)
    members[METADATA_MEMBER] = _npy_bytes(
        np.frombuffer(metadata_bytes, dtype=np.uint8)
    )
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
    return atomic_write_bytes(Path(path), destination.getvalue())


def _read_npy(payload: bytes, member: str) -> np.ndarray:
    try:
        array = np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception as exc:
        raise ArtifactValidationError(
            f"malformed or object array member: {member}"
        ) from exc
    if not isinstance(array, np.ndarray) or array.dtype.hasobject:
        raise ArtifactValidationError(f"object array member rejected: {member}")
    return np.ascontiguousarray(array)


def read_npz_members(path: Path) -> Mapping[str, bytes]:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or any(
                name.endswith("/") for name in names
            ):
                raise ArtifactValidationError("duplicate or directory ZIP member")
            return {name: archive.read(name) for name in names}
    except ArtifactValidationError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError) as exc:
        raise ArtifactValidationError("malformed or corrupted ZIP artifact") from exc


def decode_metadata(members: Mapping[str, bytes]) -> Mapping[str, object]:
    if METADATA_MEMBER not in members:
        raise ArtifactValidationError("missing metadata member")
    array = _read_npy(members[METADATA_MEMBER], METADATA_MEMBER)
    if array.dtype != np.uint8 or array.ndim != 1:
        raise ArtifactValidationError("metadata member must be a uint8 vector")
    try:
        metadata = json.loads(array.tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError("malformed metadata JSON") from exc
    if not isinstance(metadata, dict):
        raise ArtifactValidationError("metadata JSON must be an object")
    return metadata


def load_array_member(members: Mapping[str, bytes], name: str) -> np.ndarray:
    member = f"{name}.npy"
    return _read_npy(members[member], member)
