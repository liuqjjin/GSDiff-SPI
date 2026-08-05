"""Strict Draft 2020-12 validation for canonical versioned JSON documents."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import json
from pathlib import Path
import re
from types import MappingProxyType

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from referencing import Registry, Resource
from referencing.exceptions import CannotDetermineSpecification, Unresolvable

from .identity import canonical_json_bytes


_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_DIRECTORY = _ROOT / "schemas"
_VERSION = re.compile(
    r"[a-z0-9]+(?:-[a-z0-9]+)*-v(?:0|[1-9][0-9]*)\Z",
    re.ASCII,
)


class VersionedJSONError(ValueError):
    """A JSON document or one of its versioned schemas is invalid."""


def validate_versioned_json(
    value: object,
    expected_schema_version: str,
    *,
    schema_directory: Path = DEFAULT_SCHEMA_DIRECTORY,
) -> dict[str, object]:
    """Return one plain JSON object after exact versioned schema validation."""

    version = _schema_version(expected_schema_version)
    if type(value) is not dict:
        raise VersionedJSONError("versioned JSON document must be an exact object")
    try:
        payload = canonical_json_bytes(value)
        normalized = _decode_json(payload, noun="versioned JSON document")
    except (TypeError, ValueError, UnicodeError) as error:
        raise VersionedJSONError(
            "versioned JSON document is not finite exact JSON"
        ) from error
    if type(normalized) is not dict:
        raise VersionedJSONError("versioned JSON document must be an exact object")
    if normalized.get("schema_version") != version:
        raise VersionedJSONError("versioned JSON schema_version is invalid")
    validator = _validator_for(_schema_directory_key(schema_directory), version)
    try:
        errors = sorted(
            validator.iter_errors(normalized),
            key=lambda error: (
                tuple(str(part) for part in error.absolute_path),
                error.message,
            ),
        )
    except Unresolvable as error:
        raise VersionedJSONError("versioned JSON schema resolution failed") from error
    if errors:
        details = "; ".join(
            _schema_error_message(error.absolute_path, error.message)
            for error in errors
        )
        raise VersionedJSONError(f"versioned JSON schema validation failed: {details}")
    return normalized


def validate_canonical_versioned_json_bytes(
    payload: bytes,
    expected_schema_version: str,
    *,
    noun: str = "versioned JSON document",
    schema_directory: Path = DEFAULT_SCHEMA_DIRECTORY,
) -> dict[str, object]:
    """Validate exact canonical bytes and return their plain JSON object."""

    if type(payload) is not bytes:
        raise TypeError("versioned JSON payload must be exact bytes")
    try:
        value = _decode_json(payload, noun=noun)
        canonical = canonical_json_bytes(value)
    except (TypeError, ValueError, UnicodeError) as error:
        raise VersionedJSONError(f"{noun} is not valid finite JSON") from error
    if payload != canonical:
        raise VersionedJSONError(f"{noun} is not canonical JSON")
    return validate_versioned_json(
        value,
        expected_schema_version,
        schema_directory=schema_directory,
    )


def load_canonical_versioned_json(
    path: Path,
    expected_schema_version: str,
    *,
    noun: str = "versioned JSON document",
    schema_directory: Path = DEFAULT_SCHEMA_DIRECTORY,
) -> tuple[dict[str, object], bytes]:
    """Read and validate one exact canonical versioned JSON file."""

    if not isinstance(path, Path):
        raise TypeError("versioned JSON path must be a Path")
    payload = path.read_bytes()
    value = validate_canonical_versioned_json_bytes(
        payload,
        expected_schema_version,
        noun=noun,
        schema_directory=schema_directory,
    )
    return value, payload


def _schema_version(value: object) -> str:
    if type(value) is not str or _VERSION.fullmatch(value) is None:
        raise VersionedJSONError("expected schema version is not canonical")
    return value


def _schema_directory_key(path: object) -> str:
    if not isinstance(path, Path):
        raise TypeError("schema_directory must be a Path")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise VersionedJSONError("schema directory is unavailable") from error
    if not resolved.is_dir():
        raise VersionedJSONError("schema directory is not a directory")
    return str(resolved)


@lru_cache(maxsize=16)
def _schema_bundle(
    directory_key: str,
) -> tuple[Mapping[str, dict[str, object]], Registry]:
    directory = Path(directory_key)
    schemas: dict[str, dict[str, object]] = {}
    resources: list[tuple[str, Resource]] = []
    try:
        paths = sorted(directory.glob("*.schema.json"), key=lambda path: path.name)
    except OSError as error:
        raise VersionedJSONError("schema directory cannot be enumerated") from error
    if not paths:
        raise VersionedJSONError("schema directory contains no JSON schemas")
    for path in paths:
        try:
            payload = path.read_bytes()
            schema = _decode_json(payload, noun=f"schema {path.name}")
        except (OSError, TypeError, ValueError, UnicodeError) as error:
            raise VersionedJSONError(f"schema {path.name} is not valid JSON") from error
        if type(schema) is not dict:
            raise VersionedJSONError(f"schema {path.name} is not an exact object")
        schema_id = schema.get("$id")
        if schema_id is not None and schema_id != path.name:
            raise VersionedJSONError(
                f"schema {path.name} declares a mismatched $id"
            )
        try:
            Draft202012Validator.check_schema(schema)
            resource = Resource.from_contents(schema)
        except (CannotDetermineSpecification, SchemaError, ValueError) as error:
            raise VersionedJSONError(f"schema {path.name} is invalid") from error
        schemas[path.name] = schema
        resources.append((path.name, resource))
    registry = Registry().with_resources(resources)
    return MappingProxyType(schemas), registry


@lru_cache(maxsize=64)
def _validator_for(
    directory_key: str,
    expected_schema_version: str,
) -> Draft202012Validator:
    schemas, registry = _schema_bundle(directory_key)
    filename = f"{expected_schema_version}.schema.json"
    schema = schemas.get(filename)
    if schema is None:
        raise VersionedJSONError(
            f"schema file is missing for {expected_schema_version}"
        )
    return Draft202012Validator(schema, registry=registry)


def _decode_json(payload: bytes, *, noun: str) -> object:
    if type(payload) is not bytes:
        raise TypeError(f"{noun} payload must be exact bytes")
    return json.loads(
        payload.decode("utf-8", errors="strict"),
        object_pairs_hook=_unique_object,
        parse_constant=lambda token: _reject_constant(noun, token),
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, child in pairs:
        if key in value:
            raise VersionedJSONError(f"duplicate JSON key: {key!r}")
        value[key] = child
    return value


def _reject_constant(noun: str, token: str) -> object:
    raise VersionedJSONError(f"{noun} contains non-finite constant {token}")


def _schema_error_message(path: object, message: str) -> str:
    components = [str(part) for part in path]
    location = "$" if not components else "$." + ".".join(components)
    return f"{location}: {message}"


__all__ = [
    "DEFAULT_SCHEMA_DIRECTORY",
    "VersionedJSONError",
    "load_canonical_versioned_json",
    "validate_canonical_versioned_json_bytes",
    "validate_versioned_json",
]
