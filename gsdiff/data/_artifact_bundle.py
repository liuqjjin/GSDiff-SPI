"""Deterministic payload codecs and canonical dataset manifests."""

from __future__ import annotations

import hashlib
import io
import json
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np
from PIL import Image

from ._artifact_dataset import (
    BLIND_ACQUISITION_SPEC_SCHEMA,
    _validate_acquisition_identity,
    acquisition_npz_bytes,
    load_acquisition_data_bytes,
)
from ._artifact_identity import (
    ArtifactValidationError,
    canonical_json_bytes,
    validate_exact_json_native,
    validate_path_free_opaque_id,
    validate_sha256,
)
from ._artifact_models import EvaluationTruth, SPIAcquisitionData
from ._artifact_truth import (
    evaluation_truth_npz_bytes,
    load_evaluation_truth_bytes,
)
from ._corrected_generation import (
    CorrectedDataset,
    _CALIBRATION_RECORD_KEYS,
    _validate_calibration_reference_descriptor,
    _validate_corrected_config,
    validate_corrected_truth,
    validate_dataset_identity_spec,
)


DATASET_MANIFEST_SCHEMA = "dataset-manifest-v1"
PREVIEW_SCHEMA = "dataset-preview-v1"
MAX_DATASET_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_DATASET_PREVIEW_BYTES = 64 * 1024 * 1024
MAX_DATASET_NPZ_BYTES = 1024 * 1024 * 1024
_PAYLOAD_NAMES = {
    "measurements.npz",
    "evaluation-truth.npz",
    "preview.png",
}
_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "dataset_identity_sha256",
    "dataset_identity_spec",
    "resolved_generator_config",
    "noise_calibration_record",
    "files",
}
_FILE_KEYS = {
    "role",
    "schema_version",
    "sha256",
    "size_bytes",
}
_FILE_CONTRACTS = {
    "measurements.npz": (
        "blind-measurements",
        "measurements-blind-v1",
    ),
    "evaluation-truth.npz": (
        "evaluation-truth",
        "evaluation-truth-v2",
    ),
    "preview.png": ("preview", PREVIEW_SCHEMA),
}


def _payload_byte_limit(name: str) -> int:
    if name == "preview.png":
        return MAX_DATASET_PREVIEW_BYTES
    if name in {"measurements.npz", "evaluation-truth.npz"}:
        return MAX_DATASET_NPZ_BYTES
    raise ArtifactValidationError(f"unknown dataset payload: {name}")


def _validate_payload_byte_bounds(payloads: Mapping[str, bytes]) -> None:
    for name, payload in payloads.items():
        if len(payload) > _payload_byte_limit(name):
            raise ArtifactValidationError(
                f"{name} payload exceeds its role byte bound"
            )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _native_copy(value: object) -> object:
    validate_exact_json_native(value)
    if type(value) is dict:
        return {
            key: _native_copy(child)
            for key, child in value.items()
        }
    if type(value) is list:
        return [_native_copy(child) for child in value]
    if type(value) is tuple:
        return [_native_copy(child) for child in value]
    if type(value) is MappingProxyType:
        return {
            key: _native_copy(child)  # type: ignore[union-attr]
            for key, child in value.items()  # type: ignore[union-attr]
        }
    return value


def _exact_dict(
    value: object,
    field: str,
    keys: set[str],
) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{field} must be an exact dict")
    validate_exact_json_native(value, field)
    actual = set(value)
    if actual != keys:
        raise ArtifactValidationError(
            f"{field} keys mismatch; "
            f"missing={sorted(keys - actual)}, "
            f"extra={sorted(actual - keys)}"
        )
    return value


def _png_bytes(image: np.ndarray) -> bytes:
    if type(image) is not np.ndarray or image.dtype != np.uint8:
        raise TypeError("preview image must be an exact uint8 ndarray")
    if image.ndim != 2 or not image.flags.c_contiguous:
        raise ArtifactValidationError(
            "preview image must be a C-contiguous grayscale matrix"
        )
    destination = io.BytesIO()
    Image.fromarray(image, mode="L").save(
        destination,
        format="PNG",
        optimize=False,
        compress_level=9,
        pnginfo=None,
    )
    return destination.getvalue()


def _quantized_preview(truth: EvaluationTruth) -> np.ndarray:
    return np.ascontiguousarray(
        np.rint(
            np.clip(truth.canonical_image, 0.0, 1.0) * 255.0
        ).astype(np.uint8)
    )


def _decode_preview(
    payload: bytes,
    *,
    H: int,
    W: int,
) -> np.ndarray:
    if type(payload) is not bytes:
        raise TypeError("preview payload must be exact bytes")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if (
                image.format != "PNG"
                or image.mode != "L"
                or image.size != (W, H)
                or getattr(image, "is_animated", False)
                or image.info
            ):
                raise ArtifactValidationError(
                    "preview PNG mode, dimensions, or metadata mismatch"
                )
            decoded = np.ascontiguousarray(
                np.asarray(image, dtype=np.uint8)
            )
    except ArtifactValidationError:
        raise
    except (OSError, ValueError) as error:
        raise ArtifactValidationError("malformed preview PNG") from error
    if _png_bytes(decoded) != payload:
        raise ArtifactValidationError("preview PNG is not canonical")
    return decoded


def build_dataset_payloads(
    generated: CorrectedDataset,
) -> dict[str, bytes]:
    if type(generated) is not CorrectedDataset:
        raise TypeError("generated must be an exact CorrectedDataset")
    _validate_acquisition_identity(generated.acquisition)
    validate_corrected_truth(generated.truth)
    if (
        generated.acquisition.dataset_identity_sha256
        != generated.dataset_identity_sha256
        or generated.truth.dataset_identity_sha256
        != generated.dataset_identity_sha256
    ):
        raise ArtifactValidationError(
            "generated payload identities disagree"
        )
    return {
        "measurements.npz": acquisition_npz_bytes(
            generated.acquisition
        ),
        "evaluation-truth.npz": evaluation_truth_npz_bytes(
            generated.truth
        ),
        "preview.png": _png_bytes(
            _quantized_preview(generated.truth)
        ),
    }


def _validate_payload_mapping(
    payloads: object,
) -> dict[str, bytes]:
    if type(payloads) is not dict:
        raise TypeError("payloads must be an exact dict")
    if set(payloads) != _PAYLOAD_NAMES:
        raise ArtifactValidationError(
            "payload names must match the fixed dataset inventory"
        )
    for name, payload in payloads.items():
        if type(name) is not str or type(payload) is not bytes:
            raise TypeError("payload names and values must be exact native types")
    _validate_payload_byte_bounds(payloads)
    return payloads


def build_dataset_manifest(
    generated: CorrectedDataset,
    payloads: Mapping[str, bytes],
) -> dict[str, object]:
    if type(generated) is not CorrectedDataset:
        raise TypeError("generated must be an exact CorrectedDataset")
    exact_payloads = _validate_payload_mapping(payloads)
    validate_corrected_truth(generated.truth)
    manifest = {
        "schema_version": DATASET_MANIFEST_SCHEMA,
        "status": "complete",
        "dataset_identity_sha256": generated.dataset_identity_sha256,
        "dataset_identity_spec": generated.dataset_identity_spec,
        "resolved_generator_config": (
            generated.resolved_generator_config
        ),
        "noise_calibration_record": (
            generated.noise_calibration_record
        ),
        "files": {
            name: {
                "role": _FILE_CONTRACTS[name][0],
                "schema_version": _FILE_CONTRACTS[name][1],
                "sha256": _sha256(exact_payloads[name]),
                "size_bytes": len(exact_payloads[name]),
            }
            for name in sorted(exact_payloads)
        },
    }
    validate_dataset_manifest(manifest)
    return manifest


def _same_json(left: object, right: object) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _validate_calibration_record(
    record: object,
    *,
    identity: Mapping[str, object],
    config: Mapping[str, object],
) -> None:
    mapping = _exact_dict(
        record,
        "noise_calibration_record",
        _CALIBRATION_RECORD_KEYS,
    )
    if mapping["schema_version"] != "noise-calibration-record-v1":
        raise ArtifactValidationError(
            "noise calibration record schema mismatch"
        )
    calibration = _exact_dict(
        mapping["calibration"],
        "noise calibration descriptor",
        {"id", "registry_entry_sha256"},
    )
    validate_path_free_opaque_id(
        calibration["id"], "noise calibration id"
    )
    validate_sha256(
        calibration["registry_entry_sha256"],
        "noise calibration registry entry",
    )
    validate_sha256(
        mapping["reference_cell_sha256"],
        "noise calibration reference cell",
    )
    _validate_calibration_reference_descriptor(
        mapping["reference_measurements"],
        expected_count=config["dimensions"]["K"],
    )
    dimensions = config["dimensions"]
    realized = _exact_dict(
        mapping["realized_snr_db"],
        "realized SNR",
        {"train", "holdout"},
    )
    for name in ("train", "holdout"):
        value = realized[name]
        if value is not None and (
            type(value) not in (int, float) or not math.isfinite(value)
        ):
            raise ArtifactValidationError(
                f"realized SNR {name} must be finite or null"
            )
    for name in (
        "requested_snr_db",
        "reference_variance",
        "sigma_absolute",
    ):
        value = mapping[name]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ArtifactValidationError(f"{name} must be finite")
    if (
        mapping["reference_variance"] < 0
        or mapping["sigma_absolute"] < 0
        or type(mapping["ddof"]) is not int
        or mapping["ddof"] != 0
    ):
        raise ArtifactValidationError(
            "noise calibration numeric contract mismatch"
        )
    expected_sigma = math.sqrt(mapping["reference_variance"]) * 10.0 ** (
        -mapping["requested_snr_db"] / 20.0
    )
    acquisition = config["acquisition"]
    if (
        mapping["sigma_absolute"] != expected_sigma
        or mapping["requested_snr_db"] != acquisition["snr_db"]
        or calibration["id"] != acquisition["noise_calibration_id"]
    ):
        raise ArtifactValidationError(
            "noise calibration disagrees with generator config"
        )
    if (
        not _same_json(
            mapping["scientific_contract"],
            identity["scientific_contract"],
        )
        or mapping["target_id"] != identity["target"]["id"]
        or mapping["motion_id"] != identity["motion"]["id"]
        or mapping["seed"] != identity["seed"]
        or not _same_json(mapping["generator"], identity["generator"])
        or not _same_json(mapping["runtime"], identity["runtime"])
        or mapping["generator_config_sha256"]
        != identity["generator_config_sha256"]
        or calibration["id"] != identity["noise_calibration"]["id"]
        or _sha256(canonical_json_bytes(mapping))
        != identity["noise_calibration"]["sha256"]
    ):
        raise ArtifactValidationError(
            "noise calibration disagrees with dataset identity"
        )


def validate_dataset_manifest(
    value: object,
) -> dict[str, object]:
    manifest = _exact_dict(
        value, "dataset manifest", _MANIFEST_KEYS
    )
    if (
        manifest["schema_version"] != DATASET_MANIFEST_SCHEMA
        or manifest["status"] != "complete"
    ):
        raise ArtifactValidationError(
            "dataset manifest schema or status mismatch"
        )
    validate_sha256(
        manifest["dataset_identity_sha256"], "dataset identity"
    )
    identity = validate_dataset_identity_spec(
        manifest["dataset_identity_spec"]
    )
    if (
        _sha256(canonical_json_bytes(identity))
        != manifest["dataset_identity_sha256"]
    ):
        raise ArtifactValidationError(
            "dataset manifest identity hash mismatch"
        )
    config = manifest["resolved_generator_config"]
    if type(config) is not dict:
        raise TypeError("resolved_generator_config must be an exact dict")
    _validate_corrected_config(config, identity)
    _validate_calibration_record(
        manifest["noise_calibration_record"],
        identity=identity,
        config=config,
    )
    files = _exact_dict(
        manifest["files"], "dataset manifest files", _PAYLOAD_NAMES
    )
    for name, contract in _FILE_CONTRACTS.items():
        entry = _exact_dict(
            files[name],
            f"dataset manifest file {name}",
            _FILE_KEYS,
        )
        if (
            entry["role"] != contract[0]
            or entry["schema_version"] != contract[1]
        ):
            raise ArtifactValidationError(
                f"dataset manifest file contract mismatch: {name}"
            )
        validate_sha256(entry["sha256"], f"{name} payload")
        if (
            type(entry["size_bytes"]) is not int
            or entry["size_bytes"] < 1
        ):
            raise ArtifactValidationError(
                f"{name} size must be a positive exact integer"
            )
        if entry["size_bytes"] > _payload_byte_limit(name):
            raise ArtifactValidationError(
                f"{name} declared size exceeds its role byte bound"
            )
    return manifest


def dataset_manifest_bytes(manifest: Mapping[str, object]) -> bytes:
    if type(manifest) is not dict:
        raise TypeError("dataset manifest must be an exact dict")
    validate_dataset_manifest(manifest)
    payload = canonical_json_bytes(manifest)
    if len(payload) > MAX_DATASET_MANIFEST_BYTES:
        raise ArtifactValidationError(
            "dataset manifest exceeds its byte bound"
        )
    return payload


def _unique_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactValidationError(
                f"duplicate JSON key: {key}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ArtifactValidationError(
        f"non-finite JSON constant rejected: {value}"
    )


def parse_dataset_manifest_bytes(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes:
        raise TypeError("manifest payload must be exact bytes")
    if len(payload) > MAX_DATASET_MANIFEST_BYTES:
        raise ArtifactValidationError(
            "dataset manifest exceeds its byte bound"
        )
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except ArtifactValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(
            "dataset manifest is not strict UTF-8 JSON"
        ) from error
    manifest = validate_dataset_manifest(value)
    try:
        canonical = canonical_json_bytes(manifest)
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise ArtifactValidationError(
            "dataset manifest contains invalid native JSON"
        ) from error
    if payload != canonical:
        raise ArtifactValidationError(
            "dataset manifest bytes are not canonical"
        )
    native = _native_copy(manifest)
    if type(native) is not dict:
        raise RuntimeError("validated dataset manifest is not an object")
    return native


def verify_dataset_payload_bytes(
    payloads: Mapping[str, bytes],
    manifest: Mapping[str, object],
) -> tuple[SPIAcquisitionData, EvaluationTruth, np.ndarray]:
    exact_payloads = _validate_payload_mapping(payloads)
    _validate_payload_byte_bounds(exact_payloads)
    exact_manifest = validate_dataset_manifest(manifest)
    files = exact_manifest["files"]
    for name, payload in exact_payloads.items():
        entry = files[name]
        if (
            _sha256(payload) != entry["sha256"]
            or len(payload) != entry["size_bytes"]
        ):
            raise ArtifactValidationError(
                f"dataset payload hash or size mismatch: {name}"
            )
    identity_sha256 = exact_manifest["dataset_identity_sha256"]
    config = exact_manifest["resolved_generator_config"]
    calibration = exact_manifest["noise_calibration_record"]
    expected_acquisition_spec = {
        "schema_version": BLIND_ACQUISITION_SPEC_SCHEMA,
        "dimensions": config["dimensions"],
        "acquisition": {
            "pattern_family": config["acquisition"]["pattern_family"],
            "pattern_values": config["acquisition"]["pattern_values"],
            "pattern_order": config["acquisition"]["pattern_order"],
            "time_assignment": config["acquisition"]["time_assignment"],
            "holdout_pattern_family": config["acquisition"][
                "holdout_pattern_family"
            ],
            "noise_convention": "detector-absolute",
            "noise_sigma_absolute": calibration["sigma_absolute"],
        },
    }
    acquisition = load_acquisition_data_bytes(
        exact_payloads["measurements.npz"],
        expected_dataset_identity_sha256=identity_sha256,
        expected_acquisition_spec=expected_acquisition_spec,
    )
    truth = load_evaluation_truth_bytes(
        exact_payloads["evaluation-truth.npz"],
        expected_dataset_identity_sha256=identity_sha256,
    )
    if (
        not _same_json(
            truth.dataset_identity_spec,
            exact_manifest["dataset_identity_spec"],
        )
        or not _same_json(
            truth.evaluator_metadata["resolved_generator_config"],
            exact_manifest["resolved_generator_config"],
        )
        or not _same_json(
            truth.evaluator_metadata["noise_calibration_record"],
            exact_manifest["noise_calibration_record"],
        )
    ):
        raise ArtifactValidationError(
            "truth payload disagrees with dataset manifest"
        )
    preview = _decode_preview(
        exact_payloads["preview.png"],
        H=truth.H,
        W=truth.W,
    )
    if not np.array_equal(preview, _quantized_preview(truth)):
        raise ArtifactValidationError(
            "preview content disagrees with evaluation truth"
        )
    return acquisition, truth, preview
