"""Cross-lock verified dataset identity semantics to experiment protocol."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib

from .identity import canonical_json_bytes


def dataset_measurement_record(
    dataset_manifest: Mapping[str, object],
) -> dict[str, object]:
    config = dataset_manifest["resolved_generator_config"]
    calibration = dataset_manifest["noise_calibration_record"]
    assert isinstance(config, Mapping) and isinstance(calibration, Mapping)
    dimensions = config["dimensions"]
    acquisition = config["acquisition"]
    realized = calibration["realized_snr_db"]
    assert all(
        isinstance(value, Mapping)
        for value in (dimensions, acquisition, realized)
    )
    return {
        "train_count": dimensions["K"],
        "holdout_count": dimensions["holdout_K"],
        "pattern_family": acquisition["pattern_family"],
        "requested_snr_db": acquisition["snr_db"],
        "noise_calibration_id": acquisition["noise_calibration_id"],
        "noise_calibration_sha256": hashlib.sha256(
            canonical_json_bytes(calibration)
        ).hexdigest(),
        "noise_sigma_absolute": calibration["sigma_absolute"],
        "realized_train_snr_db": realized["train"],
        "realized_holdout_snr_db": realized["holdout"],
    }


def build_dataset_input_contract(verified_dataset) -> dict[str, object]:
    return {
        "dataset_manifest_sha256": verified_dataset.dataset_manifest_sha256,
        "measurements_file_sha256": verified_dataset.payload_evidence[
            "measurements.npz"
        ].sha256,
        "evaluation_truth_file_sha256": verified_dataset.payload_evidence[
            "evaluation-truth.npz"
        ].sha256,
        "measurement": dataset_measurement_record(verified_dataset.manifest),
    }


def validate_dataset_protocol_binding(
    dataset_manifest: Mapping[str, object],
    *,
    scientific_contract_id: str,
    scientific_contract_sha256: str,
    target_id: str,
    motion_id: str,
    seed: int,
    assets_sha256: Mapping[str, str],
) -> None:
    identity = dataset_manifest.get("dataset_identity_spec")
    if not isinstance(identity, Mapping):
        raise ValueError("dataset identity spec is missing")
    scientific = identity.get("scientific_contract")
    target = identity.get("target")
    motion = identity.get("motion")
    if not all(isinstance(value, Mapping) for value in (scientific, target, motion)):
        raise ValueError("dataset identity protocol projection is invalid")
    observed = {
        "scientific_contract_id": scientific.get("id"),
        "scientific_contract_sha256": scientific.get("sha256"),
        "target_id": target.get("id"),
        "motion_id": motion.get("id"),
        "seed": identity.get("seed"),
    }
    expected = {
        "scientific_contract_id": scientific_contract_id,
        "scientific_contract_sha256": scientific_contract_sha256,
        "target_id": target_id,
        "motion_id": motion_id,
        "seed": seed,
    }
    if observed != expected:
        raise ValueError(
            "dataset identity semantics disagree with experiment protocol"
        )
    config = dataset_manifest.get("resolved_generator_config")
    if not isinstance(config, Mapping):
        raise ValueError("dataset resolved generator config is missing")
    resolved_target = config.get("target")
    if not isinstance(resolved_target, Mapping):
        raise ValueError("dataset resolved target is invalid")
    observed_assets = resolved_target.get("assets_sha256")
    descriptor = resolved_target.get("descriptor")
    if not isinstance(observed_assets, Mapping):
        raise ValueError("dataset target assets are invalid")
    projected_assets = dict(observed_assets)
    if (
        type(descriptor) is str
        and descriptor in observed_assets
        and len(observed_assets) == 1
    ):
        projected_assets = {target_id: observed_assets[descriptor]}
    if projected_assets != dict(assets_sha256):
        raise ValueError(
            "dataset target assets disagree with experiment inputs"
        )
