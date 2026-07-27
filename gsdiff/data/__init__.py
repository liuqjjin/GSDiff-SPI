"""Data API with capability-sensitive imports resolved on demand."""

from importlib import import_module


_EXPORTS = {
    "generate_spi_data": (".simulation", "generate_spi_data"),
    "SPIData": (".simulation", "SPIData"),
    "generate_patterns": (".patterns", "generate_patterns"),
    "dgi_reconstruct": (".dgi", "dgi_reconstruct"),
    "ArtifactValidationError": (
        "._artifact_identity",
        "ArtifactValidationError",
    ),
    "CorrectedDataset": ("._corrected_generation", "CorrectedDataset"),
    "DatasetDirectoryDiscovery": (
        "._artifact_persistence",
        "DatasetDirectoryDiscovery",
    ),
    "DatasetPayloadEvidence": (
        "._artifact_persistence",
        "DatasetPayloadEvidence",
    ),
    "DatasetPublication": (
        "._artifact_persistence",
        "DatasetPublication",
    ),
    "EvaluationTruth": ("._artifact_models", "EvaluationTruth"),
    "MethodExecutionPolicy": (
        "._artifact_models",
        "MethodExecutionPolicy",
    ),
    "ReconstructionOutput": (
        "._artifact_models",
        "ReconstructionOutput",
    ),
    "SPIAcquisitionData": ("._artifact_models", "SPIAcquisitionData"),
    "TargetSnapshot": ("._corrected_generation", "TargetSnapshot"),
    "VerifiedDatasetDirectory": (
        "._artifact_persistence",
        "VerifiedDatasetDirectory",
    ),
    "acquisition_rng": ("._corrected_generation", "acquisition_rng"),
    "artifact_sha256": ("._artifact_io", "artifact_sha256"),
    "blind_acquisition_spec": (
        "._artifact_dataset",
        "blind_acquisition_spec",
    ),
    "build_dataset_manifest": (
        "._artifact_bundle",
        "build_dataset_manifest",
    ),
    "build_dataset_payloads": (
        "._artifact_bundle",
        "build_dataset_payloads",
    ),
    "dataset_manifest_bytes": (
        "._artifact_bundle",
        "dataset_manifest_bytes",
    ),
    "discover_dataset_directories": (
        "._artifact_persistence",
        "discover_dataset_directories",
    ),
    "generate_corrected_dataset": (
        "._corrected_generation",
        "generate_corrected_dataset",
    ),
    "resolve_corrected_dataset_request": (
        "._corrected_generation",
        "resolve_corrected_dataset_request",
    ),
    "load_acquisition_data": (
        "._artifact_dataset",
        "load_acquisition_data",
    ),
    "load_evaluation_truth": (
        "._artifact_truth",
        "load_evaluation_truth",
    ),
    "load_reconstruction_output": (
        "._artifact_outputs",
        "load_reconstruction_output",
    ),
    "method_execution_policy": (
        "._artifact_outputs",
        "method_execution_policy",
    ),
    "parse_dataset_manifest_bytes": (
        "._artifact_bundle",
        "parse_dataset_manifest_bytes",
    ),
    "publish_dataset": (
        "._artifact_persistence",
        "publish_dataset",
    ),
    "require_promotion_eligible": (
        "._artifact_outputs",
        "require_promotion_eligible",
    ),
    "resolve_target_snapshot": (
        "._corrected_generation",
        "resolve_target_snapshot",
    ),
    "save_acquisition_data": (
        "._artifact_dataset",
        "save_acquisition_data",
    ),
    "save_evaluation_truth": (
        "._artifact_truth",
        "save_evaluation_truth",
    ),
    "split_spi_data": ("._artifact_dataset", "split_spi_data"),
    "validate_dataset_identity_spec": (
        "._corrected_generation",
        "validate_dataset_identity_spec",
    ),
    "validate_evaluation_inputs": (
        "._artifact_outputs",
        "validate_evaluation_inputs",
    ),
    "verify_dataset_payload_bytes": (
        "._artifact_bundle",
        "verify_dataset_payload_bytes",
    ),
    "verify_dataset_directory": (
        "._artifact_persistence",
        "verify_dataset_directory",
    ),
    "verify_dataset_directory_discovery": (
        "._artifact_persistence",
        "verify_dataset_directory_discovery",
    ),
    "write_method_child_outputs": (
        "._artifact_outputs",
        "write_method_child_outputs",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(
        import_module(module_name, package=__name__),
        attribute_name,
    )
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
