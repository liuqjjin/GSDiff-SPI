"""Public API for immutable, capability-separated SPI artifacts.

The measurements artifact is sufficient for reconstruction but contains no
evaluation target. The truth artifact is resolved only when an evaluator
explicitly requests that capability.
"""

from importlib import import_module


__all__ = [
    "ArtifactValidationError",
    "CorrectedDataset",
    "DatasetDirectoryDiscovery",
    "DatasetPayloadEvidence",
    "DatasetPublication",
    "EvaluationTruth",
    "MethodExecutionPolicy",
    "ReconstructionOutput",
    "SPIAcquisitionData",
    "TargetSnapshot",
    "VerifiedDatasetDirectory",
    "acquisition_rng",
    "artifact_sha256",
    "blind_acquisition_spec",
    "build_dataset_manifest",
    "build_dataset_payloads",
    "dataset_manifest_bytes",
    "discover_dataset_directories",
    "generate_corrected_dataset",
    "load_acquisition_data",
    "load_evaluation_truth",
    "load_reconstruction_output",
    "method_execution_policy",
    "parse_dataset_manifest_bytes",
    "publish_dataset",
    "require_promotion_eligible",
    "resolve_corrected_dataset_request",
    "resolve_target_snapshot",
    "save_acquisition_data",
    "save_evaluation_truth",
    "split_spi_data",
    "validate_dataset_identity_spec",
    "validate_evaluation_inputs",
    "verify_dataset_payload_bytes",
    "verify_canonical_dataset_directory_discovery",
    "verify_dataset_directory",
    "verify_dataset_directory_discovery",
    "write_method_child_outputs",
]

_BUNDLE_EXPORTS = {
    "build_dataset_manifest",
    "build_dataset_payloads",
    "dataset_manifest_bytes",
    "parse_dataset_manifest_bytes",
    "verify_dataset_payload_bytes",
}
_CORRECTED_EXPORTS = {
    "CorrectedDataset",
    "TargetSnapshot",
    "acquisition_rng",
    "generate_corrected_dataset",
    "resolve_corrected_dataset_request",
    "resolve_target_snapshot",
    "validate_dataset_identity_spec",
}
_PERSISTENCE_EXPORTS = {
    "DatasetDirectoryDiscovery",
    "DatasetPayloadEvidence",
    "DatasetPublication",
    "VerifiedDatasetDirectory",
    "discover_dataset_directories",
    "publish_dataset",
    "verify_canonical_dataset_directory_discovery",
    "verify_dataset_directory",
    "verify_dataset_directory_discovery",
}
_DATASET_EXPORTS = {
    "blind_acquisition_spec",
    "load_acquisition_data",
    "save_acquisition_data",
    "split_spi_data",
}
_TRUTH_EXPORTS = {
    "load_evaluation_truth",
    "save_evaluation_truth",
}
_MODEL_EXPORTS = {
    "EvaluationTruth",
    "MethodExecutionPolicy",
    "ReconstructionOutput",
    "SPIAcquisitionData",
}
_OUTPUT_EXPORTS = {
    "load_reconstruction_output",
    "method_execution_policy",
    "require_promotion_eligible",
    "validate_evaluation_inputs",
    "write_method_child_outputs",
}


def __getattr__(name):
    if name in _BUNDLE_EXPORTS:
        module_name = "._artifact_bundle"
    elif name in _PERSISTENCE_EXPORTS:
        module_name = "._artifact_persistence"
    elif name in _CORRECTED_EXPORTS:
        module_name = "._corrected_generation"
    elif name in _DATASET_EXPORTS:
        module_name = "._artifact_dataset"
    elif name in _TRUTH_EXPORTS:
        module_name = "._artifact_truth"
    elif name == "ArtifactValidationError":
        module_name = "._artifact_identity"
    elif name == "artifact_sha256":
        module_name = "._artifact_io"
    elif name in _MODEL_EXPORTS:
        module_name = "._artifact_models"
    elif name in _OUTPUT_EXPORTS:
        module_name = "._artifact_outputs"
    else:
        raise AttributeError(name)
    value = getattr(import_module(module_name, package=__package__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
