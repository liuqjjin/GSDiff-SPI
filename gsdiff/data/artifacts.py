"""Public API for immutable, capability-separated SPI artifacts.

The measurements artifact is sufficient for reconstruction but contains no
evaluation target. The truth artifact is resolved only when an evaluator
explicitly requests that capability.
"""

from importlib import import_module


__all__ = [
    "ArtifactValidationError",
    "EvaluationTruth",
    "MethodExecutionPolicy",
    "ReconstructionOutput",
    "SPIAcquisitionData",
    "artifact_sha256",
    "load_acquisition_data",
    "load_evaluation_truth",
    "load_reconstruction_output",
    "method_execution_policy",
    "require_promotion_eligible",
    "save_acquisition_data",
    "save_evaluation_truth",
    "split_spi_data",
    "validate_evaluation_inputs",
    "write_method_child_outputs",
]

_DATASET_EXPORTS = {
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
    if name in _DATASET_EXPORTS:
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
