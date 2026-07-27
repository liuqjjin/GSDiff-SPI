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
    "artifact_sha256": ("._artifact_io", "artifact_sha256"),
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
    "require_promotion_eligible": (
        "._artifact_outputs",
        "require_promotion_eligible",
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
    "validate_evaluation_inputs": (
        "._artifact_outputs",
        "validate_evaluation_inputs",
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
