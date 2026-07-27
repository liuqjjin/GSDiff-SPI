"""Public API for immutable, capability-separated SPI artifacts.

The measurements artifact is sufficient for reconstruction but contains no
evaluation target.  The truth artifact is opened only by an evaluator that
already knows the expected dataset identity.
"""

from ._artifact_dataset import (
    load_acquisition_data,
    load_evaluation_truth,
    save_acquisition_data,
    save_evaluation_truth,
    split_spi_data,
)
from ._artifact_identity import ArtifactValidationError
from ._artifact_io import artifact_sha256
from ._artifact_models import (
    EvaluationTruth,
    MethodExecutionPolicy,
    ReconstructionOutput,
    SPIAcquisitionData,
)
from ._artifact_outputs import (
    load_reconstruction_output,
    method_execution_policy,
    require_promotion_eligible,
    validate_evaluation_inputs,
    write_method_child_outputs,
)


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
