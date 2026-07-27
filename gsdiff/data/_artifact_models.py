"""Frozen public data shapes for capability-separated SPI artifacts."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ._artifact_identity import (
    ArtifactValidationError,
    json_native,
    optional_readonly_array,
    readonly_array,
)


@dataclass(frozen=True)
class SPIAcquisitionData:
    dataset_identity_sha256: str
    dataset_identity_spec: Mapping[str, object]
    patterns: np.ndarray
    measurements: np.ndarray
    frame_indices: np.ndarray
    time_grid: np.ndarray
    holdout_patterns: np.ndarray | None
    holdout_measurements: np.ndarray | None
    holdout_frame_indices: np.ndarray | None
    H: int
    W: int
    T: int
    K: int
    resolved_generation_config: Mapping[str, object]
    generator_code_version: str
    target_asset_sha256: str
    seed: int
    pattern_family: str
    pattern_order: str
    time_assignment_mode: str
    noise_convention: str
    noise_parameters: Mapping[str, object]
    motion_model: str
    motion_parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        for field in ("patterns", "measurements", "frame_indices", "time_grid"):
            object.__setattr__(
                self, field, readonly_array(getattr(self, field), field)
            )
        for field in (
            "holdout_patterns",
            "holdout_measurements",
            "holdout_frame_indices",
        ):
            object.__setattr__(
                self,
                field,
                optional_readonly_array(getattr(self, field), field),
            )
        for field in (
            "dataset_identity_spec",
            "resolved_generation_config",
            "noise_parameters",
            "motion_parameters",
        ):
            native = json_native(getattr(self, field))
            if not isinstance(native, dict):
                raise ArtifactValidationError(f"{field} must be a mapping")
            object.__setattr__(self, field, native)

    @property
    def frame_idx(self) -> np.ndarray:
        return self.frame_indices

    @property
    def t_grid(self) -> np.ndarray:
        return self.time_grid

    @property
    def eval_patterns(self) -> np.ndarray | None:
        return self.holdout_patterns

    @property
    def eval_measurements(self) -> np.ndarray | None:
        return self.holdout_measurements

    @property
    def eval_frame_idx(self) -> np.ndarray | None:
        return self.holdout_frame_indices


@dataclass(frozen=True)
class EvaluationTruth:
    dataset_identity_sha256: str
    dataset_identity_spec: Mapping[str, object]
    canonical_image: np.ndarray
    gt_frames: np.ndarray
    translation_trajectory: np.ndarray
    rotation_trajectory: np.ndarray
    gt_velocity: np.ndarray
    gt_acceleration: np.ndarray
    gt_omega: float
    gt_beta: float
    motion_model: str
    H: int
    W: int
    T: int
    evaluator_metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        for field in (
            "canonical_image",
            "gt_frames",
            "translation_trajectory",
            "rotation_trajectory",
            "gt_velocity",
            "gt_acceleration",
        ):
            object.__setattr__(
                self, field, readonly_array(getattr(self, field), field)
            )
        for field in ("dataset_identity_spec", "evaluator_metadata"):
            native = json_native(getattr(self, field))
            if not isinstance(native, dict):
                raise ArtifactValidationError(f"{field} must be a mapping")
            object.__setattr__(self, field, native)


@dataclass(frozen=True)
class MethodExecutionPolicy:
    execution_class: str
    truth_access: str
    promotion_eligible: bool


@dataclass(frozen=True)
class ReconstructionOutput:
    dataset_identity_sha256: str
    reconstruction: np.ndarray
    dgi: np.ndarray | None
    estimated_motion_trajectory: np.ndarray
    frame_indices: np.ndarray
    time_grid: np.ndarray
    method_name: str
    method_metadata: Mapping[str, object]
    execution_policy: MethodExecutionPolicy

    def __post_init__(self) -> None:
        for field in (
            "reconstruction",
            "estimated_motion_trajectory",
            "frame_indices",
            "time_grid",
        ):
            object.__setattr__(
                self, field, readonly_array(getattr(self, field), field)
            )
        object.__setattr__(
            self, "dgi", optional_readonly_array(self.dgi, "dgi")
        )
        native = json_native(self.method_metadata)
        if not isinstance(native, dict):
            raise ArtifactValidationError("method_metadata must be a mapping")
        object.__setattr__(self, "method_metadata", native)
