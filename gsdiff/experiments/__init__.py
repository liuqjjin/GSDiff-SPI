"""Reproducibility identity helpers for experiments."""

from importlib import import_module

from .identity import (
    canonical_json_bytes,
    collect_environment_fingerprint,
    collect_runtime_metadata,
    sha256_bytes,
)
from .methods import (
    AlgorithmSeed,
    CheckpointRequirement,
    ResolvedMethod,
    canonical_method_id,
    derive_algorithm_seed,
    resolve_method_semantics,
)
from .objectives import (
    BlindObjective,
    heldout_normalized_l2,
    select_by_heldout_normalized_l2,
)
from .child_outputs import (
    MethodChildResult,
    ReconstructionV2,
    load_reconstruction_v2,
    validate_method_child_outputs_v2,
    write_method_child_outputs_v2,
)
from .adapters import (
    BASELINE_METHOD_IDS,
    run_baseline_method,
    run_canonical_method,
)
from .execution import (
    MaterializedMethodExecution,
    MaterializedMethodRequest,
    load_materialized_method_request,
    materialize_method_execution,
)
from .audit import validate_audit_log

_LAZY_GSDIFF_EXPORTS = {
    "GSDIFF_METHOD_IDS",
    "run_gsdiff_method",
}


def __getattr__(name):
    if name not in _LAZY_GSDIFF_EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(".gsdiff_adapter", package=__name__), name)
    globals()[name] = value
    return value

__all__ = [
    "canonical_json_bytes",
    "collect_environment_fingerprint",
    "collect_runtime_metadata",
    "sha256_bytes",
    "AlgorithmSeed",
    "CheckpointRequirement",
    "ResolvedMethod",
    "canonical_method_id",
    "derive_algorithm_seed",
    "resolve_method_semantics",
    "BlindObjective",
    "heldout_normalized_l2",
    "select_by_heldout_normalized_l2",
    "MethodChildResult",
    "ReconstructionV2",
    "load_reconstruction_v2",
    "validate_method_child_outputs_v2",
    "write_method_child_outputs_v2",
    "BASELINE_METHOD_IDS",
    "run_baseline_method",
    "GSDIFF_METHOD_IDS",
    "run_gsdiff_method",
    "run_canonical_method",
    "MaterializedMethodExecution",
    "MaterializedMethodRequest",
    "load_materialized_method_request",
    "materialize_method_execution",
    "validate_audit_log",
]
