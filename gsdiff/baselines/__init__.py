"""Baseline reconstructors for the GSDiff-SPI comparison suite.

Every baseline consumes the SAME simulated SPIData that GSDiff-SPI trains on
(patterns, measurements, frame_idx, t_grid) and returns a reconstruction
`recon` of shape [T, H, W] plus a small info dict. All baselines are scored
with the IDENTICAL per-frame normalize_01 + psnr_fn pipeline used by
train.evaluate() (gsdiff.baselines.common.evaluate_video), so numbers are
directly comparable to the main method — never with a more generous metric.

Hyperparameters that are not physically fixed are selected GROUND-TRUTH-FREE
on a held-out measurement residual (common.select_by_holdout), never on PSNR.
"""
from importlib import import_module

__all__ = ["build_operator", "apply_operator", "adjoint", "admm_tv",
           "dgi_image", "evaluate_video", "select_by_holdout",
           "holdout_residual", "cs", "monin", "gidc", "inr", "recinr", "tv3d"]

_COMMON_EXPORTS = set(__all__[:8])
_MODULE_EXPORTS = set(__all__[8:])


def __getattr__(name):
    if name in _COMMON_EXPORTS:
        value = getattr(import_module(".common", package=__name__), name)
    elif name in _MODULE_EXPORTS:
        value = import_module(f".{name}", package=__name__)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
