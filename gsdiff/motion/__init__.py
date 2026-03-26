"""SE(2) motion: rotation around image center + translation.

μ_m(t) = R(ωt) · (μ_m − c) + c + v·t
Σ_m(t) = R(ωt) · Σ_m · R(ωt)ᵀ

where c = (H/2, W/2) is the image center.
"""
from .se2 import SE2Motion
