"""SE(2) motion: rotation around image center + translation.

μ_m(t) = R(ωt) · (μ_m − c) + c + v·t
Σ_m(t) = R(ωt) · Σ_m · R(ωt)ᵀ

where c = ((H-1)/2, (W-1)/2) is the image center (matches scipy.ndimage.rotate).
"""
from .se2 import SE2Motion
