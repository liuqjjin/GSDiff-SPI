"""Differential Ghost Imaging (DGI) reconstruction.

DGI reconstructs an image from bucket measurements and patterns using
an incremental averaging formula that subtracts background correlation.
"""
import numpy as np


def dgi_reconstruct(patterns, measurements):
    """DGI reconstruction.

    Parameters
    ----------
    patterns     : [K, H, W]  measurement patterns
    measurements : [K]        bucket signals

    Returns
    -------
    recon : [H, W]  reconstructed image (z-scored)
    """
    K = patterns.shape[0]
    B_avg = 0.0; SI_avg = 0.0; R_avg = 0.0; RI_avg = 0.0

    for k in range(K):
        pat = patterns[k]
        y_k = measurements[k]
        n = k + 1
        SI_avg = (SI_avg * k + pat * y_k) / n
        B_avg = (B_avg * k + y_k) / n
        R_avg = (R_avg * k + np.sum(pat)) / n
        RI_avg = (RI_avg * k + np.sum(pat) * pat) / n

    # DGI formula
    recon = SI_avg - (B_avg / (R_avg + 1e-12)) * RI_avg

    # Z-score normalize
    std = recon.std()
    if std > 1e-12:
        recon = (recon - recon.mean()) / std
    return recon
