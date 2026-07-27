"""TV prior: Chambolle proximal operator.

2D: prox_{τ·TV}(v) = argmin_z TV2D(z) + 1/(2τ) ||z - v||²
3D: prox_{τ·TV3D}(v) = argmin_z TV3D(z) + 1/(2τ) ||z - v||²
    TV3D(z) = Σ_{t,i,j} sqrt((α·Δt)² + (Δy)² + (Δx)²)   [isotropic]
"""
import torch


def _gradient3d(video: torch.Tensor, alpha: float) -> torch.Tensor:
    T, H, W = video.shape
    grad = video.new_zeros(T, H, W, 3)
    grad[:-1, :, :, 0] = float(alpha) * (video[1:] - video[:-1])
    grad[:, :-1, :, 1] = video[:, 1:] - video[:, :-1]
    grad[:, :, :-1, 2] = video[:, :, 1:] - video[:, :, :-1]
    return grad


def isotropic_tv2d_sum(video: torch.Tensor) -> torch.Tensor:
    """Pointwise isotropic spatial TV, sum-reduced over a video."""
    v = video[:, 0]
    grad = v.new_zeros(*v.shape, 2)
    grad[:, :-1, :, 0] = v[:, 1:] - v[:, :-1]
    grad[:, :, :-1, 1] = v[:, :, 1:] - v[:, :, :-1]
    return torch.linalg.vector_norm(grad, dim=-1).sum()


def isotropic_tv3d_sum(
    video: torch.Tensor, temporal_weight: float
) -> torch.Tensor:
    """Pointwise isotropic spatiotemporal TV, sum-reduced over a video."""
    grad = _gradient3d(video[:, 0], temporal_weight)
    return torch.linalg.vector_norm(grad, dim=-1).sum()


def anisotropic_tv_mean(
    video: torch.Tensor, temporal_weight: float = 0.0
) -> torch.Tensor:
    """Componentwise anisotropic TV with historical per-axis means."""

    def mean_abs_or_zero(difference):
        if difference.numel() == 0:
            return difference.sum()
        return difference.abs().mean()

    dy = mean_abs_or_zero(video[:, :, 1:, :] - video[:, :, :-1, :])
    dx = mean_abs_or_zero(video[:, :, :, 1:] - video[:, :, :, :-1])
    spatial = dy + dx
    if temporal_weight > 0:
        dt = mean_abs_or_zero(video[1:] - video[:-1])
        return spatial + float(temporal_weight) * dt
    return spatial


def _divergence3d(field: torch.Tensor, alpha: float) -> torch.Tensor:
    T, H, W, _ = field.shape
    div = field.new_zeros(T, H, W)
    a = float(alpha)
    div[1:] += a * (field[1:, :, :, 0] - field[:-1, :, :, 0])
    div[0] += a * field[0, :, :, 0]
    div[:, 1:] += field[:, 1:, :, 1] - field[:, :-1, :, 1]
    div[:, 0] += field[:, 0, :, 1]
    div[:, :, 1:] += field[:, :, 1:, 2] - field[:, :, :-1, 2]
    div[:, :, 0] += field[:, :, 0, 2]
    return div


class TVPrior:

    def __init__(self, max_iter=50):
        self.max_iter = max_iter

    @staticmethod
    def _chambolle(img, weight, max_iter=50):
        """TV denoising of a 2D image via Chambolle dual projection.
        img: [H,W], weight: scalar → denoised [H,W]."""
        H, W = img.shape
        p = img.new_zeros(H, W, 2)
        tau = 1.0 / 8.0
        for _ in range(max_iter):
            div_p = img.new_zeros(H, W)
            div_p[1:] += p[1:, :, 0] - p[:-1, :, 0]
            div_p[0]  += p[0, :, 0]
            div_p[:, 1:] += p[:, 1:, 1] - p[:, :-1, 1]
            div_p[:, 0]  += p[:, 0, 1]

            x = img + weight * div_p
            g = img.new_zeros(H, W, 2)
            g[:-1, :, 0] = x[1:] - x[:-1]
            g[:, :-1, 1] = x[:, 1:] - x[:, :-1]

            pn = p + tau * g / weight
            norm = torch.sqrt(pn[..., 0]**2 + pn[..., 1]**2).clamp(min=1.0)
            p = pn / norm.unsqueeze(-1)

        div_p = img.new_zeros(H, W)
        div_p[1:] += p[1:, :, 0] - p[:-1, :, 0]
        div_p[0]  += p[0, :, 0]
        div_p[:, 1:] += p[:, 1:, 1] - p[:, :-1, 1]
        div_p[:, 0]  += p[:, 0, 1]
        return img + weight * div_p

    def proximal(self, x, weight):
        """Apply TV prox to video [T,1,H,W]."""
        T = x.shape[0]
        out = torch.empty_like(x)
        for t in range(T):
            out[t, 0] = self._chambolle(x[t, 0], weight, self.max_iter)
        return out

    def energy(self, x):
        """TV(x) for [T,1,H,W]."""
        return isotropic_tv2d_sum(x).item()


class TVPrior3D(TVPrior):
    """3D isotropic TV prior (spatial + temporal).

    TV3D(V) = Σ_{t,i,j} sqrt((α·ΔtV)² + (ΔyV)² + (ΔxV)²)

    temporal_weight α controls how strongly temporal smoothness is enforced
    relative to spatial TV.  α=0 → same as TVPrior (2D per-frame only).
    """

    def __init__(self, max_iter=50, temporal_weight=1.0):
        super().__init__(max_iter)
        self.temporal_weight = float(temporal_weight)

    @staticmethod
    def _chambolle3d(video, weight, temporal_weight=1.0, max_iter=50):
        """3D isotropic TV denoising via Chambolle dual projection.

        video          : [T, H, W]  (no channel dim)
        weight         : proximal weight λ  (scalar)
        temporal_weight: α scaling the temporal gradient component
        Returns        : denoised [T, H, W]

        Step size: tau = 1 / (8 + 4·α²)  derived from ||K_weighted||² ≤ 8+4α²
        """
        T, H, W = video.shape
        # Dual variable: component 0=temporal, 1=vertical, 2=horizontal
        p = video.new_zeros(T, H, W, 3)
        alpha = float(temporal_weight)
        tau = 1.0 / (8.0 + 4.0 * alpha * alpha)  # step size

        for _ in range(max_iter):
            # ── divergence of p (adjoint of gradient) ──
            div_p = _divergence3d(p, alpha)

            x = video + weight * div_p

            # ── forward gradient of x ──
            g = _gradient3d(x, alpha)

            pn = p + tau * g / weight
            # isotropic projection onto unit ball
            norm = torch.sqrt(
                pn[..., 0]**2 + pn[..., 1]**2 + pn[..., 2]**2
            ).clamp(min=1.0)
            p = pn / norm.unsqueeze(-1)

        # final divergence for the output
        div_p = _divergence3d(p, alpha)

        return video + weight * div_p

    def proximal(self, x, weight):
        """3D TV prox applied to full video [T,1,H,W] jointly."""
        v = x[:, 0]   # [T, H, W]
        v_den = self._chambolle3d(v, weight, self.temporal_weight, self.max_iter)
        return v_den.unsqueeze(1)   # [T, 1, H, W]

    def energy(self, x):
        """TV3D(x) for [T,1,H,W] — isotropic pointwise norm, sum-reduced."""
        return isotropic_tv3d_sum(x, self.temporal_weight).item()
