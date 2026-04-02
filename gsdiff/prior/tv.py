"""TV prior: Chambolle proximal operator.

2D: prox_{τ·TV}(v) = argmin_z TV2D(z) + 1/(2τ) ||z - v||²
3D: prox_{τ·TV3D}(v) = argmin_z TV3D(z) + 1/(2τ) ||z - v||²
    TV3D(z) = Σ_{t,i,j} sqrt((α·Δt)² + (Δy)² + (Δx)²)   [isotropic]
"""
import torch


class TVPrior:

    def __init__(self, max_iter=50):
        self.max_iter = max_iter

    @staticmethod
    def _chambolle(img, weight, max_iter=50):
        """TV denoising of a 2D image via Chambolle dual projection.
        img: [H,W], weight: scalar → denoised [H,W]."""
        H, W = img.shape; dev = img.device
        p = torch.zeros(H, W, 2, device=dev)
        tau = 1.0 / 8.0
        for _ in range(max_iter):
            div_p = torch.zeros(H, W, device=dev)
            div_p[1:] += p[1:, :, 0] - p[:-1, :, 0]
            div_p[0]  += p[0, :, 0]
            div_p[:, 1:] += p[:, 1:, 1] - p[:, :-1, 1]
            div_p[:, 0]  += p[:, 0, 1]

            x = img + weight * div_p
            g = torch.zeros(H, W, 2, device=dev)
            g[:-1, :, 0] = x[1:] - x[:-1]
            g[:, :-1, 1] = x[:, 1:] - x[:, :-1]

            pn = p + tau * g / weight
            norm = torch.sqrt(pn[..., 0]**2 + pn[..., 1]**2).clamp(min=1.0)
            p = pn / norm.unsqueeze(-1)

        div_p = torch.zeros(H, W, device=dev)
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
        dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().sum()
        dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().sum()
        return (dy + dx).item()


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
        dev = video.device
        # Dual variable: component 0=temporal, 1=vertical, 2=horizontal
        p = torch.zeros(T, H, W, 3, device=dev)
        alpha = float(temporal_weight)
        tau = 1.0 / (8.0 + 4.0 * alpha * alpha)  # step size

        for _ in range(max_iter):
            # ── divergence of p (adjoint of gradient) ──
            div_p = torch.zeros(T, H, W, device=dev)
            div_p[1:,  :,  :] += p[1:, :, :, 0] - p[:-1, :, :, 0]
            div_p[0,   :,  :] += p[0, :, :, 0]
            div_p[:,  1:,  :] += p[:, 1:, :, 1] - p[:, :-1, :, 1]
            div_p[:,   0,  :] += p[:, 0, :, 1]
            div_p[:,  :, 1:]  += p[:, :, 1:, 2] - p[:, :, :-1, 2]
            div_p[:,  :,  0]  += p[:, :, 0, 2]

            x = video + weight * div_p

            # ── forward gradient of x ──
            g = torch.zeros(T, H, W, 3, device=dev)
            g[:-1, :, :, 0] = alpha * (x[1:] - x[:-1])   # temporal
            g[:,  :-1, :, 1] = x[:, 1:] - x[:, :-1]       # vertical
            g[:,  :, :-1, 2] = x[:, :, 1:] - x[:, :, :-1] # horizontal

            pn = p + tau * g / weight
            # isotropic projection onto unit ball
            norm = torch.sqrt(
                pn[..., 0]**2 + pn[..., 1]**2 + pn[..., 2]**2
            ).clamp(min=1.0)
            p = pn / norm.unsqueeze(-1)

        # final divergence for the output
        div_p = torch.zeros(T, H, W, device=dev)
        div_p[1:,  :,  :] += p[1:, :, :, 0] - p[:-1, :, :, 0]
        div_p[0,   :,  :] += p[0, :, :, 0]
        div_p[:,  1:,  :] += p[:, 1:, :, 1] - p[:, :-1, :, 1]
        div_p[:,   0,  :] += p[:, 0, :, 1]
        div_p[:,  :, 1:]  += p[:, :, 1:, 2] - p[:, :, :-1, 2]
        div_p[:,  :,  0]  += p[:, :, 0, 2]

        return video + weight * div_p

    def proximal(self, x, weight):
        """3D TV prox applied to full video [T,1,H,W] jointly."""
        v = x[:, 0]   # [T, H, W]
        v_den = self._chambolle3d(v, weight, self.temporal_weight, self.max_iter)
        return v_den.unsqueeze(1)   # [T, 1, H, W]

    def energy(self, x):
        """TV3D(x) for [T,1,H,W] — isotropic pointwise norm, sum-reduced."""
        v = x[:, 0]   # [T, H, W]
        alpha = self.temporal_weight
        # spatial
        dy = (v[:, 1:, :] - v[:, :-1, :]).abs().sum()
        dx = (v[:, :, 1:] - v[:, :, :-1]).abs().sum()
        # temporal
        dt = alpha * (v[1:] - v[:-1]).abs().sum()
        return (dy + dx + dt).item()
