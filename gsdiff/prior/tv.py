"""TV prior: Chambolle proximal operator.

prox_{τ·TV}(v) = argmin_z TV(z) + 1/(2τ) ||z - v||²
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
