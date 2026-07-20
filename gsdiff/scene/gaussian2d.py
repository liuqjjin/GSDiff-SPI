"""Canonical 2D Gaussian scene: M Gaussians → pixel image.

Each Gaussian has 6 params: amplitude, center_y, center_x, log_sy, log_sx, angle.
Rendering: s(u) = Σ_m  a_m · exp(-½ (u-μ_m)ᵀ Σ_m⁻¹ (u-μ_m))
"""
import math, torch, torch.nn as nn, torch.nn.functional as F


class GaussianScene2D(nn.Module):

    def __init__(self, M: int, H: int, W: int, init_scale: float = 1.5,
                 min_scale: float = 0.0):
        """
        M: number of Gaussians
        H, W: image size
        init_scale: initial std-dev of each Gaussian in pixels
        min_scale: lower clamp on scales in pixels (0 = off). Sub-pixel
            Gaussians sampled on the pixel grid bias integral-type
            measurements (SPI inner products) — ~0.3 px is a sane guard.
        """
        super().__init__()
        self.M, self.H, self.W = M, H, W
        self.min_scale = float(min_scale)

        # Centers uniform in [0, H) x [0, W)
        self.centers = nn.Parameter(torch.rand(M, 2) * torch.tensor([float(H), float(W)]))

        # Log-scales → actual scale = exp(log_s), init so scale ≈ init_scale pixels
        init_ls = math.log(init_scale)
        self.log_scales = nn.Parameter(torch.full((M, 2), init_ls))

        # Rotation angles (radians)
        self.angles = nn.Parameter(torch.zeros(M))

        # Raw amplitudes → softplus for positivity, init near 1.0
        self.raw_amps = nn.Parameter(torch.zeros(M))  # softplus(0) ≈ 0.69

    def get_scales(self):
        """Returns [M, 2] positive scale values."""
        s = torch.exp(self.log_scales)
        if self.min_scale > 0:
            s = s.clamp(min=self.min_scale)
        return s

    def get_amplitudes(self):
        """Returns [M] positive amplitudes."""
        return F.softplus(self.raw_amps)

    def get_covariances(self):
        """Returns [M, 2, 2] covariance matrices Σ_m = R diag(s²) Rᵀ."""
        s = self.get_scales()  # [M, 2]
        c, sn = torch.cos(self.angles), torch.sin(self.angles)  # [M]
        R = torch.stack([c, -sn, sn, c], -1).reshape(-1, 2, 2)  # [M,2,2]
        D = torch.diag_embed(s ** 2)  # [M,2,2]
        return R @ D @ R.transpose(-1, -2)  # [M,2,2]

    def render(self, H=None, W=None):
        """Render canonical scene → [1, 1, H, W] non-negative image."""
        H = H or self.H; W = W or self.W
        dev = self.centers.device

        # Pixel grid [N, 2] where N = H*W, coords = (row, col)
        gy, gx = torch.meshgrid(
            torch.arange(H, device=dev, dtype=torch.float32),
            torch.arange(W, device=dev, dtype=torch.float32), indexing='ij')
        coords = torch.stack([gy.reshape(-1), gx.reshape(-1)], -1)  # [N, 2]

        # Precision matrices Σ⁻¹
        Sigma = self.get_covariances()  # [M,2,2]
        Sinv = torch.linalg.inv(Sigma + 1e-6 * torch.eye(2, device=dev))  # [M,2,2]

        amps = self.get_amplitudes()  # [M]

        # diff[m, n] = coords[n] - centers[m]  → [M, N, 2]
        diff = coords.unsqueeze(0) - self.centers.unsqueeze(1)

        # Mahalanobis: (diff @ Sinv) * diff summed over last dim → [M, N]
        quad = (torch.einsum('mni,mij->mnj', diff, Sinv) * diff).sum(-1)

        # Weighted Gaussians
        vals = amps[:, None] * torch.exp(-0.5 * quad)  # [M, N]

        img = vals.sum(0).reshape(1, 1, H, W)  # [1,1,H,W]
        return F.relu(img)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def init_from_image(self, dgi_img):
        """Scale Gaussian amplitudes so mean rendering ≈ dgi_img mean (方案B).

        Uses the exact softplus inverse so get_amplitudes() output scales by
        (target_mean / current_mean) without changing centers or scales.

        dgi_img : [H, W] numpy array or torch Tensor with values in [0, 1].
        """
        import numpy as np
        dev = self.centers.device
        if not isinstance(dgi_img, torch.Tensor):
            dgi_t = torch.tensor(np.asarray(dgi_img), device=dev, dtype=torch.float32)
        else:
            dgi_t = dgi_img.to(dev).float()

        with torch.no_grad():
            rendered      = self.render()[0, 0]                    # [H, W]
            current_mean  = rendered.mean().clamp(min=1e-8).item()
            target_mean   = dgi_t.mean().clamp(min=1e-8).item()
            scale         = target_mean / current_mean

            # Exact inversion: new_raw = softplus_inv(scale * softplus(raw_amps))
            # softplus_inv(y) = log(expm1(y))   for y > 0
            old_amps = self.get_amplitudes()                        # [M], all > 0
            new_amps = (old_amps * scale).clamp(min=1e-6)
            new_raw  = torch.log(torch.expm1(new_amps))
            self.raw_amps.data.copy_(new_raw)

        print(f"  Scene init (dgi): mean {current_mean:.4f} → {target_mean:.4f} "
              f"(×{scale:.3f})")

    def init_adaptive(self, dgi_img, floor=0.2):
        """Content-adaptive init (Image-GS style): sample centers from
        |∇DGI| + uniform floor, set amplitudes from local DGI intensity.

        Uses the global numpy RNG (seeded by set_seed) for reproducibility.
        Call init_from_image afterwards for the global amplitude scale.
        """
        import numpy as np
        img = np.asarray(dgi_img, dtype=np.float64)
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)
        gy, gx = np.gradient(img)
        w = np.hypot(gy, gx)
        w = w / (w.max() + 1e-12) + floor
        p = (w / w.sum()).ravel()
        idx = np.random.choice(img.size, size=self.M, p=p)
        ys, xs = np.unravel_index(idx, img.shape)
        centers = np.stack([ys, xs], 1) + (np.random.rand(self.M, 2) - 0.5)
        amps = np.clip(img[ys, xs], 0.05, None)
        with torch.no_grad():
            self.centers.data.copy_(torch.tensor(centers, dtype=torch.float32))
            self.raw_amps.data.copy_(
                torch.log(torch.expm1(torch.tensor(amps, dtype=torch.float32))))
        print(f"  Scene init (adaptive): centers ∝ |∇DGI|+{floor}, "
              f"amps from local intensity")
