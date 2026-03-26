"""Canonical 2D Gaussian scene: M Gaussians → pixel image.

Each Gaussian has 6 params: amplitude, center_y, center_x, log_sy, log_sx, angle.
Rendering: s(u) = Σ_m  a_m · exp(-½ (u-μ_m)ᵀ Σ_m⁻¹ (u-μ_m))
"""
import math, torch, torch.nn as nn, torch.nn.functional as F


class GaussianScene2D(nn.Module):

    def __init__(self, M: int, H: int, W: int, init_scale: float = 1.5):
        """
        M: number of Gaussians
        H, W: image size
        init_scale: initial std-dev of each Gaussian in pixels
        """
        super().__init__()
        self.M, self.H, self.W = M, H, W

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
        return torch.exp(self.log_scales)

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
