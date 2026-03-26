"""SE(2) shared rigid-body motion.

μ_m(t) = R(ω·t) · (μ_m − c_img) + c_img + v·t
Σ_m(t) = R(ω·t) · Σ_m · R(ω·t)ᵀ

Learnable: velocity [2], omega (scalar).
"""
import torch, torch.nn as nn


class SE2Motion(nn.Module):

    def __init__(self, img_center: tuple[float, float], enable_rotation=True):
        super().__init__()
        # Image center (not learnable)
        self.register_buffer('center', torch.tensor(img_center, dtype=torch.float32))
        self.velocity = nn.Parameter(torch.zeros(2))  # [v_y, v_x] pixels over t∈[0,1]
        self.enable_rotation = enable_rotation
        if enable_rotation:
            self.omega = nn.Parameter(torch.tensor(0.0))  # radians over t∈[0,1]

    def _R(self, t):
        """Rotation matrix R(ω·t). t: [T] → R: [T,2,2]."""
        if not self.enable_rotation:
            I = torch.eye(2, device=t.device)
            return I.unsqueeze(0).expand(t.shape[0], -1, -1)
        angle = self.omega * t  # [T]
        c, s = torch.cos(angle), torch.sin(angle)
        return torch.stack([c, -s, s, c], -1).reshape(-1, 2, 2)

    def transform_centers(self, centers, t):
        """centers: [M,2], t: [T] → [T,M,2]."""
        R = self._R(t)          # [T,2,2]
        d = self.velocity.unsqueeze(0) * t.unsqueeze(1)  # [T,2]
        c = self.center         # [2]
        # μ_m(t) = R(ωt) · (μ_m - c) + c + v·t
        rel = centers - c       # [M,2]  (relative to image center)
        rotated = torch.einsum('tij,mj->tmi', R, rel)  # [T,M,2]
        return rotated + c.unsqueeze(0).unsqueeze(0) + d.unsqueeze(1)

    def transform_covariances(self, Sigma, t):
        """Sigma: [M,2,2], t: [T] → [T,M,2,2]."""
        R = self._R(t)  # [T,2,2]
        RS = torch.einsum('tij,mjk->tmik', R, Sigma)
        return torch.einsum('tmij,tkj->tmik', RS, R)

    def get_params_dict(self):
        d = {"velocity": self.velocity.detach().cpu().tolist()}
        if self.enable_rotation:
            d["omega"] = self.omega.detach().cpu().item()
        return d

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
