"""Shared low-DOF motion with exact Gaussian transport.

Base SE(2):
    μ_m(t) = R(ω·t) · (μ_m − c_img) + c_img + v·t
    Σ_m(t) = R(ω·t) · Σ_m · R(ω·t)ᵀ

Optional extensions (flag-gated; defaults reproduce plain SE(2) exactly):
  poly_degree=2 : accelerated motion — d(t) = v·t + a·t², angle(t) = ω·t + β·t²
  enable_affine : linear part A(t) = R(angle(t)) · (I + t·L_sym), with L_sym a
                  learnable symmetric 2x2 (3 DOF: scale-y, scale-x, shear).
                  Polar-decomposition style — the symmetric factor avoids gauge
                  redundancy with ω. A Gaussian stays Gaussian under any affine
                  map, so transport remains exact: Σ(t) = A(t) Σ A(t)ᵀ.

Learnable DOF: 2 (v) [+1 ω] [+2 a, +1 β with poly] [+3 L with affine].
"""
import torch, torch.nn as nn


class SE2Motion(nn.Module):

    def __init__(self, img_center: tuple, enable_rotation=True,
                 poly_degree=1, enable_affine=False):
        super().__init__()
        # Image center (not learnable)
        self.register_buffer('center', torch.tensor(img_center, dtype=torch.float32))
        self.velocity = nn.Parameter(torch.zeros(2))  # [v_y, v_x] pixels over t∈[0,1]
        self.enable_rotation = enable_rotation
        assert poly_degree in (1, 2)
        self.poly_degree = poly_degree
        self.enable_affine = enable_affine
        if enable_rotation:
            self.omega = nn.Parameter(torch.tensor(0.0))  # radians over t∈[0,1]
        if poly_degree == 2:
            self.accel = nn.Parameter(torch.zeros(2))     # [a_y, a_x] pixels·t⁻²
            if enable_rotation:
                self.beta = nn.Parameter(torch.tensor(0.0))
        if enable_affine:
            # symmetric linear rate: [[lyy, lyx], [lyx, lxx]], init 0 → identity
            self.lin = nn.Parameter(torch.zeros(3))       # lyy, lxx, lyx

    def _angle(self, t):
        if not self.enable_rotation:
            return None
        angle = self.omega * t
        if self.poly_degree == 2:
            angle = angle + self.beta * t * t
        return angle

    def _A(self, t):
        """Linear part A(t) = R(angle(t)) · (I + t·L_sym). t: [T] → [T,2,2]."""
        angle = self._angle(t)
        if angle is None:
            R = torch.eye(2, device=t.device).unsqueeze(0).expand(t.shape[0], 2, 2)
        else:
            c, s = torch.cos(angle), torch.sin(angle)
            R = torch.stack([c, -s, s, c], -1).reshape(-1, 2, 2)
        if not self.enable_affine:
            return R
        lyy, lxx, lyx = self.lin[0], self.lin[1], self.lin[2]
        L = torch.stack([lyy, lyx, lyx, lxx]).reshape(2, 2)
        S = torch.eye(2, device=t.device) + t.reshape(-1, 1, 1) * L  # [T,2,2]
        return torch.einsum('tij,tjk->tik', R, S)

    def _displacement(self, t):
        d = self.velocity.unsqueeze(0) * t.unsqueeze(1)   # [T,2]
        if self.poly_degree == 2:
            d = d + self.accel.unsqueeze(0) * (t * t).unsqueeze(1)
        return d

    def transform_centers(self, centers, t):
        """centers: [M,2], t: [T] → [T,M,2]."""
        A = self._A(t)                    # [T,2,2]
        d = self._displacement(t)         # [T,2]
        c = self.center                   # [2]
        # μ_m(t) = A(t) · (μ_m - c) + c + d(t)
        rel = centers - c                 # [M,2]  (relative to image center)
        moved = torch.einsum('tij,mj->tmi', A, rel)  # [T,M,2]
        return moved + c.unsqueeze(0).unsqueeze(0) + d.unsqueeze(1)

    def transform_covariances(self, Sigma, t):
        """Sigma: [M,2,2], t: [T] → [T,M,2,2].  Σ(t) = A Σ Aᵀ."""
        A = self._A(t)  # [T,2,2]
        AS = torch.einsum('tij,mjk->tmik', A, Sigma)
        return torch.einsum('tmij,tkj->tmik', AS, A)

    def get_params_dict(self):
        d = {"velocity": self.velocity.detach().cpu().tolist()}
        if self.enable_rotation:
            d["omega"] = self.omega.detach().cpu().item()
        if self.poly_degree == 2:
            d["accel"] = self.accel.detach().cpu().tolist()
            if self.enable_rotation:
                d["beta"] = self.beta.detach().cpu().item()
        if self.enable_affine:
            d["lin"] = self.lin.detach().cpu().tolist()
        return d

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
