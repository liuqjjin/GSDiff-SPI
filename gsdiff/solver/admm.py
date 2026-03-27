"""ADMM solver with correct Boyd et al. sign convention.

Boyd et al. 2011, §3.1.1 (scaled form):

    L_ρ(θ, z, u) = f(θ) + g(z) + (ρ/2)||R(θ) - z + u||²

    θ-step:  min_θ  f(θ)  + (ρ/2)||R(θ) - (z - u)||²     ← target = z - u
    z-step:  z = prox_{g/ρ}( R(θ) + u )                    ← input = R(θ) + u
    u-step:  u ← u + R(θ) - z

loss_norm='zscore'     (original):
    f(θ) = ½ MSE(zscore(ŷ), zscore(y))   — independent z-score of pred and target
loss_norm='target_std' (方案A):
    f(θ) = ½ MSE(ŷ / σ_y,  y / σ_y)     — fixed normalization by target std only
g(z) = λ · TV(z)                          — video-domain regularization
"""
import torch, torch.nn.functional as F


def zscore(y):
    """Independent z-score: (y - mean) / std."""
    return (y - y.mean()) / (y.std() + 1e-8)


class ADMMSolver:

    def __init__(self, fwd, prior, patterns, y_target, frame_idx, t_grid,
                 rho=0.1, tv_weight=0.005, lr_scene=5e-3, lr_motion=1e-2,
                 n_inner=100, rho_growth=1.05, loss_norm='zscore', device='cpu'):
        """
        loss_norm : 'zscore'     → original independent z-score loss (default)
                    'target_std' → normalize by fixed target std only (方案A)
        """
        self.fwd = fwd
        self.prior = prior
        self.patterns = patterns.to(device)
        self.frame_idx = frame_idx.to(device)
        self.t_grid = t_grid.to(device)
        self.loss_norm = loss_norm
        self.rho = rho
        self.tv_weight = tv_weight
        self.n_inner = n_inner
        self.rho_growth = rho_growth
        self.device = device

        y = y_target.to(device)
        # --- loss_norm='zscore' (original) ---
        self.y_target_zs = zscore(y)
        # --- loss_norm='target_std' (方案A) ---
        # σ_y is computed once from the fixed GT target and never updated.
        _scale = y.std() + 1e-8
        self.y_norm_scale  = _scale       # scalar tensor, fixed
        self.y_target_norm = y / _scale   # [K] tensor, fixed

        T, H, W = t_grid.shape[0], fwd.H, fwd.W
        self.z = torch.zeros(T, 1, H, W, device=device)
        self.u = torch.zeros(T, 1, H, W, device=device)

        # Persistent optimizer (warm-start across ADMM iterations)
        sp = list(fwd.scene.parameters())
        mp = list(fwd.motion.parameters())
        self.optimizer = torch.optim.Adam([
            {"params": sp, "lr": lr_scene},
            {"params": mp, "lr": lr_motion}])

    # ------------------------------------------------------------------
    def _data_loss(self, y_pred):
        """Data fidelity term, controlled by self.loss_norm."""
        if self.loss_norm == 'zscore':
            return 0.5 * F.mse_loss(zscore(y_pred), self.y_target_zs)
        # 'target_std': normalize both by the fixed target std
        return 0.5 * F.mse_loss(y_pred / self.y_norm_scale,
                                 self.y_target_norm)

    # ------------------------------------------------------------------
    def theta_step(self):
        # CORRECT: target = z - u  (Boyd scaled form)
        target_vid = (self.z - self.u).detach()

        for _ in range(self.n_inner):
            self.optimizer.zero_grad()
            y_pred, video = self.fwd(self.patterns, self.frame_idx, self.t_grid)
            loss_d = self._data_loss(y_pred)
            loss_c = 0.5 * self.rho * F.mse_loss(video, target_vid)
            (loss_d + loss_c).backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.fwd.parameters()), 5.0)
            self.optimizer.step()

        with torch.no_grad():
            yf, vf = self.fwd(self.patterns, self.frame_idx, self.t_grid)
            ld = self._data_loss(yf).item()
            lc = 0.5 * self.rho * F.mse_loss(vf, target_vid).item()
        return {"loss_data": ld, "loss_consist": lc, "video": vf}

    def z_step(self, video):
        # CORRECT: z = prox(R(θ) + u)  (Boyd scaled form)
        self.z = self.prior.proximal(
            video + self.u, self.tv_weight / (self.rho + 1e-8))

    def u_step(self, video):
        # u ← u + R(θ) - z  (correct in all conventions)
        self.u = self.u + video - self.z

    def step(self):
        info = self.theta_step()
        vid = info["video"]
        self.z_step(vid)
        self.u_step(vid)
        self.rho *= self.rho_growth
        info["prim_res"] = F.mse_loss(vid, self.z).item()
        info["tv"] = self.prior.energy(self.z)
        info["rho"] = self.rho
        return info

    def get_z(self):
        return self.z.detach()

    def get_render(self):
        with torch.no_grad():
            return self.fwd.render_video(self.t_grid)
