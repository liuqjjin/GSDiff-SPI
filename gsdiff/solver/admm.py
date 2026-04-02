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


def _soft_tv(video, temporal_weight=0.0):
    """Differentiable L1 TV mean-reduced (same as SGDSolver.tv_loss).
    temporal_weight > 0 adds frame-to-frame temporal term (3D TV).
    """
    dy = (video[:, :, 1:, :] - video[:, :, :-1, :]).abs().mean()
    dx = (video[:, :, :, 1:] - video[:, :, :, :-1]).abs().mean()
    spatial = dy + dx
    if temporal_weight > 0:
        dt = temporal_weight * (video[1:] - video[:-1]).abs().mean()
        return spatial + dt
    return spatial


class ADMMSolver:

    def __init__(self, fwd, prior, patterns, y_target, frame_idx, t_grid,
                 rho=0.1, tv_weight=0.005, lr_scene=5e-3, lr_motion=1e-2,
                 n_inner=100, rho_growth=1.05, loss_norm='zscore', device='cpu',
                 n_warmup=0, n_outer=50, soft_tv_weight=0.0,
                 temporal_tv_weight=0.0):
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
        self.soft_tv_weight = soft_tv_weight
        self.temporal_tv_weight = temporal_tv_weight
        self.n_inner = n_inner
        self.rho_growth = rho_growth
        self.n_warmup = n_warmup
        self._outer_iter = 0
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
        # CosineAnnealing over ALL steps (warmup + ADMM), matches SGD behaviour
        total_steps = n_outer * n_inner
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=lr_scene * 0.1)

    # ------------------------------------------------------------------
    def _data_loss(self, y_pred):
        """Data fidelity term, controlled by self.loss_norm."""
        if self.loss_norm == 'zscore':
            return 0.5 * F.mse_loss(zscore(y_pred), self.y_target_zs)
        # 'target_std': normalize both by the fixed target std
        return 0.5 * F.mse_loss(y_pred / self.y_norm_scale,
                                 self.y_target_norm)

    # ------------------------------------------------------------------
    def theta_step(self, in_warmup=False):
        # CORRECT: target = z - u  (Boyd scaled form)
        target_vid = (self.z - self.u).detach()

        for _ in range(self.n_inner):
            self.optimizer.zero_grad()
            y_pred, video = self.fwd(self.patterns, self.frame_idx, self.t_grid)
            loss_d = self._data_loss(y_pred)
            loss_tv = self.soft_tv_weight * _soft_tv(video, self.temporal_tv_weight)
            # During warmup: skip consistency term so velocity can converge freely
            loss_c = 0.0 if in_warmup else 0.5 * self.rho * F.mse_loss(video, target_vid)
            (loss_d + loss_tv + loss_c).backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.fwd.parameters()), 5.0)
            self.optimizer.step()
            self.scheduler.step()

        with torch.no_grad():
            yf, vf = self.fwd(self.patterns, self.frame_idx, self.t_grid)
            ld = self._data_loss(yf).item()
            lc = 0.0 if in_warmup else 0.5 * self.rho * F.mse_loss(vf, target_vid).item()
            # report combined data+soft_tv as loss_data for monitoring
            ld = ld + self.soft_tv_weight * _soft_tv(vf, self.temporal_tv_weight).item()
        return {"loss_data": ld, "loss_consist": lc, "video": vf}

    def z_step(self, video):
        # CORRECT: z = prox(R(θ) + u)  (Boyd scaled form)
        self.z = self.prior.proximal(
            video + self.u, self.tv_weight / (self.rho + 1e-8))

    def u_step(self, video):
        # u ← u + R(θ) - z  (correct in all conventions)
        self.u = self.u + video - self.z

    def step(self):
        self._outer_iter += 1
        in_warmup = self._outer_iter <= self.n_warmup
        info = self.theta_step(in_warmup=in_warmup)
        vid = info["video"]
        if not in_warmup:
            self.z_step(vid)
            self.u_step(vid)
            self.rho *= self.rho_growth
        info["prim_res"] = F.mse_loss(vid, self.z).item()
        info["tv"] = self.prior.energy(self.z)
        info["rho"] = self.rho
        info["in_warmup"] = in_warmup
        return info

    def get_z(self):
        return self.z.detach()

    def get_render(self):
        with torch.no_grad():
            return self.fwd.render_video(self.t_grid)
