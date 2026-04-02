"""SGD baseline: direct optimization without ADMM.

loss_norm='zscore'     (original):
    loss = ½·MSE(zscore(ŷ), zscore(y)) + λ·TV(video)
loss_norm='target_std' (方案A):
    loss = ½·MSE(ŷ/σ_y,    y/σ_y)     + λ·TV(video)

TV is differentiable L1 total variation computed directly on the
rendered video, backpropagated through the rendering pipeline.
"""
import torch, torch.nn.functional as F


def zscore(y):
    return (y - y.mean()) / (y.std() + 1e-8)


def tv_loss(video, temporal_weight=0.0):
    """Differentiable L1 TV for video [T,1,H,W].
    temporal_weight > 0 adds the frame-to-frame temporal term (3D TV).
    """
    dy = (video[:, :, 1:, :] - video[:, :, :-1, :]).abs().mean()
    dx = (video[:, :, :, 1:] - video[:, :, :, :-1]).abs().mean()
    spatial = dy + dx
    if temporal_weight > 0:
        dt = temporal_weight * (video[1:] - video[:-1]).abs().mean()
        return spatial + dt
    return spatial


class SGDSolver:
    """Direct Adam optimization (no ADMM, no variable splitting)."""

    def __init__(self, fwd, patterns, y_target, frame_idx, t_grid,
                 tv_weight=0.005, lr_scene=3e-3, lr_motion=1e-2,
                 n_steps=800, loss_norm='zscore', temporal_tv_weight=0.0,
                 device='cpu'):
        """
        loss_norm : 'zscore'     → original independent z-score loss (default)
                    'target_std' → normalize by fixed target std only (方案A)
        """
        self.fwd = fwd
        self.patterns = patterns.to(device)
        self.frame_idx = frame_idx.to(device)
        self.t_grid = t_grid.to(device)
        self.loss_norm = loss_norm
        self.tv_weight = tv_weight
        self.temporal_tv_weight = temporal_tv_weight
        self.lr_scene = lr_scene
        self.lr_motion = lr_motion
        self.n_steps = n_steps
        self.device = device

        y = y_target.to(device)
        # --- loss_norm='zscore' (original) ---
        self.y_target_zs = zscore(y)
        # --- loss_norm='target_std' (方案A) ---
        _scale = y.std() + 1e-8
        self.y_norm_scale  = _scale       # scalar tensor, fixed
        self.y_target_norm = y / _scale   # [K] tensor, fixed

        sp = list(fwd.scene.parameters())
        mp = list(fwd.motion.parameters())
        self.optimizer = torch.optim.Adam([
            {"params": sp, "lr": lr_scene},
            {"params": mp, "lr": lr_motion}])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_steps, eta_min=lr_scene * 0.1)

    # ------------------------------------------------------------------
    def _data_loss(self, y_pred):
        """Data fidelity term, controlled by self.loss_norm."""
        if self.loss_norm == 'zscore':
            return 0.5 * F.mse_loss(zscore(y_pred), self.y_target_zs)
        # 'target_std': normalize both by the fixed target std
        return 0.5 * F.mse_loss(y_pred / self.y_norm_scale,
                                 self.y_target_norm)

    # ------------------------------------------------------------------
    def step(self):
        """One gradient step. Returns info dict."""
        self.optimizer.zero_grad()
        y_pred, video = self.fwd(self.patterns, self.frame_idx, self.t_grid)
        loss_d = self._data_loss(y_pred)
        loss_tv = self.tv_weight * tv_loss(video, self.temporal_tv_weight)
        loss = loss_d + loss_tv
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.fwd.scene.parameters()) + list(self.fwd.motion.parameters()), 5.0)
        self.optimizer.step()
        self.scheduler.step()
        return {
            "loss_data": loss_d.item(),
            "tv": loss_tv.item(),
            "loss_total": loss.item(),
        }

    def get_render(self):
        with torch.no_grad():
            return self.fwd.render_video(self.t_grid)
