"""SGD baseline: direct optimization without ADMM.

loss = ½·MSE(zscore(ŷ), zscore(y)) + λ·TV(video)

TV is differentiable L1 total variation computed directly on the
rendered video, backpropagated through the rendering pipeline.
"""
import torch, torch.nn.functional as F


def zscore(y):
    return (y - y.mean()) / (y.std() + 1e-8)


def tv_loss(video):
    """Differentiable L1 TV for video [T,1,H,W]."""
    dy = (video[:, :, 1:, :] - video[:, :, :-1, :]).abs().mean()
    dx = (video[:, :, :, 1:] - video[:, :, :, :-1]).abs().mean()
    return dy + dx


class SGDSolver:
    """Direct Adam optimization (no ADMM, no variable splitting)."""

    def __init__(self, fwd, patterns, y_target, frame_idx, t_grid,
                 tv_weight=0.005, lr_scene=3e-3, lr_motion=1e-2,
                 n_steps=800, device='cpu'):
        self.fwd = fwd
        self.patterns = patterns.to(device)
        self.frame_idx = frame_idx.to(device)
        self.t_grid = t_grid.to(device)
        self.y_target_zs = zscore(y_target.to(device))
        self.tv_weight = tv_weight
        self.lr_scene = lr_scene
        self.lr_motion = lr_motion
        self.n_steps = n_steps
        self.device = device

        sp = list(fwd.scene.parameters())
        mp = list(fwd.motion.parameters())
        self.optimizer = torch.optim.Adam([
            {"params": sp, "lr": lr_scene},
            {"params": mp, "lr": lr_motion}])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_steps, eta_min=lr_scene * 0.1)

    def step(self):
        """One gradient step. Returns info dict."""
        self.optimizer.zero_grad()
        y_pred, video = self.fwd(self.patterns, self.frame_idx, self.t_grid)
        loss_d = 0.5 * F.mse_loss(zscore(y_pred), self.y_target_zs)
        loss_tv = self.tv_weight * tv_loss(video)
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
