"""ADMM solver with correct Boyd et al. sign convention.

Boyd et al. 2011, §3.1.1 (scaled form):

    L_ρ(θ, z, u) = f(θ) + g(z) + (ρ/2)||R(θ) - z + u||²

    θ-step:  min_θ  f(θ)  + (ρ/2)||R(θ) - (z - u)||²     ← target = z - u
    z-step:  z = prox_{g/ρ}( R(θ) + u )                    ← input = R(θ) + u
    u-step:  u ← u + R(θ) - z

f(θ) = ½ MSE(zscore(ŷ), zscore(y))   (data fidelity, z-scored)
g(z) = λ · TV(z)                       (video-domain regularization)
"""
import torch, torch.nn.functional as F


def zscore(y):
    """Independent z-score: (y - mean) / std."""
    return (y - y.mean()) / (y.std() + 1e-8)


class ADMMSolver:

    def __init__(self, fwd, prior, patterns, y_target, frame_idx, t_grid,
                 rho=0.1, tv_weight=0.005, lr_scene=5e-3, lr_motion=1e-2,
                 n_inner=100, rho_growth=1.05, device='cpu'):
        self.fwd = fwd
        self.prior = prior
        self.patterns = patterns.to(device)
        self.frame_idx = frame_idx.to(device)
        self.t_grid = t_grid.to(device)
        self.y_target_zs = zscore(y_target.to(device))
        self.rho = rho
        self.tv_weight = tv_weight
        self.n_inner = n_inner
        self.rho_growth = rho_growth
        self.device = device

        T, H, W = t_grid.shape[0], fwd.H, fwd.W
        self.z = torch.zeros(T, 1, H, W, device=device)
        self.u = torch.zeros(T, 1, H, W, device=device)

        # Persistent optimizer (warm-start across ADMM iterations)
        sp = list(fwd.scene.parameters())
        mp = list(fwd.motion.parameters())
        self.optimizer = torch.optim.Adam([
            {"params": sp, "lr": lr_scene},
            {"params": mp, "lr": lr_motion}])

    def theta_step(self):
        # CORRECT: target = z - u  (Boyd scaled form)
        target_vid = (self.z - self.u).detach()

        for _ in range(self.n_inner):
            self.optimizer.zero_grad()
            y_pred, video = self.fwd(self.patterns, self.frame_idx, self.t_grid)
            loss_d = 0.5 * F.mse_loss(zscore(y_pred), self.y_target_zs)
            loss_c = 0.5 * self.rho * F.mse_loss(video, target_vid)
            (loss_d + loss_c).backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.fwd.parameters()), 5.0)
            self.optimizer.step()

        with torch.no_grad():
            yf, vf = self.fwd(self.patterns, self.frame_idx, self.t_grid)
            ld = 0.5 * F.mse_loss(zscore(yf), self.y_target_zs).item()
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
