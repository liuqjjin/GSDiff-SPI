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
import warnings

import torch, torch.nn.functional as F

from gsdiff.prior.tv import anisotropic_tv_mean
from gsdiff.solver.gradients import (
    _consumed_cosine_multiplier,
    active_parameters,
    clip_grad_groups,
)

_UNSET = object()


def zscore(y):
    """Independent z-score: (y - mean) / std."""
    return (y - y.mean()) / (y.std() + 1e-8)


def _soft_tv(video, temporal_weight=0.0):
    """Compatibility wrapper for componentwise anisotropic TV mean.

    Accepts video [T,1,H,W].
    temporal_weight > 0 adds frame-to-frame temporal term (3D TV).
    """
    return anisotropic_tv_mean(video, temporal_weight)


class ADMMSolver:

    def __init__(self, fwd, prior, patterns, y_target, frame_idx, t_grid,
                 rho=0.1, tv_weight=0.005, lr_scene=5e-3, lr_motion=1e-2,
                 n_inner=100, rho_growth=1.05, loss_norm='zscore', device='cpu',
                 splitting_warmup_outer=_UNSET, n_outer=50,
                 soft_tv_weight=0.0, temporal_tv_weight=0.0, hqs=False, *,
                 motion_warmup_outer=0, n_warmup=_UNSET):
        """
        loss_norm : 'zscore'     → original independent z-score loss (default)
                    'target_std' → normalize by fixed target std only (方案A)
        hqs       : True → u ≡ 0 (half-quadratic splitting ablation: the θ-step
                    targets z directly and the z-step input is R(θ); isolates
                    what the ADMM dual/Bregman memory actually contributes)
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
        if splitting_warmup_outer is not _UNSET and n_warmup is not _UNSET:
            raise ValueError(
                "n_warmup and splitting_warmup_outer cannot be supplied together"
            )
        if splitting_warmup_outer is _UNSET:
            splitting_warmup_outer = (
                0 if n_warmup is _UNSET else n_warmup
            )
            if n_warmup is not _UNSET:
                warnings.warn(
                    "n_warmup is deprecated; use splitting_warmup_outer",
                    DeprecationWarning,
                    stacklevel=2,
                )
        if type(n_outer) is not int or n_outer <= 0:
            raise ValueError("n_outer must be a positive integer")
        for name, value in (
            ("splitting_warmup_outer", splitting_warmup_outer),
            ("motion_warmup_outer", motion_warmup_outer),
        ):
            if type(value) is not int or value < 0 or value > n_outer:
                raise ValueError(
                    f"{name} must be an integer in [0, n_outer]"
                )
        self.splitting_warmup_outer = splitting_warmup_outer
        self.motion_warmup_outer = motion_warmup_outer
        self.n_warmup = splitting_warmup_outer
        self.hqs = hqs
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
        self._scene_params = list(fwd.scene.parameters())
        self._motion_params = list(fwd.motion.parameters())
        sp = active_parameters(self._scene_params)
        mp = active_parameters(self._motion_params)
        groups = []
        if sp:
            groups.append({"params": sp, "lr": lr_scene})
        if mp:
            groups.append({"params": mp, "lr": lr_motion})
        self.optimizer = torch.optim.Adam(groups)
        # CosineAnnealing over ALL steps (warmup + ADMM), matches SGD behaviour
        total_steps = n_outer * n_inner
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: _consumed_cosine_multiplier(
                step, total_steps
            ),
        )

    # ------------------------------------------------------------------
    def _data_loss(self, y_pred):
        """Data fidelity term, controlled by self.loss_norm."""
        if self.loss_norm == 'zscore':
            return 0.5 * F.mse_loss(zscore(y_pred), self.y_target_zs)
        # 'target_std': normalize both by the fixed target std
        return 0.5 * F.mse_loss(y_pred / self.y_norm_scale,
                                 self.y_target_norm)

    # ------------------------------------------------------------------
    def theta_step(
        self, *, in_splitting_warmup=False, in_motion_warmup=False
    ):
        # CORRECT: target = z - u  (Boyd scaled form)
        target_vid = (self.z - self.u).detach()

        for _ in range(self.n_inner):
            self.optimizer.zero_grad()
            y_pred, video = self.fwd(self.patterns, self.frame_idx, self.t_grid)
            loss_d = self._data_loss(y_pred)
            loss_tv = self.soft_tv_weight * _soft_tv(video, self.temporal_tv_weight)
            # Splitting warmup alone suppresses the consistency term.
            loss_c = (
                0.0
                if in_splitting_warmup
                else 0.5 * self.rho * F.mse_loss(video, target_vid)
            )
            (loss_d + loss_tv + loss_c).backward()
            if in_motion_warmup:
                for parameter in self._scene_params:
                    if parameter.grad is not None:
                        parameter.grad.zero_()
            clip_grad_groups(
                [self._scene_params, self._motion_params], 5.0
            )
            self.optimizer.step()
            self.scheduler.step()

        with torch.no_grad():
            yf, vf = self.fwd(self.patterns, self.frame_idx, self.t_grid)
            ld = self._data_loss(yf).item()
            lc = (
                0.0
                if in_splitting_warmup
                else 0.5 * self.rho * F.mse_loss(vf, target_vid).item()
            )
            # report combined data+soft_tv as loss_data for monitoring
            ld = ld + self.soft_tv_weight * _soft_tv(vf, self.temporal_tv_weight).item()
        return {"loss_data": ld, "loss_consist": lc, "video": vf}

    def z_step(self, video):
        # CORRECT: z = prox(R(θ) + u)  (Boyd scaled form)
        self.z = self.prior.proximal(
            video + self.u, self.tv_weight / (self.rho + 1e-8))

    def u_step(self, video):
        # u ← u + R(θ) - z  (correct in all conventions); HQS keeps u ≡ 0
        if not self.hqs:
            self.u = self.u + video - self.z

    def step(self):
        self._outer_iter += 1
        in_splitting_warmup = (
            self._outer_iter <= self.splitting_warmup_outer
        )
        in_motion_warmup = self._outer_iter <= self.motion_warmup_outer
        is_transition = (
            self._outer_iter == self.splitting_warmup_outer + 1
        )

        # 关键修补：
        # 第一个“刚结束 warmup”的 outer iter，不要立刻让 θ-step 去追未初始化的 z-u。
        # 这一轮仍按 warmup 的方式跑 θ-step，但要执行 z_step / u_step 来初始化 z 和 u。
        info = self.theta_step(
            in_splitting_warmup=(in_splitting_warmup or is_transition),
            in_motion_warmup=in_motion_warmup,
        )

        vid = info["video"]

        # 只要已经离开 warmup，就执行 z/u 更新
        # 注意：transition 轮也会进这里，从而完成 z/u 初始化
        z_prev = self.z
        if not in_splitting_warmup:
            self.z_step(vid)
            self.u_step(vid)
            self.rho *= self.rho_growth

        info["prim_res"] = F.mse_loss(vid, self.z).item()
        # Diagnostics: dual residual surrogate + dual-variable magnitude
        # (monotonically growing ‖u‖ signals manifold infeasibility)
        info["dual_res"] = (self.rho * (self.z - z_prev).pow(2).mean()).item()
        info["u_norm"] = self.u.abs().mean().item()
        info["tv"] = self.prior.energy(self.z)
        info["rho"] = self.rho
        info["in_splitting_warmup"] = in_splitting_warmup
        info["in_motion_warmup"] = in_motion_warmup
        info["in_warmup"] = in_splitting_warmup
        info["is_transition"] = is_transition
        return info

    def get_z(self):
        return self.z.detach()

    def get_render(self):
        with torch.no_grad():
            return self.fwd.render_video(self.t_grid)
