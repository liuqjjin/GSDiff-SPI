"""SPI forward: (scene, motion) → video → bucket measurements.

Physics: y_k = Σ_{i,j} P_k(i,j) · frame[f(k)](i,j)
where frame[t] is rendered from SE(2)-transformed Gaussians.
"""
import torch, torch.nn as nn, torch.nn.functional as F


class SPIForwardModel(nn.Module):

    def __init__(self, scene, motion, H, W, time_assignment_mode="uniform"):
        super().__init__()
        self.scene = scene
        self.motion = motion
        self.H, self.W = H, W
        assert time_assignment_mode in ("uniform", "interpolation"), \
            f"Unknown time_assignment_mode: {time_assignment_mode}"
        self.time_assignment_mode = time_assignment_mode

    def _render_frame(self, centers_t, Sinv_t, amps):
        """Render one frame from transformed params.
        centers_t: [M,2], Sinv_t: [M,2,2], amps: [M] → [H,W]
        """
        H, W, dev = self.H, self.W, centers_t.device
        gy, gx = torch.meshgrid(
            torch.arange(H, device=dev, dtype=torch.float32),
            torch.arange(W, device=dev, dtype=torch.float32), indexing='ij')
        coords = torch.stack([gy.reshape(-1), gx.reshape(-1)], -1)  # [N,2]
        diff = coords.unsqueeze(0) - centers_t.unsqueeze(1)  # [M,N,2]
        quad = (torch.einsum('mni,mij->mnj', diff, Sinv_t) * diff).sum(-1)  # [M,N]
        vals = amps[:, None] * torch.exp(-0.5 * quad)
        return F.relu(vals.sum(0).reshape(H, W))

    def render_video(self, t_grid):
        """Render T frames. t_grid: [T] → video: [T,1,H,W]."""
        centers = self.scene.centers        # [M,2]
        amps = self.scene.get_amplitudes()  # [M]
        Sigma = self.scene.get_covariances()  # [M,2,2]

        centers_t = self.motion.transform_centers(centers, t_grid)      # [T,M,2]
        Sigma_t = self.motion.transform_covariances(Sigma, t_grid)      # [T,M,2,2]
        eye = 1e-6 * torch.eye(2, device=Sigma_t.device)
        Sinv_t = torch.linalg.inv(Sigma_t + eye)                       # [T,M,2,2]

        frames = []
        for t in range(t_grid.shape[0]):
            frames.append(self._render_frame(centers_t[t], Sinv_t[t], amps))
        return torch.stack(frames).unsqueeze(1)  # [T,1,H,W]

    @staticmethod
    def measure(video, patterns, frame_idx):
        """y_k = <P_k, frame[f(k)]>.
        video: [T,1,H,W], patterns: [K,H,W], frame_idx: [K] → y: [K]
        """
        K, dev = patterns.shape[0], video.device
        y = torch.empty(K, device=dev)
        T = video.shape[0]
        for f in range(T):
            mask = (frame_idx == f)
            if not mask.any(): continue
            fr = video[f, 0].reshape(-1)       # [HW]
            P = patterns[mask].reshape(-1, fr.shape[0])  # [Kf, HW]
            y[mask] = P @ fr
        return y

    @staticmethod
    def measure_interpolated(video, patterns):
        """Interpolated measurement: y[k] = <patterns[k], interp_frame(t_k)>.

        t_k = k / (K-1),  u_k = t_k * (T-1),  m_k = floor(u_k),  alpha_k = u_k - m_k.
        Boundary (m_k == T-1): clamp to m_k = T-2, alpha_k = 1.0 → recovers video[T-1].
        y[k] = (1-alpha_k)*<patterns[k], video[m_k]> + alpha_k*<patterns[k], video[m_k+1]>

        video: [T,1,H,W], patterns: [K,H,W] → y: [K]
        """
        T = video.shape[0]
        K = patterns.shape[0]
        dev = video.device

        k_idx = torch.arange(K, device=dev, dtype=torch.float32)
        u = k_idx / max(K - 1, 1) * (T - 1)       # [K], continuous frame position in [0, T-1]
        m_raw = torch.floor(u).long()               # [K]
        alpha_raw = u - m_raw.float()               # [K], in [0, 1)

        # Boundary: when m_raw == T-1 (only k=K-1), set m=T-2, alpha=1.0
        boundary = (m_raw >= T - 1)
        m     = torch.where(boundary, torch.full_like(m_raw, T - 2), m_raw)        # [K]
        alpha = torch.where(boundary, torch.ones_like(alpha_raw), alpha_raw)       # [K]

        # Gather neighboring frames: [K, H, W]
        frame0 = video[m,     0]                    # [K,H,W]
        frame1 = video[m + 1, 0]                    # [K,H,W]

        # Linear interpolation then inner product (numerically equivalent to interp then dot)
        HW = patterns.shape[1] * patterns.shape[2]
        P  = patterns.reshape(K, HW)                # [K, HW]
        a  = alpha.unsqueeze(1)                     # [K, 1]
        y  = (1.0 - a) * (P * frame0.reshape(K, HW)).sum(-1) \
           +        a  * (P * frame1.reshape(K, HW)).sum(-1)  # [K]
        return y

    def forward(self, patterns, frame_idx, t_grid):
        """Full forward. Returns (y_pred [K], video [T,1,H,W])."""
        video = self.render_video(t_grid)
        if self.time_assignment_mode == "uniform":
            y = self.measure(video, patterns, frame_idx)
        else:  # interpolation
            y = self.measure_interpolated(video, patterns)
        return y, video
