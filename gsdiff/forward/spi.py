"""SPI forward: (scene, motion) → video → bucket measurements.

Physics: y_k = Σ_{i,j} P_k(i,j) · frame[f(k)](i,j)
where frame[t] is rendered from SE(2)-transformed Gaussians.
"""
import torch, torch.nn as nn, torch.nn.functional as F


class SPIForwardModel(nn.Module):

    def __init__(self, scene, motion, H, W):
        super().__init__()
        self.scene = scene
        self.motion = motion
        self.H, self.W = H, W

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

    def forward(self, patterns, frame_idx, t_grid):
        """Full forward. Returns (y_pred [K], video [T,1,H,W])."""
        video = self.render_video(t_grid)
        y = self.measure(video, patterns, frame_idx)
        return y, video
