"""Train a pixel-space video diffusion prior (DDPM, epsilon-prediction).

Loss:
    L = E_{x0, sigma, eps} [ || eps_theta(x0 + sigma*eps, sigma) - eps ||^2 ]

where sigma ~ LogUniform(sigma_min, sigma_max).

Usage:
    python scripts/train_diffusion_prior.py
    python scripts/train_diffusion_prior.py --config configs/diffusion_prior.yaml
"""
import argparse, copy, os, sys, time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gsdiff.prior.unet3d import UNet3D
from gsdiff.prior.noise_schedule import NoiseSchedule
from gsdiff.utils import set_seed, get_device


# ─── Dataset ──────────────────────────────────────────────────

class VideoDataset(Dataset):
    """Loads pre-generated video tensor from .pt file."""

    def __init__(self, path: str):
        data = torch.load(path, map_location='cpu', weights_only=True)
        self.videos = data["videos"]   # [N, T, H, W] float32

    def __len__(self):
        return self.videos.shape[0]

    def __getitem__(self, idx):
        return self.videos[idx]        # [T, H, W]


# ─── EMA helper ───────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for sp, mp in zip(self.shadow.parameters(), model.parameters()):
            sp.data.mul_(self.decay).add_(mp.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


# ─── Training ─────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/diffusion_prior.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    dev = get_device()

    # Dataset
    ds = VideoDataset(cfg["dataset"]["path"])
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                        shuffle=True, num_workers=0, drop_last=True)
    print(f"Dataset: {len(ds)} videos, shape per video: {ds[0].shape}")

    # Model
    mc = cfg["model"]
    model = UNet3D(
        in_channels=mc["in_channels"],
        base_channels=mc["base_channels"],
        channel_mults=mc["channel_mults"],
        emb_dim=mc["emb_dim"],
    ).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet3D: {n_params / 1e6:.2f}M parameters")

    # Noise schedule
    nc = cfg["noise"]
    schedule = NoiseSchedule(sigma_min=nc["sigma_min"], sigma_max=nc["sigma_max"])

    # Optimizer
    tc = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"],
                                  weight_decay=tc["weight_decay"])

    # EMA
    ema_decay = tc.get("ema_decay", 0)
    ema = EMA(model, decay=ema_decay) if ema_decay > 0 else None

    # Output
    oc = cfg["output"]
    os.makedirs(oc["checkpoint_dir"], exist_ok=True)

    # Training loop
    log_every = tc.get("log_every", 100)
    global_step = 0
    t0 = time.time()

    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            # batch: [B, T, H, W]
            x0 = batch.unsqueeze(1).to(dev)  # [B, 1, T, H, W]
            B = x0.shape[0]

            # Sample noise level: log-uniform in [sigma_min, sigma_max]
            t = torch.rand(B, device=dev)
            sigma = schedule.sigma(t)                            # [B]
            eps = torch.randn_like(x0)                           # [B,1,T,H,W]
            x_noisy = x0 + sigma.view(B, 1, 1, 1, 1) * eps     # [B,1,T,H,W]

            eps_pred = model(x_noisy, sigma)                     # [B,1,T,H,W]
            loss = F.mse_loss(eps_pred, eps)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if ema is not None:
                ema.update(model)

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % log_every == 0:
                print(f"  step {global_step:6d}  loss={loss.item():.6f}")

        avg = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{tc['epochs']}  avg_loss={avg:.6f}  elapsed={elapsed:.0f}s")

        # Save checkpoint
        save_every = oc.get("save_every_epoch", 50)
        if epoch % save_every == 0 or epoch == tc["epochs"]:
            sd = ema.state_dict() if ema is not None else model.state_dict()
            path = os.path.join(oc["checkpoint_dir"], f"diffusion_prior_ep{epoch}.pt")
            torch.save(sd, path)
            print(f"  -> saved {path}")

    # Final checkpoint (always save)
    sd = ema.state_dict() if ema is not None else model.state_dict()
    final_path = os.path.join(oc["checkpoint_dir"], "diffusion_prior.pt")
    torch.save(sd, final_path)
    print(f"\nTraining done. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
