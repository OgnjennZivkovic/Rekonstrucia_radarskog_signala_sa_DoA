"""Evaluate a trained checkpoint and draw a reconstruction example.

    python evaluate.py --ckpt checkpoints/model.pt --data data/radar.csv

Prints the test-set NMSE and saves `reconstruction_example.png` showing the
true vs. reconstructed real/imag values across the 29 antennas for one pulse,
with the masked antennas highlighted.
"""

import argparse
import numpy as np
import torch

from config import Config
from data import build_dataloaders
from model import ReconMLP, nmse_on_masked


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["config"])
    model = ReconMLP(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg, ckpt["mean"], ckpt["std"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/model.pt")
    ap.add_argument("--data", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, mean, std = load_checkpoint(args.ckpt, device)
    if args.data:
        cfg.data_csv = args.data

    _, _, test_loader, _ = build_dataloaders(cfg)

    tot, n = 0.0, 0
    sample = None
    for masked, mask, target in test_loader:
        masked, mask, target = masked.to(device), mask.to(device), target.to(device)
        recon = model(masked, mask)
        tot += nmse_on_masked(recon, target, mask).item() * masked.size(0)
        n += masked.size(0)
        if sample is None:
            sample = (masked[0].cpu(), mask[0].cpu(), target[0].cpu(), recon[0].cpu())

    test_nmse = tot / max(n, 1)
    print(f"Test NMSE (masked antennas): {test_nmse:.4f}  ({10*np.log10(test_nmse):.1f} dB)")

    # ---- plot one example (de-normalized back to physical units) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping figure.")
        return

    masked, mask, target, recon = sample
    mean_t = torch.from_numpy(mean).squeeze(0)   # (1,2)
    std_t = torch.from_numpy(std).squeeze(0)
    target = target * std_t + mean_t
    recon = recon * std_t + mean_t
    hidden = (mask == 0).numpy()
    ants = np.arange(cfg.M)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, ch, name in ((axes[0], 0, "real part"), (axes[1], 1, "imag part")):
        ax.plot(ants, target[:, ch], "-o", color="#1D9E75", ms=4, label="true")
        ax.plot(ants, recon[:, ch], "--s", color="#534AB7", ms=4, label="reconstructed")
        ax.scatter(ants[hidden], target[hidden, ch], facecolors="none",
                   edgecolors="#D85A30", s=120, linewidths=1.8, label="masked antenna")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    axes[1].set_xlabel("antenna index")
    axes[0].set_title("Reconstruction example (orange circles = antennas the model had to predict)")
    fig.tight_layout()
    fig.savefig("reconstruction_example.png", dpi=150)
    print("Saved reconstruction_example.png")


if __name__ == "__main__":
    main()
