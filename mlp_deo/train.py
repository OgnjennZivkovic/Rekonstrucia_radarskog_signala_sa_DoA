"""Train the reconstruction model.

    python train.py                       # uses Config defaults
    python train.py --epochs 50 --data data/radar.csv

Saves a checkpoint containing the weights, the normalization stats, and the
config, so evaluate.py / infer.py can reload everything consistently.
"""

import os
import argparse
import numpy as np
import torch

from config import default_config
from data import build_dataloaders
from model import ReconMLP, masked_reconstruction_loss, nmse_on_masked


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot_nmse, n = 0.0, 0
    for masked, mask, target in loader:
        masked, mask, target = masked.to(device), mask.to(device), target.to(device)
        recon = model(masked, mask)
        tot_nmse += nmse_on_masked(recon, target, mask).item() * masked.size(0)
        n += masked.size(0)
    return tot_nmse / max(n, 1)


def plot_reconstruction(model, sample, stats, epoch, val_nmse, figdir, device):
    """Save a true-vs-reconstructed figure for one fixed sample (one per epoch)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  [warn] matplotlib not installed -> run `pip install matplotlib` "
              "(skipping per-epoch figure)")
        return

    masked, mask, target = sample
    was_training = model.training
    model.eval()
    with torch.no_grad():
        recon = model(masked.unsqueeze(0).to(device), mask.unsqueeze(0).to(device))[0].cpu()
    if was_training:
        model.train()

    mean2 = stats["mean"].reshape(1, 2)
    std2 = stats["std"].reshape(1, 2)
    t = target.numpy() * std2 + mean2      # de-normalize to physical units
    r = recon.numpy() * std2 + mean2
    hidden = (mask.numpy() == 0)
    ants = np.arange(t.shape[0])

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, ch, name in ((axes[0], 0, "real part"), (axes[1], 1, "imag part")):
        ax.plot(ants, t[:, ch], "-o", color="#1D9E75", ms=4, label="true")
        ax.plot(ants, r[:, ch], "--s", color="#534AB7", ms=4, label="reconstructed")
        ax.scatter(ants[hidden], t[hidden, ch], facecolors="none",
                   edgecolors="#D85A30", s=120, linewidths=1.8, label="masked antenna")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    axes[1].set_xlabel("antenna index")
    axes[0].set_title(f"epoch {epoch}  |  val NMSE {val_nmse:.4f}")
    os.makedirs(figdir, exist_ok=True)
    path = os.path.join(figdir, f"epoch_{epoch:03d}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  saved figure -> {os.path.abspath(path)}")


def main():
    cfg = default_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=cfg.data_csv)
    ap.add_argument("--epochs", type=int, default=cfg.epochs)
    ap.add_argument("--batch_size", type=int, default=cfg.batch_size)
    ap.add_argument("--lr", type=float, default=cfg.lr)
    ap.add_argument("--ckpt", default=cfg.ckpt_path)
    ap.add_argument("--P", type=int, default=cfg.P)
    ap.add_argument("--figdir", default="figures",
                    help="directory to save a reconstruction figure each epoch")
    args = ap.parse_args()

    cfg.data_csv, cfg.epochs, cfg.batch_size, cfg.lr, cfg.ckpt_path, cfg.P = (
        args.data, args.epochs, args.batch_size, args.lr, args.ckpt, args.P
    )

    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, stats = build_dataloaders(cfg)
    print(f"Train/val/test pulses: "
          f"{len(train_loader.dataset)}/{len(val_loader.dataset)}/{len(test_loader.dataset)}")

    model = ReconMLP(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # one fixed sample (same pulse + same mask every epoch) so the figures are comparable
    fixed_sample = val_loader.dataset[0]

    best_val = float("inf")
    os.makedirs(os.path.dirname(cfg.ckpt_path) or ".", exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for masked, mask, target in train_loader:
            masked, mask, target = masked.to(device), mask.to(device), target.to(device)
            recon = model(masked, mask)
            loss, hidden_loss = masked_reconstruction_loss(
                recon, target, mask, cfg.obs_loss_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += hidden_loss.item() * masked.size(0)

        train_mse = running / len(train_loader.dataset)
        val_nmse = evaluate(model, val_loader, device)
        print(f"epoch {epoch:3d} | train hidden-MSE {train_mse:.4f} | val NMSE {val_nmse:.4f}")
        plot_reconstruction(model, fixed_sample, stats, epoch, val_nmse, args.figdir, device)

        if val_nmse < best_val:
            best_val = val_nmse
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg.to_dict(),
                    "mean": stats["mean"],
                    "std": stats["std"],
                },
                cfg.ckpt_path,
            )

    test_nmse = evaluate(model, test_loader, device)
    print(f"\nBest val NMSE: {best_val:.4f}")
    print(f"Test NMSE (masked antennas): {test_nmse:.4f}  ({10*np.log10(test_nmse):.1f} dB)")
    print(f"Saved best checkpoint to {cfg.ckpt_path}")


if __name__ == "__main__":
    main()
