"""Shared plotting helpers (matplotlib, headless)."""

import os
import numpy as np


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_reconstruction(recon, true, mask, stats, path, title):
    """recon, true: (2M,) or (M,2) normalized; mask: (M,). De-normalizes and plots."""
    plt = _mpl()
    M = mask.shape[0]
    mean2 = stats["mean"].reshape(1, 2)
    std2 = stats["std"].reshape(1, 2)
    r = recon.reshape(M, 2) * std2 + mean2
    t = true.reshape(M, 2) * std2 + mean2
    hidden = (mask == 0)
    ants = np.arange(M)

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
    axes[0].set_title(title)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_model_comparison(names, nmse_db, path, title="Stacking ensemble vs members"):
    plt = _mpl()
    colors = ["#85B7EB"] * (len(names) - 1) + ["#1D9E75"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(names, nmse_db, color=colors)
    ax.set_ylabel("NMSE on masked antennas (dB)  -- lower is better")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate(nmse_db):
        ax.annotate(f"{v:.1f}", (i, v), ha="center",
                    va="top" if v < 0 else "bottom", fontsize=9)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_nmse_by_maskcount(by_count_per_model, path):
    """by_count_per_model: dict model -> {count: nmse_ratio}."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, d in by_count_per_model.items():
        cs = sorted(d.keys())
        ax.plot(cs, [10 * np.log10(d[c] + 1e-12) for c in cs], "-o", ms=4, label=name)
    ax.set_xlabel("number of masked antennas")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title("Reconstruction error vs amount of missing data")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_doa_spectrum(doa_result, path):
    """MUSIC pseudo-spectrum (dB) vs angle for one representative scenario.

    doa_result must contain keys:
      'spectrum_scan'    -- angle grid (deg)
      'spectrum_recon'   -- MUSIC spectrum from reconstructed signal
      'spectrum_true'    -- MUSIC spectrum from true (non-masked) signal
      'spectrum_angles'  -- true angles for this scenario (deg)
    """
    plt = _mpl()
    scan = doa_result["spectrum_scan"]
    spec_r = doa_result["spectrum_recon"]
    spec_t = doa_result["spectrum_true"]
    truths = doa_result["spectrum_angles"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(scan, 10 * np.log10(np.maximum(spec_r, 1e-12)),
            color="#534AB7", lw=1.2, label="reconstruction")
    ax.plot(scan, 10 * np.log10(np.maximum(spec_t, 1e-12)),
            color="#1D9E75", lw=1.0, ls="--", label="true array")
    for i, a in enumerate(truths):
        ax.axvline(a, color="#D85A30", ls=":", lw=1.2,
                   label="true angle" if i == 0 else None)
    ax.set_xlabel("angle (deg)")
    ax.set_ylabel("MUSIC pseudo-spectrum (dB)")
    ax.set_title("MUSIC DoA spectrum — reconstruction vs true array")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_doa_scatter(doa_result, path):
    """Scatter plot: true angles vs estimated angles from reconstruction."""
    plt = _mpl()
    true_all = [a for scenario in doa_result["angles_true"] for a in scenario]
    rec_all = [a for scenario in doa_result["angles_recon"] for a in scenario]
    ref_all = [a for scenario in doa_result["angles_true_ref"] for a in scenario]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(true_all, rec_all, alpha=0.5, s=16, color="#534AB7", label="reconstruction")
    lim = max(abs(np.array(true_all + rec_all)).max(), 1)
    ax1.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax1.set_xlabel("true angle (deg)")
    ax1.set_ylabel("estimated angle (deg)")
    ax1.set_title("Reconstruction → MUSIC DoA")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.scatter(true_all, ref_all, alpha=0.5, s=16, color="#1D9E75", label="true array")
    ax2.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
    ax2.set_xlabel("true angle (deg)")
    ax2.set_ylabel("estimated angle (deg)")
    ax2.set_title("True array → MUSIC DoA (reference)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    fig.suptitle(f"DoA angle estimates ({len(doa_result['angles_true'])} scenarios)", fontsize=12)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
