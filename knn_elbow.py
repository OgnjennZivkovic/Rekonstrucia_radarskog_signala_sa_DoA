#Pravi lakat krivu za KNN 

import os
import time
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from config import default_config
from features import build_arrays
from metrics import reconstruction_metrics
import viz


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def main():
    cfg = default_config()
    K_VALUES = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]

    print("Loading data...")
    splits, stats, _test_scen = build_arrays(cfg)
    Xtr, Ytr, _ = splits["train"]
    Xva, Yva, Mva = splits["val"]
    print(f"Train pulses: {len(Xtr)}, Val pulses: {len(Xva)}")

    results = []  # list of (k, nmse_db, elapsed_sec)
    for k in K_VALUES:
        print(f"\n--- KNN k={k} ---")
        t0 = time.time()
        knn = KNeighborsRegressor(n_neighbors=k, n_jobs=-1)
        knn.fit(Xtr, Ytr)
        elapsed = time.time() - t0

        Yhat = knn.predict(Xva).astype(np.float32)
        metrics = reconstruction_metrics(Yhat, Yva, Mva, stats)
        nmse_db = metrics["NMSE_dB"]
        results.append((k, nmse_db, elapsed))
        print(f"  k={k:>3}  NMSE={nmse_db:.2f} dB  time={elapsed:.1f}s")

    # ---- Save CSV ----
    out_dir = cfg.results_dir
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results, columns=["k", "NMSE_dB", "time_sec"])
    csv_path = os.path.join(out_dir, "knn_elbow.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # ---- Plot elbow curve ----
    plt = _mpl()
    ks = [r[0] for r in results]
    nmse_vals = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(ks, nmse_vals, "-o", color="#534AB7", ms=7, lw=1.8)
    ax.set_xlabel("k (number of neighbors)", fontsize=12)
    ax.set_ylabel("NMSE on masked antennas (dB) — lower is better", fontsize=12)
    ax.set_title("KNN elbow curve: validation NMSE vs k", fontsize=14)
    ax.grid(alpha=0.3)

    # annotate each point
    for k, v in zip(ks, nmse_vals):
        ax.annotate(f"{v:.2f}", (k, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9)

    # mark the best k
    best_idx = int(np.argmin(nmse_vals))
    best_k, best_nmse = ks[best_idx], nmse_vals[best_idx]
    ax.scatter([best_k], [best_nmse], color="#D85A30", s=140, zorder=5,
               label=f"best k={best_k} ({best_nmse:.2f} dB)")
    ax.legend(fontsize=10)

    fig.tight_layout()
    png_path = os.path.join(out_dir, "knn_elbow.png")
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {png_path}")

    print(f"\n=== Best k = {best_k} (NMSE = {best_nmse:.2f} dB) ===")


if __name__ == "__main__":
    main()
