"""Podesavanje hiper parametara


    python tune.py --data data/radar.csv --member mlp   --trials 8 --epochs 20
    python tune.py --data data/radar.csv --member ridge
    python tune.py --data data/radar.csv --member rf
    python tune.py --data data/radar.csv --member xgb
    python tune.py --data data/radar.csv --member knn
    python tune.py --data data/radar.csv --meta         


"""

import os
import copy
import time
import argparse
import itertools
import numpy as np
import pandas as pd

from config import default_config
from features import build_arrays
from members import build_member
from metrics import reconstruction_metrics


# search grids per member (only the fields that matter for each one)
GRIDS = {
    "mlp":   {"hidden": [128, 256, 512], "n_blocks": [2, 3, 4],
              "dropout": [0.0, 0.1, 0.2], "lr": [1e-3, 3e-4]},
    "ridge": {"ridge_alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
    "rf":    {"rf_max_depth": [8, 16, 24], "rf_estimators": [100, 200]},
    "xgb":   {"xgb_max_depth": [4, 6, 8], "xgb_estimators": [200, 400]},
    "knn":   {"knn_neighbors": [5, 10, 20, 40]},
}


def grid_combos(grid):
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]


def eval_member(name, cfg, splits, stats):
    """Train one member on TRAIN, score reconstruction NMSE (dB) on VAL."""
    Xtr, Ytr, _ = splits["train"]
    Xva, Yva, Mva = splits["val"]
    mem = build_member(name, cfg)
    t = time.time()
    mem.fit(Xtr, Ytr)
    dt = time.time() - t
    pred = mem.predict(Xva)
    met = reconstruction_metrics(pred, Yva, Mva, stats)
    return met["NMSE_dB"], met["ExplVar"], dt


def tune_member(name, base_cfg, splits, stats, trials, rng):
    combos = grid_combos(GRIDS[name])
    if trials and len(combos) > trials:
        sel = rng.choice(len(combos), size=trials, replace=False)
        combos = [combos[i] for i in sel]
    rows = []
    for i, combo in enumerate(combos, 1):
        cfg = copy.deepcopy(base_cfg)
        for k, v in combo.items():
            setattr(cfg, k, v)
        nmse_db, evs, dt = eval_member(name, cfg, splits, stats)
        rows.append({**combo, "val_NMSE_dB": round(nmse_db, 3),
                     "val_ExplVar": round(evs, 3), "fit_s": round(dt, 1)})
        print(f"  [{i}/{len(combos)}] {combo} -> val NMSE {nmse_db:.2f} dB")
    df = pd.DataFrame(rows).sort_values("val_NMSE_dB").reset_index(drop=True)
    return df


def tune_meta(base_cfg, splits, stats, rng):
    """Train all members on TRAIN, then for each meta_alpha fit the Ridge meta-model
    on one half of VAL (masked entries) and score it on the other half."""
    from sklearn.linear_model import Ridge
    M = base_cfg.M
    Xtr, Ytr, _ = splits["train"]
    Xva, Yva, Mva = splits["val"]

    names = [n for n in base_cfg.members if n != "huber"]   # huber is useless
    members = []
    for n in names:
        mem = build_member(n, base_cfg)
        print(f"  training member '{n}' ...")
        mem.fit(Xtr, Ytr)
        members.append(mem)
    preds = {m.name: m.predict(Xva) for m in members}

    nva = Xva.shape[0]
    perm = rng.permutation(nva)
    a_rows, b_rows = perm[:nva // 2], perm[nva // 2:]

    def build_FY(rows):
        mask = Mva[rows]
        hid = (mask == 0)
        F = np.concatenate([preds[n][rows].reshape(len(rows), M, 2)[hid] for n in names], axis=1)
        Ym = Yva[rows].reshape(len(rows), M, 2)[hid]
        return F, Ym

    FA, YA = build_FY(a_rows)
    FB, YB = build_FY(b_rows)

    rows = []
    for alpha in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        meta = Ridge(alpha=alpha).fit(FA, YA)
        pred = meta.predict(FB)
        nmse = float(((pred - YB) ** 2).sum() / ((YB ** 2).sum() + 1e-12))
        rows.append({"meta_alpha": alpha, "val_NMSE_dB": round(10 * np.log10(nmse + 1e-12), 3)})
        print(f"  meta_alpha={alpha:<7} -> held-out val NMSE {10*np.log10(nmse+1e-12):.2f} dB")
    return pd.DataFrame(rows).sort_values("val_NMSE_dB").reset_index(drop=True)


def main():
    cfg = default_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=cfg.data_csv)
    ap.add_argument("--P", type=int, default=cfg.P)
    ap.add_argument("--member", choices=list(GRIDS), help="which member to tune")
    ap.add_argument("--meta", action="store_true", help="tune the meta-model alpha instead")
    ap.add_argument("--members", nargs="+", default=None,
                    help="override member list (used by --meta)")
    ap.add_argument("--trials", type=int, default=8, help="max configs for large grids (mlp)")
    ap.add_argument("--epochs", type=int, default=20, help="MLP epochs during the search")
    ap.add_argument("--results", default=cfg.results_dir)
    args = ap.parse_args()

    cfg.data_csv, cfg.P, cfg.results_dir = args.data, args.P, args.results
    cfg.mlp_member_epochs = args.epochs
    if args.members:
        cfg.members = tuple(args.members)
    os.makedirs(cfg.results_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    print("Loading data ...")
    splits, stats, _ = build_arrays(cfg)
    print(f"Train/val pulses: {len(splits['train'][0])}/{len(splits['val'][0])}")

    if args.meta:
        df = tune_meta(cfg, splits, stats, rng)
        out = os.path.join(cfg.results_dir, "tune_meta.csv")
    elif args.member:
        df = tune_member(args.member, cfg, splits, stats, args.trials, rng)
        out = os.path.join(cfg.results_dir, f"tune_{args.member}.csv")
    else:
        ap.error("choose --member <name> or --meta")

    df.to_csv(out, index=False)
    print("\nRanked results (best first):")
    print(df.to_string(index=False))
    print(f"\nBest: {df.iloc[0].to_dict()}")
    print(f"Saved -> {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
