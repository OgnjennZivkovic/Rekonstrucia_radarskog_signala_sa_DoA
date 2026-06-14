"""Train the stacking ensemble (option 2) and write all outputs to files.

    python run_ensemble.py --data data/radar.csv

Outputs (under results_dir, default 'results/'):
    metrics.csv               -- full metric set, one row per model + ENSEMBLE
    nmse_by_maskcount.csv      -- NMSE (dB) vs number of masked antennas, per model
    doa.csv                    -- DoA angle error / hit-rate (recon vs true ref)
    ensemble_comparison.png    -- NMSE bar chart
    nmse_by_maskcount.png      -- error vs missing-data curve
    reconstruction_example.png -- ensemble true-vs-reconstructed for one sample
    mlp_member/epoch_XXX.png   -- reconstruction per epoch while the MLP trains
Also saves the trained ensemble to ensemble_path so test_ensemble.py can reuse it.
"""
import os
import argparse
import pandas as pd
from config import default_config
from features import build_arrays
from ensemble import Stacker
from metrics import reconstruction_metrics
"""
import os
import argparse
import numpy as np
import pandas as pd

from config import default_config
from features import build_arrays
from ensemble import Stacker, make_reconstruct_fn
from metrics import reconstruction_metrics, doa_metric
import viz


def main():
    cfg = default_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=cfg.data_csv)
    ap.add_argument("--P", type=int, default=cfg.P)
    ap.add_argument("--members", nargs="+", default=None)
    ap.add_argument("--results", default=cfg.results_dir)
    ap.add_argument("--no_doa", action="store_true")
    args = ap.parse_args()
    cfg.data_csv, cfg.P, cfg.results_dir = args.data, args.P, args.results
    if args.members:
        cfg.members = tuple(args.members)

    os.makedirs(cfg.results_dir, exist_ok=True)
    print(f"Members: {cfg.members}")
    splits, stats, test_scen = build_arrays(cfg)
    Xte, Yte, Mte = splits["test"]
    print(f"Train/val/test pulses: "
          f"{len(splits['train'][0])}/{len(splits['val'][0])}/{len(Xte)}")

    stack = Stacker(cfg).fit(
        splits, stats=stats, mlp_figdir=os.path.join(cfg.results_dir, "mlp_member"))
    stack.save(cfg.ensemble_path)

    member_preds = stack.member_predictions(Xte)
    ens_pred = stack.ensemble_predict(Xte, member_preds)

    results = {n: reconstruction_metrics(p, Yte, Mte, stats) for n, p in member_preds.items()}
    results["ENSEMBLE"] = reconstruction_metrics(ens_pred, Yte, Mte, stats)
    names = list(member_preds.keys()) + ["ENSEMBLE"]

    # ---- metrics.csv ----
    cols = ["NMSE", "NMSE_dB", "ExplVar", "MAE", "PhaseErrDeg", "ComplexCorr", "P95", "MaxErr"]
    df = pd.DataFrame({c: [results[n][c] for n in names] for c in cols}, index=names)
    df.index.name = "model"
    df.to_csv(os.path.join(cfg.results_dir, "metrics.csv"))

    # ---- nmse_by_maskcount.csv ----
    by_model = {n: results[n]["NMSE_by_masked"] for n in names}
    counts = sorted({c for d in by_model.values() for c in d})
    bc = pd.DataFrame(
        {n: [10 * np.log10(by_model[n].get(c, np.nan) + 1e-12) for c in counts] for n in names},
        index=counts)
    bc.index.name = "n_masked"
    bc.to_csv(os.path.join(cfg.results_dir, "nmse_by_maskcount.csv"))

    # ---- figures ----
    viz.plot_model_comparison(names, [results[n]["NMSE_dB"] for n in names],
                              os.path.join(cfg.results_dir, "ensemble_comparison.png"))
    viz.plot_nmse_by_maskcount(by_model, os.path.join(cfg.results_dir, "nmse_by_maskcount.png"))
    viz.plot_reconstruction(ens_pred[0], Yte[0], Mte[0], stats,
                            os.path.join(cfg.results_dir, "reconstruction_example.png"),
                            "ENSEMBLE reconstruction (orange = masked antennas)")

    # ---- DoA ----
    if not args.no_doa:
        d = doa_metric(make_reconstruct_fn(stack, cfg), test_scen, cfg, stats)
        pd.DataFrame([d]).to_csv(os.path.join(cfg.results_dir, "doa.csv"), index=False)

    print(f"\nDone. ENSEMBLE NMSE = {results['ENSEMBLE']['NMSE_dB']:.2f} dB")
    print(f"All results written to: {os.path.abspath(cfg.results_dir)}/")
    print("  metrics.csv, nmse_by_maskcount.csv, doa.csv, *.png, mlp_member/epoch_*.png")


if __name__ == "__main__":
    main()"""
"""TRAIN the stacking ensemble (option 2). Training only -- no evaluation."""




def main():
    cfg = default_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=cfg.data_csv)
    ap.add_argument("--P", type=int, default=cfg.P)
    ap.add_argument("--members", nargs="+", default=None)
    ap.add_argument("--results", default=cfg.results_dir)
    ap.add_argument("--ensemble", default=cfg.ensemble_path)
    args = ap.parse_args()
    cfg.data_csv, cfg.P, cfg.results_dir, cfg.ensemble_path = (
        args.data, args.P, args.results, args.ensemble)
    if args.members:
        cfg.members = tuple(args.members)

    os.makedirs(cfg.results_dir, exist_ok=True)
    print(f"Members: {cfg.members}")

    splits, stats, _test_scen = build_arrays(cfg)
    print(f"Train/val pulses: {len(splits['train'][0])}/{len(splits['val'][0])}")

    stack = Stacker(cfg).fit(
        splits, stats=stats, mlp_figdir=os.path.join(cfg.results_dir, "mlp_member"))
    stack.save(cfg.ensemble_path)

    # ---- validation metrics ----
    Xva, Yva, Mva = splits["val"]
    member_preds = stack.member_predictions(Xva)
    ens_pred = stack.ensemble_predict(Xva, member_preds)

    results = {n: reconstruction_metrics(p, Yva, Mva, stats) for n, p in member_preds.items()}
    results["ENSEMBLE"] = reconstruction_metrics(ens_pred, Yva, Mva, stats)
    names = list(member_preds.keys()) + ["ENSEMBLE"]

    cols = ["NMSE", "NMSE_dB", "ExplVar", "MAE", "PhaseErrDeg", "ComplexCorr", "P95", "MaxErr"]
    df = pd.DataFrame({c: [results[n][c] for n in names] for c in cols}, index=names)
    df.index.name = "model"
    train_metrics_path = os.path.join(cfg.results_dir, "train_metrics.csv")
    df.to_csv(train_metrics_path)
    print(f"\nValidation metrics saved to: {os.path.abspath(train_metrics_path)}")
    print(f"  ENSEMBLE val NMSE = {results['ENSEMBLE']['NMSE_dB']:.2f} dB")

    print("\nTraining done.")
    print(f"  ensemble saved to : {os.path.abspath(cfg.ensemble_path)}")
    print(f"  per-epoch figures : {os.path.abspath(os.path.join(cfg.results_dir, 'mlp_member'))}/")
    print("  now run:  python test_ensemble.py "
          f"--ensemble {cfg.ensemble_path} --data {cfg.data_csv}")


if __name__ == "__main__":
    main()
