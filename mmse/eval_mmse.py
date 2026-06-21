"""Evaluate the per-scenario MMSE reconstructor and compare it with the ensemble.

    python eval_mmse.py --ensemble result_e/checkpoints/ensemble.joblib --data data/radar.csv

Loads the trained ensemble (for the config + normalization stats), rebuilds the
same held-out test split, and evaluates BOTH the ensemble members and the MMSE
reconstructor on the SAME test pulses/masks. Writes a combined metrics table and
a comparison bar chart (now including 'mmse'). Optionally runs the DoA check for
MMSE too.
"""

import os
import argparse
import numpy as np
import pandas as pd

from features import build_arrays
from ensemble import load_stacker, make_reconstruct_fn
from metrics import reconstruction_metrics, doa_metric
from mmse import mmse_reconstruct, make_mmse_reconstruct_fn
import viz


def mmse_test_predictions(Yte, Mte, P, ridge):
    """Reconstruct the whole test split with MMSE, scenario by scenario.
    Yte: (n, 2M) normalized true; Mte: (n, M). Returns (n, 2M) normalized predictions."""
    n, twoM = Yte.shape
    M = twoM // 2
    n_scen = n // P
    yt = Yte.reshape(n_scen, P, M, 2)
    mk = Mte.reshape(n_scen, P, M)
    preds = np.empty_like(yt)
    for s in range(n_scen):
        masked_signal = yt[s] * mk[s][:, :, None]          # zero the masked antennas
        preds[s] = mmse_reconstruct(masked_signal, mk[s], ridge)
    return preds.reshape(n, twoM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", default="result_e/checkpoints/ensemble.joblib")
    ap.add_argument("--data", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--ridge", type=float, default=1e-2, help="diagonal loading for MMSE")
    ap.add_argument("--no_doa", action="store_true")
    args = ap.parse_args()

    stack = load_stacker(args.ensemble)
    cfg = stack.cfg
    if args.data:
        cfg.data_csv = args.data
    out = args.results or cfg.results_dir
    os.makedirs(out, exist_ok=True)

    splits, stats, test_scen = build_arrays(cfg)
    Xte, Yte, Mte = splits["test"]
    print(f"Test pulses: {len(Xte)}  ({len(Xte)//cfg.P} scenarios)")

    # ---- ensemble members + ENSEMBLE ----
    member_preds = stack.member_predictions(Xte)
    ens_pred = stack.ensemble_predict(Xte, member_preds)
    results = {n: reconstruction_metrics(p, Yte, Mte, stats) for n, p in member_preds.items()}
    results["ENSEMBLE"] = reconstruction_metrics(ens_pred, Yte, Mte, stats)

    # ---- MMSE ----
    print("Running MMSE reconstruction ...")
    mmse_pred = mmse_test_predictions(Yte, Mte, cfg.P, args.ridge)
    results["mmse"] = reconstruction_metrics(mmse_pred, Yte, Mte, stats)

    names = list(member_preds.keys()) + ["mmse", "ENSEMBLE"]
    cols = ["NMSE", "NMSE_dB", "ExplVar", "MAE", "PhaseErrDeg", "ComplexCorr", "P95", "MaxErr"]
    df = pd.DataFrame({c: [results[n][c] for n in names] for c in cols}, index=names)
    df.index.name = "model"
    df.to_csv(os.path.join(out, "metrics_with_mmse.csv"))

    viz.plot_model_comparison(names, [results[n]["NMSE_dB"] for n in names],
                              os.path.join(out, "comparison_with_mmse.png"),
                              title="Members vs MMSE vs ensemble")

    # ---- DoA for MMSE (optional) ----
    if not args.no_doa:
        d = doa_metric(make_mmse_reconstruct_fn(cfg, args.ridge), test_scen, cfg, stats)
        d.pop("by_K", None)
        pd.DataFrame([d]).to_csv(os.path.join(out, "doa_mmse.csv"), index=False)

    print("\nReconstruction NMSE (dB):")
    for n in names:
        print(f"  {n:10s} {results[n]['NMSE_dB']:7.2f}   ExplVar {results[n]['ExplVar']:.3f}")
    print(f"\nWritten to: {os.path.abspath(out)}/  (metrics_with_mmse.csv, comparison_with_mmse.png)")


if __name__ == "__main__":
    main()
