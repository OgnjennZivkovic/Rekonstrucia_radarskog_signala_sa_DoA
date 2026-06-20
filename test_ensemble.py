"""Test an already-trained ensemble (no retraining).

    python test_ensemble.py --ensemble checkpoints/ensemble.joblib --data data/radar.csv

Loads the saved Stacker, rebuilds the test split from the CSV (same seed -> same
held-out test set), evaluates the full metric set + DoA, and writes everything to
files under results_dir/test/. Use the SAME CSV that was used for training (the
split and normalization are tied to it).
"""
import os
import argparse
import numpy as np
import pandas as pd

from features import build_arrays
from ensemble import load_stacker, make_reconstruct_fn
from metrics import reconstruction_metrics, doa_metric
import viz
"""
import os
import argparse
import numpy as np
import pandas as pd

from features import build_arrays
from ensemble import load_stacker, make_reconstruct_fn
from metrics import reconstruction_metrics, doa_metric
import viz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", default="checkpoints/ensemble.joblib")
    ap.add_argument("--data", default=None, help="CSV (defaults to the one in the saved config)")
    ap.add_argument("--out", default=None, help="output dir (default: <results_dir>/test)")
    ap.add_argument("--no_doa", action="store_true")
    args = ap.parse_args()

    stack = load_stacker(args.ensemble)
    cfg = stack.cfg
    if args.data:
        cfg.data_csv = args.data
    out = args.out or os.path.join(cfg.results_dir, "test")
    os.makedirs(out, exist_ok=True)

    splits, stats, test_scen = build_arrays(cfg)
    Xte, Yte, Mte = splits["test"]
    print(f"Loaded ensemble with members {cfg.members}; test pulses: {len(Xte)}")

    member_preds = stack.member_predictions(Xte)
    ens_pred = stack.ensemble_predict(Xte, member_preds)

    results = {n: reconstruction_metrics(p, Yte, Mte, stats) for n, p in member_preds.items()}
    results["ENSEMBLE"] = reconstruction_metrics(ens_pred, Yte, Mte, stats)
    names = list(member_preds.keys()) + ["ENSEMBLE"]

    cols = ["NMSE", "NMSE_dB", "ExplVar", "MAE", "PhaseErrDeg", "ComplexCorr", "P95", "MaxErr"]
    df = pd.DataFrame({c: [results[n][c] for n in names] for c in cols}, index=names)
    df.index.name = "model"
    df.to_csv(os.path.join(out, "metrics.csv"))

    viz.plot_model_comparison(names, [results[n]["NMSE_dB"] for n in names],
                              os.path.join(out, "ensemble_comparison.png"))
    viz.plot_reconstruction(ens_pred[0], Yte[0], Mte[0], stats,
                            os.path.join(out, "reconstruction_example.png"),
                            "ENSEMBLE reconstruction (test)")

    if not args.no_doa:
        d = doa_metric(make_reconstruct_fn(stack, cfg), test_scen, cfg, stats)
        pd.DataFrame([d]).to_csv(os.path.join(out, "doa.csv"), index=False)

    print(f"ENSEMBLE NMSE = {results['ENSEMBLE']['NMSE_dB']:.2f} dB")
    print(f"Test results written to: {os.path.abspath(out)}/")


if __name__ == "__main__":
    main()"""
"""TEST an already-trained ensemble (option 2). Evaluation only -- no training."""




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", default="result_e/checkpoints/ensemble.joblib")
    ap.add_argument("--data", default=None, help="CSV (defaults to the one in the saved config)")
    ap.add_argument("--results", default=None, help="output dir (default: saved results_dir)")
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
    print(f"Loaded ensemble {cfg.members}; test pulses: {len(Xte)}")

    member_preds = stack.member_predictions(Xte)
    ens_pred = stack.ensemble_predict(Xte, member_preds)

    results = {n: reconstruction_metrics(p, Yte, Mte, stats) for n, p in member_preds.items()}
    results["ENSEMBLE"] = reconstruction_metrics(ens_pred, Yte, Mte, stats)
    names = list(member_preds.keys()) + ["ENSEMBLE"]

    cols = ["NMSE", "NMSE_dB", "ExplVar", "MAE", "PhaseErrDeg", "ComplexCorr", "P95", "MaxErr"]
    df = pd.DataFrame({c: [results[n][c] for n in names] for c in cols}, index=names)
    df.index.name = "model"
    df.to_csv(os.path.join(out, "metrics.csv"))

    by_model = {n: results[n]["NMSE_by_masked"] for n in names}
    counts = sorted({c for d in by_model.values() for c in d})
    bc = pd.DataFrame(
        {n: [10 * np.log10(by_model[n].get(c, np.nan) + 1e-12) for c in counts] for n in names},
        index=counts)
    bc.index.name = "n_masked"
    bc.to_csv(os.path.join(out, "nmse_by_maskcount.csv"))

    viz.plot_model_comparison(names, [results[n]["NMSE_dB"] for n in names],
                              os.path.join(out, "ensemble_comparison.png"))
    viz.plot_nmse_by_maskcount(by_model, os.path.join(out, "nmse_by_maskcount.png"))
    viz.plot_reconstruction(ens_pred[0], Yte[0], Mte[0], stats,
                            os.path.join(out, "reconstruction_example.png"),
                            "ENSEMBLE reconstruction (orange = masked antennas)")

    if not args.no_doa:
        d = doa_metric(make_reconstruct_fn(stack, cfg), test_scen, cfg, stats,
                       figdir=os.path.join(out, "doa_spectra"))
        by_k = d.pop("by_K")
        pd.DataFrame([d]).to_csv(os.path.join(out, "doa.csv"), index=False)
        pd.DataFrame(list(by_k.values())).to_csv(os.path.join(out, "doa_by_k.csv"), index=False)

    print(f"ENSEMBLE NMSE = {results['ENSEMBLE']['NMSE_dB']:.2f} dB")
    print(f"All test results written to: {os.path.abspath(out)}/")


if __name__ == "__main__":
    main()
