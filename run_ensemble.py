# glavna skripta za treniranje celog ansambla
import os
import argparse
import pandas as pd
from config import default_config
from features import build_arrays
from ensemble import Stacker
from metrics import reconstruction_metrics




def main():
    cfg = default_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=cfg.data_csv)
    ap.add_argument("--P", type=int, default=cfg.P)
    ap.add_argument("--members", nargs="+", default=None) # mogucnost rucnog izbora clanova
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

    #Treniranje 
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
