# Radar antenna-signal reconstruction

A small, complete PyTorch project that learns to reconstruct **masked
(missing) antennas** in a radar array — the task described by the dataset.

## The task

- Each scenario is a radar measurement over **P pulses**, each pulse giving one
  complex value per antenna for **M = 29** antennas.
- Some antennas are **masked** (broken / switched off / dropped).
- **Input:** the antenna readings with a subset masked, plus a mask telling the
  model which antennas are missing.
- **Output:** the reconstructed values at the masked antennas.

The reconstruction is well-posed because each pulse's 29 antenna values lie in a
low-dimensional subspace set by the targets (each target adds one phase ramp
across the array), so the missing antennas can be inferred from the rest.

## Files

| file               | purpose |
|--------------------|---------|
| `config.py`        | all hyperparameters and paths in one place |
| `generate_data.py` | simulate a dataset in the exact CSV format (run before the real data) |
| `data.py`          | parse the complex-string CSV, group pulses into scenarios, apply masking |
| `model.py`         | the reconstruction network (residual MLP) + masked loss / NMSE metric |
| `train.py`         | training loop with validation and checkpointing |
| `evaluate.py`      | test-set NMSE + a true-vs-reconstructed figure |
| `infer.py`         | fill in missing antennas for an arbitrary input |

## Quickstart

```bash
pip install -r requirements.txt

# 1) make synthetic data in the dataset's format (or skip and use the real CSV)
python generate_data.py --out data/radar.csv --scenarios 300 --pulses 64

# 2) train
python train.py --data data/radar.csv --P 64 --epochs 25

# 3) evaluate (prints NMSE, saves reconstruction_example.png)
python evaluate.py --ckpt checkpoints/model.pt --data data/radar.csv

# 4) reconstruct specific hidden antennas for one pulse
python infer.py --ckpt checkpoints/model.pt --data data/radar.csv --mask 4 9 15
```

## Using the real dataset

Point `--data` at the real CSV and set `--P` to the true pulses-per-scenario
(**256** for this dataset). The loader expects the columns
`angle, ant_1, ..., ant_29`, with `ant_*` written as complex strings like
`(0.15+0.21j)` and `angle` as a bracketed list like `[58 -57 21]`. Rows are
grouped into scenarios in fixed blocks of `P`. For the full set:

```bash
python train.py --data path/to/real.csv --P 256 --epochs 50
```

## How it works (model)

The masked signal `(M, 2)` is flattened to `2M` real numbers and concatenated
with the `M`-long mask, so the network always knows which antennas to invent.
A residual MLP maps this to the full `2M` reconstruction. The loss is mean
squared error on the **masked** antennas (with a small term on the observed
ones for stability), and quality is reported as **NMSE** on the masked antennas
(lower is better; e.g. -7 dB means the residual error is ~19% of the signal
power).

## Ideas to extend

- **Per-antenna transformer** (a masked-autoencoder over the 29 antennas) for
  stronger handling of arbitrary mask patterns.
- **Use all P pulses jointly** (estimate the spatial covariance) rather than
  reconstructing each pulse independently — more information per scenario.
- **Per-scenario masks** (fixed broken antennas across all pulses) if that
  matches how masking is applied in the real data.
- **Auxiliary angle head** that also predicts the targets' angles, so
  reconstruction is regularized to preserve direction-of-arrival information.

---

## Stacking ensemble (option 2)

Beyond the single MLP, the project includes a **stacking ensemble**: several
different models each reconstruct the masked antennas, and a meta-model learns
how to combine them.

- **Members** (`config.py: members`): `linear`, `ridge`, `huber`, `rf`
  (RandomForest), `xgb` (XGBoost), `knn`, `mlp` (the ReconMLP).
- **Meta-model**: a Ridge regressor trained **only on the masked entries** of
  the held-out validation split, so it can favour the strongest member rather
  than averaging everything. Training on validation (not train) is what stops
  it from trusting overfit members.

Run it:

```bash
python run_ensemble.py --data data/radar.csv                 # all members
python run_ensemble.py --data data/radar.csv --members linear ridge mlp
python run_ensemble.py --data data/radar.csv --no_doa        # skip the DoA check
```

It prints a comparison table of every member and the ensemble across the full
metric set, an NMSE-by-mask-count breakdown, the DoA result, and saves
`ensemble_comparison.png`.

On a quick 300-scenario test the ensemble beat the best single member on every
metric (e.g. NMSE -5.4 dB vs the MLP's -5.0 dB).

### Cost note
`huber` trains 58 separate robust regressions (one per output) and is by far
the slowest member; `rf`/`xgb` are also heavy. They are capped to
`tree_sample_cap` training rows. On the full 512k-row dataset, start with
`--members linear ridge mlp` to get going fast, then add the tree members.

## Metrics (same set for every model)

1. **NMSE** on masked antennas (ratio + dB) -- the headline number
2. **Explained Variance Score**
3. **MAE** on the complex error magnitude (robust to outliers)
4. **Mean absolute phase error** (degrees) -- domain metric
5. **Complex correlation** (0..1, scale-independent)
6. **P95 / max error** -- tail behaviour
7. **NMSE by number of masked antennas** -- how error grows with missing data
8. **DoA check (MUSIC)** -- angle error and detection rate on the reconstructed
   array vs the true array, so you see how much reconstruction degraded the
   direction information (the absolute angle error depends on how close the
   targets are, so read it as recon-vs-true, not as an absolute).

## Preprocessing / augmentation toggles (`config.py`)

- `per_pulse_power_norm` -- scale each pulse to unit power (all splits)
- `aug_phase_rot` -- train-only random global phase rotation
- `aug_mirror` -- train-only antenna-order flip + conjugate

All default to off, so behaviour is unchanged unless you enable them.

---

## Saving, separate testing, and file outputs (ensemble)

`run_ensemble.py` now writes everything to files instead of only printing, saves
the trained ensemble, and produces figures during training:

```
results/
  metrics.csv                 # full metric set, one row per member + ENSEMBLE
  nmse_by_maskcount.csv        # NMSE (dB) vs number of masked antennas, per model
  doa.csv                      # DoA angle error / hit-rate (recon vs true)
  ensemble_comparison.png      # NMSE bar chart
  nmse_by_maskcount.png         # error-vs-missing-data curve
  reconstruction_example.png   # ENSEMBLE true-vs-reconstructed for one sample
  mlp_member/epoch_001.png ...  # reconstruction per epoch while the MLP trains
checkpoints/ensemble.joblib    # the trained ensemble (members + meta-model)
```

Test an already-trained ensemble without retraining:

```bash
python test_ensemble.py --ensemble checkpoints/ensemble.joblib --data data/radar.csv
```

It loads the saved ensemble, rebuilds the held-out test split (same seed -> same
test set), and writes `results/test/metrics.csv`, a comparison chart, a
reconstruction figure, and `doa.csv`. Use the same CSV that was used for training
(the split and normalization are tied to it).

### Console vs files
Both scripts print only short progress lines and a one-line summary; the full
numbers live in the CSVs and the figures.
