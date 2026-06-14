"""Central configuration for the radar antenna-reconstruction project.

Every script imports `Config` from here, so changing a value in one place
propagates everywhere. The defaults are sized for a quick end-to-end run;
the comments show the full-dataset values from the dataset description.
"""

from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # ---- array / dataset geometry ----
    M: int = 29                 # number of antennas
    P: int = 256                # pulses (snapshots) per scenario -- matches the real dataset
    n_scenarios: int = 400      # only used by generate_data.py (real dataset has 2000)
    min_targets: int = 1        # K varies between 1 ...
    max_targets: int = 5        # ... and 5 targets per scenario
    angle_min: float = -60.0    # angle of arrival range (degrees)
    angle_max: float = 60.0
    snr_db: float = 20.0        # signal-to-noise ratio
    d_over_lambda: float = 0.5  # element spacing in wavelengths (half-wavelength ULA)

    # ---- masking (the reconstruction task) ----
    mask_min: int = 1           # fewest antennas hidden per sample
    mask_max: int = 10          # most antennas hidden per sample ("varying levels of masking")

    # ---- preprocessing / augmentation toggles ----
    per_pulse_power_norm: bool = False  # scale every pulse to unit average power (all splits)
    aug_phase_rot: bool = False         # train-only: random global phase rotation e^{j phi}
    aug_mirror: bool = False            # train-only: random antenna-order flip + conjugate

    # ---- model ----
    hidden: int = 256
    n_blocks: int = 3
    dropout: float = 0.1

    # ---- training ----
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    obs_loss_weight: float = 0.1   # how much to also fit the *observed* antennas
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 0

    # ---- ensemble (option 2: stacking) ----
    members: tuple = ("linear", "ridge", "huber", "rf", "xgb", "mlp")
    meta_alpha: float = 0.01     # Ridge alpha for the meta-model (low: let it favor strong members)
    ridge_alpha: float = 1.0
    huber_epsilon: float = 1.35
    rf_estimators: int = 100
    rf_max_depth: int = 16
    xgb_estimators: int = 200
    xgb_max_depth: int = 6
    knn_neighbors: int = 10
    tree_sample_cap: int = 60000   # cap train rows for heavy members (rf/xgb/huber/knn)
    mlp_member_epochs: int = 100    # epochs for the MLP when used as an ensemble member
    doa_eval_scenarios: int = 60   # how many test scenarios to run MUSIC on (cost control)

    # ---- paths ----
    data_csv: str = "complex.csv"
    ckpt_path: str = "result_e/checkpoints/model.pt"
    ensemble_path: str = "result_e/checkpoints/ensemble.joblib"
    results_dir: str = "result_e"

    def to_dict(self):
        return asdict(self)


def default_config() -> Config:
    return Config()
