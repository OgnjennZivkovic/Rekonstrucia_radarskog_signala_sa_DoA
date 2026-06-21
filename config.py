#Konfiguracija hiperparametara, modela

from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # Parametri dataseta
    M: int = 29                 # Broj antena
    P: int = 256                # broj pulseva po scenariju
    n_scenarios: int = 400      # broj scenarija koji se obradjuje (koristi se samo kada je pravljen vestacki dataset za proveru rada koda)
    min_targets: int = 1        # K broj meta 1 (min broj)
    max_targets: int = 5        # K broj meta 5 (max broj)
    angle_min: float = -60.0    # Maks i min vrednost opsega uglova
    angle_max: float = 60.0
    snr_db: float = 20.0        # snr 20 db
    d_over_lambda: float = 0.5  # uniformno rastojanje izmedju antena (talasna duzina)

    # Maskiranje
    mask_min: int = 1           # koliko se antena maskira min-max
    mask_max: int = 10         

    # Agumentacije
    per_pulse_power_norm: bool = False  # snaga po pulsevima se izjedncuje
    aug_phase_rot: bool = False         # fazno rotiranje antena
    aug_mirror: bool = False            # obrtanje redosleda antena 1-29 => 29-1

    # JMLP
    hidden: int = 256
    n_blocks: int = 4
    dropout: float = 0.0

    # Trening svih modela
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    obs_loss_weight: float = 0.1   # koli da obraca paznju na loss kod stvarnih vrednosti , on se koristi kako ne bi dobili neke vrednosti koje bas odskacu od skupa
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 0 # koristi se pri generisanju maski

    # Ansamble
    members: tuple = ("linear", "ridge", "huber", "rf","knn", "xgb", "mlp")
    meta_alpha: float = 0.01     # koliko paznje meta model pridaje losijim rezultatima
    ridge_alpha: float = 1.0
    huber_epsilon: float = 1.35
    rf_estimators: int = 100
    rf_max_depth: int = 16
    xgb_estimators: int = 200  #xgb_max_depth  xgb_estimators  val_NMSE_dB  val_ExplVar  fit_s
    xgb_max_depth: int = 6      #           8             400       -1.164        0.235  749.9
    knn_neighbors: int = 10
    tree_sample_cap: int = 60000   # ogranicenje na koliko se redova vrsi trening za robusne modele (rf/xgb/huber/knn)
    mlp_member_epochs: int = 100    # broj epoha za MLP
    doa_eval_scenarios: int = 60   # Koliko se test scenarija koristi da se na njima odradi MUSIC

    # Putanje
    data_csv: str = "complex.csv"
    ckpt_path: str = "result_e/checkpoints/model.pt"
    ensemble_path: str = "result_e/checkpoints/ensemble.joblib"
    results_dir: str = "result_e"

    def to_dict(self):
        return asdict(self)


def default_config() -> Config:
    return Config()
