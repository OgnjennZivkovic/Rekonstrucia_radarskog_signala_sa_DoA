# Rekonstrukcija signala sa maskiranih antena radarskog niza

Projekat iz predmeta *Softverski algoritmi u sistemima automatskog upravljanja*.
Cilj je rekonstrukcija radarskog signala kada deo antena „otkaže" — maskirane
antene se postave na nulu, a model treba da im obnovi kompleksne (I/Q) vrednosti.
Problem je rešen kao regresija po pulsu, korišćenjem **stacking ansambla**, uz
domensku proveru kvaliteta preko procene dolaznih uglova (DoA, MUSIC).

## Skup podataka

CSV sa 2000 scenarija × 256 pulseva = **512.000 redova**, 30 kolona:

- `angle` — dolazni uglovi meta u sceni (1–5 meta po scenariju), u stepenima.
- `ant_1 … ant_29` — kompleksna I/Q merenja svake od 29 antena, kao string `(re+imj)`.

Niz je uniformni linearni (ULA), razmak pola talasne dužine, uglovi u opsegu
[−60°, +60°]. Očekivana putanja fajla: `data/radar.csv`.

## Struktura projekta

| Fajl | Uloga |
|------|-------|
| `config.py` | Sva podešavanja (broj antena M=29, pulseva P=256, lista članova, putanje, hiperparametri). |
| `data.py` | Učitavanje i parsiranje CSV-a, keširanje. |
| `features.py` | Razdvajanje na re/im, standardizacija (statistike samo iz traina), maskiranje, podela 80/10/10 po scenariju. |
| `model.py` | `ReconMLP` — rezidualni MLP za rekonstrukciju. |
| `members.py` | Članovi ansambla: `SklearnMember` (linear, ridge, huber, rf, knn, xgb) i `MLPMember`. |
| `ensemble.py` | `Stacker` — stacking; meta-model (Ridge) uči samo na maskiranim antenama validacionog skupa. |
| `metrics.py` | Rekonstrukcione metrike + MUSIC i DoA metrika. |
| `viz.py` | Svi grafici (poređenje modela, primer rekonstrukcije, NMSE po maskiranju, DoA spektar i scatter). |
| `run_ensemble.py` | **Trening** — obučava članove i meta-model, snima `ensemble.joblib`. |
| `test_ensemble.py` | **Evaluacija** — metrike, grafici i DoA na test skupu. |
| `tune.py` | Pretraga hiperparametara za pojedinačne članove i meta-model. |
| `profile_dataset.py` | Profilisanje skupa (raspodela K, statistike, procena SNR-a). |
| `app.py` | Veb aplikacija (Flask) za demonstraciju modela. |
| `requirements.txt` | Zavisnosti. |

## Instalacija

```bash
pip install -r requirements.txt
```

Potreban je Python 3.10+. Glavne zavisnosti: numpy, pandas, scikit-learn,
xgboost, torch, matplotlib, joblib, flask.

## Pokretanje

Redosled je: **trening → evaluacija → (opciono) aplikacija**.

```bash
# 1) Trening ansambla (snima model u result_e/checkpoints/ensemble.joblib)
python run_ensemble.py --data data/radar.csv

# 2) Evaluacija na test skupu (metrike, grafici, DoA)
python test_ensemble.py --ensemble result_e/checkpoints/ensemble.joblib --data data/radar.csv

# 3) Korisnička aplikacija (otvara http://127.0.0.1:5000)
python app.py --ensemble result_e/checkpoints/ensemble.joblib
```

### Podešavanje hiperparametara

```bash
python tune.py --member mlp  --data data/radar.csv   # mlp / ridge / rf / xgb / knn
python tune.py --meta        --data data/radar.csv   # meta-model (Ridge alpha)
```

Rezultati pretrage se snimaju kao `tune_<član>.csv` (rangirano po validacionom NMSE-u).

### Profilisanje skupa

```bash
python profile_dataset.py --data data/radar.csv
```

## Izlazi

Sve se piše u `result_e/`:

- `metrics.csv` — rekonstrukcione metrike po modelu (NMSE, ExplVar, MAE, fazna greška, korelacija, percentilne greške).
- `nmse_by_maskcount.csv` — greška u zavisnosti od broja maskiranih antena.
- `doa.csv`, `doa_by_k.csv` — DoA rezultati (rekonstruisani vs pravi niz, ukupno i po broju meta).
- `ensemble_comparison.png`, `reconstruction_example.png`, `nmse_by_maskcount.png`, `doa_spectrum.png`, `doa_scatter.png` — grafici.
- `checkpoints/ensemble.joblib` — istrenirani model.

## Modeli

Na prvom nivou se porede klase algoritama: linearna i Ridge regresija, Huber,
RandomForest, KNN, XGBoost i MLP. Na drugom nivou meta-model (Ridge) ih
kombinuje, učeći isključivo na maskiranim antenama da bi favorizovao najjačeg
člana. Lista aktivnih članova se podešava u `config.py` (`members`).

## Rezultati (ukratko)

- MLP ubedljivo najbolji (~−6.6 dB NMSE, ~78% objašnjene varijanse); ansambl ga prati.
- Među klasičnim modelima KNN je najbolji (~−1.5 dB), ostali su slabi (−0.5 do −0.8 dB), Huber praktično beskoristan.
- DoA: rekonstruisani niz daje praktično istu tačnost kao pravi niz (~21° vs ~21°), tj. rekonstrukcija čuva ugaonu informaciju.
- Analiza pokazuje da podaci nisu čisto rang-K i da je efektivni SNR niži od deklarisanog, što ograničava dostižni kvalitet.

## Opciono / eksperimentalno: MMSE

`mmse.py` i `eval_mmse.py` sadrže statistički, model-based rekonstruktor (MMSE
po scenariju, koristi kovarijansu svih 256 pulseva). Nije deo glavnog pipeline-a
ni dokumentacije; pokreće se zasebno radi poređenja:

```bash
python eval_mmse.py --ensemble result_e/checkpoints/ensemble.joblib --data data/radar.csv
```

## Napomena

Trening MLP-a koristi nasumičnost, pa se tačni brojevi mogu neznatno razlikovati
između pokretanja; seme je fiksirano u `config.py`.