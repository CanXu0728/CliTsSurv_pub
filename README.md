# CliTsSurv

Core PyTorch implementation of CliTsSurv for survival prediction from irregular longitudinal measurements and static covariates. Includes preprocessing, adversarial autoencoder pretraining, Transformer/LSTM/GRU backbones, training and evaluation.

## Installation

Use an isolated Python 3.12 environment. The pipeline was smoke-tested with Python 3.12.14, PyTorch 2.7.1 (CUDA 12.8) and NumPy 1.26.4. A CUDA-capable GPU is recommended.

From the repository root:

```bash
python -m pip install -r requirements.txt
```

For the CUDA 12.8 build used in the smoke test, install PyTorch first:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

## Data preparation

Place these input files in `data/`:

- `seq.csv`: longitudinal records with patient identifier `id`, elapsed time `date` in months, and numerical measurement columns.
- `static.csv`: patient records with `id`, `survival_months`, `survival_status` (1 for death, 0 for censoring), and covariates `bloodtype_A`, `bloodtype_B`, `RH`, `VTE`, `PE`, `CE` and `DM`.

Identifiers must match across files. Follow the schema in `data/data_preprocess_clitssurv.py`; this is not a general-purpose importer for arbitrary clinical datasets. Local patient data are excluded by `.gitignore` and must not be redistributed without authorization.

```bash
python data/data_preprocess_clitssurv.py
```

Preprocessing cleans and merges records, adds inter-visit intervals and comorbidity counts, normalizes features and creates 50 time bins. It writes merged CSV/Parquet data, metadata and time-bin boundaries to `data/CliTsSurv_data/`. Normalization statistics and bin boundaries are computed at this stage, before the trainer's patient-level split.

## Training

The main entry point is `scripts/train_clitssurv.py`. It defaults to **20 runs**, with seeds 1000–1019 and patient-level 70/15/15 training/validation/test partitions.

For one run in Windows PowerShell:

```powershell
$env:CLITS_N_RUNS = "1"
$env:CLITS_SEED_START = "1000"
$env:CLITS_BACKBONE = "transformer"
python scripts/train_clitssurv.py
```

On a POSIX shell:

```bash
CLITS_N_RUNS=1 CLITS_SEED_START=1000 CLITS_BACKBONE=transformer python scripts/train_clitssurv.py
```

The trainer automatically pretrains both encoders using the outer training partition and caches the weights. Pretraining uses 200 epochs; survival training uses up to 40 epochs with early stopping. One run includes both stages and is not a quick unit test.

### Configuration

Set environment variables before starting Python:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CLITS_N_RUNS` | `20` | Number of seeds |
| `CLITS_SEED_START` | `1000` | First seed |
| `CLITS_BACKBONE` | `transformer` | `transformer`, `lstm` or `gru` |
| `CLITS_TIME_ENCODER` | `multiscale_sincos` | Also supports `sincos`, `time2vec` and `ctlpe` |
| `CLITS_TRAINING_SCHEME` | `balanced` | Also supports `natural_balanced_ranking` |
| `CLITS_GAMMA` | `0.5` | Brier-loss weight |
| `CLITS_SAVE_DIR` | Backbone-specific path under `checkpoints/` | Model output directory |
| `CLITS_CSV_PATH` | Path under `results/clitssurv/` | Per-seed result CSV |

The `natural_balanced_ranking` scheme uses natural sampling for the main stream and a balanced stream for ranking. Other hyperparameters are defined at the top of the main script. The default Transformer uses six-dimensional time encoding and feed-forward dimension 128.

Use the main entry point for a shared workflow across all three backbones. The separate LSTM/GRU sinusoidal scripts are retained alternative configurations; they do not inherit all main-trainer settings. Standalone `pretrain_latent_encoders.py` uses the full dataset, unlike the main trainer's split-specific pretraining.

## Evaluation and outputs

Checkpoint selection uses validation mean IPCW C-index, with IBS as a tie-breaker. Evaluation uses training-event Q25/Q50/Q75 horizons, conditional survival probabilities and patients whose last observation precedes each horizon. Metrics include IPCW C-index, Harrell's C-index, dynamic AUC, Brier score and IBS.

- `checkpoints/pretrained_train_only/seed_<seed>/`: cached pretrained encoders.
- `checkpoints/clitssurv*/seed_<seed>/`: best weights and training/selection histories.
- `results/clitssurv/`: per-seed evaluation results by default.

Completed seeds may be skipped or reused from existing checkpoints/results. Use distinct output paths for independent experiments and check that cached pretraining weights match the intended settings. Fixed seeds improve repeatability, but hardware and software differences can still affect results. A single-run smoke test establishes executability, not reproduction of a multi-seed paper result.

## Source layout

```text
data/                     Preprocessing and cleaning utilities
models/CliTsSurv/         Model, losses, backbones, datasets and augmenters
scripts/                  Pretraining, training and evaluation utilities
requirements.txt          Tested dependency versions
```

See [Architecture](models/CliTsSurv/ARCHITECTURE.md) and [LICENSE](LICENSE). This is research software, not a clinical decision-making system.
