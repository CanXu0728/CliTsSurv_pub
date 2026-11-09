# CliTsSurv architecture

Configuration values below refer to `scripts/train_clitssurv.py`. Constructor defaults and standalone recurrent scripts may differ.

## Inputs

`CliTsSurvDataset` groups records by patient and associates each sequence with static covariates and survival labels. `collate_fn` pads variable-length sequences and provides lengths and reconstruction masks. Raw elapsed times are encoded inside the model.

For batch size B, padded length T, longitudinal feature count F and static feature count S, principal input shapes are `(B, T, F)`, `(B, S)` and `(B, T, 1)` for longitudinal values, static values and elapsed times.

## Representations and backbones

Separate `LatentEncoder` instances encode longitudinal and static inputs. Each contains an encoder, decoder and optional discriminator for Gaussian prior alignment. These are adversarial autoencoders, not variational autoencoders. The main configuration uses longitudinal encoder widths `[64, 32]` and static encoder widths `[32, 16]`, with decoder widths `[32, 64]` and `[16, 32]`.

Time encoders include sinusoidal, multiscale sinusoidal, Time2Vec and CTLPE variants. The main configuration uses six-dimensional multiscale sinusoidal encoding, taking the first three wavelengths from `(171, 60, 36, 24, 12)` months. Time embeddings are concatenated with longitudinal latent features.

The sequence backbone supports Transformer, LSTM and GRU variants. Main-trainer settings use hidden size 64 and three layers. The Transformer uses four heads, feed-forward dimension 128 and dropout 0.1. Temporal attention aggregates sequence representations, which are combined with sequence and static features in the prediction head.

## Augmentation and losses

Latent-space augmentation includes Gaussian perturbation, interpolation, removal and slicing. It is applied during training, not evaluation.

The head produces a discrete absolute-time probability distribution. `metrices.py` implements conditional survival transformations and survival losses. The objective combines survival loss, longitudinal/static reconstruction terms and a Brier term. Main-trainer defaults are `alpha=0.3`, `sigma=0.1`, reconstruction weights of 1 and Brier weight 0.5. Refer to the loss functions for exact definitions.

The `balanced` sampling scheme balances the main stream. `natural_balanced_ranking` retains natural sampling for the main stream and uses a separate balanced ranking stream.

## Training and evaluation

The main trainer fixes each run's seed and partitions patient IDs into 70/15/15 training/validation/test splits. Automatic encoder pretraining uses the outer training partition, with internal validation. Preprocessing statistics and bin boundaries are generated separately by the preprocessing script.

Validation mean IPCW C-index selects the survival checkpoint, with IBS breaking ties. Evaluation uses training-event quantile horizons and conditional survival probabilities, restricting each horizon to patients whose observation window ends before it. Shared metrics are implemented in `scripts/utils.py`.

Weights and training/selection histories are saved under the configured checkpoint directory. Per-seed results are saved incrementally; completed runs can be reused. Use distinct paths when configurations must not share checkpoints or results.

## Main files

| File | Responsibility |
| --- | --- |
| `CliTsSurv.py` | Model composition, training and prediction |
| `metrices.py` | Survival losses and conditional distributions |
| `backbones/LatentEncoder.py` | Adversarial autoencoders |
| `backbones/CliTs_backbone.py` | Sequence backbones and temporal attention |
| `backbones/time_embedding.py` | Time encoders |
| `datasets/` | Patient datasets and batching helpers |
| `augmenters/seq_augmenters.py` | Latent sequence augmentation |
