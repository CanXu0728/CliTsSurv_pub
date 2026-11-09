#!/usr/bin/env python3
"""Pretrain longitudinal and static adversarial autoencoders.

The main trainer calls ensure_seed_pretrained_encoders and caches weights
under checkpoints/pretrained_train_only/seed_<seed>. This path uses only
patients in the corresponding outer training partition. The standalone
entry point below instead uses the full dataset."""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.CliTsSurv.backbones.LatentEncoder import LatentEncoder
from scripts.utils import set_global_seed


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SeqEncoderDataset(Dataset):
    """Flatten longitudinal records into per-timestep encoder samples.

Return (x, target, valid_mask). Missing target values are replaced with zero;
valid_mask identifies observed entries for masked reconstruction. Optional
random masking affects x only and is controlled by max_drop."""

    def __init__(self, df: pd.DataFrame, feature_cols: list, max_drop: float = 0.0):
        features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.features = features
        self.feature_dim = features.shape[1]
        self.max_drop = max_drop

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        nan_mask = torch.isnan(feature)
        target = feature.masked_fill(nan_mask, 0.0)
        valid_mask = ~nan_mask  # True = valid position (used in masked loss)

        if self.max_drop > 0 and (nan_mask.sum().item() / self.feature_dim) <= self.max_drop:
            # Apply additional random masking for data augmentation.
            p = torch.rand(1).item()
            while p < self.max_drop:
                p = torch.rand(1).item()
            aug_mask = torch.rand(self.feature_dim) < (1 - p)
            drop_mask = nan_mask | aug_mask
            x = feature.masked_fill(drop_mask, 0.0)
        else:
            x = target  # Replace NaNs only; do not apply additional masking.

        return x, target, valid_mask


class StaticEncoderDataset(Dataset):
    """Use one static record per patient, without additional random masking."""

    def __init__(self, df: pd.DataFrame, feature_cols: list):
        unique_df = df.drop_duplicates(subset='id')[feature_cols]
        features = torch.tensor(unique_df.values, dtype=torch.float32)
        self.features = features
        self.feature_dim = features.shape[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        nan_mask = torch.isnan(feature)
        target = feature.masked_fill(nan_mask, 0.0)
        valid_mask = ~nan_mask
        return target, target, valid_mask  # x equals target; no additional masking.


# ---------------------------------------------------------------------------
# collate_fn
# ---------------------------------------------------------------------------

def collate_fn_enc(batch, device):
    xs, ys, masks = zip(*batch)
    return (
        torch.stack(xs).to(device),
        torch.stack(ys).to(device),
        torch.stack(masks).to(device),
    )


# ---------------------------------------------------------------------------
# Data loading and splitting.
# ---------------------------------------------------------------------------

def load_data(data_dir: Path):
    with open(data_dir / 'metadata.json', 'r') as f:
        meta = json.load(f)

    csv_path = data_dir / 'merged.csv'
    parquet_path = data_dir / 'merged.parquet'
    if csv_path.exists():
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print('  使用 CSV 格式加载')
    else:
        df = pd.read_parquet(parquet_path)
        print('  使用 Parquet 格式加载')

    return df, meta


def split_by_patient(df: pd.DataFrame, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
    """Split patient IDs into training, validation and test partitions (70/15/15)."""
    np.random.seed(seed)
    uids = df['id'].unique()
    np.random.shuffle(uids)

    n_test = int(len(uids) * test_ratio)
    n_val = int(len(uids) * val_ratio)

    test_ids = uids[:n_test]
    val_ids = uids[n_test:n_test + n_val]
    train_ids = uids[n_test + n_val:]

    train_df = df[df['id'].isin(train_ids)].copy()
    val_df = df[df['id'].isin(val_ids)].copy()
    test_df = df[df['id'].isin(test_ids)].copy()

    print(f'  训练集: {len(train_ids)} 患者 ({len(train_df)} 条记录)')
    print(f'  验证集: {len(val_ids)} 患者 ({len(val_df)} 条记录)')
    print(f'  测试集: {len(test_ids)} 患者 ({len(test_df)} 条记录)  [预训练阶段不使用]')

    return train_df, val_df, test_df


# The reusable final-run helper below performs an internal patient-level split
# within the outer training partition. The legacy standalone entry point at the
# end of this file is retained only for manual development runs.


# ---------------------------------------------------------------------------
# Pretrain using the AAE training loop in LatentEncoder.train_model.
# ---------------------------------------------------------------------------

def pretrain_encoder(
    encoder: LatentEncoder,
    train_ds: Dataset,
    val_ds: Dataset,
    batch_size: int,
    epochs: int,
    lr: float,
    save_dir: Path,
    device: str,
    name: str,
) -> Path:
    """Pretrain the adversarial autoencoder.

Each step updates the discriminator to separate Gaussian prior samples from
encoded samples, then updates the encoder/decoder with reconstruction and
adversarial losses."""
    col_fn = lambda b: collate_fn_enc(b, device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=col_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=col_fn)

    run_dir = save_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"=" * 60}')
    print(f'预训练 {name}')
    print(f'  train samples: {len(train_ds)}  |  val samples: {len(val_ds)}')
    print(f'  epochs={epochs}, lr={lr}, batch_size={batch_size}')
    print(f'{"=" * 60}')

    encoder.to(device)
    # Each step updates the discriminator once and the encoder/decoder once.
    # Load best.pt automatically after training, based on validation loss.
    encoder.train_model(train_loader, val_loader, epochs=epochs, lr=lr, save_dir=str(run_dir))

    enc_path = save_dir / f'pretrained_{name}.pt'
    torch.save(encoder.state_dict(), enc_path)
    print(f'[saved] encoder checkpoint: {enc_path}')
    return enc_path


def ensure_seed_pretrained_encoders(
    train_df: pd.DataFrame,
    meta: dict,
    seed: int,
    save_root: Path,
    device: str,
    epochs: int = 200,
    lr: float = 1e-4,
) -> tuple[Path, Path]:
    """Pretrain both ARAEs using only the outer training split.

    A deterministic 85/15 patient-level split of ``train_df`` is used for
    pretraining and checkpoint selection.  Files are cached by outer-split
    seed so the full model and all paired sensitivity variants start from the
    exact same pretrained weights.
    """
    seed_dir = Path(save_root) / f"seed_{seed}"
    seq_path = seed_dir / "pretrained_seq_encoder.pt"
    static_path = seed_dir / "pretrained_static_encoder.pt"
    if seq_path.exists() and static_path.exists():
        print(f"  Reusing train-only pretrained encoders: {seed_dir}")
        return seq_path, static_path

    set_global_seed(seed)
    patient_ids = np.asarray(train_df["id"].unique())
    rng = np.random.RandomState(seed + 7919)
    rng.shuffle(patient_ids)
    n_val = max(1, int(len(patient_ids) * 0.15))
    inner_val_ids = set(patient_ids[:n_val])
    pretrain_df = train_df[~train_df["id"].isin(inner_val_ids)].copy()
    pretrain_val_df = train_df[train_df["id"].isin(inner_val_ids)].copy()

    seq_cols = meta["seq_feature_cols"]
    static_cols = meta["static_feature_cols"]
    seq_train_ds = SeqEncoderDataset(pretrain_df, seq_cols, max_drop=0.3)
    seq_val_ds = SeqEncoderDataset(pretrain_val_df, seq_cols, max_drop=0.0)
    static_train_ds = StaticEncoderDataset(pretrain_df, static_cols)
    static_val_ds = StaticEncoderDataset(pretrain_val_df, static_cols)

    seq_encoder = LatentEncoder(
        conti_feature_in=meta["n_seq_features"], bi_feature_in=0,
        enc_dims=[64, 32], dec_dims=[32, 64], disc_dims=[32, 64, 32],
        beta=0.5,
    )
    static_encoder = LatentEncoder(
        conti_feature_in=meta["n_static_features"], bi_feature_in=0,
        enc_dims=[32, 16], dec_dims=[16, 32], disc_dims=[16, 32, 16],
        beta=0.5,
    )
    seed_dir.mkdir(parents=True, exist_ok=True)
    if not seq_path.exists():
        seq_path = pretrain_encoder(
            seq_encoder, seq_train_ds, seq_val_ds, 256, epochs, lr, seed_dir,
            device, "seq_encoder",
        )
    else:
        print(f"  Reusing train-only sequence encoder: {seq_path}")
    if not static_path.exists():
        static_path = pretrain_encoder(
            static_encoder, static_train_ds, static_val_ds, 64, epochs, lr,
            seed_dir, device, "static_encoder",
        )
    else:
        print(f"  Reusing train-only static encoder: {static_path}")
    return seq_path, static_path


# ---------------------------------------------------------------------------
# Check alignment of the latent distribution.
# ---------------------------------------------------------------------------

def check_latent_alignment(encoder: LatentEncoder, val_ds: Dataset, device: str, name: str):
    """Check whether validation latents have mean near zero and standard deviation near one."""
    col_fn = lambda b: collate_fn_enc(b, device)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, collate_fn=col_fn)

    encoder.eval()
    all_z = []
    with torch.no_grad():
        for x, _, _ in val_loader:
            z = encoder.enc(x)
            all_z.append(z.cpu())
    all_z = torch.cat(all_z, dim=0)

    mean_val = all_z.mean().item()
    std_val = all_z.std().item()
    # Report per-dimension statistics to diagnose alignment.
    dim_means = all_z.mean(dim=0)
    dim_stds = all_z.std(dim=0)
    worst_mean_dim = dim_means.abs().argmax().item()
    worst_std_dim = (dim_stds - 1).abs().argmax().item()

    print(f'  [{name}]')
    print(f'    global mean={mean_val:.4f} (目标≈0)  |  global std={std_val:.4f} (目标≈1)')
    print(f'    worst mean dim={worst_mean_dim}: {dim_means[worst_mean_dim]:.4f}')
    print(f'    worst std  dim={worst_std_dim}: {dim_stds[worst_std_dim]:.4f}')


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    set_global_seed(1000)
    # ------------------------------------------------------------------
    # Keep encoder dimensions consistent with create_model() in the training script.
    # ------------------------------------------------------------------
    DATA_DIR = Path('/Volumes/Workspace/code/causalTS/data/CliTsDeepHit_data')
    SAVE_DIR = Path(__file__).parent.parent / 'checkpoints' / 'pretrained'
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Use a larger batch for the per-timestep sequence encoder.
    SEQ_BATCH_SIZE = 256
    # Use a smaller batch for the one-record-per-patient static encoder.
    STATIC_BATCH_SIZE = 64

    EPOCHS = 300
    LR = 1e-4
    MAX_DROP = 0.3  # Maximum random-mask fraction for the sequence encoder.

    # ------------------------------------------------------------------
    # Load data.
    # ------------------------------------------------------------------
    print('[1/4] 加载数据...')
    df, meta = load_data(DATA_DIR)

    seq_cols = meta['seq_feature_cols']
    static_cols = meta['static_feature_cols']
    n_seq = meta['n_seq_features']
    n_static = meta['n_static_features']

    print(f'  时序特征维度: {n_seq}  |  静态特征维度: {n_static}')
    print(f'  设备: {DEVICE}')

    # ------------------------------------------------------------------
    # Construct datasets from the full data without a train/validation/test split.
    # ------------------------------------------------------------------
    print('\n[2/4] 构建 Datasets（全量数据）...')

    # Flatten longitudinal records into individual timesteps.
    seq_train_ds = SeqEncoderDataset(df, seq_cols, max_drop=MAX_DROP)
    seq_val_ds = SeqEncoderDataset(df, seq_cols, max_drop=0.0)  # Reuse the same data for validation to retain the best.pt mechanism.
    print(f'  SeqEncoder    — {len(seq_train_ds):,} timesteps (全量)')

    # Use one static record per patient.
    static_train_ds = StaticEncoderDataset(df, static_cols)
    static_val_ds = StaticEncoderDataset(df, static_cols)
    print(f'  StaticEncoder — {len(static_train_ds):,} patients (全量)')

    # ------------------------------------------------------------------
    # Create LatentEncoder instances.
    # Keep enc_dims and dec_dims consistent with encoder_args in create_model().
    # Discriminator dimensions used during pretraining.
    # ------------------------------------------------------------------
    print('\n[3/4] 初始化 LatentEncoder...')

    seq_encoder = LatentEncoder(
        conti_feature_in=n_seq,
        bi_feature_in=0,
        enc_dims=[64, 32],       # Latent dimension 32, matching create_model().
        dec_dims=[32, 64],
        disc_dims=[32, 64, 32],  # Enable the discriminator for Gaussian latent alignment.
        beta=0.5,                # Initially emphasize reconstruction to avoid an overly strong discriminator.
    )

    static_encoder = LatentEncoder(
        conti_feature_in=n_static,
        bi_feature_in=0,
        enc_dims=[32, 16],       # Latent dimension 16, matching create_model().
        dec_dims=[16, 32],
        disc_dims=[16, 32, 16],
        beta=0.5,
    )

    # ------------------------------------------------------------------
    # Pretrain encoders.
    # ------------------------------------------------------------------
    print('\n[4/4] 开始预训练...')

    seq_path = pretrain_encoder(
        encoder=seq_encoder,
        train_ds=seq_train_ds,
        val_ds=seq_val_ds,
        batch_size=SEQ_BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        save_dir=SAVE_DIR,
        device=DEVICE,
        name='seq_encoder',
    )

    static_path = pretrain_encoder(
        encoder=static_encoder,
        train_ds=static_train_ds,
        val_ds=static_val_ds,
        batch_size=STATIC_BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        save_dir=SAVE_DIR,
        device=DEVICE,
        name='static_encoder',
    )

    # ------------------------------------------------------------------
    # Check latent alignment.
    # ------------------------------------------------------------------
    print('\n[验证] 检查 latent 分布对齐情况（目标：mean≈0，std≈1）...')
    check_latent_alignment(seq_encoder, seq_val_ds, DEVICE, 'seq_encoder')
    check_latent_alignment(static_encoder, static_val_ds, DEVICE, 'static_encoder')

    # ------------------------------------------------------------------
    # Finished.
    # ------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('预训练完成！')
    print(f'  seq encoder    → {seq_path}')
    print(f'  static encoder → {static_path}')
    print('=' * 60)
    print('\n在主训练脚本 create_model() 完成后加载：')
    print("  model.encoder.enc.load_state_dict(")
    print(f"      torch.load('{seq_path}', map_location=device))")
    print("  model.static_encoder.enc.load_state_dict(")
    print(f"      torch.load('{static_path}', map_location=device))")


def standalone_train_only_main():
    """Standalone reproducible pretraining for seed 1000.

    The outer validation and test patients are deliberately discarded before
    the helper performs its internal split of the outer training patients.
    """
    seed = 1000
    set_global_seed(seed)
    data_dir = Path(__file__).resolve().parents[1] / 'data' / 'CliTsSurv_data'
    save_root = Path(__file__).resolve().parents[1] / 'checkpoints' / 'pretrained_train_only'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df, meta = load_data(data_dir)
    outer_train, _, _ = split_by_patient(df, seed=seed)
    seq_path, static_path = ensure_seed_pretrained_encoders(
        outer_train, meta, seed, save_root, device, epochs=200, lr=1e-4
    )
    print(f'seq encoder: {seq_path}')
    print(f'static encoder: {static_path}')


if __name__ == '__main__':
    standalone_train_only_main()
