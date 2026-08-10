#!/usr/bin/env python3
"""
预训练 CliTsDeepHit 的两个 LatentEncoder（序列特征 & 静态特征）
对齐标准高斯先验分布（AAE 结构）

训练完成后仅保存 encoder 权重（decoder / discriminator 预训练后丢弃）：
  checkpoints/pretrained/pretrained_seq_encoder.pt
  checkpoints/pretrained/pretrained_static_encoder.pt

主训练脚本加载方式：
  model.encoder.enc.load_state_dict(torch.load('.../pretrained_seq_encoder.pt'))
  model.static_encoder.enc.load_state_dict(torch.load('.../pretrained_static_encoder.pt'))
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.CliTsSurv.backbones.LatentEncoder import LatentEncoder


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class SeqEncoderDataset(Dataset):
    """
    将时序 DataFrame 展平为 per-timestep 样本 (N_total_steps, K)。
    和 CliTsRNN.forward() 里 x.view((B*T, K)) 的调用方式对齐。

    返回 (x, target, valid_mask)：
      - target: NaN 填 0 后的原始特征（重建目标）
      - valid_mask: True 表示该位置有效（非 NaN），用于 masked MSE
      - x: 在 target 基础上可额外随机 mask（数据增强，max_drop 控制比例）
    """

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
            # 随机额外 mask 做数据增强（参考 CliTs-rnn LatentEncoderDataset）
            p = torch.rand(1).item()
            while p < self.max_drop:
                p = torch.rand(1).item()
            aug_mask = torch.rand(self.feature_dim) < (1 - p)
            drop_mask = nan_mask | aug_mask
            x = feature.masked_fill(drop_mask, 0.0)
        else:
            x = target  # 只填 NaN，不做额外 mask

        return x, target, valid_mask


class StaticEncoderDataset(Dataset):
    """
    静态特征：每个 unique patient 一条记录（drop_duplicates by 'id'）。
    静态特征不做随机 mask 增强。
    """

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
        return target, target, valid_mask  # x == target，不做额外 mask


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
# 数据加载 & 划分
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
    """按患者 ID 划分，比例 0.70 / 0.15 / 0.15，和其他训练脚本对齐。"""
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


# NOTE: LatentEncoder 预训练使用全量数据，不划分 train/val/test。
# train_model() 内部需要 test_dataloader 来选 best checkpoint，
# 因此将同一份全量 dataset 同时传给 train 和 val——两者 loss 相同，
# best.pt 机制照常工作，只是不再有独立 val 集的泛化评估。


# ---------------------------------------------------------------------------
# 预训练通用函数（复用 LatentEncoder.train_model 的 AAE 训练循环）
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
    """
    复用 LatentEncoder.train_model() 的完整 AAE 训练循环：
      step 1 — Discriminator 更新（区分 N(0,I) 与 encoder 输出）
      step 2 — Encoder+Decoder 更新（重建 loss + 对抗 loss，欺骗 D）

    """
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
    # train_model 内部：D 每 step 更新 1 次，enc+dec 每 step 更新 1 次
    # 完成后自动 load best.pt（val loss 最优的权重）
    encoder.train_model(train_loader, val_loader, epochs=epochs, lr=lr, save_dir=str(run_dir))

    enc_path = save_dir / f'pretrained_{name}.pt'
    torch.save(encoder.state_dict(), enc_path)
    print(f'[✓] encoder 权重已保存 → {enc_path}')
    return enc_path


# ---------------------------------------------------------------------------
# 验证 latent 分布对齐质量
# ---------------------------------------------------------------------------

def check_latent_alignment(encoder: LatentEncoder, val_ds: Dataset, device: str, name: str):
    """
    验证 val 集 latent 的 mean 和 std 是否接近 N(0, 1)。
    目标：mean ≈ 0，std ≈ 1。
    """
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
    # per-dim 统计（方便排查哪个维度没对齐）
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
    # ------------------------------------------------------------------
    # 配置（维度要和 train_clits_deephit_example.py 的 create_model() 对齐）
    # ------------------------------------------------------------------
    DATA_DIR = Path('/Volumes/Workspace/code/causalTS/data/CliTsDeepHit_data')
    SAVE_DIR = Path(__file__).parent.parent / 'checkpoints' / 'pretrained'
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 序列 encoder：per-timestep 样本量大，batch 开大一点
    SEQ_BATCH_SIZE = 256
    # 静态 encoder：每人一条，样本少，batch 小一点
    STATIC_BATCH_SIZE = 64

    EPOCHS = 300
    LR = 1e-4
    MAX_DROP = 0.3  # 序列 encoder 随机 mask 增强比例上限

    # ------------------------------------------------------------------
    # 加载数据
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
    # 构建 Datasets（全量数据训练，不划分 train/val/test）
    # ------------------------------------------------------------------
    print('\n[2/4] 构建 Datasets（全量数据）...')

    # 序列：每行是一个时间步（展平）
    seq_train_ds = SeqEncoderDataset(df, seq_cols, max_drop=MAX_DROP)
    seq_val_ds = SeqEncoderDataset(df, seq_cols, max_drop=0.0)  # 同数据传给 val，保留 best.pt 机制
    print(f'  SeqEncoder    — {len(seq_train_ds):,} timesteps (全量)')

    # 静态：每人一条
    static_train_ds = StaticEncoderDataset(df, static_cols)
    static_val_ds = StaticEncoderDataset(df, static_cols)
    print(f'  StaticEncoder — {len(static_train_ds):,} patients (全量)')

    # ------------------------------------------------------------------
    # 创建 LatentEncoder
    # enc_dims / dec_dims 和 create_model() 里的 encoder_args 保持一致
    # disc_dims 仅预训练阶段使用
    # ------------------------------------------------------------------
    print('\n[3/4] 初始化 LatentEncoder...')

    seq_encoder = LatentEncoder(
        conti_feature_in=n_seq,
        bi_feature_in=0,
        enc_dims=[64, 32],       # neck=32，和 create_model() 一致
        dec_dims=[32, 64],
        disc_dims=[32, 64, 32],  # 开 discriminator → AAE 对齐高斯
        beta=0.5,                # 初期让 reconstruction 主导，避免 D 过强
    )

    static_encoder = LatentEncoder(
        conti_feature_in=n_static,
        bi_feature_in=0,
        enc_dims=[32, 16],       # neck=16，和 create_model() 一致
        dec_dims=[16, 32],
        disc_dims=[16, 32, 16],
        beta=0.5,
    )

    # ------------------------------------------------------------------
    # 预训练
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
    # 验证对齐效果
    # ------------------------------------------------------------------
    print('\n[验证] 检查 latent 分布对齐情况（目标：mean≈0，std≈1）...')
    check_latent_alignment(seq_encoder, seq_val_ds, DEVICE, 'seq_encoder')
    check_latent_alignment(static_encoder, static_val_ds, DEVICE, 'static_encoder')

    # ------------------------------------------------------------------
    # 完成
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


if __name__ == '__main__':
    main()
