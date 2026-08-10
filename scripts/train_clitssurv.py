#!/usr/bin/env python3
"""
CliTsSurv 训练脚本
基于 CliTsDeepHit 训练脚本修改，适配 CliTsSurv 模型
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler

# 导入 CliTsSurv 组件
import sys
sys.path.insert(0, '/Volumes/Workspace/code/causalTS')
sys.path.insert(0, str(Path(__file__).parent))
from utils import evaluate_survival_metrics

from models.CliTsSurv.CliTsSurv import CliTsSurv
from models.CliTsSurv.datasets import CliTsSurvDataset, collate_fn


# ------------------------------------------------------------------ #
#  Hyperparameters
# ------------------------------------------------------------------ #
# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 配置
DATA_DIR     = Path(__file__).parent.parent / 'data' / 'CliTsSurv_data'
BASE_SAVE_DIR = Path(__file__).parent.parent / 'checkpoints' / 'CliTsSurv' / 'learn_time_sincos'
BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
# 保存每轮结果到 CSV
RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'clitssurv'
CSV_PATH = RESULTS_DIR / 'repeated_training_results_sincos.csv'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Repeated training
N_RUNS     = 1
SEED_START = 1000
PRETRAINED_ENCODER = "./checkpoints/pretrained/pretrained_seq_encoder.pt"  # 可选预训练编码器路径
PRETRAINED_STATIC_ENCODER = "./checkpoints/pretrained/pretrained_static_encoder.pt"  # 可选预训练静态编码器路径

# Training
BATCH_SIZE = 32
EPOCHS     = 40
LR         = 0.0001

# Data split
TEST_RATIO  = 0.15
VAL_RATIO   = 0.15

# Time encoding
TIME_DIM          = 6
TIME_ENCODER_TYPE = 'sincos' # 'sincos' | 'multiscale_sincos' | 'time2vec' | 'ctlpe'

# Encoder (temporal VAE)
ENCODER_ARGS = {
    'enc_dims': [64, 32],
    'dec_dims': [32, 64],
    'disc_dims': [],
    'beta': 1.0
}

# Static encoder
STATIC_ENCODER_ARGS = {
    'enc_dims': [32, 16],
    'dec_dims': [16, 32],
    'disc_dims': [],
    'beta': 1.0
}

# RNN backbone
# Backbone: rnn_type 可选 'lstm' | 'gru' | 'transformer'
# nhead / dim_feedforward / dropout 仅 transformer 使用，rnn 时忽略
NET_ARGS = {
    'hidden_size': 64,
    'num_layers': 3,
    'rnn_type': 'transformer',
    'nhead': 4,
    'dim_feedforward': 256,
    'dropout': 0.1,
}

# Survival head
HEAD_ARGS = {
    'hidden_dims': [64, 32]
}

# Loss weights
BETA1 = 1
BETA2 = 1
ALPHA = 0.3
SIGMA = 0.1
GAMMA = 0.5  # Brier calibration loss weight

# Data augmentation
AUG_CONFIGS = [
    {'type': 'AddGaussianNoise', 'args': {'p': 0.3}},
    {'type': 'InnerInterpolate', 'args': {'p': 0.7}},
    {'type': 'RandomRemove',     'args': {'p': 0.5}},
    {'type': 'RandomSlice',      'args': {'p': 0.1}},
]


def load_data(data_dir: Path):
    """加载预处理后的数据"""
    print("[1/4] 加载数据...")
    
    # 加载元数据
    with open(data_dir / "metadata.json", 'r') as f:
        meta = json.load(f)
    
    # 加载合并后的数据 (支持 CSV 或 Parquet)
    csv_path = data_dir / "merged.csv"
    parquet_path = data_dir / "merged.parquet"
    if csv_path.exists():
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print("  使用 CSV 格式加载")
    else:
        df = pd.read_parquet(parquet_path)
        print("  使用 Parquet 格式加载")
    
    # 加载时间区间边界
    time_bin_edges = np.load(data_dir / "time_bin_edges.npy")
    
    print(f"  样本数: {meta['n_unique_ids']} 患者")
    print(f"  时序特征: {meta['n_seq_features']}")
    print(f"  静态特征: {meta['n_static_features']}")
    print(f"  时间区间: {len(time_bin_edges)-1} 个")
    
    return df, meta, time_bin_edges


def split_train_test(df: pd.DataFrame, test_ratio=0.15, val_ratio=0.15, random_seed=42):
    """划分训练/验证/测试集"""
    print("\n[2/4] 划分数据集...")
    
    np.random.seed(random_seed)
    
    # 按患者ID划分
    unique_ids = df['id'].unique()
    np.random.shuffle(unique_ids)
    
    n_test = int(len(unique_ids) * test_ratio)
    n_val = int(len(unique_ids) * val_ratio)
    
    test_ids = unique_ids[:n_test]
    val_ids = unique_ids[n_test:n_test + n_val]
    train_ids = unique_ids[n_test + n_val:]
    
    train_df = df[df['id'].isin(train_ids)].copy()
    val_df = df[df['id'].isin(val_ids)].copy()
    test_df = df[df['id'].isin(test_ids)].copy()
    
    print(f"  训练集: {len(train_ids)} 患者 ({len(train_df)} 条记录)")
    print(f"  验证集: {len(val_ids)} 患者 ({len(val_df)} 条记录)")
    print(f"  测试集: {len(test_ids)} 患者 ({len(test_df)} 条记录)")
    
    return train_df, val_df, test_df


def create_dataloaders(train_df, val_df, test_df, meta, batch_size=32, device='cuda'):
    """创建 DataLoader"""
    print("\n[3/4] 创建 DataLoaders...")
    
    # Dataset 参数
    dataset_kwargs = {
        'feature_cols': meta['seq_feature_cols'],
        'time_to_event_col': 'time_to_event',
        'event_indicator_col': 'event_indicator',
        'time_col': meta['time_col'],
        'static_feature_cols': meta['static_feature_cols'],
        'infer': False,
    }
    
    # 创建 Datasets
    train_dataset = CliTsSurvDataset(train_df, **dataset_kwargs)
    val_dataset = CliTsSurvDataset(val_df, **dataset_kwargs)
    test_dataset = CliTsSurvDataset(test_df, **dataset_kwargs)

    num_events = int(np.sum(train_dataset.event_indicator))
    train_weights = np.where(np.array(train_dataset.event_indicator) == 1, 
                             (len(train_dataset.event_indicator)-num_events)/num_events, 1)
    train_sampler = WeightedRandomSampler(train_weights, len(train_dataset.event_indicator), replacement=True)
    # train_sampler = None  # 目前不使用采样器，直接 shuffle=True
    
    # 创建 DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        collate_fn=lambda b: collate_fn(b, infer=False, device=device)
    ) if train_sampler is not None else DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, infer=False, device=device)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, infer=False, device=device)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, infer=False, device=device)
    )
    
    print(f"  batch_size: {batch_size}")
    
    return train_loader, val_loader, test_loader


def create_model(meta, num_time_bins=10, device='cuda'):
    """创建 CliTsSurv 模型"""
    print("\n[4/4] 创建模型...")

    # 特征维度
    conti_feature_in = meta['n_seq_features']
    bi_feature_in = 0
    static_conti_feature_in = meta['n_static_features']
    static_bi_feature_in = 0
    
    model = CliTsSurv(
        conti_feature_in=conti_feature_in,
        bi_feature_in=bi_feature_in,
        static_conti_feature_in=static_conti_feature_in,
        static_bi_feature_in=static_bi_feature_in,
        time_dim=TIME_DIM,
        encoder_args=ENCODER_ARGS,
        static_encoder_args=STATIC_ENCODER_ARGS,
        net_args=NET_ARGS,
        head_args=HEAD_ARGS,
        beta1=BETA1,
        beta2=BETA2,
        aug_configs=AUG_CONFIGS,
        num_time_bins=num_time_bins,
        alpha=ALPHA,
        sigma=SIGMA,
        gamma=GAMMA,
        time_encoder_type=TIME_ENCODER_TYPE,
        device=device
    )

    model = model.to(device)
    
    if PRETRAINED_ENCODER and Path(PRETRAINED_ENCODER).exists():
        print(f"  加载预训练编码器权重: {PRETRAINED_ENCODER}")
        state   = torch.load(PRETRAINED_ENCODER, weights_only=True, map_location=device)
        current = model.encoder.state_dict()
        compat  = {k: v for k, v in state.items() if k in current and v.shape == current[k].shape}
        model.encoder.load_state_dict(compat, strict=False)
        print(f"    加载 {len(compat)}/{len(state)} keys")

    if PRETRAINED_STATIC_ENCODER and Path(PRETRAINED_STATIC_ENCODER).exists():
        print(f"  加载预训练静态编码器权重: {PRETRAINED_STATIC_ENCODER}")
        state   = torch.load(PRETRAINED_STATIC_ENCODER, weights_only=True, map_location=device)
        current = model.static_encoder.state_dict()
        compat  = {k: v for k, v in state.items() if k in current and v.shape == current[k].shape}
        model.static_encoder.load_state_dict(compat, strict=False)
        print(f"    加载 {len(compat)}/{len(state)} keys")

    print(f"  时序输入维度: {conti_feature_in}")
    print(f"  静态输入维度: {static_conti_feature_in}")
    print(f"  时间区间数: {num_time_bins}")
    
    return model


def train_model(model, train_loader, val_loader, time_bin_edges, 
                epochs=100, lr=0.001, save_dir='./checkpoints',
                optimizer_type='adam', optimizer_kwargs=None):
    """训练模型"""
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    
    # 设置 time_bin_edges 用于 predict_survival/predict_risk
    model.time_bin_edges = time_bin_edges
    
    model.train_model(
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        epochs=epochs,
        lr=lr,
        save_dir=save_dir,
        time_bin_edges=time_bin_edges,
        optimizer_type=optimizer_type,
        optimizer_kwargs=optimizer_kwargs,
    )
    
    print("\n训练完成!")


def _collect_survival_data_with_times(model, loader, time_bin_edges, device, eval_times):
    """收集每个评估时间点的风险评分（与 DDH 对齐）
    
    Args:
        model: CliTsSurv 模型
        loader: DataLoader
        time_bin_edges: 时间区间边界
        device: 设备
        eval_times: 评估时间点数组 (n_times,)
    
    Returns:
        times: (N,) 真实时间
        events: (N,) 事件指示
        risk_scores: (N, n_times) 每个评估时间点的风险评分
    """
    model.eval()
    all_times, all_events = [], []
    all_risk_scores = []
    
    with torch.no_grad():
        for batch in loader:
            x, static_x = batch[0], batch[1]
            tte, evt = batch[2], batch[3]
            lens, times_emb = batch[4], batch[5]
            
            # 转换数据格式用于 predict_risk
            x_np = x.cpu().numpy()
            static_x_np = static_x.cpu().numpy()
            # lens 可能是 list，需要检查
            if isinstance(lens, list):
                lens_np = np.array(lens)
            else:
                lens_np = lens.cpu().numpy()
            times_np = times_emb.cpu().numpy()
            
            # 使用 predict_risk 获取每个时间点的风险评分
            risk_scores = model.predict_risk(
                x=x_np,
                t=eval_times,
                static_x=static_x_np,
                lens=lens_np,
                times=times_np,
                bs=128
            )
            
            all_times.extend(tte.cpu().numpy().tolist())
            all_events.extend(evt.cpu().numpy().tolist())
            all_risk_scores.append(risk_scores)
    
    return np.array(all_times), np.array(all_events), np.concatenate(all_risk_scores, axis=0)


def _collect_survival_probs_with_times(model, loader, time_bin_edges, device, eval_times):
    """收集每个评估时间点的生存概率
    
    Args:
        model: CliTsSurv 模型
        loader: DataLoader
        time_bin_edges: 时间区间边界
        device: 设备
        eval_times: 评估时间点数组 (n_times,)
    
    Returns:
        survival_probs: (N, n_times) 每个评估时间点的生存概率
    """
    model.eval()
    all_survival_probs = []
    
    with torch.no_grad():
        for batch in loader:
            x, static_x = batch[0], batch[1]
            tte, evt = batch[2], batch[3]
            lens, times_emb = batch[4], batch[5]
            
            # 转换数据格式
            x_np = x.cpu().numpy()
            static_x_np = static_x.cpu().numpy()
            if isinstance(lens, list):
                lens_np = np.array(lens)
            else:
                lens_np = lens.cpu().numpy()
            times_np = times_emb.cpu().numpy()
            
            # 使用 predict_survival 获取每个时间点的生存概率
            survival_probs = model.predict_survival(
                x=x_np,
                t=eval_times,
                static_x=static_x_np,
                lens=lens_np,
                times=times_np,
                bs=128
            )
            
            all_survival_probs.append(survival_probs)
    
    return np.concatenate(all_survival_probs, axis=0)


def _evaluate_ipcw(model, train_loader, test_loader, time_bin_edges, device, save_dir=None):
    """用 concordance_index_ipcw 评估，和 DDH 的指标对齐"""

    # 先收集训练集数据用于 IPCW 计算
    t_train_list, e_train_list = [], []
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            tte, evt = batch[2], batch[3]
            t_train_list.extend(tte.cpu().numpy().tolist())
            e_train_list.extend(evt.cpu().numpy().tolist())
    t_train = np.array(t_train_list)
    e_train = np.array(e_train_list)
    
    # 收集测试集数据（时间和事件）
    t_test_list, e_test_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            tte, evt = batch[2], batch[3]
            t_test_list.extend(tte.cpu().numpy().tolist())
            e_test_list.extend(evt.cpu().numpy().tolist())
    t_test = np.array(t_test_list)
    e_test = np.array(e_test_list)

    # 三个评估时间点：25% / 50% / 75% 分位数（仅用训练集，避免测试集泄漏）
    horizons = [0.25, 0.50, 0.75]
    valid_times = np.quantile(t_train[e_train == 1], horizons)
    # 确保评估时间点在训练集范围内（IPCW 要求）
    t_max_train = t_train.max()
    valid_times = np.clip(valid_times, t_train.min(), t_max_train * 0.99)

    # 收集每个评估时间点的风险评分和生存概率
    _, _, risk_scores = _collect_survival_data_with_times(
        model, test_loader, time_bin_edges, device, valid_times
    )
    survival_probs = _collect_survival_probs_with_times(
        model, test_loader, time_bin_edges, device, valid_times
    )

    metrics = evaluate_survival_metrics(
        t_train=t_train, e_train=e_train,
        t_eval=t_test,   e_eval=e_test,
        risk_scores=risk_scores,
        survival_probs=survival_probs,
        eval_times=valid_times,
        horizons=horizons,
        split_name='Test',
    )

    # 写入平均 IPCW c-index
    if metrics['mean_ipcw'] is not None and save_dir is not None:
        avg_ipcw = metrics['mean_ipcw']
        ipcw_path = Path(save_dir) / "ipcw_score"
        ipcw_path.write_text(f"{avg_ipcw:.6f}\n")
        print(f"\n[ratchet] avg IPCW c-index = {avg_ipcw:.4f} → 已写入 {ipcw_path}")

    return metrics


def save_hparams(save_dir: Path):
    hparams = {
        "device": DEVICE,
        "n_runs": N_RUNS,
        "seed_start": SEED_START,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "test_ratio": TEST_RATIO,
        "val_ratio": VAL_RATIO,
        "time_dim": TIME_DIM,
        "time_encoder_type": TIME_ENCODER_TYPE,
        "encoder_args": ENCODER_ARGS,
        "static_encoder_args": STATIC_ENCODER_ARGS,
        "net_args": NET_ARGS,
        "head_args": HEAD_ARGS,
        "beta1": BETA1,
        "beta2": BETA2,
        "alpha": ALPHA,
        "sigma": SIGMA,
        "gamma": GAMMA,
        "aug_configs": AUG_CONFIGS,
        "data_dir": str(DATA_DIR),
        "base_save_dir": str(BASE_SAVE_DIR),
        "csv_path": str(CSV_PATH),
    }
    out_path = save_dir / 'hparams.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(hparams, f, indent=2, ensure_ascii=False)
    print(f"超参已保存到: {out_path}")


def main():

    print("="*60)
    print("CliTsSurv 训练脚本")
    print(f"设备: {DEVICE}  |  重复次数: {N_RUNS}")
    print("="*60)

    save_hparams(BASE_SAVE_DIR)

    # 1. 加载数据（只加载一次）
    df, meta, time_bin_edges = load_data(DATA_DIR)

    all_metrics = []

    for run in range(N_RUNS):
        seed = SEED_START + run
        SAVE_DIR = BASE_SAVE_DIR / f"run_{run:02d}"
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Run {run+1}/{N_RUNS}  (seed={seed})")
        print('='*60)

        # 2. 划分数据集
        train_df, val_df, test_df = split_train_test(
            df, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, random_seed=seed
        )

        # 3. 创建 DataLoaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_df, val_df, test_df, meta, BATCH_SIZE, DEVICE
        )

        # 4. 创建模型
        model = create_model(meta, num_time_bins=len(time_bin_edges)-1, device=DEVICE)
        model.time_bin_edges = time_bin_edges

        # 5. 训练
        train_model(model, train_loader, val_loader, time_bin_edges,
                    epochs=EPOCHS, lr=LR, save_dir=str(SAVE_DIR),
                    optimizer_type='adam',
                    optimizer_kwargs=None)

        # 6. 评估
        metrics = _evaluate_ipcw(
            model, train_loader, test_loader, time_bin_edges, DEVICE,
            save_dir=str(SAVE_DIR)
        )
        if metrics is not None:
            all_metrics.append(metrics)

    # 汇总结果
    print("\n" + "="*60)
    print(f"汇总结果 ({len(all_metrics)}/{N_RUNS} runs 成功)")
    print("="*60)

    if all_metrics:
        mean_ipcws  = [m['mean_ipcw']  for m in all_metrics if m.get('mean_ipcw')  is not None]
        mean_aucs   = [m['mean_auc']   for m in all_metrics if m.get('mean_auc')   is not None]
        overall_cis = [m['overall_ci'] for m in all_metrics if m.get('overall_ci') is not None]
        ibss        = [m['ibs']        for m in all_metrics if m.get('ibs')        is not None]
        if mean_ipcws:
            print(f"  Mean IPCW C-index : {np.mean(mean_ipcws):.4f} ± {np.std(mean_ipcws):.4f}")
        if mean_aucs:
            print(f"  Mean AUC          : {np.mean(mean_aucs):.4f} ± {np.std(mean_aucs):.4f}")
        if overall_cis:
            print(f"  Harrell C-index   : {np.mean(overall_cis):.4f} ± {np.std(overall_cis):.4f}")
        if ibss:
            print(f"  IBS               : {np.mean(ibss):.4f} ± {np.std(ibss):.4f}")

        rows = []
        for run_idx, m in enumerate(all_metrics):
            row = {'run': run_idx, 'seed': SEED_START + run_idx}
            for k, v in m.items():
                row[f'test_{k}'] = v
            rows.append(row)
        results_df = pd.DataFrame(rows)
        results_df.to_csv(CSV_PATH, index=False)
        print(f"\n✅ 详细结果已保存到: {CSV_PATH}")

    print(f"\n模型保存在: {BASE_SAVE_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
