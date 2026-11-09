#!/usr/bin/env python3
"""Train CliTsSurv for discrete-time survival prediction."""

import json
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, WeightedRandomSampler

# Import CliTsSurv components.
import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (evaluate_two_prediction_protocols, set_global_seed,
                   split_patient_indices)

from models.CliTsSurv.CliTsSurv import CliTsSurv
from models.CliTsSurv.datasets import CliTsSurvDataset, collate_fn
from pretrain_latent_encoders import ensure_seed_pretrained_encoders


# ------------------------------------------------------------------ #
#  Hyperparameters
# ------------------------------------------------------------------ #
# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Configuration.
DATA_DIR     = Path(__file__).parent.parent / 'data' / 'CliTsSurv_data'
BACKBONE = os.environ.get('CLITS_BACKBONE', 'transformer').strip().lower()
if BACKBONE not in {'transformer', 'lstm', 'gru'}:
    raise ValueError(f"Unsupported CLITS_BACKBONE={BACKBONE!r}; use transformer, lstm, or gru")
TRAINING_SCHEME = os.environ.get('CLITS_TRAINING_SCHEME', 'balanced').strip().lower()
if TRAINING_SCHEME not in {'balanced', 'natural_balanced_ranking'}:
    raise ValueError(
        f"Unsupported CLITS_TRAINING_SCHEME={TRAINING_SCHEME!r}; "
        "use balanced or natural_balanced_ranking"
    )

default_checkpoint_name = 'clitssurv' if BACKBONE == 'transformer' else f'clitssurv_{BACKBONE}'
BASE_SAVE_DIR = Path(os.environ.get(
    'CLITS_SAVE_DIR',
    Path(__file__).parent.parent / 'checkpoints' / default_checkpoint_name,
))
BASE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
# Save each run's results to CSV.
RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'clitssurv'
CSV_PATH = Path(os.environ.get(
    'CLITS_CSV_PATH',
    RESULTS_DIR / (
        'repeated_training_results_multiscale_sincos_trainonly_pretrain_final.csv'
        if BACKBONE == 'transformer'
        else f'repeated_training_results_{BACKBONE}_multiscale_sincos_trainonly_pretrain_final.csv'
    )
))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

# Repeated training
N_RUNS     = int(os.environ.get('CLITS_N_RUNS', '20'))
SEED_START = int(os.environ.get('CLITS_SEED_START', '1000'))
PRETRAINED_ROOT = Path(__file__).parent.parent / 'checkpoints' / 'pretrained_train_only'
PRETRAIN_EPOCHS = 200

# Training
BATCH_SIZE = 32
EPOCHS     = 40
LR         = 0.0001 if BACKBONE == 'transformer' else 0.001

# Data split
TEST_RATIO  = 0.15
VAL_RATIO   = 0.15

# Time encoding
TIME_DIM          = 6
TIME_ENCODER_TYPE = os.environ.get(
    'CLITS_TIME_ENCODER', 'multiscale_sincos'
).strip().lower()
if TIME_ENCODER_TYPE not in {'sincos', 'time2vec', 'ctlpe', 'multiscale_sincos'}:
    raise ValueError(
        f"Unsupported CLITS_TIME_ENCODER={TIME_ENCODER_TYPE!r}; use sincos, "
        "time2vec, ctlpe, or multiscale_sincos"
    )
# With d_t=6, MultiSinCos uses wavelengths 171, 60, and 36 months.

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
# Backbone rnn_type: 'lstm', 'gru' or 'transformer'.
# nhead, dim_feedforward and dropout configure the Transformer path.
NET_ARGS = {
    'hidden_size': 64,
    'num_layers': 3,
    'rnn_type': BACKBONE,
    'nhead': 4,
    'dim_feedforward': 128,
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
GAMMA = float(os.environ.get('CLITS_GAMMA', '0.5'))  # Brier calibration loss weight

# Data augmentation
AUG_CONFIGS = [
    {'type': 'AddGaussianNoise', 'args': {'p': 0.3}},
    {'type': 'InnerInterpolate', 'args': {'p': 0.7}},
    {'type': 'RandomRemove',     'args': {'p': 0.5}},
    {'type': 'RandomSlice',      'args': {'p': 0.1}},
]


def load_data(data_dir: Path):
    """Load preprocessed data and metadata."""
    print("[1/4] 加载数据...")
    
    # Load metadata.
    with open(data_dir / "metadata.json", 'r') as f:
        meta = json.load(f)
    
    # Load merged data from CSV or Parquet.
    csv_path = data_dir / "merged.csv"
    parquet_path = data_dir / "merged.parquet"
    if csv_path.exists():
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print("  使用 CSV 格式加载")
    else:
        df = pd.read_parquet(parquet_path)
        print("  使用 Parquet 格式加载")
    
    # Load time-bin boundaries.
    time_bin_edges = np.load(data_dir / "time_bin_edges.npy")
    
    print(f"  样本数: {meta['n_unique_ids']} 患者")
    print(f"  时序特征: {meta['n_seq_features']}")
    print(f"  静态特征: {meta['n_static_features']}")
    print(f"  时间区间: {len(time_bin_edges)-1} 个")
    
    return df, meta, time_bin_edges


def split_train_test(df: pd.DataFrame, test_ratio=0.15, val_ratio=0.15, random_seed=42):
    """Create patient-level training, validation and test partitions."""
    print("\n[2/4] 划分数据集...")
    
    # Split by patient ID.
    unique_ids = df['id'].unique()
    train_idx, val_idx, test_idx = split_patient_indices(
        unique_ids, random_seed, test_ratio, val_ratio
    )
    train_ids = unique_ids[train_idx]
    val_ids = unique_ids[val_idx]
    test_ids = unique_ids[test_idx]
    
    train_df = df[df['id'].isin(train_ids)].copy()
    val_df = df[df['id'].isin(val_ids)].copy()
    test_df = df[df['id'].isin(test_ids)].copy()
    
    print(f"  训练集: {len(train_ids)} 患者 ({len(train_df)} 条记录)")
    print(f"  验证集: {len(val_ids)} 患者 ({len(val_df)} 条记录)")
    print(f"  测试集: {len(test_ids)} 患者 ({len(test_df)} 条记录)")
    
    return train_df, val_df, test_df


def create_dataloaders(train_df, val_df, test_df, meta, batch_size=32,
                       device='cuda', seed=42, sampler_mode='balanced'):
    """Create data loaders for the patient partitions."""
    print("\n[3/4] 创建 DataLoaders...")
    
    # Dataset parameters.
    dataset_kwargs = {
        'feature_cols': meta['seq_feature_cols'],
        'time_to_event_col': 'time_to_event',
        'event_indicator_col': 'event_indicator',
        'time_col': meta['time_col'],
        'static_feature_cols': meta['static_feature_cols'],
        'infer': False
    }
    
    # Create datasets.
    train_dataset = CliTsSurvDataset(train_df, **dataset_kwargs)
    val_dataset = CliTsSurvDataset(val_df, **dataset_kwargs)
    test_dataset = CliTsSurvDataset(test_df, **dataset_kwargs)

    num_events = int(np.sum(train_dataset.event_indicator))
    train_weights = np.where(np.array(train_dataset.event_indicator) == 1, 
                             (len(train_dataset.event_indicator)-num_events)/num_events, 1)
    loader_generator = torch.Generator().manual_seed(seed)
    if sampler_mode not in {'balanced', 'natural'}:
        raise ValueError("sampler_mode must be 'balanced' or 'natural'")
    train_sampler = None
    if sampler_mode == 'balanced':
        train_sampler = WeightedRandomSampler(
            train_weights, len(train_dataset.event_indicator), replacement=True,
            generator=loader_generator,
        )
    # train_sampler = None  # Alternative: use shuffle=True without a sampler.
    
    # Create data loaders.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        generator=loader_generator,
        collate_fn=lambda b: collate_fn(b, infer=False, device=device)
    ) if train_sampler is not None else DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        generator=loader_generator,
        collate_fn=lambda b: collate_fn(b, infer=False, device=device)
    )

    # Evaluation must use every training patient exactly once.  The weighted
    # sampler is a training device and must not alter IPCW estimation or the
    # event-time quantiles used by the common evaluation protocol.
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, infer=False, device=device),
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
    print(f"  sampler_mode: {sampler_mode}")
    
    return train_loader, train_eval_loader, val_loader, test_loader


def create_model(meta, num_time_bins=10, device='cuda',
                 pretrained_encoder=None, pretrained_static_encoder=None):
    """Construct the CliTsSurv model."""
    print("\n[4/4] 创建模型...")

    # Feature dimensions.
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
    
    if pretrained_encoder and Path(pretrained_encoder).exists():
        print(f"  加载预训练编码器权重: {pretrained_encoder}")
        state   = torch.load(pretrained_encoder, weights_only=True, map_location=device)
        current = model.encoder.state_dict()
        compat  = {k: v for k, v in state.items() if k in current and v.shape == current[k].shape}
        model.encoder.load_state_dict(compat, strict=False)
        print(f"    加载 {len(compat)}/{len(state)} keys")

    if pretrained_static_encoder and Path(pretrained_static_encoder).exists():
        print(f"  加载预训练静态编码器权重: {pretrained_static_encoder}")
        state   = torch.load(pretrained_static_encoder, weights_only=True, map_location=device)
        current = model.static_encoder.state_dict()
        compat  = {k: v for k, v in state.items() if k in current and v.shape == current[k].shape}
        model.static_encoder.load_state_dict(compat, strict=False)
        print(f"    加载 {len(compat)}/{len(state)} keys")

    print(f"  时序输入维度: {conti_feature_in}")
    print(f"  静态输入维度: {static_conti_feature_in}")
    print(f"  时间区间数: {num_time_bins}")
    
    return model


def train_model(model, train_loader, train_eval_loader, val_loader, time_bin_edges,
                epochs=100, lr=0.001, save_dir='./checkpoints',
                optimizer_type='adam', optimizer_kwargs=None,
                ranking_loader=None):
    """Train the model."""
    print("\n" + "="*60)
    print("开始训练")
    print("="*60)
    
    # Set time_bin_edges for predict_survival and predict_risk.
    model.time_bin_edges = time_bin_edges
    
    # Validation is inference-only, so a larger batch reduces Python/DataLoader
    # overhead without changing predictions or the risk set.
    validation_loader = DataLoader(
        val_loader.dataset,
        batch_size=max(256, val_loader.batch_size or 1),
        shuffle=False,
        collate_fn=val_loader.collate_fn,
    )

    def collect_outcomes(loader):
        times, events, observations = [], [], []
        current_mode = model.training
        model.eval()
        with torch.no_grad():
            for batch in loader:
                _, observation = model._observation_bins(
                    batch[0], batch[4], batch[5], time_bin_edges
                )
                times.extend(batch[2].cpu().numpy().tolist())
                events.extend(batch[3].cpu().numpy().tolist())
                observations.extend(observation.cpu().numpy().tolist())
        model.train(current_mode)
        return np.asarray(times), np.asarray(events), np.asarray(observations)

    t_train, e_train, a_train = collect_outcomes(train_eval_loader)
    t_val, e_val, a_val = collect_outcomes(validation_loader)
    horizons = [0.25, 0.50, 0.75]
    eval_times = np.quantile(t_train[e_train == 1], horizons)
    eval_times = np.clip(eval_times, t_train.min(), t_train.max() * 0.99)

    def validation_metric_fn(current_model):
        survival_probs = _collect_survival_probs_with_times(
            current_model, validation_loader, time_bin_edges, DEVICE, eval_times
        )
        _, metrics = evaluate_two_prediction_protocols(
            t_train=t_train, e_train=e_train, a_train=a_train,
            t_eval=t_val, e_eval=e_val, a_eval=a_val,
            native_risk_scores=1.0 - survival_probs,
            native_survival_probs=survival_probs,
            native_is_conditional=True,
            eval_times=eval_times, horizons=horizons,
            print_results=False, split_name='Validation',
        )
        return metrics['mean_ipcw'], metrics['ibs']

    model.train_model(
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        epochs=epochs,
        lr=lr,
        save_dir=save_dir,
        time_bin_edges=time_bin_edges,
        optimizer_type=optimizer_type,
        optimizer_kwargs=optimizer_kwargs,
        ranking_dataloader=ranking_loader,
        validation_metric_fn=validation_metric_fn,
    )
    
    print("\n训练完成!")


def _collect_survival_data_with_times(model, loader, time_bin_edges, device, eval_times):
    """Collect outcomes and risk scores at the requested evaluation times.

Return times (N,), events (N,) and risk_scores (N, n_times)."""
    model.eval()
    all_times, all_events = [], []
    all_risk_scores = []
    
    with torch.no_grad():
        for batch in loader:
            x, static_x = batch[0], batch[1]
            tte, evt = batch[2], batch[3]
            lens, times_emb = batch[4], batch[5]
            
            # Convert inputs for predict_risk.
            x_np = x.cpu().numpy()
            static_x_np = static_x.cpu().numpy()
            # Sequence lengths may be a list rather than a tensor.
            if isinstance(lens, list):
                lens_np = np.array(lens)
            else:
                lens_np = lens.cpu().numpy()
            times_np = times_emb.cpu().numpy()
            
            # Get risk scores at each evaluation time using predict_risk.
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
    """Collect survival probabilities with shape (N, n_times)."""
    model.eval()
    all_survival_probs = []
    
    with torch.no_grad():
        for batch in loader:
            x, static_x = batch[0], batch[1]
            tte, evt = batch[2], batch[3]
            lens, times_emb = batch[4], batch[5]
            
            # Convert input formats.
            x_np = x.cpu().numpy()
            static_x_np = static_x.cpu().numpy()
            if isinstance(lens, list):
                lens_np = np.array(lens)
            else:
                lens_np = lens.cpu().numpy()
            times_np = times_emb.cpu().numpy()
            
            # Get survival probabilities at each evaluation time using predict_survival.
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


def _evaluate_ipcw(model, train_loader, test_loader, time_bin_edges, device,
                   save_dir=None, print_results=True, split_name='Test'):
    """Evaluate survival predictions using IPCW concordance and related metrics."""

    # Collect training outcomes for IPCW estimation.
    t_train_list, e_train_list, a_train_list = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            tte, evt = batch[2], batch[3]
            _, observation_times = model._observation_bins(
                batch[0], batch[4], batch[5], time_bin_edges
            )
            t_train_list.extend(tte.cpu().numpy().tolist())
            e_train_list.extend(evt.cpu().numpy().tolist())
            a_train_list.extend(observation_times.cpu().numpy().tolist())
    t_train = np.array(t_train_list)
    e_train = np.array(e_train_list)
    a_train = np.array(a_train_list)
    
    # Collect evaluation times and event indicators.
    t_test_list, e_test_list, a_test_list = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            tte, evt = batch[2], batch[3]
            _, observation_times = model._observation_bins(
                batch[0], batch[4], batch[5], time_bin_edges
            )
            t_test_list.extend(tte.cpu().numpy().tolist())
            e_test_list.extend(evt.cpu().numpy().tolist())
            a_test_list.extend(observation_times.cpu().numpy().tolist())
    t_test = np.array(t_test_list)
    e_test = np.array(e_test_list)
    a_test = np.array(a_test_list)

    # Use training-derived Q25, Q50 and Q75 evaluation horizons.
    horizons = [0.25, 0.50, 0.75]
    valid_times = np.quantile(t_train[e_train == 1], horizons)
    # Keep evaluation horizons within the training range for IPCW.
    t_max_train = t_train.max()
    valid_times = np.clip(valid_times, t_train.min(), t_max_train * 0.99)

    # Collect risk scores and survival probabilities at each evaluation time.
    _, _, risk_scores = _collect_survival_data_with_times(
        model, test_loader, time_bin_edges, device, valid_times
    )
    survival_probs = _collect_survival_probs_with_times(
        model, test_loader, time_bin_edges, device, valid_times
    )

    metrics, conditional_metrics = evaluate_two_prediction_protocols(
        t_train=t_train, e_train=e_train, a_train=a_train,
        t_eval=t_test, e_eval=e_test, a_eval=a_test,
        native_risk_scores=risk_scores,
        native_survival_probs=survival_probs,
        native_is_conditional=True,
        eval_times=valid_times, horizons=horizons,
        print_results=print_results, split_name=split_name,
    )
    for k, v in conditional_metrics.items():
        metrics[f'conditional_{k}'] = v

    # Store mean IPCW C-index.
    if metrics['mean_ipcw'] is not None and save_dir is not None:
        avg_ipcw = metrics['mean_ipcw']
        ipcw_path = Path(save_dir) / "ipcw_score"
        ipcw_path.write_text(f"{avg_ipcw:.6f}\n")
        print(f"\n[ratchet] avg IPCW c-index = {avg_ipcw:.4f}; saved to {ipcw_path}")

    return metrics


def save_hparams(save_dir: Path):
    hparams = {
        "device": DEVICE,
        "backbone": BACKBONE,
        "training_scheme": TRAINING_SCHEME,
        "n_runs": N_RUNS,
        "seed_start": SEED_START,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "test_ratio": TEST_RATIO,
        "val_ratio": VAL_RATIO,
        "time_dim": TIME_DIM,
        "time_encoder_type": TIME_ENCODER_TYPE,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "pretraining_data": "outer training split only; inner 85/15 patient split",
        "encoder_args": ENCODER_ARGS,
        "static_encoder_args": STATIC_ENCODER_ARGS,
        "net_args": NET_ARGS,
        "head_args": HEAD_ARGS,
        "beta1": BETA1,
        "beta2": BETA2,
        "alpha": ALPHA,
        "sigma": SIGMA,
        "gamma": GAMMA,
        "output_time_scale": "absolute",
        "prediction_conditioning": "T>a using last non-zero observation",
        "model_selection": (
            "validation conditional mean IPCW C-index; "
            "lower conditional IBS tie-break"
        ),
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

    # 1. Load data once.
    df, meta, time_bin_edges = load_data(DATA_DIR)

    if CSV_PATH.exists():
        result_rows = pd.read_csv(CSV_PATH).to_dict('records')
    else:
        result_rows = []
    completed_seeds = {int(row['seed']) for row in result_rows}

    for run in range(N_RUNS):
        seed = SEED_START + run
        if seed in completed_seeds:
            print(f"Skipping completed seed={seed}")
            continue
        set_global_seed(seed)
        SAVE_DIR = BASE_SAVE_DIR / f"seed_{seed}"
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Run {run+1}/{N_RUNS}  (seed={seed})")
        print('='*60)

        # 2. Split the dataset.
        train_df, val_df, test_df = split_train_test(
            df, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO, random_seed=seed
        )

        # 3. Pretrain both ARAEs using this run's outer training split only.
        seq_pretrained, static_pretrained = ensure_seed_pretrained_encoders(
            train_df=train_df, meta=meta, seed=seed,
            save_root=PRETRAINED_ROOT, device=DEVICE,
            epochs=PRETRAIN_EPOCHS, lr=LR,
        )
        # Pretraining consumes random draws only when the cache is first
        # created. Reset here so end-to-end initialization is identical
        # whether the seed-specific pretrained weights were created or reused.
        set_global_seed(seed)

        # 4. Create the natural probability/reconstruction stream and,
        # when requested, an independent event-balanced ranking stream.
        sampler_mode = (
            'natural' if TRAINING_SCHEME == 'natural_balanced_ranking'
            else 'balanced'
        )
        train_loader, train_eval_loader, val_loader, test_loader = create_dataloaders(
            train_df, val_df, test_df, meta, BATCH_SIZE, DEVICE, seed,
            sampler_mode=sampler_mode,
        )
        ranking_loader = None
        if TRAINING_SCHEME == 'natural_balanced_ranking':
            ranking_loader = create_dataloaders(
                train_df, val_df, test_df, meta, BATCH_SIZE, DEVICE, seed,
                sampler_mode='balanced',
            )[0]

        # 5. Create the model.
        model = create_model(
            meta, num_time_bins=len(time_bin_edges)-1, device=DEVICE,
            pretrained_encoder=seq_pretrained,
            pretrained_static_encoder=static_pretrained,
        )
        model.time_bin_edges = time_bin_edges

        # 6. Train, or reuse a checkpoint only after a prior run reached the
        # normal end of train_model (identified by both best.pt and loss_hist).
        best_checkpoint = SAVE_DIR / 'best.pt'
        completed_training = best_checkpoint.exists() and (SAVE_DIR / 'loss_hist').exists()
        if completed_training:
            print(f"Reusing completed checkpoint: {best_checkpoint}")
            model.load_state_dict(torch.load(
                best_checkpoint, weights_only=True, map_location=DEVICE
            ))
        else:
            train_model(model, train_loader, train_eval_loader, val_loader,
                        time_bin_edges,
                        epochs=EPOCHS, lr=LR, save_dir=str(SAVE_DIR),
                        optimizer_type='adam',
                        optimizer_kwargs=None,
                        ranking_loader=ranking_loader)

        # 7. Evaluate.
        metrics = _evaluate_ipcw(
            model, train_eval_loader, test_loader, time_bin_edges, DEVICE,
            save_dir=str(SAVE_DIR)
        )
        if metrics is not None:
            row = {'run': run, 'seed': seed}
            row.update({f'test_{k}': v for k, v in metrics.items()})
            result_rows = [r for r in result_rows if int(r['seed']) != seed]
            result_rows.append(row)
            pd.DataFrame(result_rows).sort_values('seed').to_csv(CSV_PATH, index=False)
            print(f"Incremental result saved to: {CSV_PATH}")

    # Summarize results.
    print("\n" + "="*60)
    results_df = pd.DataFrame(result_rows).sort_values('seed')
    print(f"汇总结果 ({len(results_df)}/{N_RUNS} runs 成功)")
    print("="*60)

    if not results_df.empty:
        mean_ipcws  = results_df['test_mean_ipcw'].dropna().tolist()
        mean_aucs   = results_df['test_mean_auc'].dropna().tolist()
        overall_cis = results_df['test_overall_ci'].dropna().tolist()
        ibss        = results_df['test_ibs'].dropna().tolist()
        if mean_ipcws:
            print(f"  Mean IPCW C-index : {np.mean(mean_ipcws):.4f} ± {np.std(mean_ipcws):.4f}")
        if mean_aucs:
            print(f"  Mean AUC          : {np.mean(mean_aucs):.4f} ± {np.std(mean_aucs):.4f}")
        if overall_cis:
            print(f"  Harrell C-index   : {np.mean(overall_cis):.4f} ± {np.std(overall_cis):.4f}")
        if ibss:
            print(f"  IBS               : {np.mean(ibss):.4f} ± {np.std(ibss):.4f}")

        print(f"\nResults saved to: {CSV_PATH}")

    print(f"\n模型保存在: {BASE_SAVE_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
