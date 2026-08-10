#!/usr/bin/env python3
"""
CliTsSurv 数据预处理脚本
将 seq.csv（时序数据）和 static.csv（静态+生存数据）合并为训练格式

处理流程:
  1. 加载 seq.csv 和 static.csv
  2. 过滤掉生存数据缺失的样本
  3. 合并时序和静态数据
  4. 保存为 merged.parquet（供 CliTsSurvDataset 使用）
  5. 计算时间区间边界 (time_bin_edges)

输出:
  - CliTsSurv_data/merged.parquet: 合并后的完整数据
  - CliTsSurv_data/time_bin_edges.npy: 时间区间边界
  - CliTsSurv_data/metadata.json: 数据元信息
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_and_merge_data(data_dir: Path) -> pd.DataFrame:
    """加载并合并 seq.csv 和 static.csv"""
    print("\n[1/4] 加载原始数据...")
    
    seq_df = pd.read_csv(data_dir / "seq.csv", encoding='utf-8-sig')
    static_df = pd.read_csv(data_dir / "static.csv", encoding='utf-8-sig')

    print(f"  seq.csv: {seq_df.shape}, 唯一ID: {seq_df['id'].nunique()}")
    print(f"  static.csv: {static_df.shape}, 唯一ID: {static_df['id'].nunique()}")
    
    # 过滤掉生存数据缺失的样本
    valid_static = static_df[
        static_df['survival_months'].notna() & 
        static_df['survival_status'].notna()
    ].copy()
    
    print(f"  过滤缺失生存数据后: {len(valid_static)}")
    
    # 创建生存标签: 1=事件(死亡), 0=删失(存活或失访)
    # survival_status=1 表示死亡事件
    valid_static['event_indicator'] = (valid_static['survival_status'] == 1).astype(int)
    valid_static['time_to_event'] = valid_static['survival_months']
    
    # 过滤 time_to_event <= 0 的患者（与 DDH 对齐）
    valid_static = valid_static[valid_static['time_to_event'] > 0].copy()
    print(f"  过滤 time_to_event <= 0 后: {len(valid_static)}")
    
    # 只保留在 seq 中也存在的 ID
    common_ids = set(seq_df['id'].unique()) & set(valid_static['id'].unique())
    print(f"  共同ID数: {len(common_ids)}")
    
    seq_df = seq_df[seq_df['id'].isin(common_ids)].copy()
    valid_static = valid_static[valid_static['id'].isin(common_ids)].copy()
    
    # 合并数据: 将 static 的生存信息添加到 seq 的每一行
    merge_cols = ['id', 'event_indicator', 'time_to_event', 'bloodtype_A', 'bloodtype_B', 'RH', 
                  'VTE', 'PE', 'CE', 'DM']
    static_subset = valid_static[merge_cols]
    
    # Left merge: 每个时序记录都带上静态特征和生存标签
    merged = seq_df.merge(static_subset, on='id', how='left')
    
    print(f"  合并后: {merged.shape}")
    
    return merged


def add_feature_engineering(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    特征工程步骤（在归一化之前执行）:
    1. 删除缺失率 >90% 的特征（噪声大，对模型几乎无贡献）
    2. Per-patient 前向填充（ffill），减少需要 mask 的位置
    3. 新增 delta_t 特征（相邻访问时间间隔，随访密度是预后信号）
    4. 新增 comorbidity_count 静态特征（VTE+PE+CE+DM 合并病症数）
    """
    print("\n[FE] 特征工程...")
    
    # # 1. 删除缺失率 >90% 的特征
    # HIGH_MISSING_COLS = [
    #     'NK%', 'BLY%', 'NK', 'ABLC',   # ~98% 缺失
    #     'TLY%', 'CD8%', 'CD8', 'CD4%', 'CD4', 'CD4/CD8', 'ATLC', 'ALC',  # ~98% 缺失
    #     'BNP',   # ~95% 缺失
    #     'cTnI',  # ~90% 缺失
    # ]
    # cols_to_drop = [c for c in HIGH_MISSING_COLS if c in merged_df.columns]
    # merged_df = merged_df.drop(columns=cols_to_drop)
    # print(f"  删除高缺失特征 ({len(cols_to_drop)}个): {cols_to_drop}")

    # 2. Per-patient 前向填充（只对时序数值特征）
    # exclude_from_ffill = {
    #     'id', 'date', 'event_indicator', 'time_to_event',
    #     'bloodtype_A', 'bloodtype_B', 'RH', 'VTE', 'PE', 'CE', 'DM', 'comorbidity_count'
    # }
    # ffill_cols = [c for c in merged_df.columns
    #               if c not in exclude_from_ffill
    #               and merged_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    # # 按 (id, date) 排序，然后按患者 ffill
    # merged_df = merged_df.sort_values(['id', 'date']).reset_index(drop=True)
    # merged_df[ffill_cols] = merged_df.groupby('id')[ffill_cols].ffill()
    # # 残余 NaN（第一次访问就缺失）保留，后续归一化时仍为 NaN，collate_fn 会 mask 掉
    # nan_ratio_after = merged_df[ffill_cols].isnull().mean().mean()
    # print(f"  Per-patient ffill 后，平均缺失率: {nan_ratio_after:.2%}")

    # 3. 新增 delta_t 特征（相邻访问时间间隔，单位：月）
    merged_df['delta_t'] = merged_df.groupby('id')['date'].diff().fillna(0.0)
    merged_df['delta_t'] = np.log1p(merged_df['delta_t'])  # log1p 压缩长尾，避免极端 z-score
    print(f"  新增 delta_t 特征（log1p 变换后，相邻访问间隔），范围: {merged_df['delta_t'].min():.2f}~{merged_df['delta_t'].max():.2f}")

    # 4. 新增 comorbidity_count 静态特征（每个患者的合并病症数，0~4）
    comorbidity_cols = [c for c in ['VTE', 'PE', 'CE', 'DM'] if c in merged_df.columns]
    if comorbidity_cols:
        merged_df['comorbidity_count'] = merged_df[comorbidity_cols].sum(axis=1)
        print(f"  新增 comorbidity_count 静态特征，分布: {merged_df['comorbidity_count'].value_counts().sort_index().to_dict()}")

    return merged_df


def split_feature_columns(merged_df: pd.DataFrame) -> dict:
    """分离不同类型的特征列"""
    print("\n[2/4] 分离特征列...")

    # 时序特征: 在 seq.csv 中的检验指标（排除id, date, 静态特征）
    exclude_cols = {
        'id', 'date', 'event_indicator', 'time_to_event',
        'bloodtype_A', 'bloodtype_B', 'RH', 'VTE', 'PE', 'CE', 'DM', 'comorbidity_count',
        'survival_status', 'survival_months',
        'death_time', 'death_in_hospital', 'death_from_cancer',
        'lost_to_followup_status', 'lost_to_followup_time', 'lost_to_followup_reason',
        'last_contact_time', 'desc'
    }

    # 时序特征列（包含新增的 delta_t）
    seq_feature_cols = [c for c in merged_df.columns
                        if c not in exclude_cols and merged_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

    # 静态特征列（从 static 合并过来的 + comorbidity_count）
    static_feature_cols = ['bloodtype_A', 'bloodtype_B', 'RH', 'VTE', 'PE', 'CE', 'DM', 'comorbidity_count']
    static_feature_cols = [c for c in static_feature_cols if c in merged_df.columns]

    print(f"  时序特征: {len(seq_feature_cols)} (含 delta_t)")
    print(f"  静态特征: {len(static_feature_cols)} (含 comorbidity_count)")
    print(f"  时序特征示例: {seq_feature_cols[:5]}")

    return {
        'seq_feature_cols': seq_feature_cols,
        'static_feature_cols': static_feature_cols,
        'time_col': 'date',
        'time_to_event_col': 'time_to_event',
        'event_indicator_col': 'event_indicator',
        'id_col': 'id'
    }


def compute_time_bins(merged_df: pd.DataFrame, num_bins: int = 10) -> np.ndarray:
    """计算时间区间边界用于 DeepHit 离散化"""
    print(f"\n[3/4] 计算时间区间边界 (num_bins={num_bins})...")
    
    tte = merged_df['time_to_event'].values
    tte_min, tte_max = tte.min(), tte.max()
    
    # 等宽分箱
    bin_edges = np.linspace(tte_min, tte_max * 1.2, num_bins + 1)  # 留 20% 余量
    
    print(f"  时间范围: [{tte_min:.1f}, {tte_max:.1f}] 月")
    print(f"  区间边界: {bin_edges}")
    print(f"  事件数: {merged_df['event_indicator'].sum():.0f} ({merged_df['event_indicator'].mean():.2%})")
    print(f"  删失数: {(merged_df['event_indicator'] == 0).sum():.0f}")
    
    return bin_edges


def normalize_and_save(merged_df: pd.DataFrame, feature_info: dict,
                       bin_edges: np.ndarray, output_dir: Path):
    """归一化并保存数据"""
    print("\n[4/4] 归一化并保存...")

    output_dir.mkdir(exist_ok=True)

    # 分离时序特征和静态特征
    seq_cols = feature_info['seq_feature_cols']
    static_cols = feature_info['static_feature_cols']

    # 计算归一化参数（只从时序数据中计算）
    # 时序特征: 按列计算均值和标准差
    seq_means = merged_df[seq_cols].mean()
    seq_stds = merged_df[seq_cols].std()
    seq_stds = seq_stds.replace(0, 1)  # 避免除以0

    # 静态特征: 同样归一化
    static_means = merged_df[static_cols].mean()
    static_stds = merged_df[static_cols].std()
    static_stds = static_stds.replace(0, 1)

    # 应用归一化
    for col in seq_cols:
        merged_df[col] = (merged_df[col] - seq_means[col]) / seq_stds[col]
    for col in static_cols:
        merged_df[col] = (merged_df[col] - static_means[col]) / static_stds[col]

    # 保存合并后的数据 (CSV 格式方便查看，parquet 格式方便快速读取)
    csv_path = output_dir / "merged.csv"
    merged_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  合并数据 (CSV) -> {csv_path}")

    parquet_path = output_dir / "merged.parquet"
    merged_df.to_parquet(parquet_path, index=False)
    print(f"  合并数据 (Parquet) -> {parquet_path}")

    # 保存时间区间边界
    bin_edges_path = output_dir / "time_bin_edges.npy"
    np.save(bin_edges_path, bin_edges)
    print(f"  时间区间边界 -> {bin_edges_path}")

    # 保存元数据
    meta = {
        'n_samples': len(merged_df),
        'n_unique_ids': merged_df['id'].nunique(),
        'n_seq_features': len(seq_cols),
        'n_static_features': len(static_cols),
        'seq_feature_cols': seq_cols,
        'static_feature_cols': static_cols,
        'time_col': feature_info['time_col'],
        'time_to_event_col': feature_info['time_to_event_col'],
        'event_indicator_col': feature_info['event_indicator_col'],
        'time_range': [float(merged_df['time_to_event'].min()),
                       float(merged_df['time_to_event'].max())],
        'event_rate': float(merged_df['event_indicator'].mean()),
        'num_time_bins': len(bin_edges) - 1,
        'time_bin_edges': bin_edges.tolist(),
        'seq_means': {k: float(v) for k, v in seq_means.items()},
        'seq_stds': {k: float(v) for k, v in seq_stds.items()},
        'static_means': {k: float(v) for k, v in static_means.items()},
        'static_stds': {k: float(v) for k, v in static_stds.items()},
        'data_files': {
            'csv': 'merged.csv',
            'parquet': 'merged.parquet',
            'time_bin_edges': 'time_bin_edges.npy'
        }
    }

    meta_path = output_dir / "metadata.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  元数据 -> {meta_path}")

    return meta


def main():
    print("=" * 60)
    print("CliTsSurv 数据预处理")
    print("=" * 60)
    
    # 路径设置
    data_dir = Path(__file__).parent
    output_dir = data_dir / "CliTsSurv_data"
    
    # 运行处理流程
    merged_df = load_and_merge_data(data_dir)
    merged_df = add_feature_engineering(merged_df)
    feature_info = split_feature_columns(merged_df)
    bin_edges = compute_time_bins(merged_df, num_bins=50)
    meta = normalize_and_save(merged_df, feature_info, bin_edges, output_dir)
    
    print("\n" + "=" * 60)
    print("预处理完成！")
    print(f"输出目录: {output_dir}")
    print(f"样本数: {meta['n_unique_ids']} 患者, {meta['n_samples']} 条记录")
    print(f"特征: {meta['n_seq_features']} 时序 + {meta['n_static_features']} 静态")
    print(f"事件率: {meta['event_rate']:.2%}")
    print("=" * 60)
    
    print("\n使用示例:")
    print("""
from models.CliTsSurv.datasets import CliTsSurvDataset, collate_fn
from torch.utils.data import DataLoader
import pandas as pd
import json

# 加载数据
meta = json.load(open('data/CliTsSurv_data/metadata.json'))
df = pd.read_parquet('data/CliTsSurv_data/merged.parquet')
time_bin_edges = meta['time_bin_edges']

# 创建 Dataset
dataset = CliTsSurvDataset(
    df=df,
    feature_cols=meta['seq_feature_cols'],
    time_to_event_col='time_to_event',
    event_indicator_col='event_indicator',
    time_col='date',
    static_feature_cols=meta['static_feature_cols'],
    time_embedding=True,
    embedding_dim=6
)

# 创建 DataLoader
dataloader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, infer=False, device='cuda')
)

# 训练
model.train_model(dataloader, ..., time_bin_edges=time_bin_edges)
    """)


if __name__ == "__main__":
    main()
