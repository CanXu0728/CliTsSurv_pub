# CliTsSurv 模型架构详解

> **模型定位**: 生存分析模型，融合时序动态特征与静态基线特征，采用 VAE 风格编码器进行特征压缩，支持 RNN/Transformer 双骨干架构与多种数据增强策略。

---

## 1. 架构总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CliTsSurv 整体架构                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入层                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                       │
│  │ 时序特征 x    │  │ 静态特征     │  │ 时间嵌入     │                       │
│  │ (B,T,K)      │  │ static_x     │  │ times        │                       │
│  │ 连续+二值    │  │ (B, S)       │  │ (B,T,Dt)     │                       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                       │
│         │                 │                 │                                │
│         ▼                 ▼                 │                                │
│  ┌─────────────────┐  ┌─────────────────┐  │                               │
│  │ 时序 LatentEncoder│  │静态 LatentEncoder│  │                            │
│  │  (VAE 编码器)     │  │  (VAE 编码器)     │  │                            │
│  │                 │  │                 │  │                               │
│  │  Encoder: MLP   │  │  Encoder: MLP   │  │                               │
│  │  Neck: z (latent)│  │  Neck: z_s      │  │                               │
│  │  Decoder: 重构    │  │  Decoder: 重构    │  │                               │
│  │  Discriminator  │  │  Discriminator  │  │                               │
│  └──────┬──────────┘  └──────┬──────────┘  │                               │
│         │                      │            │                               │
│         ▼                      │            ▼                               │
│  ┌─────────────────────────────────────────────┐                          │
│  │          序列拼接: [z, times]                │                          │
│  │              (B, T, neck_dim + time_dim)      │                          │
│  └──────────────────────┬────────────────────────┘                          │
│                         │                                                   │
│                         ▼                                                   │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │              CliTsSurv_backbone (时序骨干网络)                  │          │
│  │                                                               │          │
│  │   ┌──────────────┐   ┌─────────────────┐   ┌────────────┐  │          │
│  │   │   RNN 分支   │   │ Transformer 分支  │   │  Attention │  │          │
│  │   │ (GRU/LSTM)   │   │  (CLS Token)      │   │  (DDH风格) │  │          │
│  │   │              │   │                   │   │            │  │          │
│  │   │ pack_padded  │   │ input_proj + cls  │   │ hidden +   │  │          │
│  │   │   → RNN      │   │ → TransformerEnc  │   │ x_last 拼接 │  │          │
│  │   │   → unpack   │   │ → CLS output      │   │ → MLP打分  │  │          │
│  │   │              │   │                   │   │ → softmax  │  │          │
│  │   └──────┬───────┘   └────────┬────────┘   └─────┬──────┘  │          │
│  │          │                    │                  │          │          │
│  │          └────────────────────┴──────────────────┘          │          │
│  │                         │                                │          │
│  │                         ▼                                │          │
│  │              [hidden_attentive, x_last]                   │          │
│  │                   (B, hidden_size + feat)               │          │
│  └─────────────────────────┬─────────────────────────────────┘          │
│                            │                                              │
│                            ▼                                              │
│  ┌──────────────────────────────────────────────────────────────┐        │
│  │         特征融合: concat([backbone_out, static_z])            │        │
│  │              (B, hidden_size + feat + static_neck)           │        │
│  └────────────────────────────┬─────────────────────────────────┘        │
│                               │                                           │
│                               ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐       │
│  │              CliTsSurv_head (预测头)                   │       │
│  │                                                               │       │
│  │   MLP Hidden Layers → Linear(num_time_bins + 1) → Softmax    │       │
│  │                                                               │       │
│  │   输出: 离散时间概率分布                                       │       │
│  │   - bins[0..K-1]: 各时间区间的事件发生概率                    │       │
│  │   - bin[K]: 生存概率 (无事件发生)                             │       │
│  └──────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  损失函数 (多任务加权)                                                        │
│  ┌─────────────────────────────────────────────────────────────┐         │
│  │  L_total = L_deephit + β₁·L_vae_seq + β₂·L_vae_static       │         │
│  │            + γ·L_brier                                      │         │
│  │                                                             │         │
│  │  L_deephit = NLL + α·Ranking Loss                           │         │
│  │  L_vae     = MSE Reconstruction + β·GAN Loss                │         │
│  │  L_brier   = Integrated Brier Score                         │         │
│  └─────────────────────────────────────────────────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心模块详解

### 2.1 LatentEncoder — VAE 风格特征编码器

**文件**: `backbones/LatentEncoder.py`

为时序特征和静态特征分别提供独立的编码器。采用**编码-瓶颈-解码**架构，支持可选的对抗判别器进行隐空间正则化。

#### 2.1.1 组成结构

```
LatentEncoder
├── LatentEnc (Encoder)
│   ├── MLP Layers: feature_in → hidden_dims → neck_dim
│   └── 输出: latent vector z (neck_dim)
│
├── LatentDec (Decoder)
│   ├── MLP Layers: neck_dim → dec_dims
│   ├── conti_out_layer: 连续特征重构 (conti_feature_out)
│   └── bi_out_layer: 二值特征重构 (bi_feature_out, Sigmoid激活)
│
└── Discriminator (可选)
    ├── MLP: neck_dim → disc_dims → 1
    └── Sigmoid → 判别隐向量来自 Encoder 还是噪声
```

#### 2.1.2 关键参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `conti_feature_in` | int | 连续输入特征维度 |
| `bi_feature_in` | int | 二值输入特征维度 |
| `enc_dims` | list[int] | Encoder 隐藏层维度，最后一维为 neck_dim |
| `dec_dims` | list[int] | Decoder 隐藏层维度 |
| `disc_dims` | list[int] | 判别器隐藏层维度（空列表则禁用） |
| `beta` | float | GAN 损失权重 |

#### 2.1.3 损失函数

```python
loss = MSE(masked_reconstruction) + beta * BCE(discriminator_pred, label=1)
```

- **重构损失**: 对原始输入进行重构，支持 NaN 掩码（只计算非缺失值）
- **GAN 损失**: 判别器试图区分 Encoder 输出的隐向量和随机高斯噪声，Encoder 试图欺骗判别器

#### 2.1.4 在本模型中的使用

- **时序编码器** (`self.encoder`): 对 `(B×T, K)` 的展平时序特征编码，输出 `z` 形状 `(B, T, neck_dim)`
- **静态编码器** (`self.static_encoder`): 对 `(B, S)` 的静态特征编码，输出 `static_z` 形状 `(B, static_neck_dim)`

---

### 2.2 CliTsSurv_backbone — 时序骨干网络

**文件**: `backbones/CliTs_backbone.py`

处理编码后的时序隐向量，提取时间依赖性。支持三种模式：**GRU**、**LSTM**、**Transformer**。

#### 2.2.1 RNN 分支 (GRU/LSTM)

```
CliTs_backbone
├── pack_padded_sequence(x, lens)   # 变长序列打包
├── RNN (GRU/LSTM)
│   ├── input_size: neck_dim + time_dim
│   ├── hidden_size: configurable
│   ├── num_layers: configurable
│   └── batch_first=True
├── pad_packed_sequence()           # 解包回填充序列
└── 输出: r_out (B, T, hidden_size)
```

配合 **CliTs_attention** 使用：

```
CliTs_attention
├── 输入: hidden (B,T,H), x (B,T,F), lens (B,)
├── 构建 mask: 标记超过实际长度的位置
├── 提取 last observation: x_last (B,F)
├── 拼接 [hidden, x_last] → attention_net → attention score (B,T)
├── Softmax 归一化 (仅对有效位置)
├── 加权求和 → hidden_attentive (B,H)
└── 输出: [hidden_attentive, x_last] → (B, H+F)
```

**Attention 机制细节**:

- 对每个时间步，将 RNN 输出与最后观测值 `x_last` 拼接
- 通过 MLP 计算注意力分数
- 对超过实际序列长度的位置 mask 为 `-1e10`
- 仅对序列长度 > 1 的样本进行 softmax 归一化
- 单观测样本注意力为 0，依赖 `x_last`

#### 2.2.2 Transformer 分支

```
CliTs_transformer_backbone
├── input_proj: Linear(feature_in → hidden_size)
├── CLS Token: 可学习参数 (1, 1, hidden_size)
├── 拼接: [CLS, x_proj] → (B, T+1, hidden_size)
├── TransformerEncoder
│   ├── nhead: 多头注意力头数
│   ├── num_layers: Encoder 层数
│   ├── dim_feedforward: FFN 维度
│   └── dropout:  dropout 率
├── key_padding_mask: 标记填充位置
└── 输出: out[:, 0, :] → CLS token 输出 (B, hidden_size)
```

与 RNN 分支的差异：

- 不需要单独的 Attention 模块，全局自注意力天然处理时序依赖
- 使用 CLS token 聚合序列信息，而非 RNN last hidden + attention
- 为了统一输出维度，Transformer 分支也会提取 `x_last` 与 CLS 输出拼接

#### 2.2.3 统一输出接口

无论哪种骨干类型，`CliTsSurv_backbone.forward()` 的输出统一为：

```
(B, hidden_size + feature_in)
= concat([hidden_attentive 或 CLS_output, x_last])
```

---

### 2.3 CliTsSurv_head

**文件**: `CliTsSurv.py`

将融合后的特征映射到离散时间概率分布。

```
CliTsSurv_head
├── MLP Layers
│   ├── 输入: hidden_size + neck_dim + time_dim + static_neck_dim
│   └── 可选隐藏层: hidden_dims[] → ReLU
├── 输出层: Linear(last_hidden, num_time_bins + 1)
└── Softmax(dim=-1)
```

**输出语义**:

| 维度 | 含义 |
|------|------|
| `pred[:, 0:K]` | 在第 `k` 个时间区间发生事件的概率 |
| `pred[:, K]` | 在整个观测期间**不发生**事件的概率（生存概率）|

其中 `K = num_time_bins`。

---

### 2.4 CliTsSurv — 主模型类

**文件**: `CliTsSurv.py`

将所有模块组合为端到端可训练模型。

#### 2.4.1 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `conti_feature_in` | int | — | 时序连续特征维度 |
| `bi_feature_in` | int | — | 时序二值特征维度 |
| `static_conti_feature_in` | int | — | 静态连续特征维度 |
| `static_bi_feature_in` | int | — | 静态二值特征维度 |
| `time_dim` | int | — | 时间嵌入维度 |
| `encoder_args` | dict | — | 时序编码器配置 `{enc_dims, dec_dims, disc_dims, beta}` |
| `static_encoder_args` | dict | — | 静态编码器配置（同 encoder_args 结构） |
| `net_args` | dict | — | 骨干网络配置 `{hidden_size, num_layers, rnn_type, nhead, dim_feedforward, dropout}` |
| `head_args` | dict | — | 预测头配置 `{hidden_dims}` |
| `beta1` | float | 1.0 | 时序 VAE 损失权重 |
| `beta2` | float | 1.0 | 静态 VAE 损失权重 |
| `aug_configs` | list[dict] | `[]` | 数据增强配置列表 |
| `num_time_bins` | int | 10 | 离散时间区间数量 |
| `alpha` | float | 0.1 | ranking loss 权重 |
| `sigma` | float | 0.1 | Ranking loss 指数核带宽 |
| `gamma` | float | 0.0 | Brier loss 权重 |
| `device` | str | `'cuda'` | 计算设备 |

#### 2.4.2 前向传播流程

```python
def forward(self, x, static_x, lens, times, labels=None):
    # Step 1: 特征编码
    z, recon, disc_pred = self.encoder(x.view((B*T, K)))
    static_z, static_recon, static_disc_pred = self.static_encoder(static_x)

    # Step 2: 重构掩码与 reshape
    mask_recon = build_mask(lens, T, K)
    recon = recon.view((B, T, K)) * mask_recon
    z = z.view((B, T, neck_dim)) * mask

    # Step 3: 数据增强 (仅训练时)
    if training and augmenters:
        z, lens, times = augment(z, lens, times, labels)

    # Step 4: 时序建模
    z = concat([z, times], dim=-1)        # (B, T, neck_dim + time_dim)
    z = self.net(z, lens)                  # (B, hidden_size + feature_in)

    # Step 5: 静态特征融合
    z = concat([z, static_z], dim=-1)      # (B, ... + static_neck_dim)

    # Step 6: 预测
    out = self.head(z)                     # (B, num_time_bins + 1)

    return out, recon, disc_pred, static_recon, static_disc_pred
```

---

## 3. 数据增强模块

**文件**: `augmenters/seq_augmenters.py`

在**隐空间**（而非原始特征空间）执行数据增强，作用于编码器输出的 `z`。

| 增强器 | 参数 | 操作 | 训练效果 |
|--------|------|------|---------|
| `AddGaussianNoise` | `p=0.3` | 以概率 `p` 在随机时间步添加高斯噪声 | 提升鲁棒性 |
| `InnerInterpolate` | `p=0.3` | 以概率 `p` 在两个相邻时间步间线性插值一个新点，序列长度+1 | 数据平滑、密度增加 |
| `RandomRemove` | `p=0.3` | 以概率 `p` 随机删除一个时间步，序列长度-1 | 模拟缺失观测 |
| `RandomSlice` | `p=0.3` | 以概率 `p` 随机截取序列的一个子片段 | 模拟不同观察窗口 |

**设计要点**:

- 增强发生在隐空间 `z`，而非原始输入 `x`，避免破坏原始特征分布
- 同步修改 `times` 和 `lens`，保持时间对齐
- 仅在 `self.training == True` 时生效
- 通过 `aug_configs` 列表灵活配置，支持堆叠多种增强器

---

## 4. 损失函数详解

**文件**: `metrices.py`

### 4.1 DeepHit Loss (NLL + rank)

DeepHit 的核心损失，由两部分组成：

```
L_deephit = L_NLL + α · L_rank
```

#### 4.1.1 负对数似然 (NLL)

```python
# 事件样本: -log P(事件发生在确切的时间区间)
if event_mask.any():
    event_prob = pred[event_mask, time_bins[event_mask]]
    loss += -log(event_prob + 1e-8)

# 删失样本: -log P(生存时间超过删失时间)
if censored_mask.any():
    survival_prob = sum(pred[censored, bins > censored_time])
    loss += -log(survival_prob + 1e-8)
```

#### 4.1.2 Ranking Loss (Concordance)

```python
# 对每一对事件样本 (i, j):
# 如果 t_i < t_j，期望 F(t_i|x_i) > F(t_j|x_i)
# 即风险高的样本应该更早发生事件

cdf = cumsum(event_probs, dim=-1)  # F(t|x)
diff = F_i(t_i) - F_j(t_i)       # (n_events, batch)
rank_loss = sum(exp(-diff / sigma) * indicator(t_i < t_j)) / n_pairs
```

### 4.2 Brier Loss (Integrated Brier Score)

```python
# 对每个时间区间 k:
# - 存活样本 (T > k): target = 1
# - 事件样本 (T ≤ k, E=1): target = 0
# - 删失样本 (T ≤ k, E=0): 不参与计算

survival = 1.0 - cumsum(event_probs)  # S(k|x)
bs_per_bin = mean((target - survival)^2, valid_samples)
ibs = mean(bs_per_bin)
```

### 4.3 VAE Reconstruction Loss

```python
# 时序重构
loss_enc = MSE(recon * mask, x * mask) + beta * BCE(disc_pred, 1)

# 静态重构
loss_static_enc = MSE(static_recon * mask, static_x * mask) + beta * BCE(static_disc_pred, 1)
```

### 4.4 总损失

```python
L_total = L_deephit
        + β₁ · L_vae_seq
        + β₂ · L_vae_static
        + γ · L_brier
```

---

## 5. 数据集与数据流

### 5.1 数据集

**文件**: `datasets/CliTs_dataset.py`

#### 5.1.1 CliTsSurvDataset

| 参数 | 说明 |
|------|------|
| `df` | 输入 DataFrame（已合并时序 + 静态） |
| `feature_cols` | 时序特征列名列表 |
| `time_to_event_col` | 生存时间列名 |
| `event_indicator_col` | 事件指示列名（1=事件发生，0=删失） |
| `time_col` | 时间戳列名 |
| `static_feature_cols` | 静态特征列名列表 |
| `time_embedding` | 是否对时间戳做位置编码 |
| `embedding_dim` | 位置编码维度 |
| `incremental` | 是否生成增量序列（[x[:1], x[:2], ..., x[:T]]） |
| `balance` | 测试集是否平衡事件/删失样本 |

**特殊处理**:

- 如果 `time_to_event > last_follow_up_time`，追加一个全 0 特征行和时间戳，确保模型能看到事件发生时刻
- 支持增量模式（`incremental=True`），为每个患者生成 T 个不同长度的序列
- 测试集 `balance=True` 时，随机下采样删失样本以匹配事件样本数量

#### 5.1.2 collate_fn

处理变长序列的批处理：

```python
# 输入: list of (x, static_x, time_to_event, event_indicator, times)
# 输出: (X, static_X, time_to_event, event_indicator, lengths, times_tensor, mask_nan, mask_nan_static)

# 关键操作:
# 1. 计算 batch 内最大序列长度
# 2. 对短序列零填充
# 3. 生成 NaN 掩码，用 fillna 填充
# 4. 所有 tensor 转移到指定 device
```

### 5.2 输入数据格式

| 张量 | 形状 | 说明 |
|------|------|------|
| `x` | `(B, T, K)` | 时序特征（连续+二值拼接） |
| `static_x` | `(B, S)` | 静态特征 |
| `time_to_event` | `(B,)` | 连续生存时间（月） |
| `event_indicator` | `(B,)` | 1=事件发生，0=删失 |
| `lens` | `(B,)` | 每个样本的实际序列长度 |
| `times` | `(B, T, Dt)` | 时间嵌入（位置编码） |
| `mask_nan` | `(B, T, K)` | 时序特征 NaN 位置掩码 |
| `mask_nan_static` | `(B, S)` | 静态特征 NaN 位置掩码 |

---

## 6. 训练流程

### 6.1 训练循环

```python
for epoch in range(epochs):
    for batch in train_dataloader:
        # 1. 前向传播
        pred, recon, disc_pred, static_recon, static_disc_pred = model.forward(...)

        # 2. 连续时间离散化
        time_bins = searchsorted(time_bin_edges, time_to_event).clamp(0, K-1)

        # 3. 计算多任务损失
        loss = L_deephit + β₁·L_vae_seq + β₂·L_vae_static + γ·L_brier

        # 4. 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 验证
    cindex, val_loss, val_survival_loss, val_brier = model.test_model(...)

    # 早停策略: 基于综合得分
    score = mean([cindex, 1 - val_brier])
    if score > best_score:
        save_checkpoint('best.pt')
```

### 6.2 优化器配置

支持三种优化器，通过 `optimizer_type` 指定：

| 优化器 | 参数 | 适用场景 |
|--------|------|---------|
| Adam | `lr` | 默认，通用场景 |
| AdamW | `lr`, `weight_decay` | 需要权重衰减的正则化 |
| SGD | `lr`, `momentum` | 大规模数据，需要精细调参 |

参数分组：

```python
param_groups = [
    {'params': encoder.enc},       # 时序编码器
    {'params': encoder.dec},       # 时序解码器
    {'params': static_encoder.enc}, # 静态编码器
    {'params': static_encoder.dec}, # 静态解码器
    {'params': net},               # 骨干网络
    {'params': head},              # 预测头
]
```

### 6.3 模型选择策略

```python
score = mean([cindex, 1 - val_brier_loss])
```

同时优化**区分能力**（C-index 越高越好）和**概率校准**（Brier score 越低越好）。

---

## 7. 预测与评估

### 7.1 预测生存函数

```python
survival_probs = model.predict_survival(x, t, static_x, lens, times)
# 返回: (n_samples, len(t))
```

**步骤**:

1. 将连续时间 `t` 通过 `time_bin_edges` 转换为区间索引 `t_bins`
2. 分 batch 前向传播
3. 计算累积事件概率 `F(t) = cumsum(pred[:, :-1], dim=-1)`
4. 生存概率 `S(t) = 1 - F(t)`
5. 在 `t_bins` 指定位置提取 `S(t)`

### 7.2 预测风险

```python
risk_probs = model.predict_risk(x, t, ...)
# risk(t) = 1 - S(t) = F(t)
```

### 7.3 评估指标

| 指标 | 说明 | 计算方式 |
|------|------|---------|
| **C-index** | 一致性指数，衡量模型区分能力 | `sksurv.metrics.concordance_index_censored` |
| **Brier Score** | 概率校准度，衡量预测概率的准确性 | `metrices.brier_loss` |
| **IBS** | Integrated Brier Score，Brier 的全时间平均 | 在 `test_model` 中计算 |

---

## 8. 文件结构与模块依赖

```
models/CliTsSurv/
├── __init__.py                          # 模块入口（空）
│
├── CliTsSurv.py                         # 主模型 + 预测头 + 训练/测试/预测逻辑
│   ├── class CliTsSurv                  # 主模型
│   │   ├── __init__                     # 组装编码器、骨干、头、增强器
│   │   ├── loss_fn                      # 多任务损失计算
│   │   ├── build_mask                   # 变长序列掩码构建
│   │   ├── augment                      # 数据增强编排
│   │   ├── forward                      # 前向传播
│   │   ├── train_model                  # 训练循环
│   │   ├── test_model                   # 验证评估
│   │   ├── predict_survival             # 生存函数预测
│   │   ├── predict_risk                 # 风险预测
│   │   └── _preprocess_test_data        # 测试数据预处理
│   └── class CliTsSurv_head             # 预测头
│
├── metrices.py                          # 损失函数与评估指标
│   ├── deephit_loss                     # NLL + Ranking
│   ├── brier_loss                       # Integrated Brier Score
│   └── c_index                          # 手动实现的 C-index
│
├── backbones/
│   ├── __init__.py
│   ├── LatentEncoder.py                 # VAE 编码器
│   │   ├── class LatentEnc              # MLP 编码器
│   │   ├── class LatentDec              # MLP 解码器（支持连续+二值输出）
│   │   ├── class Discriminator          # 对抗判别器
│   │   └── class LatentEncoder          # 组合封装 + 损失计算
│   └── CliTs_backbone.py                # 时序骨干网络
│       ├── class CliTs_attention        # DDH 风格注意力
│       ├── class CliTs_backbone         # RNN 分支 (GRU/LSTM)
│       ├── class CliTs_transformer_backbone  # Transformer 分支
│       └── class CliTsSurv_backbone     # 统一接口封装
│
├── augmenters/
│   ├── __init__.py
│   └── seq_augmenters.py                # 隐空间序列增强
│       ├── AddGaussianNoise             # 高斯噪声
│       ├── InnerInterpolate             # 内插值
│       ├── RandomRemove                 # 随机删除
│       └── RandomSlice                  # 随机切片
│
└── datasets/
    ├── __init__.py
    ├── CliTs_dataset.py                 # DeepHit 版本 Dataset
    │   └── class CliTsSurvDataset
    └── utils.py                         # 数据工具函数
        ├── positional_encoding          # 正弦/余弦位置编码
        ├── collate_fn                   # 变长序列批处理
        ├── collate_fn_latentenc         # 预训练编码器用 collate
        ├── norm_df_ignorena             # 忽略 NaN 的归一化
        └── normalize_df                 # 标准归一化
```

---

## 9. 设计亮点与特色

### 9.1 双编码器架构

- **时序编码器**: 对每个时间步独立编码，捕获局部特征模式
- **静态编码器**: 对整个患者的静态属性编码，提供人口统计学/基线信息
- 两者均为 VAE 结构，隐空间具有可解释性和正则化效果

### 9.2 隐空间数据增强

- 在压缩后的隐空间操作，避免破坏原始特征语义
- 增强器可堆叠、可配置，适应不同数据特性
- 同步处理时间对齐和长度变化

### 9.3 多任务损失平衡

- DeepHit 损失确保生存分析核心能力
- VAE 重构损失作为正则化，防止过拟合
- Brier loss 提升概率校准度
- 通过 `beta1`, `beta2`, `gamma` 灵活调整各损失权重

### 9.4 RNN / Transformer 双骨干

- **RNN + Attention**: 适合短序列、数据量有限的场景，参数量小
- **Transformer**: 适合长序列、数据量充足的场景，全局依赖捕获能力强
- 统一接口，切换只需改 `rnn_type` 参数

### 9.5 与 DDH 对齐

- Attention 机制参照 DDH 实现
- `predict_survival` / `predict_risk` API 与 DDH 保持一致
- 时间离散化逻辑兼容 DDH 的 `discretize` 方法

---

## 10. 配置示例

```python
model = CliTsSurv(
    # 特征维度
    conti_feature_in=46,          # 时序连续特征
    bi_feature_in=0,              # 时序二值特征
    static_conti_feature_in=7,    # 静态连续特征
    static_bi_feature_in=0,       # 静态二值特征
    time_dim=10,                  # 时间嵌入维度

    # 时序编码器: 46 → 64 → 32 → 16 (neck)
    encoder_args={
        'enc_dims': [64, 32, 16],
        'dec_dims': [32, 64],
        'disc_dims': [32, 64, 32],  # 启用判别器
        'beta': 1.0
    },

    # 静态编码器: 7 → 32 → 8 (neck)
    static_encoder_args={
        'enc_dims': [32, 8],
        'dec_dims': [32],
        'disc_dims': [],             # 禁用判别器
        'beta': 1.0
    },

    # 骨干网络: GRU
    net_args={
        'hidden_size': 32,
        'num_layers': 2,
        'rnn_type': 'gru'            # 或 'lstm', 'transformer'
    },

    # 预测头
    head_args={
        'hidden_dims': [16]
    },

    # loss 参数
    num_time_bins=10,
    alpha=0.1,        # ranking loss 权重
    sigma=0.1,        # ranking kernel 带宽
    gamma=0.0,        # brier loss 权重 (0=禁用)

    # 多任务权重
    beta1=1.0,        # 时序重构权重
    beta2=1.0,        # 静态重构权重

    # 数据增强
    aug_configs=[
        {'type': 'AddGaussianNoise', 'args': {'p': 0.3}},
        {'type': 'RandomSlice', 'args': {'p': 0.3}}
    ],

    device='cuda'
)
```

---

*文档版本: 1.0 | 基于 CliTsSurv 代码库自动生成*
