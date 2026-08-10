"""
scripts/utils.py

统一生存分析评估工具。

主要接口
--------
evaluate_survival_metrics(t_train, e_train, t_eval, e_eval,
                           risk_scores, survival_probs, eval_times, ...)
    -> dict  键名格式: '{horizon}_ipcw', '{horizon}_brier', '{horizon}_auc', 'mean_auc'

interpolate_survival_at_times(surv_curves, duration_index, eval_times)
    -> (survival_probs, risk_scores)
    用于 SurvTRACE / SurvTimeSurvival 等离散时间模型，将其 predict_surv 输出
    插值到指定评估时间点。
"""

import numpy as np
from sksurv.metrics import (
    concordance_index_ipcw,
    concordance_index_censored,
    brier_score,
    integrated_brier_score,
    cumulative_dynamic_auc,
)


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _make_et(t, e):
    """把 (time, event) numpy 数组打包成 sksurv 结构化数组。"""
    t = np.asarray(t, dtype=float)
    e = np.asarray(e, dtype=bool)
    return np.array(list(zip(e, t)), dtype=[('e', bool), ('t', float)])


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def evaluate_survival_metrics(
    t_train,
    e_train,
    t_eval,
    e_eval,
    risk_scores,
    survival_probs,
    eval_times,
    horizons=None,
    filter_by_train_max=True,
    print_results=True,
    split_name='Eval',
):
    """
    统一生存模型评估（基于 sksurv.metrics）。

    Parameters
    ----------
    t_train, e_train : array-like, shape (N_train,)
        训练集的观测时间和事件指示（0/1 或 bool）。
    t_eval, e_eval : array-like, shape (N_eval,)
        评估集（验证集或测试集）的观测时间和事件指示。
    risk_scores : ndarray, shape (N_eval, n_times)
        每个评估时间点的累积风险评分（值越高 = 风险越高）。
    survival_probs : ndarray, shape (N_eval, n_times)
        每个评估时间点的生存概率（值越高 = 风险越低）。
    eval_times : array-like, shape (n_times,)
        要计算指标的时间点。
    horizons : list of float, optional
        分位数列表，如 [0.25, 0.5, 0.75]，用作返回 dict 的键名前缀。
        若为 None，则以位置索引（'0', '1', ...）作为前缀。
    filter_by_train_max : bool
        若为 True，则过滤掉 t_eval > max(t_train) 的样本再计算 Brier Score，
        以满足 sksurv 的 IPCW 要求。
        注意：C-index 和 AUC 始终使用原始 et_eval（sksurv 内部会处理）。
    print_results : bool
        是否打印格式化结果表格。
    split_name : str
        打印时显示的数据集名称，如 'Validation' 或 'Test'。

    Returns
    -------
    dict
        键名格式：
            '{h}_ipcw'   - concordance_index_ipcw
            '{h}_brier'  - brier_score
            '{h}_auc'    - cumulative_dynamic_auc
            'mean_auc'   - 各时间点 AUC 的均值
            'mean_ipcw'  - 各时间点 IPCW C-index 的均值
            'overall_ci' - Harrell's C-index（全局，concordance_index_censored）
            'ibs'        - Integrated Brier Score（时间加权积分）
        其中 h = str(horizons[i]) 或 str(i)。
    """
    t_train = np.asarray(t_train, dtype=float)
    e_train = np.asarray(e_train, dtype=bool)
    t_eval  = np.asarray(t_eval,  dtype=float)
    e_eval  = np.asarray(e_eval,  dtype=bool)
    eval_times = np.asarray(eval_times, dtype=float)
    n_times = len(eval_times)

    if horizons is None:
        keys = [str(i) for i in range(n_times)]
    else:
        keys = [str(h) for h in horizons]

    et_train = _make_et(t_train, e_train)
    et_eval  = _make_et(t_eval,  e_eval)

    # 将评估时间点裁剪到训练集事件时间范围内（sksurv IPCW 要求）
    t_max_train = float(t_train.max())
    eval_times_clipped = np.clip(eval_times, None, t_max_train * 0.999)

    # Brier Score：可选地过滤超出训练集最大时间的评估样本
    if filter_by_train_max:
        keep = t_eval <= t_max_train
        et_eval_brier  = et_eval[keep]
        surv_brier     = survival_probs[keep]
    else:
        et_eval_brier  = et_eval
        surv_brier     = survival_probs

    try:
        _, brier_scores = brier_score(
            et_train, et_eval_brier, surv_brier, eval_times_clipped
        )
    except Exception:
        brier_scores = [None] * n_times

    results = {}
    cis, aucs, brs = [], [], []

    for i, (key, t) in enumerate(zip(keys, eval_times_clipped)):
        try:
            ci = concordance_index_ipcw(
                et_train, et_eval, risk_scores[:, i], t
            )[0]
        except Exception:
            ci = None

        try:
            auc = cumulative_dynamic_auc(
                et_train, et_eval, risk_scores[:, i], t
            )[0][0]
        except Exception:
            auc = None

        bs = brier_scores[i] if not isinstance(brier_scores[i], type(None)) else None

        results[f'{key}_ipcw']  = ci
        results[f'{key}_brier'] = bs
        results[f'{key}_auc']   = auc
        cis.append(ci)
        aucs.append(auc)
        brs.append(bs)

    valid_aucs = [a for a in aucs if a is not None]
    valid_cis  = [c for c in cis  if c is not None]
    results['mean_auc']  = float(np.mean(valid_aucs)) if valid_aucs else None
    results['mean_ipcw'] = float(np.mean(valid_cis))  if valid_cis  else None

    # Overall Harrell's C-index（单一风险分，不依赖特定时间点）
    try:
        mean_risk = risk_scores.mean(axis=1)
        overall_ci = concordance_index_censored(e_eval, t_eval, mean_risk)[0]
    except Exception:
        overall_ci = None
    results['overall_ci'] = overall_ci

    # Integrated Brier Score（对给定时间点做梯形积分，不同于 3 点均值）
    try:
        ibs = integrated_brier_score(
            et_train, et_eval_brier, surv_brier, eval_times_clipped
        )
    except Exception:
        ibs = None
    results['ibs'] = ibs

    if print_results:
        horizon_labels = horizons if horizons is not None else list(range(n_times))
        print(f"\n{'='*62}")
        print(f"  {split_name}")
        print(f"{'='*62}")
        print(f"  {'分位数':<8} {'时间':>8}  {'C-index(IPCW)':>14}  {'Brier Score':>12}  {'AUC':>10}")
        print(f"  {'-'*58}")
        for i, (h, t) in enumerate(zip(horizon_labels, eval_times)):
            ci_s  = f"{cis[i]:.4f}"  if cis[i]  is not None else "N/A"
            bs_s  = f"{brs[i]:.4f}"  if brs[i]  is not None else "N/A"
            auc_s = f"{aucs[i]:.4f}" if aucs[i] is not None else "N/A"
            print(f"  {str(h):<8} {t:>8.2f}  {ci_s:>14}  {bs_s:>12}  {auc_s:>10}")
        print(f"  {'-'*58}")
        mean_auc_s = f"{results['mean_auc']:.4f}" if results['mean_auc'] is not None else "N/A"
        print(f"  {'Mean':<8} {'':>8}  {'':>14}  {'':>12}  {mean_auc_s:>10}")
        print(f"  {'-'*58}")
        overall_ci_s = f"{overall_ci:.4f}" if overall_ci is not None else "N/A"
        ibs_s        = f"{ibs:.4f}"        if ibs        is not None else "N/A"
        print(f"  {'Overall':<8} {'':>8}  {overall_ci_s:>14}  {ibs_s:>12}  {'':>10}")
        print(f"  {'':43}  (Harrell C)   (IBS)")

    return results


def interpolate_survival_at_times(surv_curves, duration_index, eval_times):
    """
    将 SurvTRACE / SurvTimeSurvival 式离散时间模型的生存曲线，
    插值（阶梯函数）到指定评估时间点。

    Parameters
    ----------
    surv_curves : ndarray, shape (N, K+1)
        模型 predict_surv 输出（经 pad_col 在开头添加了 1 列）。
        第 0 列 = S(0) = 1，第 i 列 (i>=1) = S after i-th time bin。
    duration_index : array-like, shape (K+1,)
        LabelTransform 的 cuts 数组，对应 K+1 个时间断点。
        第 i 列对应于 duration_index[i]（即 surv[:,i] = S(duration_index[i])）。
    eval_times : array-like, shape (n_eval,)
        目标评估时间点。

    Returns
    -------
    survival_probs : ndarray, shape (N, n_eval)
    risk_scores    : ndarray, shape (N, n_eval)
        risk_scores = 1 - survival_probs
    """
    duration_index = np.asarray(duration_index, dtype=float)
    eval_times     = np.asarray(eval_times,     dtype=float)

    # 列 i -> t = duration_index[i]
    # surv[:,i] = S(duration_index[i])，与 pad_col 在开头插入 S(0)=1 后的对齐方式一致
    time_grid = np.asarray(duration_index, dtype=float)

    N      = surv_curves.shape[0]
    n_eval = len(eval_times)
    survival_probs = np.zeros((N, n_eval), dtype=float)

    for j, t in enumerate(eval_times):
        # 阶梯函数：取最后一个 <= t 的 time_grid 索引
        idx = int(np.searchsorted(time_grid, t, side='right')) - 1
        idx = max(0, min(idx, len(time_grid) - 1))
        survival_probs[:, j] = surv_curves[:, idx]

    risk_scores = 1.0 - survival_probs
    return survival_probs, risk_scores
