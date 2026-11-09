"""Shared survival-evaluation utilities.

Includes structured outcome conversion, horizon-specific metrics and
step-function sampling of discrete survival curves."""

import os
import random
from pathlib import Path

import numpy as np
import torch
from sksurv.metrics import (
    concordance_index_ipcw,
    concordance_index_censored,
    brier_score,
    integrated_brier_score,
    cumulative_dynamic_auc,
)


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------

def _make_et(t, e):
    """Convert time and event arrays to the structured format expected by scikit-survival."""
    t = np.asarray(t, dtype=float)
    e = np.asarray(e, dtype=bool)
    return np.array(list(zip(e, t)), dtype=[('e', bool), ('t', float)])


# ---------------------------------------------------------------------------
# Public interface.
# ---------------------------------------------------------------------------

def set_global_seed(seed):
    """Seed Python, NumPy and PyTorch and return a DataLoader generator."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def checkpoint_dir(model_name, seed):
    """Return the standard persistent checkpoint directory for one run."""
    path = Path(__file__).resolve().parents[1] / "checkpoints" / model_name / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def split_patient_indices(ids, seed, test_ratio=0.15, val_ratio=0.15):
    """Canonical patient-ID split shared by every model.

    Sorting before permutation makes the split independent of preprocessing
    row order.  Returned indices preserve the local dataset order.
    """
    ids = np.asarray(ids)
    if len(np.unique(ids)) != len(ids):
        raise ValueError('split_patient_indices expects one row per patient')
    canonical = np.sort(ids.copy())
    rng = np.random.RandomState(seed)
    shuffled = canonical[rng.permutation(len(canonical))]
    n_test = int(len(shuffled) * test_ratio)
    n_val = int(len(shuffled) * val_ratio)
    test_ids = set(shuffled[:n_test].tolist())
    val_ids = set(shuffled[n_test:n_test + n_val].tolist())
    train_ids = set(shuffled[n_test + n_val:].tolist())
    train_idx = np.flatnonzero(np.isin(ids, list(train_ids)))
    val_idx = np.flatnonzero(np.isin(ids, list(val_ids)))
    test_idx = np.flatnonzero(np.isin(ids, list(test_ids)))
    return train_idx, val_idx, test_idx


def save_torch_checkpoint(model_name, seed, model_state_dict, **metadata):
    """Persist a model state and the minimum metadata needed for inference."""
    path = checkpoint_dir(model_name, seed) / "best.pt"
    torch.save({
        "model_name": model_name,
        "seed": int(seed),
        "model_state_dict": model_state_dict,
        **metadata,
    }, path)
    return path


def load_torch_checkpoint(model_name, seed, map_location="cpu"):
    """Load a checkpoint written by :func:`save_torch_checkpoint`."""
    path = checkpoint_dir(model_name, seed) / "best.pt"
    return torch.load(path, map_location=map_location, weights_only=True)

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
    observation_times_train=None,
    observation_times_eval=None,
):
    """Evaluate survival predictions with scikit-survival metrics.

Training and evaluation outcomes are one-dimensional time/event arrays.
Risk scores and survival probabilities have shape (N_eval, n_times); higher
risk scores indicate greater risk. eval_times specifies the horizons, and
horizons optionally supplies labels for result keys.

Returns horizon-specific IPCW concordance, Brier scores and dynamic AUC,
plus mean_ipcw, mean_auc, overall_ci (Harrell concordance) and ibs.
filter_by_train_max controls filtering for Brier evaluation. print_results
and split_name control console reporting."""
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

    if (observation_times_train is None) != (observation_times_eval is None):
        raise ValueError(
            "observation_times_train and observation_times_eval must be supplied together"
        )
    masked_dynamic_evaluation = observation_times_eval is not None
    if masked_dynamic_evaluation:
        observation_times_train = np.asarray(observation_times_train, dtype=float)
        observation_times_eval = np.asarray(observation_times_eval, dtype=float)
        if len(observation_times_train) != len(t_train):
            raise ValueError("observation_times_train length does not match t_train")
        if len(observation_times_eval) != len(t_eval):
            raise ValueError("observation_times_eval length does not match t_eval")

    # Clip evaluation horizons to the training-event time range.
    t_max_train = float(t_train.max())
    eval_times_clipped = np.clip(eval_times, None, t_max_train * 0.999)

    # Legacy full-cohort Brier calculation.  For dynamic evaluation the risk
    # set differs at every absolute time, so Brier is computed inside the loop.
    if not masked_dynamic_evaluation:
        if filter_by_train_max:
            keep = t_eval <= t_max_train
            et_eval_brier = et_eval[keep]
            surv_brier = survival_probs[keep]
        else:
            et_eval_brier = et_eval
            surv_brier = survival_probs
        try:
            _, brier_scores = brier_score(
                et_train, et_eval_brier, surv_brier, eval_times_clipped
            )
        except Exception:
            brier_scores = [None] * n_times
    else:
        brier_scores = [None] * n_times

    results = {}
    cis, aucs, brs = [], [], []

    for i, (key, t) in enumerate(zip(keys, eval_times_clipped)):
        if masked_dynamic_evaluation:
            # A prediction at absolute time t is evaluable only when it was
            # issued strictly before t.  t <= a is already-observed survival.
            train_keep = observation_times_train < t
            eval_keep = observation_times_eval < t
            et_train_i = et_train[train_keep]
            et_eval_i = et_eval[eval_keep]
            risk_i = risk_scores[eval_keep, i]
            results[f'{key}_n'] = int(eval_keep.sum())
        else:
            et_train_i = et_train
            et_eval_i = et_eval
            risk_i = risk_scores[:, i]
            results[f'{key}_n'] = int(len(et_eval))

        try:
            ci = concordance_index_ipcw(
                et_train_i, et_eval_i, risk_i, t
            )[0]
        except Exception:
            ci = None

        try:
            auc = cumulative_dynamic_auc(
                et_train_i, et_eval_i, risk_i, t
            )[0][0]
        except Exception:
            auc = None

        if masked_dynamic_evaluation:
            try:
                brier_keep = eval_keep.copy()
                if filter_by_train_max:
                    train_max_i = float(t_train[train_keep].max())
                    brier_keep &= t_eval <= train_max_i
                _, bs_arr = brier_score(
                    et_train_i,
                    et_eval[brier_keep],
                    survival_probs[brier_keep, i:i + 1],
                    np.asarray([t], dtype=float),
                )
                bs = float(bs_arr[0])
            except Exception:
                bs = None
        else:
            bs = brier_scores[i] if brier_scores[i] is not None else None

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

    # Overall Harrell's C-index.  Under dynamic masking, use the common risk
    # set that is eligible at every reported time so no known pre-a outcomes
    # enter the summary.
    try:
        if masked_dynamic_evaluation:
            common_keep = observation_times_eval < float(eval_times_clipped.min())
        else:
            common_keep = np.ones(len(t_eval), dtype=bool)
        mean_risk = risk_scores[common_keep].mean(axis=1)
        overall_ci = concordance_index_censored(
            e_eval[common_keep], t_eval[common_keep], mean_risk
        )[0]
    except Exception:
        overall_ci = None
    results['overall_ci'] = overall_ci

    # With a time-varying eligible cohort, integrate the independently
    # estimated masked Brier scores.  The legacy path retains sksurv's IBS.
    if masked_dynamic_evaluation:
        try:
            valid = np.asarray([b is not None for b in brs], dtype=bool)
            bt = eval_times_clipped[valid]
            bv = np.asarray([b for b in brs if b is not None], dtype=float)
            ibs = (float(np.trapz(bv, bt) / (bt[-1] - bt[0]))
                   if len(bt) >= 2 and bt[-1] > bt[0] else None)
        except Exception:
            ibs = None
    else:
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


def observation_times_for_ids(ids, seq_path=None, static_path=None):
    """Return the last real observation time ``a`` for patient IDs."""
    import pandas as pd

    if seq_path is None:
        seq_path = Path(__file__).resolve().parents[1] / 'data' / 'seq.csv'
    if static_path is None:
        static_path = Path(__file__).resolve().parents[1] / 'data' / 'static.csv'
    seq = pd.read_csv(seq_path, usecols=['id', 'date'])
    static = pd.read_csv(static_path, usecols=['id', 'survival_months'])
    outcome = static.drop_duplicates('id').set_index('id')['survival_months']
    seq['__T'] = seq['id'].map(outcome)
    seq = seq[seq['date'].notna() & seq['__T'].notna() & (seq['date'] < seq['__T'])]
    last = seq.groupby('id', sort=False)['date'].max()
    values = pd.Series(np.asarray(ids)).map(last)
    if values.isna().any():
        missing = pd.Series(np.asarray(ids))[values.isna()].head(5).tolist()
        raise KeyError(f'missing observation times for IDs: {missing}')
    return values.to_numpy(dtype=float)


def outcomes_for_ids(ids, static_path=None):
    """Return continuous absolute T and death indicator in patient-ID order."""
    import pandas as pd

    if static_path is None:
        static_path = Path(__file__).resolve().parents[1] / 'data' / 'static.csv'
    static = pd.read_csv(
        static_path, usecols=['id', 'survival_months', 'survival_status']
    ).drop_duplicates('id').set_index('id')
    ordered = static.reindex(np.asarray(ids))
    if ordered[['survival_months', 'survival_status']].isna().any().any():
        raise KeyError('missing survival outcomes for one or more patient IDs')
    t = ordered['survival_months'].to_numpy(dtype=float)
    e = (ordered['survival_status'].to_numpy() == 1).astype(int)
    return t, e


def survival_at_individual_times(surv_curves, time_grid, observation_times):
    """Step-function lookup of each subject's S(a_i) on a common grid."""
    surv_curves = np.asarray(surv_curves, dtype=float)
    time_grid = np.asarray(time_grid, dtype=float)
    observation_times = np.asarray(observation_times, dtype=float)
    if surv_curves.shape[0] != len(observation_times):
        raise ValueError('surv_curves and observation_times have different lengths')
    idx = np.searchsorted(time_grid, observation_times, side='right') - 1
    idx = np.clip(idx, 0, len(time_grid) - 1)
    return surv_curves[np.arange(len(observation_times)), idx]


def _pc_survival_lookup(surv_curves, time_grid, query_times, individual=False):
    """Interpolate cumulative hazard, not survival, within PC-hazard intervals.

    Boundary curves include S(0)=1. Queries outside the grid are clamped to
    its endpoints; no tail hazard extrapolation is assumed.
    """
    curves = np.asarray(surv_curves, dtype=float)
    grid = np.asarray(time_grid, dtype=float)
    query = np.asarray(query_times, dtype=float)
    if (grid.ndim != 1 or len(grid) < 2 or not np.all(np.isfinite(grid))
            or grid[0] != 0 or np.any(np.diff(grid) <= 0)):
        raise ValueError('PC-hazard grid must start at zero and strictly increase')
    if (curves.ndim != 2 or curves.shape[1] != len(grid)
            or not np.all(np.isfinite(curves)) or np.any(curves < 0)
            or np.any(curves > 1) or np.any(np.diff(curves, axis=1) > 1e-7)
            or not np.allclose(curves[:, 0], 1)):
        raise ValueError('invalid PC-hazard boundary survival curves')
    if query.ndim != 1 or not np.all(np.isfinite(query)):
        raise ValueError('query times must be a finite vector')
    if individual and len(query) != len(curves):
        raise ValueError('one query time is required per subject')
    query = np.clip(query, grid[0], grid[-1])
    left = np.clip(np.searchsorted(grid, query, side='right') - 1, 0, len(grid)-2)
    weight = (query-grid[left]) / (grid[left+1]-grid[left])
    cumulative = -np.log(np.clip(curves, np.finfo(float).tiny, 1))
    if individual:
        rows = np.arange(len(curves))
        h = (1-weight)*cumulative[rows, left] + weight*cumulative[rows, left+1]
    else:
        h = (1-weight)*cumulative[:, left] + weight*cumulative[:, left+1]
    return np.exp(-h)


def pc_survival_at_times(surv_curves, time_grid, eval_times):
    survival = _pc_survival_lookup(surv_curves, time_grid, eval_times)
    return survival, 1-survival


def pc_survival_at_individual_times(surv_curves, time_grid, observation_times):
    return _pc_survival_lookup(surv_curves, time_grid, observation_times, individual=True)


def conditionalize_survival(survival_probs, survival_at_observation,
                            eval_times, observation_times, eps=1e-8):
    """Convert absolute S(t) to S(t)/S(a), retaining absolute query times."""
    survival_probs = np.asarray(survival_probs, dtype=float)
    denom = np.maximum(np.asarray(survival_at_observation, dtype=float), eps)
    conditional = np.clip(survival_probs / denom[:, None], 0.0, 1.0)
    before_or_at_a = (np.asarray(eval_times, dtype=float)[None, :]
                      <= np.asarray(observation_times, dtype=float)[:, None])
    conditional[before_or_at_a] = 1.0
    return conditional


def evaluate_two_prediction_protocols(
    *, t_train, e_train, a_train, t_eval, e_eval, a_eval,
    native_risk_scores, native_survival_probs, eval_times, horizons,
    survival_at_observation=None, native_is_conditional=False,
    print_results=False, split_name='Eval',
):
    """Evaluate native and uniformly conditional predictions on identical risk sets.

    Both protocols exclude subjects with ``t <= a`` at each absolute evaluation
    time.  If a model already emits conditional predictions (DDH/CliTsSurv),
    protocol 2 reuses them exactly rather than conditioning twice.
    """
    common = dict(
        t_train=t_train, e_train=e_train,
        t_eval=t_eval, e_eval=e_eval,
        eval_times=eval_times, horizons=horizons,
        observation_times_train=a_train,
        observation_times_eval=a_eval,
        print_results=print_results,
    )
    native = evaluate_survival_metrics(
        risk_scores=native_risk_scores,
        survival_probs=native_survival_probs,
        split_name=f'{split_name} / native+masked',
        **common,
    )

    if native_is_conditional:
        conditional_survival = np.asarray(native_survival_probs, dtype=float)
        conditional_risk = np.asarray(native_risk_scores, dtype=float)
    else:
        if survival_at_observation is None:
            raise ValueError('absolute predictions require survival_at_observation')
        conditional_survival = conditionalize_survival(
            native_survival_probs, survival_at_observation,
            eval_times, a_eval,
        )
        conditional_risk = 1.0 - conditional_survival

    conditional = evaluate_survival_metrics(
        risk_scores=conditional_risk,
        survival_probs=conditional_survival,
        split_name=f'{split_name} / conditional+masked',
        **common,
    )
    return native, conditional


def interpolate_survival_at_times(surv_curves, duration_index, eval_times):
    """Sample discrete survival curves using a step function.

Each column of surv_curves corresponds to duration_index, including the
leading S(0)=1 column. Return survival probabilities and risk scores at
eval_times, both with shape (N, n_eval); risk equals 1 minus survival."""
    duration_index = np.asarray(duration_index, dtype=float)
    eval_times     = np.asarray(eval_times,     dtype=float)

    # Column i corresponds to duration_index[i].
    # surv[:, i] = S(duration_index[i]), including the leading S(0)=1 column.
    time_grid = np.asarray(duration_index, dtype=float)

    N      = surv_curves.shape[0]
    n_eval = len(eval_times)
    survival_probs = np.zeros((N, n_eval), dtype=float)

    for j, t in enumerate(eval_times):
        # Step function: select the last time_grid index at or before t.
        idx = int(np.searchsorted(time_grid, t, side='right')) - 1
        idx = max(0, min(idx, len(time_grid) - 1))
        survival_probs[:, j] = surv_curves[:, idx]

    risk_scores = 1.0 - survival_probs
    return survival_probs, risk_scores
