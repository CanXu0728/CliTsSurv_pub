"""Shared validation and cleaning for longitudinal survival data."""

from __future__ import annotations

import pandas as pd


def remove_post_outcome_records(
    seq_df: pd.DataFrame,
    static_df: pd.DataFrame,
    *,
    id_col: str = "id",
    time_col: str = "date",
    outcome_time_col: str = "survival_months",
):
    """Keep patients with valid outcomes and only observations strictly before T.

    Returns cleaned copies of ``seq_df`` and ``static_df`` plus a small audit
    dictionary.  A record at ``date == T`` is excluded because the prediction
    origin ``a`` must satisfy ``a < T``.
    """
    required_seq = {id_col, time_col}
    required_static = {id_col, outcome_time_col}
    if missing := required_seq.difference(seq_df.columns):
        raise KeyError(f"seq data is missing columns: {sorted(missing)}")
    if missing := required_static.difference(static_df.columns):
        raise KeyError(f"static data is missing columns: {sorted(missing)}")
    if static_df[id_col].duplicated().any():
        raise ValueError(f"static data contains duplicate {id_col} values")

    static = static_df.loc[
        static_df[outcome_time_col].notna()
        & (static_df[outcome_time_col] > 0)
    ].copy()
    outcome = static.set_index(id_col)[outcome_time_col]

    seq = seq_df[seq_df[id_col].isin(outcome.index)].copy()
    seq["__outcome_time"] = seq[id_col].map(outcome)
    valid_record = seq[time_col].notna() & (seq[time_col] < seq["__outcome_time"])
    removed_records = int((~valid_record).sum())
    removed_patients = int(seq.loc[~valid_record, id_col].nunique())
    seq = seq.loc[valid_record].drop(columns="__outcome_time")

    retained_ids = pd.Index(seq[id_col].unique())
    static = static[static[id_col].isin(retained_ids)].copy()
    seq = seq.sort_values([id_col, time_col]).reset_index(drop=True)
    static = static.reset_index(drop=True)

    max_observed = seq.groupby(id_col)[time_col].max()
    retained_outcome = static.set_index(id_col)[outcome_time_col]
    if not (max_observed < retained_outcome.loc[max_observed.index]).all():
        raise AssertionError("post-outcome observations remain after cleaning")

    audit = {
        "removed_records": removed_records,
        "affected_patients": removed_patients,
        "retained_patients": int(len(retained_ids)),
    }
    print(
        "  清理 date >= T 的真实观测: "
        f"{removed_records} 条记录，涉及 {removed_patients} 位患者；"
        f"保留 {len(retained_ids)} 位患者"
    )
    return seq, static, audit
