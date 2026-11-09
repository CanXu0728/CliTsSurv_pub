import torch


def conditional_event_distribution(pred, observation_bins, num_time_bins, eps=1e-8):
    """Condition the absolute-time softmax mass on survival beyond ``a``.

    ``observation_bins`` is the first interval whose right endpoint is later
    than the last non-zero observation time.  The final softmax entry is the
    beyond-horizon mass and is always part of S(a).
    """
    bin_idx = torch.arange(num_time_bins, device=pred.device).unsqueeze(0)
    future_mask = bin_idx >= observation_bins.unsqueeze(1)
    future_event = pred[:, :num_time_bins] * future_mask
    denominator = future_event.sum(dim=1) + pred[:, num_time_bins]
    denominator = denominator.clamp_min(eps)
    conditional_event = future_event / denominator.unsqueeze(1)
    conditional_tail = pred[:, num_time_bins] / denominator
    return conditional_event, conditional_tail, denominator


def conditional_survival_curve(pred, observation_bins, num_time_bins, eps=1e-8):
    """Return S(t | T>a, H_a) on the absolute-time bin grid."""
    _, _, denominator = conditional_event_distribution(
        pred, observation_bins, num_time_bins, eps=eps
    )
    absolute_survival = 1.0 - torch.cumsum(pred[:, :num_time_bins], dim=1)
    conditional_survival = absolute_survival / denominator.unsqueeze(1)
    return conditional_survival.clamp(min=0.0, max=1.0)


def deephit_loss(pred, time_bins, event_indicator, alpha=0.1, sigma=0.1,
                 observation_bins=None, sample_weights=None,
                 rank_sample_weights=None, include_nll=True,
                 include_ranking=True):
    """
    DeepHit loss = NLL loss + alpha * ranking loss (concordance)

    Args:
        pred: (batch, num_time_bins+1) probability distribution
        time_bins: (batch,) time bin index where event occurs (0 to num_time_bins-1)
        event_indicator: (batch,) 1 if event occurred, 0 if censored
        observation_bins: first absolute-time bin ending after observation time a
        alpha: weight for ranking loss
        sigma: bandwidth for ranking loss exponential kernel
    """
    batch_size = pred.shape[0]
    if observation_bins is None:
        observation_bins = torch.zeros(
            batch_size, dtype=torch.long, device=pred.device
        )

    event_mask = (event_indicator == 1)
    censored_mask = (event_indicator == 0)

    if sample_weights is None:
        sample_weights = torch.ones(batch_size, device=pred.device)
    else:
        sample_weights = sample_weights.to(pred.device, dtype=pred.dtype)
    nll_terms = torch.zeros(batch_size, device=pred.device, dtype=pred.dtype)
    conditional_event, conditional_tail, _ = conditional_event_distribution(
        pred, observation_bins, pred.shape[1] - 1
    )

    # --- Conditional NLL: -log q(T) / S(a) ---
    if include_nll and event_mask.any():
        event_pred = conditional_event[event_mask]
        event_times = time_bins[event_mask].long()
        event_prob = event_pred[torch.arange(event_pred.shape[0], device=pred.device), event_times]
        nll_terms[event_mask] = -torch.log(event_prob + 1e-8)

    # --- NLL: censoring loss: -log P(T > censored_time) = -log sum(probs for bins > t) ---
    if include_nll and censored_mask.any():
        censored_pred = conditional_event[censored_mask]
        censored_tail = conditional_tail[censored_mask]
        censored_times = time_bins[censored_mask].long()
        # Vectorized: mask[i, j] = 1 if j > censored_times[i]
        bin_indices = torch.arange(censored_pred.shape[1], device=pred.device).unsqueeze(0)
        time_threshold = censored_times.unsqueeze(1)  # (n_censored, 1)
        surv_mask = (bin_indices > time_threshold).float()
        survival_probs = ((censored_pred * surv_mask).sum(dim=-1)
                          + censored_tail)
        nll_terms[censored_mask] = -torch.log(survival_probs + 1e-8)

    if include_nll:
        loss = ((nll_terms * sample_weights).sum()
                / sample_weights.sum().clamp_min(1e-8))
    else:
        loss = pred.sum() * 0.0

    # --- Ranking loss (concordance) ---
    if include_ranking and alpha > 0 and event_mask.any():
        event_idx = torch.where(event_mask)[0]
        t_i = time_bins[event_idx]  # (n_events,)

        # Ranking intentionally remains on the absolute-time distribution q.
        event_probs_all = pred[:, :-1]  # (batch, num_time_bins)
        cdf = torch.cumsum(event_probs_all, dim=-1)  # (batch, num_time_bins)

        # F(t_i | x_i): CDF of each uncensored sample at its own event time
        F_i = cdf[event_idx, t_i]  # (n_events,)

        # F(t_i | x_j): CDF of every sample j at each uncensored t_i
        # cdf[:, t_i] -> (batch, n_events), transpose -> (n_events, batch)
        F_j_at_ti = cdf[:, t_i].T  # (n_events, batch)

        # Indicator I(t_i < t_j) for all pairs
        t_j = time_bins  # (batch,)
        indicator = (t_i.unsqueeze(1) < t_j.unsqueeze(0)).float()  # (n_events, batch)

        # Ranking loss: sum over permissible pairs of exp(-(F_i(t_i) - F_j(t_i)) / sigma)
        diff = F_i.unsqueeze(1) - F_j_at_ti  # (n_events, batch)
        if rank_sample_weights is None:
            rank_sample_weights = torch.ones_like(sample_weights)
        else:
            rank_sample_weights = rank_sample_weights.to(
                pred.device, dtype=pred.dtype
            )
        pair_weights = (rank_sample_weights[event_idx].unsqueeze(1)
                        * rank_sample_weights.unsqueeze(0))
        weighted_indicator = indicator * pair_weights
        rank_loss = (weighted_indicator * torch.exp(-diff / sigma)).sum()

        n_pairs = weighted_indicator.sum()
        if n_pairs > 0:
            loss = loss + alpha * rank_loss / n_pairs

    return loss


def brier_loss(pred, time_bins, event_indicator, num_time_bins,
               observation_bins=None, sample_weights=None):
    """
    Integrated Brier Score loss (differentiable, no IPCW weighting)

    For each time bin k:
    - alive (T_i > k):                target = 1
    - event at or before k (E_i = 1): target = 0
    - censored before k (E_i = 0):    excluded

    IBS = mean over k of BS(k)
    BS(k) = mean over valid samples of (I(T_i > k) - S(k|x_i))^2

    Args:
        pred:             (batch, num_time_bins+1) softmax probabilities
        time_bins:        (batch,) discretised event/censor time bin index
        event_indicator:  (batch,) 1 = event, 0 = censored
        num_time_bins:    number of time bins
    """
    if observation_bins is None:
        observation_bins = torch.zeros(
            pred.shape[0], dtype=torch.long, device=pred.device
        )
    if sample_weights is None:
        sample_weights = torch.ones(pred.shape[0], device=pred.device)
    else:
        sample_weights = sample_weights.to(pred.device, dtype=pred.dtype)
    survival = conditional_survival_curve(
        pred, observation_bins, num_time_bins
    )                                                               # S(k | T>a, H_a)

    bin_idx = torch.arange(num_time_bins, device=pred.device).unsqueeze(0)  # (1, num_time_bins)
    t = time_bins.unsqueeze(1)           # (batch, 1)
    e = event_indicator.unsqueeze(1)     # (batch, 1)

    alive_mask = (t > bin_idx).float()                               # (batch, num_time_bins)
    event_mask = ((t <= bin_idx) & (e == 1)).float()                 # (batch, num_time_bins)
    after_observation = (bin_idx >= observation_bins.unsqueeze(1)).float()
    valid_mask = (alive_mask + event_mask) * after_observation

    weighted_valid = valid_mask * sample_weights.unsqueeze(1)
    bs = weighted_valid * (alive_mask - survival) ** 2
    n_valid = weighted_valid.sum(dim=0)
    valid_bins = n_valid > 0
    if not valid_bins.any():
        return pred.sum() * 0.0
    bs_per_bin = bs.sum(dim=0)[valid_bins] / n_valid[valid_bins]
    return bs_per_bin.mean()


def c_index(y_true_time, y_true_event, y_pred_risk):
    """
    Calculate Concordance Index (C-index)
    
    Args:
        y_true_time: (N,) actual survival/censoring times
        y_true_event: (N,) event indicator (1=event, 0=censored)
        y_pred_risk: (N,) predicted risk scores (higher = higher risk)
    
    Returns:
        c-index value between 0 and 1
    """
    n = len(y_true_time)
    concordant = 0
    permissible = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            # Check if pair is permissible
            # A pair is permissible if we know which one failed first
            if y_true_event[i] == 1 and y_true_event[j] == 1:
                # Both events observed
                permissible += 1
                if (y_true_time[i] < y_true_time[j] and y_pred_risk[i] > y_pred_risk[j]) or \
                   (y_true_time[i] > y_true_time[j] and y_pred_risk[i] < y_pred_risk[j]):
                    concordant += 1
                elif y_true_time[i] == y_true_time[j]:
                    concordant += 0.5
            elif y_true_event[i] == 1 and y_true_event[j] == 0:
                # i failed, j censored
                if y_true_time[i] <= y_true_time[j]:
                    permissible += 1
                    if y_pred_risk[i] > y_pred_risk[j]:
                        concordant += 1
                    elif y_pred_risk[i] == y_pred_risk[j]:
                        concordant += 0.5
            elif y_true_event[i] == 0 and y_true_event[j] == 1:
                # i censored, j failed
                if y_true_time[j] <= y_true_time[i]:
                    permissible += 1
                    if y_pred_risk[j] > y_pred_risk[i]:
                        concordant += 1
                    elif y_pred_risk[j] == y_pred_risk[i]:
                        concordant += 0.5
    
    return concordant / permissible if permissible > 0 else 0.5
