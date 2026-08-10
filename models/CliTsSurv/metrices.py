import torch

def deephit_loss(pred, time_bins, event_indicator, alpha=0.1, sigma=0.1):
    """
    DeepHit loss = NLL loss + alpha * ranking loss (concordance)

    Args:
        pred: (batch, num_time_bins+1) probability distribution
        time_bins: (batch,) time bin index where event occurs (0 to num_time_bins-1)
        event_indicator: (batch,) 1 if event occurred, 0 if censored
        alpha: weight for ranking loss
        sigma: bandwidth for ranking loss exponential kernel
    """
    batch_size = pred.shape[0]

    event_mask = (event_indicator == 1)
    censored_mask = (event_indicator == 0)

    loss = torch.tensor(0.0, device=pred.device)

    # --- NLL: event loss: -log P(event at time_bins[i]) ---
    if event_mask.any():
        event_pred = pred[event_mask]
        event_times = time_bins[event_mask].long()
        event_prob = event_pred[torch.arange(event_pred.shape[0], device=pred.device), event_times]
        loss = loss + (-torch.log(event_prob + 1e-8)).sum()

    # --- NLL: censoring loss: -log P(T > censored_time) = -log sum(probs for bins > t) ---
    if censored_mask.any():
        censored_pred = pred[censored_mask]
        censored_times = time_bins[censored_mask].long()
        # Vectorized: mask[i, j] = 1 if j > censored_times[i]
        bin_indices = torch.arange(censored_pred.shape[1], device=pred.device).unsqueeze(0)  # (1, num_bins)
        time_threshold = censored_times.unsqueeze(1)  # (n_censored, 1)
        surv_mask = (bin_indices > time_threshold).float()  # (n_censored, num_bins)
        survival_probs = (censored_pred * surv_mask).sum(dim=-1)  # (n_censored,)
        loss = loss + (-torch.log(survival_probs + 1e-8)).sum()

    loss = loss / batch_size

    # --- Ranking loss (concordance) ---
    if alpha > 0 and event_mask.any():
        event_idx = torch.where(event_mask)[0]
        t_i = time_bins[event_idx]  # (n_events,)

        # CDF for all samples at all time bins: F(t|x) = cumsum of event probs up to t
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
        rank_loss = (indicator * torch.exp(-diff / sigma)).sum()

        n_pairs = indicator.sum()
        if n_pairs > 0:
            loss = loss + alpha * rank_loss / n_pairs

    return loss


def brier_loss(pred, time_bins, event_indicator, num_time_bins):
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
    event_probs = pred[:, :-1]                                       # (batch, num_time_bins)
    survival = 1.0 - torch.cumsum(event_probs, dim=-1)               # S(k|x_i)

    bin_idx = torch.arange(num_time_bins, device=pred.device).unsqueeze(0)  # (1, num_time_bins)
    t = time_bins.unsqueeze(1)           # (batch, 1)
    e = event_indicator.unsqueeze(1)     # (batch, 1)

    alive_mask = (t > bin_idx).float()                               # (batch, num_time_bins)
    event_mask = ((t <= bin_idx) & (e == 1)).float()                 # (batch, num_time_bins)
    valid_mask = alive_mask + event_mask

    bs = valid_mask * (alive_mask - survival) ** 2
    n_valid = valid_mask.sum(dim=0).clamp(min=1.0)
    bs_per_bin = bs.sum(dim=0) / n_valid

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