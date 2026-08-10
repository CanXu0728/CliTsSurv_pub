import numpy as np
import torch
from torch import nn


class SinCosTimeEmbedding(nn.Module):
    """Fixed sinusoidal positional encoding (no learnable parameters).

    Replicates the behavior of positional_encoding() in datasets/utils.py,
    but as an nn.Module so it fits the unified interface in CliTsSurv.forward.

    Input:  (B, T, 1)  raw time values
    Output: (B, T, time_dim)
    """

    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        # Pre-compute angle rates: shape (1, time_dim)
        i = np.arange(time_dim)
        angle_rates = 1.0 / np.power(10000, (2 * (i // 2)) / np.float32(time_dim))
        self.register_buffer('angle_rates', torch.from_numpy(angle_rates).float())

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, T, 1)
        angle_rads = t * self.angle_rates  # (B, T, time_dim)
        out = angle_rads.clone()
        out[..., 0::2] = torch.sin(angle_rads[..., 0::2])
        out[..., 1::2] = torch.cos(angle_rads[..., 1::2])
        return out


class Time2VecEmbedding(nn.Module):
    """Time2Vec (Kazemi et al., 2019).

    t2v(τ)[0]   = ω₀τ + φ₀              (linear trend)
    t2v(τ)[i]   = sin(ωᵢτ + φᵢ)  i≥1   (periodic components)

    All ω and φ are learnable. Parameter count: 2 * time_dim.

    Input:  (B, T, 1)  raw time values
    Output: (B, T, time_dim)
    """

    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.w = nn.Parameter(torch.randn(time_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(time_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, T, 1)
        x = t * self.w + self.b          # (B, T, time_dim)
        out = torch.sin(x)
        # First dimension is linear (no sin)
        out[..., 0] = x[..., 0]
        return out


class CTLPEEmbedding(nn.Module):
    """Continuous-Time Linear Positional Embedding (Chen et al., 2024).

    p(τ)[i] = kᵢτ + bᵢ

    Each dimension learns an independent slope k and bias b.
    Theoretically optimal for irregular time series: monotone and
    translation-invariant. Parameter count: 2 * time_dim.

    Input:  (B, T, 1)  raw time values
    Output: (B, T, time_dim)
    """

    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        # Initialise slopes to small distinct values to break symmetry
        self.k = nn.Parameter(torch.randn(time_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(time_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, T, 1)
        return t * self.k + self.b       # (B, T, time_dim)


class MultiScaleSinCosEmbedding(nn.Module):
    """Multi-scale sinusoidal time encoding with clinically meaningful periods.

    Periods are selected from a fixed candidate list in descending order,
    taking the first (time_dim // 2) entries so that each period contributes
    one sin + one cos dimension.

    Default candidate list (months): [171, 60, 36, 24, 12]
      time_dim=4  → periods [171, 60]        (overall span + 5-year node)
      time_dim=6  → periods [171, 60, 36]    (+ 3-year node)
      time_dim=8  → periods [171, 60, 36, 24](+ 2-year node)
      time_dim=10 → periods [171, 60, 36, 24, 12]

    Compared with the standard SinCos encoding whose lowest frequency covers
    ~628 months (almost constant over the 0–171 month data range), these
    periods are chosen to span the actual follow-up window and clinically
    important survival milestones.

    No learnable parameters.

    Input:  (B, T, 1)  raw time values (months)
    Output: (B, T, time_dim)
    """

    CANDIDATE_PERIODS = [171, 60, 36, 24, 12]  # months, descending

    def __init__(self, time_dim: int,
                 periods: list = None):
        super().__init__()
        if time_dim % 2 != 0:
            raise ValueError(
                f"MultiScaleSinCosEmbedding requires even time_dim, got {time_dim}."
            )
        self.time_dim = time_dim
        n_periods = time_dim // 2

        if periods is None:
            if n_periods > len(self.CANDIDATE_PERIODS):
                raise ValueError(
                    f"time_dim={time_dim} requires {n_periods} periods but only "
                    f"{len(self.CANDIDATE_PERIODS)} candidates are available: "
                    f"{self.CANDIDATE_PERIODS}. Pass a custom periods list."
                )
            periods = self.CANDIDATE_PERIODS[:n_periods]

        if len(periods) != n_periods:
            raise ValueError(
                f"time_dim={time_dim} requires exactly {n_periods} periods, "
                f"got {len(periods)}: {periods}."
            )

        # angle_rates shape: (time_dim,)
        # each period T contributes two entries: 2π/T (for sin) and 2π/T (for cos)
        rates = np.array(
            [2 * np.pi / p for p in periods for _ in range(2)],
            dtype=np.float32,
        )
        self.register_buffer('angle_rates', torch.from_numpy(rates))
        self.periods = periods  # stored for inspection / hparams logging

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B, T, 1)
        angle_rads = t * self.angle_rates          # (B, T, time_dim)
        out = torch.empty_like(angle_rads)
        out[..., 0::2] = torch.sin(angle_rads[..., 0::2])
        out[..., 1::2] = torch.cos(angle_rads[..., 1::2])
        return out


TIME_ENCODER_REGISTRY = {
    'sincos':            SinCosTimeEmbedding,
    'time2vec':          Time2VecEmbedding,
    'ctlpe':             CTLPEEmbedding,
    'multiscale_sincos': MultiScaleSinCosEmbedding,
}


def build_time_encoder(time_encoder_type: str, time_dim: int) -> nn.Module:
    """Factory function.

    time_encoder_type: 'sincos' | 'time2vec' | 'ctlpe' | 'multiscale_sincos'
    """
    if time_encoder_type not in TIME_ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown time_encoder_type '{time_encoder_type}'. "
            f"Choose from {list(TIME_ENCODER_REGISTRY.keys())}."
        )
    return TIME_ENCODER_REGISTRY[time_encoder_type](time_dim)
