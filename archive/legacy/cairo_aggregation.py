"""
cairo_aggregation.py — CAIRO ensemble aggregation functions (minimal, self-contained).

Verbatim extraction from Reinforce/parallel/common.py so that the RR bridge's
signal aggregation matches the Python backtest exactly. The production
`common.py` pulls in matplotlib/torch/gymnasium, DB config, and Reinforce
constants (FORCE_RETRAIN, PARALLEL_DIR, get_feature_cols, ...) which train_dt.py
also imports — a module named `common` in the RR checkout would shadow the real
one and break Decision Transformer loading. These functions only need numpy, so
they live in their own module and the bridge imports them by this name.

Included:
  - cairo_aggregate                (plain hard vote + ETA confidence)
  - cairo_aggregate_weighted       (Sharpe-weighted hard vote)
  - cairo_aggregate_soft_weighted  (probability pooling — used by the bridge)
  - cairo_aggregate_regime         (regime-ETA hard vote)

The bridge calls cairo_aggregate_soft_weighted when model_weights are loaded
(from sizing_results.json), which is the engine_v2 CAIROEnsemble._aggregate
behaviour reproduced in the Python backtest.
"""

import numpy as np

ETA = 0.5  # confidence multiplier: weight = 1 + ETA * (votes_for_winner - 1)

# Per-regime ETA values: tune via parallel/optimize_cairo.py
# regime 0 = trending/low-vol  -> favour LSTM momentum models (PPO/IMPALA)
# regime 1 = ranging/med-vol   -> favour entropy-regularised SAC
# regime 2 = high-vol/risk-off -> conservative uniform weights
REGIME_ETA_MAP: dict = {
    0: {"PPO": 0.7, "IMPALA": 0.7, "APPO": 0.5, "SAC": 0.3, "TQC": 0.3, "GBM": 0.4, "DT": 0.4},
    1: {"PPO": 0.3, "IMPALA": 0.3, "APPO": 0.4, "SAC": 0.7, "TQC": 0.7, "GBM": 0.5, "DT": 0.4},
    2: {"PPO": 0.4, "IMPALA": 0.4, "APPO": 0.4, "SAC": 0.4, "TQC": 0.4, "GBM": 0.4, "DT": 0.4},
}


def cairo_aggregate(probs_dict: dict):
    valid, seq_len = {}, None
    for name, arr in probs_dict.items():
        p = np.asarray(arr, dtype=np.float32)
        if p.ndim != 2 or p.shape[1] != 3:
            continue
        if not np.isfinite(p).all():
            continue
        if seq_len is None:
            seq_len = p.shape[0]
        if p.shape[0] != seq_len:
            continue
        valid[name] = p

    if not valid:
        raise ValueError("No valid prob arrays for Cairo aggregation")

    stacked = np.stack([valid[a] for a in valid], axis=0)   # (n_algos, T, 3)
    votes   = stacked.argmax(axis=2)                          # (n_algos, T)
    T       = votes.shape[1]
    actions = np.zeros(T, dtype=np.int32)
    weights = np.ones(T, dtype=np.float32)

    for t in range(T):
        counts = np.bincount(votes[:, t], minlength=3)
        max_c  = counts.max()
        top    = np.where(counts == max_c)[0]
        if len(top) == 1:
            winner = int(top[0])
        else:
            winner = 1  # tie -> HOLD
        actions[t] = winner
        weights[t] = 1.0 + ETA * max(0, int(counts[winner]) - 1)

    return actions, weights, list(valid.keys())


def cairo_aggregate_weighted(probs_dict: dict, model_weights: dict):
    valid, seq_len = {}, None
    for name, arr in probs_dict.items():
        p = np.asarray(arr, dtype=np.float32)
        if p.ndim != 2 or p.shape[1] != 3:
            continue
        if not np.isfinite(p).all():
            continue
        if seq_len is None:
            seq_len = p.shape[0]
        if p.shape[0] != seq_len:
            continue
        valid[name] = p

    if not valid:
        raise ValueError("No valid prob arrays for weighted CAIRO aggregation")

    algos = list(valid.keys())
    n     = len(algos)

    raw_w = np.array([model_weights.get(a, 1.0 / n) for a in algos], dtype=np.float64)
    raw_w = np.maximum(raw_w, 0.0)
    w     = raw_w / raw_w.sum() if raw_w.sum() > 0 else np.ones(n) / n

    stacked = np.stack([valid[a] for a in algos], axis=0)   # (n_algos, T, 3)
    votes   = stacked.argmax(axis=2)                          # (n_algos, T)
    assert seq_len is not None
    T: int  = seq_len
    actions = np.zeros(T, dtype=np.int32)
    weights = np.ones(T, dtype=np.float32)

    for t in range(T):
        wt_counts = np.zeros(3, dtype=np.float64)
        for i in range(n):
            wt_counts[votes[i, t]] += w[i]

        max_wt = wt_counts.max()
        top    = np.where(wt_counts == max_wt)[0]
        winner = int(top[0]) if len(top) == 1 else 1  # tie -> HOLD

        actions[t] = winner
        effective  = float(wt_counts[winner]) * n
        weights[t] = 1.0 + ETA * max(0.0, effective - 1.0)

    return actions, weights, algos


def cairo_aggregate_soft_weighted(probs_dict: dict, model_weights: dict):
    """Soft probability-pooling CAIRO aggregation — matches engine_v2 CAIROEnsemble._aggregate."""
    valid, seq_len = {}, None
    for name, arr in probs_dict.items():
        p = np.asarray(arr, dtype=np.float32)
        if p.ndim != 2 or p.shape[1] != 3:
            continue
        if not np.isfinite(p).all():
            continue
        if seq_len is None:
            seq_len = p.shape[0]
        if p.shape[0] != seq_len:
            continue
        valid[name] = p

    if not valid:
        raise ValueError("No valid prob arrays for soft-weighted CAIRO aggregation")

    algos = list(valid.keys())
    n     = len(algos)

    raw_w = np.array([model_weights.get(a, 1.0) for a in algos], dtype=np.float64)
    raw_w = np.maximum(raw_w, 0.0)
    w     = raw_w / raw_w.sum() if raw_w.sum() > 0 else np.ones(n) / n

    assert seq_len is not None
    T: int  = seq_len
    actions = np.zeros(T, dtype=np.int32)
    weights = np.ones(T, dtype=np.float32)

    for t in range(T):
        vote_scores = np.zeros(3, dtype=np.float64)
        for i, a in enumerate(algos):
            vote_scores += w[i] * valid[a][t].astype(np.float64)

        winner = int(vote_scores.argmax())
        actions[t] = winner

        total = vote_scores.sum()
        weights[t] = float(vote_scores[winner] / total) if total > 0 else 0.0

    return actions, weights, algos


def cairo_aggregate_regime(probs_dict: dict, regime_label: int = 1):
    eta_map = REGIME_ETA_MAP.get(regime_label, REGIME_ETA_MAP[1])

    valid, seq_len = {}, None
    for name, arr in probs_dict.items():
        p = np.asarray(arr, dtype=np.float32)
        if p.ndim != 2 or p.shape[1] != 3:
            continue
        if not np.isfinite(p).all():
            continue
        if seq_len is None:
            seq_len = p.shape[0]
        if p.shape[0] != seq_len:
            continue
        valid[name] = p

    if not valid:
        raise ValueError("No valid prob arrays for regime-aware CAIRO aggregation")

    stacked = np.stack([valid[a] for a in valid], axis=0)
    votes   = stacked.argmax(axis=2)
    algos   = list(valid.keys())
    T       = votes.shape[1]
    actions = np.zeros(T, dtype=np.int32)
    weights = np.ones(T, dtype=np.float32)

    for t in range(T):
        counts = np.bincount(votes[:, t], minlength=3)
        max_c  = counts.max()
        top    = np.where(counts == max_c)[0]
        winner = int(top[0]) if len(top) == 1 else 1

        winning_algos = [algos[i] for i in range(len(algos)) if votes[i, t] == winner]
        eta_vals      = [eta_map.get(a, ETA) for a in winning_algos]
        avg_eta       = float(np.mean(eta_vals)) if eta_vals else ETA

        actions[t] = winner
        weights[t] = 1.0 + avg_eta * max(0, int(counts[winner]) - 1)

    return actions, weights, algos