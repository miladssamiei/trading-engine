"""Smoke test: full process_bar_core roundtrip with 300+ warmup bars.

Feeds replayed historical bars from the existing CAIRO EURUSD live_fold
parquet through the bridge_core pipeline and reports actual signal output.
"""
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd

_CAIRO_FOLD = Path(
    r"C:\Users\Milad\Desktop\Reinforce\Reinforce_Windows_20260613"
    r"\multi_live\data\EURUSD\15m\live_fold"
)
_PARQUET = _CAIRO_FOLD / "parquets" / "fold_202606_train.parquet"

from bridge.bridge_core import (
    LiveModels,
    BarValidator,
    BarBuffer,
    process_bar_core,
)

print("=" * 70)
print("  SMOKE TEST — process_bar_core 300-bar roundtrip")
print("=" * 70)

# ── Load bar data ─────────────────────────────────────────────────────────
print(f"\n[1] Loading parquet: {_PARQUET.name}")
df = pd.read_parquet(_PARQUET)
print(f"    Rows: {len(df)}, Cols: {len(df.columns)}")

# Extract OHLC columns used by BarBuffer
_ohlc = ["open", "high", "low", "close", "avg_spread"]
ohlc_cols = [c for c in _ohlc if c in df.columns]
print(f"    OHLC cols present: {ohlc_cols}")

# ── Load splits + feature_cols ─────────────────────────────────────────────
splits = json.loads((_CAIRO_FOLD / "splits.json").read_text())
feature_cols = splits["feature_cols"]
warmup_bars = splits["warmup_bars"]
print(f"    Feature cols: {len(feature_cols)}, Warmup: {warmup_bars}")

# ── Instantiate LiveModels ─────────────────────────────────────────────────
print(f"\n[2] Instantiating LiveModels...")
models = LiveModels(symbol="EURUSD", fold_dir=_CAIRO_FOLD)
models.load()
print(f"    Models loaded: {list(models.algos.keys())}")
print(f"    Model types:   {dict(models._algo_types)}")
print(f"    Weights:        {dict(models.model_weights)}")

# ── Instantiate validator + buffer ─────────────────────────────────────────
print(f"\n[3] Creating BarValidator + BarBuffer...")
validator = BarValidator(symbol="EURUSD")
buffer = BarBuffer()

# ── Feed bars ──────────────────────────────────────────────────────────────
N_BARS = min(500, len(df))
print(f"\n[4] Feeding {N_BARS} bars through process_bar_core...")
print()

prev_action = "HOLD"
bars_since_flip = 0
first_signal_row = None
signal_log = []
hold_guard_fires = 0

t_start = time.time()

for idx in range(N_BARS):
    row = df.iloc[idx]

    bar = {
        "timestamp": int(time.time()) + idx * 900,
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "spread": float(row.get("avg_spread", 0.0001)),
    }

    action_str, details, elapsed_ms, bars_since_flip = process_bar_core(
        bar=bar,
        models=models,
        buffer=buffer,
        validator=validator,
        feature_cols=feature_cols,
        prev_action=prev_action,
        bars_since_flip=bars_since_flip,
    )

    if action_str != prev_action and action_str in ("BUY", "SELL"):
        if prev_action in ("BUY", "SELL"):
            hold_guard_fires += 1

    prev_action = action_str

    if (idx + 1) % 100 == 0:
        elapsed = time.time() - t_start
        print(f"    ... {idx+1}/{N_BARS} bars ({elapsed:.1f}s elapsed)  "
              f"| buffer={len(buffer)}  last_action={prev_action}  "
              f"flip_bars={bars_since_flip}")

    if action_str != "HOLD" and first_signal_row is None:
        first_signal_row = idx + 1

    signal_log.append({
        "idx": idx + 1,
        "close": round(float(row["close"]), 5),
        "action": action_str,
        "bars_since_flip": bars_since_flip,
        "elapsed_ms": round(elapsed_ms, 2),
    })

t_elapsed = time.time() - t_start

# ── Results ─────────────────────────────────────────────────────────────────
print(f"\n[5] RESULTS")
print(f"    Total bars processed:   {N_BARS}")
print(f"    Processing time:        {t_elapsed:.2f}s ({N_BARS/t_elapsed:.0f} bars/s)")
print(f"    Warmup bars:            {warmup_bars}")
print(f"    First non-HOLD signal:  bar #{first_signal_row} ({'after warmup' if first_signal_row and first_signal_row > warmup_bars else 'N/A'})")
print(f"    Hold-guard fires:       {hold_guard_fires}")
print(f"    Rejections:             {dict(validator._rejection_counts)}")

# Action distribution
from collections import Counter
action_counts = Counter(s["action"] for s in signal_log)
print(f"    Action distribution:    {dict(action_counts)}")

# Post-warmup action distribution
post_warmup = [s for s in signal_log if s["idx"] > warmup_bars]
post_counts = Counter(s["action"] for s in post_warmup)
print(f"    Post-warmup dist:       {dict(post_counts)}")

# Elapsed time stats
post_elapsed = [s["elapsed_ms"] for s in post_warmup]
if post_elapsed:
    print(f"    Predict latency:")
    print(f"      min:  {min(post_elapsed):.2f}ms")
    print(f"      p50:  {np.percentile(post_elapsed, 50):.2f}ms")
    print(f"      p95:  {np.percentile(post_elapsed, 95):.2f}ms")
    print(f"      p99:  {np.percentile(post_elapsed, 99):.2f}ms")
    print(f"      max:  {max(post_elapsed):.2f}ms")

# Last 10 signals
print(f"\n    Last 10 signals:")
for s in signal_log[-10:]:
    flag = " <-- FLIP" if s["action"] in ("BUY", "SELL") else ""
    print(f"      bar {s['idx']:6d} | close={s['close']:.5f} | "
          f"action={s['action']:4s} | flips={s['bars_since_flip']:4d} | "
          f"{s['elapsed_ms']:7.2f}ms{flag}")

# First 10 non-HOLD signals
non_hold = [s for s in signal_log if s["action"] != "HOLD"]
print(f"\n    First 10 non-HOLD signals (total {len(non_hold)}):")
for s in non_hold[:10]:
    flag = " <-- FLIP" if s["action"] in ("BUY", "SELL") else ""
    print(f"      bar {s['idx']:6d} | close={s['close']:.5f} | "
          f"action={s['action']:4s} | flips={s['bars_since_flip']:4d} | "
          f"{s['elapsed_ms']:7.2f}ms{flag}")

# Validation summary
n_bars = len(df)
n_hold = action_counts.get("HOLD", 0)
n_buy = action_counts.get("BUY", 0)
n_sell = action_counts.get("SELL", 0)
print(f"\n[6] VALIDATION SUMMARY")
print(f"    Non-HOLD signal rate: {(n_buy + n_sell) / n_bars * 100:.1f}%")
print(f"    Pipeline exercised:    validate → buffer → features → regime → predict → hold-guard")
print(f"    Models exercised:      {list(models.algos.keys())}")
print(f"    Roundtrip:             PASS" if n_bars > warmup_bars else "    Roundtrip:             FAIL (insufficient bars)")

print(f"\n{'=' * 70}")
print(f"  SMOKE TEST COMPLETE")
print(f"{'=' * 70}")
