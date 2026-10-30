"""Frozen Hypothesis #96; see SPEC_EDRIS_IB.txt before any changes.

Run with the project's reinforce_rl Python. Observed candles only, causal
confirmed pivots, NY-wall H4 anchors, executable-side 1:3 M1 bracket replay.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/edris_sweep_divergence_ib"
REPORT = OUT / "REPORT_EDRIS_IB.txt"
ASSETS = (
    ("US500", ROOT / "data/results/vwap_2021_2026/us500tick_m1.csv", .51, 2021),
    ("USTECH", ROOT / "data/results/virgin_assets/ustech_m1.csv", 1.66, 2020),
    ("XAUUSD", ROOT / "data/results/xauusd_onboarding/xauusdtick_m1.csv", .41, 2020),
)
ARMS = ("ALL_HOURS", "KILLZONE_ONLY")
BOOT, BLOCK, SEED = 10_000, 5, 20261014
HOUR_NS = 3600 * 10**9
SETUP_COLUMNS = [
    "asset", "setup_id", "side", "trigger_ix", "mother_ix", "sweep_ix", "pivot_ix",
    "pivot_confirmation_ix", "anchor_ix", "current_h4_ix", "trigger_start_epoch",
    "signal_epoch", "entry_bucket_epoch", "entry_ix", "entry_epoch", "entry_bid",
    "entry_ny_wall_minute", "killzone", "anchor_close", "anchor_ema50",
    "anchor_low", "anchor_high", "anchor_end_epoch", "current_h4_start_epoch",
    "trigger_high", "trigger_low", "mother_high", "mother_low", "sweep_price",
    "sweep_rsi", "sweep_end_epoch", "pivot_price", "pivot_rsi",
    "pivot_confirmation_epoch", "stop_bid_reference", "bid_risk", "initial_risk",
    "entry_exec", "stop_exec", "target_exec", "eligible_entry", "rejection_reason",
]
TRADE_COLUMNS = [
    "asset", "arm", "trade_id", "setup_id", "side", "entry_ix", "exit_ix",
    "signal_epoch", "entry_epoch", "exit_epoch", "entry_bid", "entry_exec",
    "stop_bid_reference", "stop_exec", "target_exec", "initial_risk", "exit_bid",
    "exit_exec", "cost_floor", "gross_units", "net_units", "net_r", "exit_type",
    "collision", "adverse_gap", "hold_calendar_hours", "ny_date_crossed",
    "entry_year", "closed", "terminal_net_units_mark", "terminal_net_r_mark",
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iso(epoch):
    return pd.Timestamp(int(epoch), unit="s", tz="UTC").isoformat()


def load_source(path):
    frame = pd.read_csv(path, usecols=["time", "open", "high", "low", "close"])
    t = frame.time.to_numpy(np.int64)
    o, h, l, c = [frame[k].to_numpy(float) for k in ("open", "high", "low", "close")]
    price = np.column_stack((o, h, l, c))
    if (len(t) < 100_000 or np.any(np.diff(t) <= 0) or np.any(t % 60)
            or not np.isfinite(price).all() or np.any(price <= 0)
            or np.any(h < price.max(axis=1)) or np.any(l > price.min(axis=1))):
        raise ValueError(f"Invalid M1 OHLC/chronology: {path}")
    carried = (o == h) & (h == l) & (l == c) & np.r_[False, c[1:] == c[:-1]]
    retained = frame.loc[~carried].reset_index(drop=True)
    assert not retained.empty
    meta = {"path": str(path), "sha256": digest(path), "raw_rows": len(frame),
            "retained_rows": len(retained), "removed_carried": int(carried.sum()),
            "first_utc": iso(retained.time.iloc[0]), "last_m1_open_utc": iso(retained.time.iloc[-1]),
            "first_row_flat_unclassifiable": bool(o[0] == h[0] == l[0] == c[0]),
            "max_retained_gap_hours": float(np.diff(retained.time).max() / 3600)}
    return retained, meta


def wall_h4_keys(times):
    ny = pd.to_datetime(times, unit="s", utc=True).tz_convert("America/New_York")
    wall = ny.tz_localize(None).asi8
    return wall - wall % (4 * HOUR_NS)


def local_boundary_epochs(keys):
    starts = pd.to_datetime(keys).tz_localize("America/New_York").tz_convert("UTC").asi8 // 10**9
    ends = pd.to_datetime(keys + 4 * HOUR_NS).tz_localize("America/New_York").tz_convert("UTC").asi8 // 10**9
    return starts.astype(np.int64), ends.astype(np.int64)


def aggregate(tape, keys):
    work = tape.assign(key=keys, ix=np.arange(len(tape)))
    return work.groupby("key", sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), first_ix=("ix", "first"), last_ix=("ix", "last"),
        observed_m1=("ix", "size"))


def wilder_rsi(close):
    close = np.asarray(close, float)
    out = np.full(len(close), np.nan)
    if len(close) <= 14:
        return out
    change = np.diff(close)
    gain, loss = np.maximum(change, 0), np.maximum(-change, 0)
    ag, al = float(gain[:14].mean()), float(loss[:14].mean())
    for i in range(14, len(close)):
        if i > 14:
            ag = (13 * ag + gain[i - 1]) / 14
            al = (13 * al + loss[i - 1]) / 14
        out[i] = (50.0 if ag == 0 else 100.0) if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def fractals(high, low):
    n = len(high)
    ph, pl = np.zeros(n, bool), np.zeros(n, bool)
    for i in range(2, n - 2):
        ph[i] = all(high[i] > high[i + d] for d in (-2, -1, 1, 2))
        pl[i] = all(low[i] < low[i + d] for d in (-2, -1, 1, 2))
    return ph, pl


def build_grids(tape):
    t = tape.time.to_numpy(np.int64)
    m30 = aggregate(tape, t // 1800 * 1800)
    m30["start"] = m30.index.to_numpy(np.int64)
    m30["end"] = m30.start + 1800
    h4 = aggregate(tape, wall_h4_keys(t))
    h4["wall_key_ns"] = h4.index.to_numpy(np.int64)
    h4["start"], h4["end"] = local_boundary_epochs(h4.wall_key_ns.to_numpy())
    h4["ema50"] = h4.close.ewm(span=50, adjust=False, min_periods=50).mean()
    m30["rsi14"] = wilder_rsi(m30.close.to_numpy())
    m30["fractal_high"], m30["fractal_low"] = fractals(m30.high.to_numpy(), m30.low.to_numpy())
    h4_lookup = {int(key): i for i, key in enumerate(h4.wall_key_ns)}
    m30["h4_ix"] = [h4_lookup[int(k)] for k in wall_h4_keys(m30.start.to_numpy())]
    start_ny = pd.to_datetime(m30.start, unit="s", utc=True).dt.tz_convert("America/New_York")
    assert start_ny.dt.minute.isin([0, 30]).all()
    assert m30.start.is_unique and m30.start.is_monotonic_increasing
    assert h4.start.is_unique and h4.start.is_monotonic_increasing
    assert np.isin((h4.end - h4.start).to_numpy(), [3 * 3600, 4 * 3600, 5 * 3600]).all()
    # Every observation belongs to exactly one NY-wall bucket and UTC half-hour.
    assert m30.observed_m1.sum() == h4.observed_m1.sum() == len(tape)
    return m30.reset_index(drop=True), h4.reset_index(drop=True)


def signals(asset, tape, m30, h4, cost, score_year):
    high, low, rsi = [m30[k].to_numpy() for k in ("high", "low", "rsi14")]
    ph, pl = m30.fractal_high.to_numpy(), m30.fractal_low.to_numpy()
    latest_high = latest_low = -1
    sweep_low = sweep_high = -1
    low_pivot = high_pivot = -1
    current_h4 = -1
    out = []
    counts = {"completed_m30": 0, "trend_ready": 0, "strict_inside": 0,
              "sweep_and_inside": 0, "hidden_divergence_setups": 0,
              "no_next_bucket": 0, "degenerate_risk": 0, "unscored_entry": 0,
              "fractal_lows": int(pl.sum()), "fractal_highs": int(ph.sum())}
    next_lookup = {int(start): i for i, start in enumerate(m30.start)}
    covered_end = int(tape.time.iloc[-1]) + 60
    score_start = int(pd.Timestamp(f"{score_year}-01-01", tz="UTC").timestamp())
    for i, candle in enumerate(m30.itertuples(index=False)):
        if candle.end > covered_end:
            continue
        counts["completed_m30"] += 1
        # The two right neighbours are closed NOW; this center was not known
        # during either earlier candle. This snapshot is used only by new s.
        k = i - 2
        if k >= 2:
            if ph[k]:
                latest_high = k
            if pl[k]:
                latest_low = k
        hi = candle.h4_ix
        if hi != current_h4:
            current_h4 = hi
            sweep_low = sweep_high = low_pivot = high_pivot = -1
        anchor_ix = hi - 1
        valid_anchor = (anchor_ix >= 0 and h4.end.iloc[anchor_ix] <= h4.start.iloc[hi]
                        and np.isfinite(h4.ema50.iloc[anchor_ix]))
        inside = i > 0 and m30.h4_ix.iloc[i - 1] == hi and high[i] <= high[i - 1] and low[i] >= low[i - 1]
        if inside:
            counts["strict_inside"] += 1
        if valid_anchor:
            counts["trend_ready"] += 1
            anchor = h4.iloc[anchor_ix]
            side = s = p = 0
            if inside and anchor.close > anchor.ema50 and sweep_low >= 0 and low[sweep_low] < anchor.low:
                counts["sweep_and_inside"] += 1
                s, p = sweep_low, low_pivot
                if (p >= max(0, i - 40) and p + 2 <= s and np.isfinite(rsi[s]) and np.isfinite(rsi[p])
                        and low[s] > low[p] and rsi[s] < rsi[p]):
                    side = 1
            elif inside and anchor.close < anchor.ema50 and sweep_high >= 0 and high[sweep_high] > anchor.high:
                counts["sweep_and_inside"] += 1
                s, p = sweep_high, high_pivot
                if (p >= max(0, i - 40) and p + 2 <= s and np.isfinite(rsi[s]) and np.isfinite(rsi[p])
                        and high[s] < high[p] and rsi[s] > rsi[p]):
                    side = -1
            if side:
                counts["hidden_divergence_setups"] += 1
                bid_stop = min(low[i], low[i - 1], low[s]) if side == 1 else max(high[i], high[i - 1], high[s])
                next_ix = next_lookup.get(int(candle.end), -1)
                entry_ix = int(m30.first_ix.iloc[next_ix]) if next_ix >= 0 else -1
                entry_epoch = int(tape.time.iloc[entry_ix]) if entry_ix >= 0 else None
                bid = float(tape.open.iloc[entry_ix]) if entry_ix >= 0 else np.nan
                entry_exec = bid + (cost if side == 1 else 0)
                stop_exec = bid_stop + (cost if side == -1 else 0)
                bid_risk = side * (bid - bid_stop)
                risk = side * (entry_exec - stop_exec)
                target = entry_exec + side * 3 * risk
                wall = None
                killzone = False
                rejection = ""
                if entry_epoch is None:
                    rejection = "NO_EXACT_NEXT_REAL_M30_BUCKET"
                    counts["no_next_bucket"] += 1
                else:
                    local = pd.Timestamp(entry_epoch, unit="s", tz="UTC").tz_convert("America/New_York")
                    wall = local.hour * 60 + local.minute
                    killzone = 120 <= wall < 300 or 480 <= wall < 660
                    if entry_epoch < score_start:
                        rejection = "ENTRY_BEFORE_SCORE_START"
                        counts["unscored_entry"] += 1
                    elif bid_risk <= cost:
                        rejection = "BID_STOP_DISTANCE_NOT_ABOVE_COST"
                        counts["degenerate_risk"] += 1
                row = {"asset": asset, "setup_id": len(out) + 1, "side": side,
                       "trigger_ix": i, "mother_ix": i - 1, "sweep_ix": s, "pivot_ix": p,
                       "pivot_confirmation_ix": p + 2, "anchor_ix": anchor_ix, "current_h4_ix": hi,
                       "trigger_start_epoch": int(candle.start), "signal_epoch": int(candle.end),
                       "entry_bucket_epoch": int(candle.end), "entry_ix": entry_ix, "entry_epoch": entry_epoch,
                       "entry_bid": bid, "entry_ny_wall_minute": wall, "killzone": killzone,
                       "anchor_close": anchor.close, "anchor_ema50": anchor.ema50,
                       "anchor_low": anchor.low, "anchor_high": anchor.high,
                       "anchor_end_epoch": int(anchor.end), "current_h4_start_epoch": int(h4.start.iloc[hi]),
                       "trigger_high": high[i], "trigger_low": low[i], "mother_high": high[i - 1], "mother_low": low[i - 1],
                       "sweep_price": low[s] if side == 1 else high[s], "sweep_rsi": rsi[s],
                       "sweep_end_epoch": int(m30.end.iloc[s]), "pivot_price": low[p] if side == 1 else high[p],
                       "pivot_rsi": rsi[p], "pivot_confirmation_epoch": int(m30.end.iloc[p + 2]),
                       "stop_bid_reference": bid_stop, "bid_risk": bid_risk, "initial_risk": risk,
                       "entry_exec": entry_exec, "stop_exec": stop_exec, "target_exec": target,
                       "eligible_entry": not rejection, "rejection_reason": rejection}
                out.append(row)
        # Update extrema ONLY AFTER evaluating t against the state through t-1.
        if sweep_low < 0 or low[i] < low[sweep_low]:
            sweep_low = i
            low_pivot = latest_low if latest_low >= max(0, i - 40) else -1
        if sweep_high < 0 or high[i] > high[sweep_high]:
            sweep_high = i
            high_pivot = latest_high if latest_high >= max(0, i - 40) else -1
    frame = pd.DataFrame(out, columns=SETUP_COLUMNS)
    for row in frame.itertuples(index=False):
        assert row.pivot_confirmation_epoch <= row.sweep_end_epoch < row.signal_epoch
        assert row.anchor_end_epoch <= row.current_h4_start_epoch <= row.trigger_start_epoch
        assert row.pivot_ix >= row.trigger_ix - 40 and row.pivot_ix + 2 <= row.sweep_ix < row.trigger_ix
        assert row.trigger_high <= row.mother_high and row.trigger_low >= row.mother_low
        if row.eligible_entry:
            assert row.entry_epoch >= row.signal_epoch and row.entry_epoch < row.signal_epoch + 1800
            assert row.bid_risk > cost and np.isclose(row.initial_risk, row.bid_risk + cost, rtol=0, atol=1e-8)
    return frame, counts


@njit(cache=True)
def walk(opens, highs, lows, start, side, cost, stop, target):
    adjustment = cost if side == -1 else 0.0
    for j in range(start, len(opens)):
        quote_open, quote_high, quote_low = opens[j] + adjustment, highs[j] + adjustment, lows[j] + adjustment
        stopped = quote_low <= stop if side == 1 else quote_high >= stop
        targeted = quote_high >= target if side == 1 else quote_low <= target
        if stopped:
            gap = quote_open < stop if side == 1 else quote_open > stop
            fill = quote_open if gap else stop
            return j, fill - adjustment, fill, 1, targeted, gap
        if targeted:
            return j, target - adjustment, target, 2, False, False
    return -1, np.nan, np.nan, 0, False, False


def replay(asset, arm, tape, setups, cost):
    o, h, l, c = [tape[k].to_numpy(float) for k in ("open", "high", "low", "close")]
    times = tape.time.to_numpy(np.int64)
    occupied_until = -1
    records = []
    funnel = {"eligible_setups": int(setups.eligible_entry.sum()), "session_filtered": 0, "shadowed_occupied": 0}
    for row in setups.loc[setups.eligible_entry].itertuples(index=False):
        if arm == "KILLZONE_ONLY" and not row.killzone:
            funnel["session_filtered"] += 1
            continue
        if row.entry_ix <= occupied_until:
            funnel["shadowed_occupied"] += 1
            continue
        xi, exit_bid, exit_exec, kind, collision, gap = walk(o, h, l, row.entry_ix, row.side, cost, row.stop_exec, row.target_exec)
        closed = xi >= 0
        final_ix = xi if closed else len(tape) - 1
        entry_dt = pd.Timestamp(row.entry_epoch, unit="s", tz="UTC").tz_convert("America/New_York")
        exit_epoch = int(times[final_ix])
        exit_dt = pd.Timestamp(exit_epoch, unit="s", tz="UTC").tz_convert("America/New_York")
        if closed:
            gross = row.side * (exit_bid - row.entry_bid)
            net = row.side * (exit_exec - row.entry_exec)
            assert np.isclose(gross - net, cost, rtol=0, atol=1e-8)
            net_r = net / row.initial_risk
            assert kind != 2 or np.isclose(net_r, 3, rtol=0, atol=1e-7)
            assert kind != 1 or gap or np.isclose(net_r, -1, rtol=0, atol=1e-7)
            terminal_net = terminal_r = np.nan
        else:
            gross = net = net_r = np.nan
            mark_exec = c[-1] + (cost if row.side == -1 else 0)
            terminal_net = row.side * (mark_exec - row.entry_exec)
            terminal_r = terminal_net / row.initial_risk
        records.append({"asset": asset, "arm": arm, "trade_id": len(records) + 1,
                        "setup_id": row.setup_id, "side": row.side, "entry_ix": row.entry_ix, "exit_ix": xi,
                        "signal_epoch": row.signal_epoch, "entry_epoch": row.entry_epoch,
                        "exit_epoch": exit_epoch if closed else None,
                        "entry_bid": row.entry_bid, "entry_exec": row.entry_exec,
                        "stop_bid_reference": row.stop_bid_reference, "stop_exec": row.stop_exec,
                        "target_exec": row.target_exec, "initial_risk": row.initial_risk,
                        "exit_bid": exit_bid, "exit_exec": exit_exec, "cost_floor": cost,
                        "gross_units": gross, "net_units": net, "net_r": net_r,
                        "exit_type": "SL" if kind == 1 else "TP" if kind == 2 else "OPEN_AT_TAPE_END",
                        "collision": collision, "adverse_gap": gap,
                        "hold_calendar_hours": (exit_epoch - row.entry_epoch) / 3600,
                        "ny_date_crossed": entry_dt.date() != exit_dt.date(),
                        "entry_year": entry_dt.year, "closed": closed,
                        "terminal_net_units_mark": terminal_net, "terminal_net_r_mark": terminal_r})
        occupied_until = xi if closed else len(tape)
    ledger = pd.DataFrame(records, columns=TRADE_COLUMNS)
    if len(ledger) > 1:
        assert np.all(ledger.entry_ix.to_numpy()[1:] > np.where(ledger.closed, ledger.exit_ix, len(tape))[:-1])
    assert len(ledger.loc[~ledger.closed]) <= 1
    funnel["executed"] = len(ledger)
    return ledger, funnel


def mbb(values):
    values = np.asarray(values, float)
    if len(values) == 0:
        return 1.0
    block = min(BLOCK, len(values))
    count, remainder = divmod(len(values), block)
    sums = np.convolve(values, np.ones(block), mode="valid")
    prefix = np.r_[0.0, np.cumsum(values)]
    rng = np.random.default_rng(SEED)
    nonpositive = 0
    for _ in range(20):
        indices = rng.integers(len(sums), size=(500, count + bool(remainder)))
        sampled = sums[indices[:, :count]].sum(axis=1)
        if remainder:
            starts = indices[:, -1]
            sampled += prefix[starts + remainder] - prefix[starts]
        nonpositive += int((sampled <= 0).sum())
    return nonpositive / BOOT


def metrics(asset, arm, ledger):
    closed = ledger.loc[ledger.closed]
    n = len(closed)
    gross = float(closed.gross_units.mean()) if n else None
    net = float(closed.net_units.mean()) if n else None
    mean_r = float(closed.net_r.mean()) if n else None
    std = float(closed.net_r.std(ddof=1)) if n > 1 else 0.0
    sharpe = mean_r / std if n > 1 and std > 0 else None
    probability = mbb(closed.net_r)
    g1 = n >= 100
    g2 = n > 0 and net > 0 and mean_r > 0
    g3 = sharpe is not None and sharpe > 0
    g4 = probability <= .05
    annual = {str(int(year)): {"closed": len(rows), "net_units": float(rows.net_units.sum()), "net_r": float(rows.net_r.sum())}
              for year, rows in closed.groupby("entry_year")}
    return {"asset": asset, "arm": arm, "closed_trades": n, "open_at_tape_end": int((~ledger.closed).sum()),
            "win_rate": float(closed.net_units.gt(0).mean()) if n else None,
            "gross_expectancy_units": gross, "net_expectancy_units": net,
            "net_r_per_trade": mean_r, "net_r_sharpe": sharpe, "mbb_p": probability,
            "total_closed_net_units": float(closed.net_units.sum()), "total_closed_net_r": float(closed.net_r.sum()),
            "terminal_net_units_mark": float(ledger.terminal_net_units_mark.sum()),
            "terminal_net_r_mark": float(ledger.terminal_net_r_mark.sum()),
            "stop_exits": int(closed.exit_type.eq("SL").sum()), "target_exits": int(closed.exit_type.eq("TP").sum()),
            "collisions": int(closed.collision.sum()), "adverse_gaps": int(closed.adverse_gap.sum()),
            "overnight_or_weekend": int(ledger.ny_date_crossed.sum()),
            "average_hold_hours": float(closed.hold_calendar_hours.mean()) if n else None,
            "G1": bool(g1), "G2": bool(g2), "G3": bool(g3), "G4": bool(g4),
            "verdict": "PASS" if g1 and g2 and g3 and g4 else "FAIL", "annual_entry_cohorts": annual}


def check_examples():
    # DST boundaries: NY00-04 has3h in spring and5h in autumn.
    keys = pd.to_datetime(["2024-03-10 00:00", "2024-11-03 00:00"]).asi8
    starts, ends = local_boundary_epochs(keys)
    assert np.array_equal(ends - starts, [10800, 18000])
    repeated = pd.to_datetime(["2024-11-03 05:00Z", "2024-11-03 06:00Z"])
    assert repeated.tz_convert("America/New_York").hour.tolist() == [1, 1]
    assert len(np.unique(repeated.asi8 // (1800 * 10**9))) == 2
    # Closed-fractal confirmation, strict ties, and SMA-seeded RSI boundaries.
    ph, pl = fractals(np.array([1., 2, 3, 2, 1]), np.array([3., 2, 1, 2, 3]))
    assert ph[2] and pl[2]
    tied, _ = fractals(np.array([1., 2, 3, 3, 1]), np.arange(5.))
    assert not tied[2]
    assert np.isnan(wilder_rsi(np.arange(20.))[13]) and wilder_rsi(np.arange(20.))[14] == 100
    assert wilder_rsi(np.ones(20))[14] == 50
    for side in (1, -1):
        cost = .5
        bid = 100.0
        entry = bid + (cost if side == 1 else 0)
        stop = 90.0 if side == 1 else 110.5
        risk = side * (entry - stop)
        target = entry + side * 3 * risk
        both = walk(np.array([100.]), np.array([160.]), np.array([40.]), 0, side, cost, stop, target)
        assert both[3] == 1 and both[4]
        assert np.isclose(side * (both[2] - entry) / risk, -1)
        gap_open = 80.0 if side == 1 else 120.0
        gap = walk(np.array([gap_open]), np.array([gap_open + 1]), np.array([gap_open - 1]), 0, side, cost, stop, target)
        assert gap[3] == 1 and gap[5] and np.isclose(gap[1], gap_open)
        tp_open = 150.0 if side == 1 else 50.0
        tp = walk(np.array([tp_open]), np.array([tp_open + 1]), np.array([tp_open - 1]), 0, side, cost, stop, target)
        assert tp[3] == 2 and np.isclose(side * (tp[2] - entry) / risk, 3)


def fmt(value, places=5):
    return "N/A" if value is None else f"{value:.{places}f}"


def main():
    if REPORT.exists():
        raise FileExistsError("Historical report exists; freeze a new version for any change")
    spec = (OUT / "SPEC_EDRIS_IB.txt").read_text(encoding="utf-8")
    gate = (OUT / "GATE_EDRIS_IB.txt").read_text(encoding="utf-8").strip()
    assert gate == "\n".join(x for x in spec.splitlines() if x.startswith(tuple(f"G{i}:" for i in range(1, 5))))
    check_examples()
    results, sources, funnels = [], {}, {}
    for asset, path, cost, year in ASSETS:
        tape, meta = load_source(path)
        print(f"{asset}: retained {len(tape):,}, removed {meta['removed_carried']:,}; building NY grids", flush=True)
        m30, h4 = build_grids(tape)
        setups, funnel = signals(asset, tape, m30, h4, cost, year)
        meta.update(m30_bars=len(m30), h4_bars=len(h4), cost_floor=cost,
                    h4_3hour_count=int((h4.end - h4.start).eq(10800).sum()),
                    h4_5hour_count=int((h4.end - h4.start).eq(18000).sum()))
        sources[asset] = meta
        setups.to_csv(OUT / f"setups_{asset.lower()}.csv", index=False)
        m30.to_csv(OUT / f"m30_{asset.lower()}.csv", index=False)
        h4.to_csv(OUT / f"h4_{asset.lower()}.csv", index=False)
        print(f"{asset}: {len(setups):,} divergence/inside setups; {int(setups.eligible_entry.sum()):,} eligible", flush=True)
        funnels[asset] = {"signals": funnel, "arms": {}}
        for arm in ARMS:
            ledger, arm_funnel = replay(asset, arm, tape, setups, cost)
            ledger.to_csv(OUT / f"trades_{asset.lower()}_{arm.lower()}.csv", index=False)
            result = metrics(asset, arm, ledger)
            results.append(result)
            funnels[asset]["arms"][arm] = arm_funnel
            print(f"{asset}/{arm}: n={result['closed_trades']}, net R={fmt(result['net_r_per_trade'])}, "
                  f"MBB={result['mbb_p']:.4f}, {result['verdict']}", flush=True)
    assert len(results) == 6
    pd.DataFrame([{k: v for k, v in row.items() if k != "annual_entry_cohorts"} for row in results]).to_csv(OUT / "summary.csv", index=False)
    metadata = {"study_date": "2026-10-14", "sources": sources, "funnels": funnels,
                "script_sha256": digest(Path(__file__)), "spec_sha256": digest(OUT / "SPEC_EDRIS_IB.txt"),
                "gate_sha256": digest(OUT / "GATE_EDRIS_IB.txt"),
                "bootstrap": {"iterations": BOOT, "block": BLOCK, "seed_each_cell": SEED},
                "execution_examples": "PASS", "results": results,
                "scope": "Exploratory fixed-rule proxy test; no walk-forward, broker or prop validation"}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["HYPOTHESIS #96: H4 SWEEP / M30 HIDDEN RSI DIVERGENCE / STRICT INSIDE BAR - 2026-10-14",
             "EXPLORATORY PHASE 1; NO INDEPENDENT WALK-FORWARD OR LIVE/PROP VALIDATION", "",
             "Cost floors once embedded: US5000.51, USTECH1.66, XAUUSD0.41 native units.",
             "Signals: prior observed completed NY H4 EMA50 trend, current-bucket sweep through t-1,",
             "pivot confirmed by sweep s close, strictly hidden RSI14 divergence, strict M30 inside bar t.",
             "Entry: first retained M1 open in exact next M30 bucket; single occupancy per asset/arm.",
             "Risk = directional BID distance + once-embedded cost; capped executable target+3R, stop-1R unless adverse gap.",
             "Killzones: actual NY entry time [02:00,05:00) or [08:00,11:00).", "",
             "SIX-CELL PERFORMANCE MATRIX",
             "Asset   Arm             Closed Open  Win%    GrossMean   NetMean     NetR/Trade RSharpe   MBBp   Verdict"]
    for row in results:
        win = fmt(row["win_rate"] * 100 if row["win_rate"] is not None else None, 2)
        lines.append(f"{row['asset']:7s} {row['arm']:15s} {row['closed_trades']:6d} {row['open_at_tape_end']:4d} "
                     f"{win:7s} {fmt(row['gross_expectancy_units']):11s} {fmt(row['net_expectancy_units']):11s} "
                     f"{fmt(row['net_r_per_trade']):10s} {fmt(row['net_r_sharpe']):9s} {row['mbb_p']:.4f} {row['verdict']}")
        lines.append(f"  Gates G1={row['G1']} G2={row['G2']} G3={row['G3']} G4={row['G4']}; "
                     f"SL{row['stop_exits']} TP{row['target_exits']}, collisions{row['collisions']}, adverse gaps{row['adverse_gaps']}.")
        lines.append(f"  Closed net native total{row['total_closed_net_units']:.6f}, totalR{row['total_closed_net_r']:.6f}; "
                     f"unresolved mark native{row['terminal_net_units_mark']:.6f}, R{row['terminal_net_r_mark']:.6f}; "
                     f"NY-date-crossing positions{row['overnight_or_weekend']}, average hold{fmt(row['average_hold_hours'], 2)} hours.")
        lines.append(f"  Annual entry cohorts: {json.dumps(row['annual_entry_cohorts'], sort_keys=True)}")
    lines += ["", "SOURCE / GRID AUDIT"]
    for asset, meta in sources.items():
        lines += [f"{asset}: {meta['first_utc']}..{meta['last_m1_open_utc']}; raw{meta['raw_rows']:,}, "
                  f"retained{meta['retained_rows']:,}, carried removed{meta['removed_carried']:,}; "
                  f"observed M30{meta['m30_bars']:,}, H4{meta['h4_bars']:,}; DST3h{meta['h4_3hour_count']},5h{meta['h4_5hour_count']}.",
                  f"  SHA256 {meta['sha256']}", f"  Funnel {json.dumps(funnels[asset])}"]
    lines += ["", "LIMITS / HONESTY",
              "The supplied fixed floors are scenarios, not measured current broker fees, variable spreads or slippage.",
              "Gold0.41 is the explicit requested floor. No switch to0.585 or optimization after outcomes.",
              "There are no swap/margin/account-level/prop guards. Overnight/weekend holds and terminal open marks are disclosed.",
              "Carried-row predicate can also exclude genuine repeated flat observations; first row has no predecessor.",
              "Observed-bar indicators do not manufacture weekend/synthetic candles. Partial source-boundary bars do not establish complete prehistory.",
              "M30 repeated DST wall hours remain distinct; H4 bounds use actual NY localization. Pivots never use future confirmation.",
              "Within-M1 event order is unknown; adverse-first collisions and adverse open gaps are deliberately pessimistic.",
              "MBB is unadjusted conditional uncertainty on already-researched history, not a research-wide multiple-testing correction.",
              "Owner100-trade PASS is exploratory; no independent walk-forward or live profitability claim follows.",
              "Artifacts: frozen SPEC/GATE, setup and context grids, six full trade ledgers, summary.csv and metadata.json."]
    with REPORT.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
