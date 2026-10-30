"""Frozen 1H EMA200 anchor-bias ATR bracket grid on BID M1 proxy tapes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/ema200_1h_grid"
ASSETS = (
    ("US500", ROOT / "data/results/vwap_2021_2026/us500tick_m1.csv", 0.51, 9, 30, 1800),
    ("XAUUSD", ROOT / "data/results/xauusd_onboarding/xauusdtick_m1.csv", 0.41, 8, 0, 0),
)
SL_MULTS = (1.0, 2.0, 3.0)
TP_MULTS = (0.5, 1.0, 2.0, 3.0)
BOOT, BLOCK, SEED = 10_000, 5, 20260930


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_m1(path: Path) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path, usecols=["time", "open", "high", "low", "close"])
    data = {name: frame[name].to_numpy(dtype=np.int64 if name == "time" else np.float64)
            for name in ("time", "open", "high", "low", "close")}
    t, o, h, l, c = (data[name] for name in ("time", "open", "high", "low", "close"))
    if (len(t) < 100_000 or np.any(np.diff(t) <= 0) or np.any(t % 60 != 0)
            or not np.isfinite(np.column_stack((o, h, l, c))).all()
            or np.any(o <= 0) or np.any(l <= 0)
            or np.any(h < np.maximum(o, c)) or np.any(l > np.minimum(o, c))):
        raise ValueError(f"Invalid M1 BID tape: {path}")
    return data


def continuous_h1(data: dict[str, np.ndarray], offset: int) -> dict[str, np.ndarray]:
    bucket = (data["time"] - offset) // 3600
    grouped = pd.DataFrame({"bucket": bucket, "open": data["open"],
                            "high": data["high"], "low": data["low"],
                            "close": data["close"]}).groupby("bucket", sort=True).agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"))
    keys = np.arange(int(bucket[0]), int(bucket[-1]) + 1, dtype=np.int64)
    grid = grouped.reindex(keys).ffill()
    real = np.isin(keys, grouped.index.to_numpy(np.int64), assume_unique=True)
    if not np.isfinite(grid.to_numpy()).all():
        raise ValueError("Continuous H1 grid contains non-finite values")
    return {"start": keys * 3600 + offset, "real": real,
            **{field: grid[field].to_numpy(np.float64)
               for field in ("open", "high", "low", "close")}}


def ema200(close: np.ndarray) -> np.ndarray:
    out = np.empty(len(close), dtype=np.float64)
    out[0] = close[0]
    alpha = 2.0 / 201.0
    for i in range(1, len(close)):
        out[i] = alpha * close[i] + (1.0 - alpha) * out[i - 1]
    out[:199] = np.nan
    return out


def atr14(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    previous = np.r_[close[0], close[:-1]]
    tr = np.maximum(high - low, np.maximum(np.abs(high - previous),
                                            np.abs(low - previous)))
    out = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        out[13] = float(np.mean(tr[:14]))
        for i in range(14, len(tr)):
            out[i] = (13.0 * out[i - 1] + tr[i]) / 14.0
    return out


def candidates(data: dict[str, np.ndarray], grid: dict[str, np.ndarray],
               anchor_hour: int, anchor_minute: int) -> tuple[pd.DataFrame, dict]:
    wall = pd.to_datetime(data["time"], unit="s", utc=True).tz_convert("America/New_York")
    exact = np.flatnonzero((wall.hour == anchor_hour) & (wall.minute == anchor_minute))
    dates = wall[exact].strftime("%Y-%m-%d").to_numpy()
    if len(np.unique(dates)) != len(dates):
        raise AssertionError("More than one M1 anchor on a NY calendar date")
    ema = ema200(grid["close"])
    atr = atr14(grid["high"], grid["low"], grid["close"])
    hidx = np.searchsorted(grid["start"], data["time"][exact], side="left")
    if np.any(hidx >= len(grid["start"])) or np.any(grid["start"][hidx] != data["time"][exact]):
        raise AssertionError("Anchor M1 is not the current shifted H1 bucket open")
    if not np.allclose(grid["open"][hidx], data["open"][exact], atol=1e-9, rtol=0):
        raise AssertionError("Anchor M1 BID open differs from real H1 bucket open")
    prior = hidx - 1
    eligible = (prior >= 199) & np.isfinite(ema[np.maximum(prior, 0)]) & np.isfinite(atr[np.maximum(prior, 0)])
    price = data["open"][exact]
    ref_ema = ema[np.maximum(prior, 0)]
    side = np.where(price > ref_ema, 1, np.where(price < ref_ema, -1, 0))
    selected = eligible & (side != 0)
    out = pd.DataFrame({"ny_date": dates[selected], "entry_ix": exact[selected],
                        "entry_time": data["time"][exact[selected]],
                        "side": side[selected], "entry_bid": price[selected],
                        "prior_h1_start": grid["start"][prior[selected]],
                        "prior_ema200": ref_ema[selected], "prior_atr14": atr[prior[selected]]})
    if out.ny_date.duplicated().any():
        raise AssertionError("One-entry-per-NY-date rule violated")
    return out, {"real_anchor_dates": len(exact), "indicator_warmup_skips": int(np.count_nonzero(~eligible)),
                 "equal_open_skips": int(np.count_nonzero(eligible & (side == 0))),
                 "eligible_signals": len(out), "synthetic_h1_buckets": int(np.count_nonzero(~grid["real"])),
                 "total_h1_buckets": len(grid["start"])}


@njit(cache=True)
def walk_m1(open_bid: np.ndarray, high_bid: np.ndarray, low_bid: np.ndarray,
            entry_ix: np.ndarray, side: np.ndarray, entry_bid: np.ndarray,
            atr: np.ndarray, cost: float, sl_mult: float, tp_mult: float):
    n = len(entry_ix)
    exit_ix = np.full(n, -1, np.int64)
    exit_bid = np.full(n, np.nan)
    exit_exec = np.full(n, np.nan)
    exit_type = np.zeros(n, np.int8)  # 0 OPEN_END, 1 STOP, 2 TARGET
    collision = np.zeros(n, np.bool_)
    stop_gap = np.zeros(n, np.bool_)
    for i in range(n):
        risk = sl_mult * atr[i]
        reward = tp_mult * atr[i]
        if side[i] == 1:
            stop = entry_bid[i] - risk
            target = entry_bid[i] + reward
        else:
            stop = entry_bid[i] + risk
            target = entry_bid[i] - reward
        for j in range(entry_ix[i], len(open_bid)):
            if side[i] == 1:
                hit_stop = low_bid[j] <= stop
                hit_target = high_bid[j] >= target
                if hit_stop:
                    filled_bid = min(stop, open_bid[j])
                    exit_ix[i] = j
                    exit_bid[i] = filled_bid
                    exit_exec[i] = filled_bid
                    exit_type[i] = 1
                    collision[i] = hit_target
                    stop_gap[i] = open_bid[j] < stop
                    break
                if hit_target:
                    exit_ix[i] = j
                    exit_bid[i] = target
                    exit_exec[i] = target
                    exit_type[i] = 2
                    break
            else:
                ask_open = open_bid[j] + cost
                hit_stop = high_bid[j] + cost >= stop
                hit_target = low_bid[j] + cost <= target
                if hit_stop:
                    filled_ask = max(stop, ask_open)
                    exit_ix[i] = j
                    exit_bid[i] = filled_ask - cost
                    exit_exec[i] = filled_ask
                    exit_type[i] = 1
                    collision[i] = hit_target
                    stop_gap[i] = ask_open > stop
                    break
                if hit_target:
                    exit_ix[i] = j
                    exit_bid[i] = target - cost
                    exit_exec[i] = target
                    exit_type[i] = 2
                    break
    return exit_ix, exit_bid, exit_exec, exit_type, collision, stop_gap


def mbb_p(net: np.ndarray, asset_id: int, sl_id: int, tp_id: int) -> float:
    n = len(net)
    if n < BLOCK:
        return float("nan")
    rng = np.random.default_rng(np.random.SeedSequence([SEED, asset_id, sl_id, tp_id]))
    block_sums = np.convolve(net, np.ones(BLOCK), mode="valid")
    block_count, extra = divmod(n, BLOCK)
    tail = np.convolve(net, np.ones(extra), mode="valid") if extra else None
    bad = 0
    for begin in range(0, BOOT, 250):
        draws = min(250, BOOT - begin)
        starts = rng.integers(0, n - BLOCK + 1, size=(draws, block_count + int(extra > 0)))
        totals = block_sums[starts[:, :block_count]].sum(axis=1)
        if extra:
            totals += tail[starts[:, block_count]]
        bad += int(np.count_nonzero(totals <= 0.0))
    return bad / BOOT


def run_cell(asset: str, asset_id: int, data: dict[str, np.ndarray], signals: pd.DataFrame,
             cost: float, sl_mult: float, tp_mult: float, sl_id: int, tp_id: int) -> dict:
    entry_ix = signals.entry_ix.to_numpy(np.int64)
    side = signals.side.to_numpy(np.int8)
    entry_bid = signals.entry_bid.to_numpy(np.float64)
    atr = signals.prior_atr14.to_numpy(np.float64)
    exit_ix, bid_exit, exec_exit, kind, collision, stop_gap = walk_m1(
        data["open"], data["high"], data["low"], entry_ix, side, entry_bid,
        atr, cost, sl_mult, tp_mult)
    closed = kind != 0
    risk = sl_mult * atr
    entry_exec = entry_bid + np.where(side == 1, cost, 0.0)
    gross = side * (bid_exit - entry_bid)
    net = side * (exec_exit - entry_exec)
    if not np.allclose(gross[closed] - net[closed], cost, atol=1e-8, rtol=0):
        raise AssertionError(f"Cost was not embedded exactly once: {asset} {sl_mult}/{tp_mult}")
    if np.any(exit_ix[closed] < entry_ix[closed]):
        raise AssertionError("Exit predates entry")
    exit_time = np.full(len(signals), np.nan)
    exit_time[closed] = data["time"][exit_ix[closed]]
    stop = entry_bid - side * risk
    target = entry_bid + side * tp_mult * atr
    ledger = signals.copy()
    ledger.insert(0, "asset", asset)
    ledger["sl_mult"] = sl_mult
    ledger["tp_mult"] = tp_mult
    ledger["cost"] = cost
    ledger["entry_exec"] = entry_exec
    ledger["initial_risk"] = risk
    ledger["stop_exit_side"] = stop
    ledger["target_exit_side"] = target
    ledger["exit_ix"] = np.where(closed, exit_ix, np.nan)
    ledger["exit_time"] = exit_time
    ledger["exit_bid"] = bid_exit
    ledger["exit_exec"] = exec_exit
    ledger["exit_type"] = np.where(kind == 1, "STOP", np.where(kind == 2, "TARGET", "OPEN_END"))
    ledger["same_m1_collision"] = collision
    ledger["adverse_stop_gap"] = stop_gap
    ledger["gross"] = gross
    ledger["net"] = net
    ledger["net_r"] = net / risk
    ledger["entry_time_utc"] = pd.to_datetime(ledger.entry_time, unit="s", utc=True).astype(str)
    ledger["exit_time_utc"] = pd.to_datetime(ledger.exit_time, unit="s", utc=True).astype(str)
    ledger.to_csv(OUT / f"trades_{asset}_SL{sl_mult:g}_TP{tp_mult:g}.csv", index=False)
    scored = net[closed]
    scored_r = (net / risk)[closed]
    p_value = mbb_p(scored, asset_id, sl_id, tp_id)
    mean = float(np.mean(scored)) if len(scored) else float("nan")
    gate = bool(len(scored) >= 150 and mean > 0.0 and p_value <= 0.05)
    return {"asset": asset, "sl_mult": sl_mult, "tp_mult": tp_mult,
            "signals": len(signals), "trades": int(np.count_nonzero(closed)),
            "open_end": int(np.count_nonzero(~closed)),
            "win_rate": float(np.mean(scored > 0)) if len(scored) else float("nan"),
            "gross_expectancy": float(np.mean(gross[closed])) if len(scored) else float("nan"),
            "net_expectancy": mean, "net_r_expectancy": float(np.mean(scored_r)) if len(scored) else float("nan"),
            "mbb_p": p_value, "gate": "PASS" if gate else "FAIL",
            "stop_count": int(np.count_nonzero(kind == 1)),
            "target_count": int(np.count_nonzero(kind == 2)),
            "same_m1_collisions": int(np.count_nonzero(collision)),
            "adverse_stop_gaps": int(np.count_nonzero(stop_gap))}


def main() -> None:
    for name in ("SPEC_EMA200.txt", "GATE_EMA200.txt"):
        if not (OUT / name).is_file():
            raise RuntimeError(f"Predeclared document missing: {name}")
    results = []
    metadata = {"bootstrap": {"resamples": BOOT, "block_trades": BLOCK, "seed": SEED}, "assets": {}}
    for asset_id, (asset, path, cost, hour, minute, offset) in enumerate(ASSETS):
        data = load_m1(path)
        grid = continuous_h1(data, offset)
        signals, funnel = candidates(data, grid, hour, minute)
        signals.to_csv(OUT / f"signals_{asset}.csv", index=False)
        metadata["assets"][asset] = {"source": str(path), "sha256": sha256(path),
                                      "m1_rows": len(data["time"]),
                                      "first_utc": pd.to_datetime(data["time"][0], unit="s", utc=True).isoformat(),
                                      "last_utc": pd.to_datetime(data["time"][-1], unit="s", utc=True).isoformat(),
                                      "h1_utc_offset_seconds": offset, "cost": cost, "funnel": funnel}
        print(f"{asset}: {len(data['time']):,} M1 rows, {len(signals):,} entry signals", flush=True)
        for sl_id, sl in enumerate(SL_MULTS):
            for tp_id, tp in enumerate(TP_MULTS):
                result = run_cell(asset, asset_id, data, signals, cost, sl, tp, sl_id, tp_id)
                results.append(result)
                print(f"  SL {sl:g} TP {tp:g}: n={result['trades']} net={result['net_expectancy']:.4f} "
                      f"p={result['mbb_p']:.4f} {result['gate']}", flush=True)
    summary = pd.DataFrame(results)
    summary.to_csv(OUT / "summary.csv", index=False)
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report = ["EMA200 1H ANCHOR BIAS / ATR BRACKET GRID - PHASE 1",
              "All 24 cells are exploratory; no walk-forward or multiple-testing correction.",
              f"MBB P(mean net <= 0): {BOOT:,} moving-block resamples, block {BLOCK} trades, seed {SEED} with cell identifiers.",
              "PASS = closed n >= 150, net expectancy > 0, MBB p <= 0.05. OPEN_END excluded from scored metrics.",
              "Gross-minus-net equals exactly one cost floor per closed trade. Same-M1 stop wins collisions; adverse stop gaps fill at open.",
              "No time stop, so trades may hold overnight or overlap across different NY entry dates; swap = 0 by assumption.", ""]
    for asset, *_ in ASSETS:
        meta = metadata["assets"][asset]
        report.extend([f"{asset}: {meta['m1_rows']:,} M1 rows, {meta['first_utc']} to {meta['last_utc']}; cost {meta['cost']:.2f} price units.",
                       f"Anchor funnel: {meta['funnel']}",
                       "SL | TP | CLOSED | OPEN_END | WIN% | GROSS/TRADE | NET/TRADE | NET R/TRADE | MBB P | GATE"])
        for row in results:
            if row["asset"] != asset:
                continue
            report.append(f"{row['sl_mult']:>3g} | {row['tp_mult']:>3g} | {row['trades']:>6} | {row['open_end']:>8} | "
                          f"{row['win_rate']*100:>6.2f} | {row['gross_expectancy']:>11.5f} | "
                          f"{row['net_expectancy']:>9.5f} | {row['net_r_expectancy']:>11.5f} | "
                          f"{row['mbb_p']:>5.4f} | {row['gate']}")
        report.append("")
    report.extend(["Execution note: US500 H1 buckets start on the half-hour so the 09:30 ET anchor is an H1 open; XAUUSD uses clock-hour buckets at 08:00 ET.",
                   "XAUUSD uses the owner's 0.41 floor here, below the separate 0.585 all-inclusive scenario. These are proxy BID tapes, not verified FTMO fills.",
                   "A passing cell is an in-sample grid finding and needs an independently predeclared out-of-sample confirmation before it can be called validated."])
    text = "\n".join(report) + "\n"
    (OUT / "REPORT_EMA200.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
