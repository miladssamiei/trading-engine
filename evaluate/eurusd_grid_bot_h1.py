"""Frozen EURUSD volume/EMA grid experiment; see SPEC_GRID.txt.

H1 drives every order. Real M1 observations audit underwater time and locate
the adverse extreme. No cap, loss exit, financing or terminal liquidation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/results/eurusd_onboarding/eurusdtick_m1.csv"
OUT = ROOT / "data/results/eurusd_grid"
REPORT = OUT / "REPORT_GRID.txt"
PIP = 0.0001
SPREAD_PIPS = 0.3
SPACING = 0.003
MULTIPLIER = 1.5
TARGET = 30.0
EPS = 1e-9


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def utc(epoch):
    return pd.Timestamp(int(epoch), unit="s", tz="UTC").isoformat()


def load_grid():
    tape = pd.read_csv(SOURCE, usecols=["time", "open", "high", "low", "close", "tick_volume"])
    times = tape.time.to_numpy(dtype=np.int64)
    price = tape[["open", "high", "low", "close"]].to_numpy(float)
    volume = tape.tick_volume.to_numpy(float)
    if (len(tape) < 2 or np.any(np.diff(times) <= 0) or np.any(times % 60)
            or not np.isfinite(price).all() or np.any(price <= 0)
            or np.any(price[:, 1] < price.max(axis=1))
            or np.any(price[:, 2] > price.min(axis=1))
            or not np.isfinite(volume).all() or np.any(volume < 0)
            or np.any(volume != np.floor(volume))):
        raise ValueError("Invalid real BID M1 OHLC/tick-volume source")
    tape["tick_volume"] = tape.tick_volume.astype(np.int64)
    tape["key"] = times // 3600
    tape["ix"] = np.arange(len(tape))
    observed = tape.groupby("key", sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), tick_volume=("tick_volume", "sum"),
        first_ix=("ix", "first"), last_ix=("ix", "last"))
    keys = np.arange(times[0] // 3600, times[-1] // 3600 + 1)
    grid = observed.reindex(keys)
    grid["real"] = grid.first_ix.notna()
    grid[["open", "high", "low", "close"]] = grid[["open", "high", "low", "close"]].ffill()
    grid["tick_volume"] = grid.tick_volume.fillna(0).astype(np.int64)
    grid[["first_ix", "last_ix"]] = grid[["first_ix", "last_ix"]].fillna(-1).astype(np.int64)
    grid["start"] = keys * 3600
    grid["end"] = (keys + 1) * 3600
    grid["ema5"] = grid.close.ewm(span=5, adjust=False, min_periods=5).mean()
    increasing = grid.tick_volume.shift(1).gt(grid.tick_volume.shift(2)) & grid.tick_volume.shift(2).gt(grid.tick_volume.shift(3))
    eligible = increasing & grid.real & grid.real.shift(1, fill_value=False)
    grid["signal"] = np.select(
        [eligible & grid.close.shift(1).gt(grid.ema5.shift(1)),
         eligible & grid.close.shift(1).lt(grid.ema5.shift(1))], [1, -1], default=0)
    assert grid.tick_volume.sum() == tape.tick_volume.sum()
    assert (grid.loc[~grid.real, "tick_volume"] == 0).all()
    # Independent reduce-at sums check every real H1 bucket against the M1 tape.
    begins = grid.loc[grid.real, "first_ix"].to_numpy()
    assert np.array_equal(np.add.reduceat(tape.tick_volume.to_numpy(), begins), observed.tick_volume.to_numpy())
    return tape, grid.reset_index(drop=True)


def floating(basket, price):
    entries = np.array([p["entry_bid"] for p in basket["positions"]])
    sizes = np.array([p["size"] for p in basket["positions"]])
    quotes = np.asarray(price)
    gross = basket["side"] * np.sum((quotes[..., None] - entries) * sizes, axis=-1) / PIP
    return gross - SPREAD_PIPS * sizes.sum()


def basket_record(basket, end, bid, closed):
    q = sum(p["size"] for p in basket["positions"])
    net = float(floating(basket, bid))
    seconds = end - basket["entry_epoch"]
    return {
        "basket_id": basket["id"], "side": "LONG" if basket["side"] == 1 else "SHORT",
        "status": "CLOSED_TARGET" if closed else "OPEN_AT_TAPE_END",
        "entry_epoch": basket["entry_epoch"], "end_epoch": end,
        "entry_time_utc": utc(basket["entry_epoch"]), "end_time_utc": utc(end),
        "mark_bid": bid, "position_count": len(basket["positions"]), "total_size": q,
        "max_individual_size": basket["positions"][-1]["size"],
        "gross_base_pips": net + SPREAD_PIPS * q, "spread_base_pips": SPREAD_PIPS * q,
        "net_base_pips": net, "max_floating_loss_base_pips": basket["mae"],
        "worst_minute_utc": utc(basket["worst_epoch"]),
        "close_marked_peak_to_trough_base_pips": basket["close_dd"],
        "calendar_underwater_hours_estimate": basket["underwater_seconds"] / 3600,
        "observed_m1_underwater_hours": basket["negative_minutes"] / 60,
        "observed_minutes": basket["observed_minutes"], "hold_calendar_hours": seconds / 3600,
        "first_underwater_to_end_hours": seconds / 3600,
    }


def replay(tape, grid):
    times, closes = tape.time.to_numpy(), tape.close.to_numpy()
    highs, lows = tape.high.to_numpy(), tape.low.to_numpy()
    baskets, positions, curve = [], [], []
    active = None
    next_id = 0
    realized = 0.0
    shadow_opposite = shadow_spacing = adds = 0
    for i, bar in enumerate(grid.itertuples(index=False)):
        closed_id = None
        if not bar.real:
            if active is not None and active["last_net"] < -EPS:
                active["underwater_seconds"] += 3600
        else:
            first, last = bar.first_ix, bar.last_ix
            entry_epoch = int(times[first])
            if active is not None and active["last_net"] < -EPS:
                active["underwater_seconds"] += entry_epoch - bar.start
            should_order = False
            if bar.signal:
                if active is None:
                    next_id += 1
                    active = {"id": next_id, "side": bar.signal, "entry_epoch": entry_epoch,
                              "positions": [], "mae": 0.0, "worst_epoch": entry_epoch,
                              "peak_close": 0.0, "close_dd": 0.0, "underwater_seconds": 0,
                              "negative_minutes": 0, "observed_minutes": 0, "last_net": 0.0}
                    should_order = True
                elif bar.signal != active["side"]:
                    shadow_opposite += 1
                else:
                    previous = active["positions"][-1]["entry_bid"]
                    if active["side"] * (bar.open - previous) <= -SPACING + 1e-12:
                        should_order = True
                        adds += 1
                    else:
                        shadow_spacing += 1
            if should_order:
                depth = len(active["positions"]) + 1
                size = 1.0 if depth == 1 else active["positions"][-1]["size"] * MULTIPLIER
                prior = grid.iloc[i - 1]
                assert i >= 3 and prior.real and prior.tick_volume > grid.iloc[i - 2].tick_volume > grid.iloc[i - 3].tick_volume
                assert (prior.close - prior.ema5) * active["side"] > 0
                assert prior.end <= entry_epoch
                position = {"basket_id": active["id"], "level": depth,
                            "side": active["side"], "size": size, "entry_bid": bar.open,
                            "entry_exec": bar.open + (SPREAD_PIPS * PIP if active["side"] == 1 else 0),
                            "entry_epoch": entry_epoch, "entry_time_utc": utc(entry_epoch),
                            "entry_bucket": bar.start, "entry_delay_minutes": (entry_epoch - bar.start) / 60,
                            "signal_completed_utc": utc(prior.end), "prior_close": prior.close,
                            "prior_ema5": prior.ema5,
                            "volume_t1": int(prior.tick_volume),
                            "volume_t2": int(grid.iloc[i - 2].tick_volume),
                            "volume_t3": int(grid.iloc[i - 3].tick_volume)}
                if depth > 1:
                    assert active["side"] * (bar.open - active["positions"][-1]["entry_bid"]) <= -SPACING + 1e-12
                active["positions"].append(position)
            if active is not None:
                # All inventory is fixed during this hour. Its worst BID quote
                # is consequently also the basket's worst net liquidation mark.
                adverse = lows[first:last + 1] if active["side"] == 1 else highs[first:last + 1]
                offset = int(np.argmin(adverse) if active["side"] == 1 else np.argmax(adverse))
                adverse_net = float(floating(active, adverse[offset]))
                if max(0.0, -adverse_net) > active["mae"]:
                    active["mae"] = max(0.0, -adverse_net)
                    active["worst_epoch"] = int(times[first + offset])
                minute_net = floating(active, closes[first:last + 1])
                negative = minute_net < -EPS
                duration = np.diff(np.append(times[first:last + 1], bar.end))
                active["underwater_seconds"] += int(duration[negative].sum())
                active["negative_minutes"] += int(negative.sum())
                active["observed_minutes"] += last - first + 1
                active["last_net"] = float(minute_net[-1])
                active["peak_close"] = max(active["peak_close"], active["last_net"])
                active["close_dd"] = max(active["close_dd"], active["peak_close"] - active["last_net"])
                if active["last_net"] > TARGET + EPS:
                    record = basket_record(active, int(bar.end), float(bar.close), True)
                    baskets.append(record)
                    for p in active["positions"]:
                        p = p.copy()
                        p.update(exit_epoch=int(bar.end), exit_time_utc=utc(bar.end),
                                 last_quote_epoch=int(times[last]), exit_bid=float(bar.close),
                                 exit_exec=float(bar.close) + (SPREAD_PIPS * PIP if p["side"] == -1 else 0),
                                 status="CLOSED_TARGET")
                        p["gross_base_pips"] = p["side"] * (p["exit_bid"] - p["entry_bid"]) / PIP * p["size"]
                        p["spread_base_pips"] = SPREAD_PIPS * p["size"]
                        p["net_base_pips"] = p["gross_base_pips"] - p["spread_base_pips"]
                        positions.append(p)
                    realized += record["net_base_pips"]
                    closed_id = active["id"]
                    active = None
        net = active["last_net"] if active is not None else 0.0
        curve.append({"hour_start_epoch": bar.start, "hour_end_epoch": bar.end, "real": bar.real,
                      "bid_close": bar.close, "signal": bar.signal,
                      "active_basket_id": active["id"] if active else None,
                      "closed_basket_id": closed_id, "depth": len(active["positions"]) if active else 0,
                      "total_size": sum(p["size"] for p in active["positions"]) if active else 0,
                      "floating_net_base_pips": net, "realized_net_base_pips": realized,
                      "marked_equity_base_pips": realized + net,
                      "active_max_floating_loss_base_pips": active["mae"] if active else None})
    if active is not None:
        end = int(grid.iloc[-1].end)
        bid = float(tape.close.iloc[-1])
        baskets.append(basket_record(active, end, bid, False))
        for p in active["positions"]:
            p = p.copy()
            p.update(exit_epoch=None, exit_time_utc=None, last_quote_epoch=int(times[-1]),
                     exit_bid=bid, exit_exec=bid + (SPREAD_PIPS * PIP if p["side"] == -1 else 0),
                     status="OPEN_AT_TAPE_END")
            p["gross_base_pips"] = p["side"] * (bid - p["entry_bid"]) / PIP * p["size"]
            p["spread_base_pips"] = SPREAD_PIPS * p["size"]
            p["net_base_pips"] = p["gross_base_pips"] - p["spread_base_pips"]
            positions.append(p)
    return pd.DataFrame(baskets), pd.DataFrame(positions), pd.DataFrame(curve), {
        "ignored_opposite_signals": shadow_opposite, "ignored_spacing_signals": shadow_spacing,
        "grid_additions": adds}


def verify_ledgers(baskets, positions, curve):
    assert not positions.entry_bucket.duplicated().any()
    assert np.allclose(positions.gross_base_pips - positions.net_base_pips,
                       SPREAD_PIPS * positions["size"], rtol=1e-10, atol=1e-7)
    for row in baskets.itertuples(index=False):
        p = positions.loc[positions.basket_id.eq(row.basket_id)].sort_values("level")
        assert p.side.nunique() == 1 and len(p) == row.position_count
        assert np.allclose(p["size"], MULTIPLIER ** np.arange(len(p)), rtol=1e-12, atol=0)
        if len(p) > 1:
            assert np.all(p.side.iloc[0] * np.diff(p.entry_bid) <= -SPACING + 1e-12)
        assert np.isclose(p.net_base_pips.sum(), row.net_base_pips, rtol=1e-10, atol=1e-7)
        assert row.calendar_underwater_hours_estimate <= row.hold_calendar_hours + EPS
        assert row.observed_m1_underwater_hours <= row.hold_calendar_hours + EPS
        assert row.max_floating_loss_base_pips >= max(0.0, -row.net_base_pips) - 1e-7
        if row.status == "CLOSED_TARGET":
            assert row.net_base_pips > TARGET + EPS
    assert baskets.loc[baskets.status.eq("OPEN_AT_TAPE_END")].shape[0] <= 1
    if len(baskets) > 1:
        assert np.all(baskets.entry_epoch.to_numpy()[1:] >= baskets.end_epoch.to_numpy()[:-1])
    assert not curve.loc[~curve.real, "closed_basket_id"].notna().any()
    assert np.isclose(curve.marked_equity_base_pips.iloc[-1], baskets.net_base_pips.sum(), rtol=1e-10, atol=1e-7)


def main():
    if REPORT.exists():
        raise FileExistsError("Historical REPORT exists; freeze a new study for any rerun")
    spec = (OUT / "SPEC_GRID.txt").read_text(encoding="utf-8")
    gate = (OUT / "GATE_GRID.txt").read_text(encoding="utf-8").strip()
    assert gate == "\n".join(line for line in spec.splitlines() if line.startswith(tuple(f"G{i}:" for i in range(1, 6))))
    tape, grid = load_grid()
    print(f"Loaded {len(tape):,} M1 bars; {len(grid):,} continuous H1 buckets", flush=True)
    baskets, positions, curve, ignored = replay(tape, grid)
    verify_ledgers(baskets, positions, curve)
    baskets.to_csv(OUT / "baskets.csv", index=False)
    positions.to_csv(OUT / "positions.csv", index=False)
    curve.to_csv(OUT / "hourly_curve.csv", index=False)
    grid.to_csv(OUT / "h1_grid.csv", index=False)
    closed = baskets.loc[baskets.status.eq("CLOSED_TARGET")]
    opened = baskets.loc[baskets.status.eq("OPEN_AT_TAPE_END")]
    worst = baskets.loc[baskets.max_floating_loss_base_pips.idxmax()]
    stats = {
        "closed_baskets": len(closed), "open_baskets": len(opened),
        "closed_basket_win_rate": float(closed.net_base_pips.gt(0).mean()) if len(closed) else None,
        "positions_opened": len(positions),
        "max_floating_loss_base_pips": float(worst.max_floating_loss_base_pips),
        "worst_basket_id": int(worst.basket_id), "worst_basket_status": worst.status,
        "worst_basket_minute_utc": worst.worst_minute_utc,
        "max_grid_depth": int(baskets.position_count.max()),
        "max_total_size": float(baskets.total_size.max()),
        "max_individual_size": float(baskets.max_individual_size.max()),
        "average_calendar_underwater_hours_closed": float(closed.calendar_underwater_hours_estimate.mean()) if len(closed) else None,
        "average_observed_underwater_hours_closed": float(closed.observed_m1_underwater_hours.mean()) if len(closed) else None,
        "average_hold_calendar_hours_closed": float(closed.hold_calendar_hours.mean()) if len(closed) else None,
        "max_hold_calendar_hours_closed": float(closed.hold_calendar_hours.max()) if len(closed) else None,
        "total_closed_net_base_pips": float(closed.net_base_pips.sum()),
        "terminal_open_net_base_pips": float(opened.net_base_pips.sum()),
        "marked_total_net_base_pips": float(baskets.net_base_pips.sum()),
        "total_spread_base_pips_all_positions": float(positions.spread_base_pips.sum()),
        "max_close_marked_peak_to_trough_base_pips": float(baskets.close_marked_peak_to_trough_base_pips.max()),
    }
    metadata = {"study_date": "2026-10-05", "source": str(SOURCE), "source_sha256": digest(SOURCE),
                "source_first_utc": utc(tape.time.iloc[0]), "source_last_m1_open_utc": utc(tape.time.iloc[-1]),
                "source_rows": len(tape), "source_tick_volume_sum": int(tape.tick_volume.sum()),
                "continuous_hours": len(grid), "real_hours": int(grid.real.sum()),
                "synthetic_hours": int((~grid.real).sum()), "signals": int(grid.signal.ne(0).sum()),
                "script_sha256": digest(Path(__file__)), "spec_sha256": digest(OUT / "SPEC_GRID.txt"),
                "gate_sha256": digest(OUT / "GATE_GRID.txt"), "stats": stats, "ignored": ignored,
                "gates": {f"G{i}": "PASS" for i in range(1, 6)},
                "scope": "Exploratory integrity checks only; not walk-forward, broker or prop validation"}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    win = f"{stats['closed_basket_win_rate']:.2%}" if len(closed) else "N/A"
    lines = ["HYPOTHESIS #95 - EURUSD H1 VOLUME/EMA GRID - REPORT - 2026-10-05",
             "EXPLORATORY PHASE 1; INTEGRITY PASS DOES NOT VALIDATE LIVE PERFORMANCE", "",
             f"Source: {SOURCE.relative_to(ROOT)}",
             f"Actual tape: {metadata['source_first_utc']} through {metadata['source_last_m1_open_utc']} (last M1 open)",
             f"M1 rows: {len(tape):,}; real H1 hours: {metadata['real_hours']:,}; filled hours: {metadata['synthetic_hours']:,}",
             f"Source SHA256: {metadata['source_sha256']}",
             "Continuous UTC H1; EMA5; strictly rising three completed hourly tick volumes.",
             "Base size1; grid30pips; multiplier1.5; basket target strictly >30 base-unit net pips.",
             "Fixed round-trip spread0.3pips times each position's size, embedded once.", "",
             "REQUESTED METRICS",
             f"Total baskets closed: {len(closed):,}", f"Win rate of CLOSED baskets: {win}",
             f"Absolute maximum floating loss (all baskets): {stats['max_floating_loss_base_pips']:,.6f} base-unit pip equivalents",
             f"Worst basket: {stats['worst_basket_id']} ({stats['worst_basket_status']}); adverse M1 minute: {stats['worst_basket_minute_utc']}",
             f"Maximum grid depth: {stats['max_grid_depth']} positions",
             f"Maximum combined size: {stats['max_total_size']:,.6f} base units; largest individual: {stats['max_individual_size']:,.6f}"]
    if len(closed):
        lines += [f"Average calendar underwater time before closing (M1-mark/gap-carry estimate): {stats['average_calendar_underwater_hours_closed']:,.4f} hours / {stats['average_calendar_underwater_hours_closed']/24:,.4f} days",
                  f"Average observed-M1 underwater time: {stats['average_observed_underwater_hours_closed']:,.4f} hours",
                  f"Average total basket hold: {stats['average_hold_calendar_hours_closed']:,.4f} hours; maximum: {stats['max_hold_calendar_hours_closed']:,.4f} hours"]
    lines += ["", "REALIZED PROFITS AND UNRESOLVED EXPOSURE",
              f"Positions opened: {len(positions):,}; grid additions: {ignored['grid_additions']:,}",
              f"Closed net: {stats['total_closed_net_base_pips']:,.6f} base-unit pip equivalents",
              f"Baskets open at tape end: {len(opened)}; terminal floating net: {stats['terminal_open_net_base_pips']:,.6f}",
              f"Closed net + terminal mark: {stats['marked_total_net_base_pips']:,.6f}",
              f"Total modeled spread, including open positions: {stats['total_spread_base_pips_all_positions']:,.6f}",
              f"Maximum close-marked peak-to-trough floating drawdown: {stats['max_close_marked_peak_to_trough_base_pips']:,.6f}"]
    for row in opened.itertuples(index=False):
        lines.append(f"OPEN basket {row.basket_id}: {row.side}, start {row.entry_time_utc}, depth{row.position_count}, "
                     f"size{row.total_size:,.6f}, floating{row.net_base_pips:,.6f}, max loss{row.max_floating_loss_base_pips:,.6f}, "
                     f"hold{row.hold_calendar_hours:,.4f}h, underwater calendar estimate{row.calendar_underwater_hours_estimate:,.4f}h.")
    lines += ["", "INTERPRETATION AND LIMITS",
              "Closed-basket wins are mechanically selected by the positive-profit-only exit. A 100% closed win rate is not evidence of an edge.",
              "No stop, position cap, margin limit, financing or prop equity guard is included. Inventory is allowed to grow exactly as specified.",
              "All PnL/risk is in base-unit pip equivalents. Multiply by the actual cash value of one pip at the chosen base size to obtain account-currency amounts.",
              "Tick volume counts quote activity, not centralized exchange traded volume or an order book.",
              "Adverse loss includes actual intrahour extrema. Close-marked drawdown is separate and does not infer OHLC event order.",
              "Calendar underwater time is an M1-close estimate with the last mark carried over missing quotes/weekends; observed-minute time excludes missing minutes.",
              "Entries use the first real minute in a bucket; exits use completed H1 close proxies, not independently verified executable broker tick fills.",
              "End-of-tape inventory stays open and is included in maximum risk and marked PnL. This full-tape fixed-rule replay is exploratory, not walk-forward validation.",
              "", "GATE: G1-G5 PASS FOR EXECUTION/DATA INTEGRITY ONLY", *gate.splitlines(), "",
              "ARTIFACTS: frozen SPEC_GRID/GATE_GRID; baskets.csv; positions.csv; hourly_curve.csv; h1_grid.csv; metadata.json; this report."]
    with REPORT.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
