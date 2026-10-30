"""H83: frozen FTMO XAUUSD London opening-range sweep reversal screen."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/ftmo_xauusd_london_sweep_h83"
SPEC = OUT / "SPEC_FTMO_XAU_LONDON_SWEEP_H83.txt"
GATE = OUT / "GATE_FTMO_XAU_LONDON_SWEEP_H83.txt"
REPORT = OUT / "REPORT_FTMO_XAU_LONDON_SWEEP_H83.txt"
SOURCE = OUT / "ftmo_xauusd_m15.csv"
TERMINAL = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
SERVER_TZ = ZoneInfo("Europe/Athens")
LONDON = ZoneInfo("Europe/London")
PRAGUE = ZoneInfo("Europe/Prague")
SPREAD = 0.41
SEED = 20260926
RISKS = (0.005, 0.01, 0.015, 0.02)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def check_frozen() -> None:
    if REPORT.exists() or SOURCE.exists():
        raise FileExistsError("H83 source or REPORT already exists; never overwrite")
    lines = [line for line in SPEC.read_text(encoding="utf-8").splitlines()
             if line.startswith("G-LS")]
    if len(lines) != 6 or lines != GATE.read_text(encoding="utf-8").splitlines():
        raise ValueError("H83 SPEC/GATE mismatch")


def export_m15() -> tuple[pd.DataFrame, dict]:
    import MetaTrader5 as mt5

    if not mt5.initialize(TERMINAL):
        raise RuntimeError("Target FTMO terminal connection failed")
    try:
        account = mt5.account_info()
        symbol = mt5.symbol_info("XAUUSD")
        if account is None or account.server != "FTMO-Demo" or "ftmo" not in str(account.company).lower():
            raise RuntimeError("Connected broker account is not the expected FTMO-Demo")
        if symbol is None or symbol.custom or abs(symbol.trade_contract_size - 100) > 1e-6:
            raise RuntimeError("XAUUSD is not the expected native 100-unit contract")
        was_visible = bool(symbol.visible)
        if not mt5.symbol_select("XAUUSD", True):
            raise RuntimeError("Cannot subscribe to native FTMO XAUUSD")
        try:
            rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15,
                                         datetime(2024, 1, 1, tzinfo=timezone.utc),
                                         datetime(2026, 9, 24, 23, 45, tzinfo=timezone.utc))
            if rates is None or len(rates) < 50_000:
                raise RuntimeError("M15 broker history unavailable or unexpectedly short")
            frame = pd.DataFrame(rates)[["time", "open", "high", "low", "close", "tick_volume", "spread"]]
            # Same source and clock domain as the bars. This is a spot check,
            # not a proof that all historical bars equal an executable quote.
            sample_at = datetime(2025, 9, 25, 15, 0, tzinfo=timezone.utc)
            ticks = mt5.copy_ticks_range("XAUUSD", sample_at,
                                         sample_at + timedelta(seconds=5), mt5.COPY_TICKS_ALL)
        finally:
            if not was_visible and mt5.account_info() is not None and mt5.account_info().login == account.login:
                mt5.symbol_select("XAUUSD", False)
    finally:
        mt5.shutdown()

    raw = frame.time.to_numpy(np.int64)
    if not np.all(np.diff(raw) > 0):
        raise ValueError("Broker M15 wall timestamps repeat or reverse")
    corrected = np.array([
        int(datetime.fromtimestamp(int(stamp), timezone.utc).replace(tzinfo=None)
            .replace(tzinfo=SERVER_TZ).astimezone(timezone.utc).timestamp())
        for stamp in raw], dtype=np.int64)
    unique_utc = bool(np.all(np.diff(corrected) > 0) and np.all(corrected % 900 == 0))
    if not unique_utc:
        raise ValueError("M15 broker clock conversion not unique quarter-hours")
    if not ((frame.low > 0).all()
            and (frame.low <= frame[["open", "close"]].min(axis=1)).all()
            and (frame.high >= frame[["open", "close"]].max(axis=1)).all()):
        raise ValueError("Invalid M15 BID OHLC")
    frame.insert(0, "utc_epoch", corrected)
    frame.rename(columns={"time": "server_epoch"}, inplace=True)
    sample_server = int(sample_at.timestamp())
    bar = frame.loc[frame.server_epoch.eq(sample_server)]
    tick_bid = float(ticks[0]["bid"]) if ticks is not None and len(ticks) else None
    tick_delay = ((int(ticks[0]["time_msc"]) - sample_server * 1000) / 1000
                  if ticks is not None and len(ticks) else None)
    bar_open = float(bar.open.iloc[0]) if len(bar) == 1 else None
    tick_aligned = bool(bar_open is not None and tick_bid is not None and
                        0 <= tick_delay <= 5 and abs(tick_bid - bar_open) <= 0.25)
    frame.to_csv(SOURCE, index=False, float_format="%.5f")
    info = {"source": "FTMO-Demo XAUUSD native MT5 M15 BID OHLC",
            "clock": "server wall epoch localized Europe/Athens then converted UTC",
            "rows": len(frame), "sha256": sha256(SOURCE),
            "first_utc": datetime.fromtimestamp(int(corrected[0]), timezone.utc).isoformat(),
            "last_utc": datetime.fromtimestamp(int(corrected[-1]), timezone.utc).isoformat(),
            "largest_gap_hours": float(np.diff(corrected).max() / 3600),
            "non_custom": True, "contract_size": 100, "strict_unique_utc_quarters": unique_utc,
            "sample_server_bar_epoch": sample_server, "sample_bar_open_bid": bar_open,
            "sample_first_tick_bid": tick_bid, "sample_tick_delay_s": tick_delay,
            "sample_tick_aligned": tick_aligned}
    return frame, info


def signal_for_day(day: pd.DataFrame) -> tuple[dict | None, str]:
    """Use only completed M15 bars through the signal; entry is the next bar."""
    indexed = day.set_index("london_minute", drop=False)
    required = [480, 495, 510, 525]
    if not all(k in indexed.index for k in required):
        return None, "MISSING_OPENING_BAR"
    opening = indexed.loc[required]
    high = float(opening.high.max())
    low = float(opening.low.min())
    width = high - low
    if not width > 0.82:
        return None, "NARROW_OPENING_RANGE"
    state = None
    extreme = None
    scan = day[day.london_minute.between(540, 705)]
    for row in scan.itertuples(index=False):
        upper = row.high > high
        lower = row.low < low
        if state is None:
            if upper and lower:
                return None, "AMBIGUOUS_FIRST_BREAK"
            if upper:
                state, extreme = "SHORT", float(row.high)
            elif lower:
                state, extreme = "LONG", float(row.low)
        if state == "SHORT":
            extreme = max(extreme, float(row.high))
            if lower:
                return None, "OPPOSITE_EDGE_BEFORE_RETURN"
        elif state == "LONG":
            extreme = min(extreme, float(row.low))
            if upper:
                return None, "OPPOSITE_EDGE_BEFORE_RETURN"
        if state is not None and low < row.close < high:
            return {"side": state, "signal_epoch": int(row.utc_epoch),
                    "range_high": high, "range_low": low,
                    "range_width": width, "sweep_extreme": extreme}, "SIGNAL"
    return None, "NO_RETURN_SIGNAL"


def execute_one(day: pd.DataFrame, signal: dict) -> tuple[dict | None, str]:
    by_epoch = day.set_index("utc_epoch", drop=False)
    entry_epoch = signal["signal_epoch"] + 900
    if entry_epoch not in by_epoch.index:
        return None, "MISSING_ENTRY_BAR"
    entry = by_epoch.loc[entry_epoch]
    side = signal["side"]
    width = signal["range_width"]
    entry_bid = float(entry.open)
    entry_ask = entry_bid + SPREAD
    if side == "LONG":
        stop = signal["sweep_extreme"] - 0.25 * width  # BID stop
        target = signal["range_high"]  # BID target
        risk = entry_ask - stop
        reward = target - entry_ask
        valid = entry_bid > stop
    else:
        stop = signal["sweep_extreme"] + 0.25 * width  # ASK stop
        target = signal["range_low"]  # ASK target
        risk = stop - entry_bid
        reward = entry_bid - target
        valid = entry_ask < stop
    if not (risk > 0 and reward > 0 and valid):
        return None, "INVALID_ENTRY_GEOMETRY"

    path = day[(day.utc_epoch >= entry_epoch) & (day.london_minute <= 945)]
    final_exists = bool((day.london_minute == 945).any())
    previous_epoch = None
    exit_fill = None
    exit_reason = None
    exit_row = None
    used = []
    for row in path.itertuples(index=False):
        if previous_epoch is not None and row.utc_epoch != previous_epoch + 900:
            exit_reason = "UNRESOLVED_GAP"
            break
        previous_epoch = row.utc_epoch
        used.append(row)
        if side == "LONG":
            stop_hit = row.low <= stop
            target_hit = row.high >= target
            if stop_hit:
                exit_fill, exit_reason = min(stop, float(row.open)), "STOP"
            elif target_hit:
                exit_fill, exit_reason = target, "TARGET"
        else:
            stop_hit = row.high + SPREAD >= stop
            target_hit = row.low + SPREAD <= target
            if stop_hit:
                exit_fill, exit_reason = max(stop, float(row.open) + SPREAD), "STOP"
            elif target_hit:
                exit_fill, exit_reason = target, "TARGET"
        if exit_reason is not None:
            exit_row = row
            break
    if not used:
        return None, "MISSING_ENTRY_BAR"
    if exit_reason is None:
        exit_row = used[-1]
        if final_exists and exit_row.london_minute == 945:
            exit_reason = "DAY_FLAT"
        else:
            exit_reason = "UNRESOLVED_FLAT"
        exit_fill = float(exit_row.close) if side == "LONG" else float(exit_row.close) + SPREAD
    if exit_reason == "UNRESOLVED_GAP":
        exit_row = used[-1]
        exit_fill = float(exit_row.close) if side == "LONG" else float(exit_row.close) + SPREAD
    unresolved = exit_reason.startswith("UNRESOLVED")
    net = (exit_fill - entry_ask) if side == "LONG" else (entry_bid - exit_fill)
    low_seen = min(float(row.low) for row in used)
    high_seen = max(float(row.high) for row in used)
    worst = low_seen - entry_ask if side == "LONG" else entry_bid - (high_seen + SPREAD)
    entry_dt = datetime.fromtimestamp(entry_epoch, timezone.utc)
    end_dt = datetime.fromtimestamp(int(exit_row.utc_epoch) + 900 - 0.001, timezone.utc)
    row = {"london_date": entry_dt.astimezone(LONDON).date(),
           "year": entry_dt.astimezone(LONDON).year,
           "entry_prague_date": entry_dt.astimezone(PRAGUE).date(),
           "exit_prague_date": end_dt.astimezone(PRAGUE).date(),
           "side": side, "signal_utc": datetime.fromtimestamp(signal["signal_epoch"], timezone.utc).isoformat(),
           "entry_utc": entry_dt.isoformat(), "exit_bar_end_utc": end_dt.isoformat(),
           "range_high": signal["range_high"], "range_low": signal["range_low"],
           "range_width": width, "sweep_extreme": signal["sweep_extreme"],
           "entry_bid": entry_bid, "entry_ask": entry_ask,
           "initial_stop": stop, "target": target, "exit_fill": exit_fill,
           "exit_reason": exit_reason, "unresolved": unresolved,
           "risk_units": risk, "target_units": reward,
           "net_units": net, "stress_units": net - SPREAD,
           "realized_r": (net - SPREAD) / risk,
           "worst_float_r": (worst - SPREAD) / risk,
           "hold_hours_upper": (int(exit_row.utc_epoch) + 900 - entry_epoch) / 3600,
           "path_bars": len(used)}
    return row, exit_reason


def run_days(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    stamps = pd.to_datetime(bars.utc_epoch, unit="s", utc=True).dt.tz_convert("Europe/London")
    bars = bars.copy()
    bars["london_date"] = stamps.dt.date
    bars["london_minute"] = stamps.dt.hour * 60 + stamps.dt.minute
    trades = []
    days = []
    funnel: dict[str, int] = {}
    for day_date, group in bars.groupby("london_date", sort=True):
        if day_date.weekday() >= 5 or day_date < date(2024, 1, 1) or day_date > date(2026, 9, 24):
            continue
        signal, reason = signal_for_day(group)
        if signal is not None:
            trade, reason = execute_one(group, signal)
            if trade is not None:
                trades.append(trade)
        else:
            trade = None
        funnel[reason] = funnel.get(reason, 0) + 1
        days.append({"london_date": day_date, "status": reason,
                     "side": signal["side"] if signal else None,
                     "trade": trade is not None})
    return pd.DataFrame(trades), pd.DataFrame(days), funnel


def bootstrap(trades: pd.DataFrame) -> dict:
    grouped = trades.groupby("london_date", sort=True).net_units.agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(int)
    if len(sums) == 0:
        return {"date_blocks": 0, "p_mean_le_zero": 1.0}
    rng = np.random.default_rng(SEED)
    means = np.empty(10_000)
    for start in range(0, 10_000, 250):
        end = min(start + 250, 10_000)
        picks = rng.integers(0, len(sums), size=(end - start, len(sums)))
        means[start:end] = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return {"date_blocks": len(sums), "p_mean_le_zero": float(np.mean(means <= 0)),
            "ci95_low": float(np.quantile(means, 0.025)),
            "ci95_high": float(np.quantile(means, 0.975))}


def replay_window(trades: pd.DataFrame, start: date, risk: float) -> dict:
    end = start + timedelta(days=180)
    balance = 10_000.0
    day_open = balance
    last_prague_day = None
    trading_days = set()
    for trade in trades.itertuples(index=False):
        if trade.london_date < start or trade.london_date >= end:
            continue
        if trade.entry_prague_date != last_prague_day:
            day_open = balance
            last_prague_day = trade.entry_prague_date
        trading_days.add(trade.entry_prague_date)
        floor = max(9_000.0, day_open - 500.0)
        floating = balance + 10_000.0 * risk * trade.worst_float_r
        if floating < floor:
            return {"outcome": "BREACH", "days": (trade.london_date - start).days}
        balance += 10_000.0 * risk * trade.realized_r
        if balance < floor:
            return {"outcome": "BREACH", "days": (trade.london_date - start).days}
        if balance >= 11_000.0 and len(trading_days) >= 4:
            return {"outcome": "PASS", "days": (trade.london_date - start).days}
    return {"outcome": "TIMEOUT", "days": 180}


def challenge(trades: pd.DataFrame, last_source_london_day: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    starts = sorted({d for d in trades.london_date
                     if d + timedelta(days=180) <= last_source_london_day + timedelta(days=1)})
    outcomes = []
    for risk in RISKS:
        for start in starts:
            outcomes.append({"risk_pct": risk * 100, "start_date": start.isoformat(),
                             **replay_window(trades, start, risk)})
    frame = pd.DataFrame(outcomes)
    summary = []
    for pct, group in frame.groupby("risk_pct", sort=True):
        passed = group.loc[group.outcome.eq("PASS"), "days"]
        summary.append({"risk_pct": pct, "starts": len(group), "pass": len(passed),
                        "breach": int(group.outcome.eq("BREACH").sum()),
                        "timeout": int(group.outcome.eq("TIMEOUT").sum()),
                        "pass_rate": float(group.outcome.eq("PASS").mean()),
                        "breach_rate": float(group.outcome.eq("BREACH").mean()),
                        "median_pass_days": float(passed.median()) if len(passed) else None})
    return frame, pd.DataFrame(summary)


def main() -> None:
    check_frozen()
    bars, source = export_m15()
    all_trades, roster, funnel = run_days(bars)
    if all_trades.empty:
        raise ValueError("No trades; cannot write H83 metrics")
    test = all_trades[all_trades.london_date >= date(2025, 1, 1)].copy()
    if test.empty:
        raise ValueError("No gated trades")
    inference = bootstrap(test)
    last_source = datetime.fromtimestamp(int(bars.utc_epoch.iloc[-1]), timezone.utc).astimezone(LONDON).date()
    outcomes, pace = challenge(test, last_source)
    yearly = {str(y): {"n": len(group), "net_total_units": float(group.net_units.sum()),
                       "stress_mean_units": float(group.stress_units.mean())}
              for y, group in all_trades.groupby("year", sort=True)}
    unresolved = int(test.unresolved.sum())
    missing_entry = int(((roster.london_date >= date(2025, 1, 1))
                         & roster.status.eq("MISSING_ENTRY_BAR")).sum())
    same_day = bool((test.entry_prague_date == test.exit_prague_date).all()
                    and (test.hold_hours_upper < 16).all()
                    and (test.risk_units > 0).all() and (test.target_units > 0).all())
    qualifying_pace = pace[(pace.starts >= 150) & (pace.pass_rate >= 0.70)
                           & (pace.breach_rate <= 0.10)]
    gates = {"G-LS1": bool(len(test) >= 150 and test.london_date.nunique() >= 150
                           and unresolved == 0 and missing_entry == 0),
             "G-LS2": bool(test.stress_units.mean() > 0),
             "G-LS3": bool(inference["p_mean_le_zero"] <= 0.05),
             "G-LS4": bool(yearly.get("2025", {}).get("net_total_units", 0) > 0
                           and yearly.get("2026", {}).get("net_total_units", 0) > 0),
             "G-LS5": bool(len(qualifying_pace) > 0),
             "G-LS6": bool(same_day and source["non_custom"]
                           and source["strict_unique_utc_quarters"]
                           and source["sample_tick_aligned"] and len(source["sha256"]) == 64)}
    all_trades.to_csv(OUT / "trades.csv", index=False, float_format="%.8f")
    roster.to_csv(OUT / "day_roster.csv", index=False)
    outcomes.to_csv(OUT / "rolling_outcomes.csv", index=False, float_format="%.8f")
    pace.to_csv(OUT / "pace_summary.csv", index=False, float_format="%.8f")
    net = test.net_units.to_numpy(float)
    eq = np.r_[0, np.cumsum(net)]
    payload = {"generated_utc": datetime.now(timezone.utc).isoformat(),
               "spec_sha256": sha256(SPEC), "gate_sha256": sha256(GATE),
               "source": source, "funnel": funnel, "years": yearly,
               "gated": {"n": len(test), "dates": int(test.london_date.nunique()),
                         "net_mean_units": float(net.mean()),
                         "net_total_units": float(net.sum()),
                         "stress_mean_units": float(test.stress_units.mean()),
                         "mean_stressed_r": float(test.realized_r.mean()),
                         "win_rate": float(np.mean(net > 0)),
                         "profit_factor": float(net[net > 0].sum() / abs(net[net < 0].sum()))
                         if (net < 0).any() else None,
                         "max_drawdown_units": float(np.max(np.maximum.accumulate(eq) - eq)),
                         "median_hold_hours_upper": float(test.hold_hours_upper.median()),
                         "unresolved": unresolved, "missing_entry": missing_entry,
                         "side": test.groupby("side").net_units.agg(["size", "sum"]).to_dict("index"),
                         "exit_reasons": test.exit_reason.value_counts().to_dict()},
               "bootstrap": inference,
               "pace": pace.astype(object).where(pd.notna(pace), None).to_dict("records"),
               "gates": gates, "verdict": "PASS" if all(gates.values()) else "FAIL",
               "limitations": "M15 BID bar model with fixed spread, optimistic next-open and level-touch fills; no commissions, variable spread, slippage, 0.01 lot rounding or historical EUR conversion. External source did not test this exact fade rule."}
    (OUT / "run_meta.json").write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n",
                                        encoding="utf-8")
    REPORT.write_text("H83 FTMO GOLD LONDON SWEEP PHASE-1 REPORT\n"
                      + json.dumps(payload, indent=2, allow_nan=False, default=str)
                      + "\nOVERALL VERDICT: " + payload["verdict"] + "\n", encoding="utf-8")
    print(json.dumps({"verdict": payload["verdict"], "gated": payload["gated"],
                      "years": yearly, "bootstrap": inference, "pace": payload["pace"],
                      "gates": gates}, indent=2, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
