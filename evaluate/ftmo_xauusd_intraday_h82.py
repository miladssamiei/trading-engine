"""H82: predeclared FTMO-native H1 gold intraday and 180-day pace screen."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import xauusd_saleh_donchian_phase1 as saleh
from xauusd_saleh_floating_pace_audit_v1 import replay_window
from xauusd_saleh_intraday_phase1 import walk_same_day


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/ftmo_xauusd_intraday_h82"
SPEC = OUT / "SPEC_FTMO_XAU_H82.txt"
GATE = OUT / "GATE_FTMO_XAU_H82.txt"
REPORT = OUT / "REPORT_FTMO_XAU_H82.txt"
SOURCE = OUT / "ftmo_xauusd_h1.csv"
TERMINAL = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
NY = ZoneInfo("America/New_York")
PRAGUE = ZoneInfo("Europe/Prague")
SERVER_TZ = ZoneInfo("Europe/Athens")
RISKS = (0.005, 0.01, 0.015, 0.02)
SEED = 20260925
SPREAD = 0.41
START = datetime(2023, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 9, 24, 23, tzinfo=timezone.utc)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_predeclared() -> None:
    if REPORT.exists() or SOURCE.exists():
        raise FileExistsError("H82 historical artifact already exists; never overwrite")
    lines = [line for line in SPEC.read_text(encoding="utf-8").splitlines()
             if line.startswith("G-FT")]
    if len(lines) != 6 or lines != GATE.read_text(encoding="utf-8").splitlines():
        raise ValueError("H82 SPEC/GATE lines differ")


def export_ftmo_h1() -> tuple[pd.DataFrame, dict]:
    import MetaTrader5 as mt5

    if not mt5.initialize(TERMINAL):
        raise RuntimeError("FTMO terminal connection failed")
    try:
        account = mt5.account_info()
        symbol = mt5.symbol_info("XAUUSD")
        if account is None or account.server != "FTMO-Demo" or "ftmo" not in str(account.company).lower():
            raise RuntimeError("Connected MT5 account is not the expected FTMO-Demo server")
        if symbol is None or symbol.custom or abs(symbol.trade_contract_size - 100) > 1e-6:
            raise RuntimeError("XAUUSD is not the expected broker contract")
        was_visible = bool(symbol.visible)
        if not mt5.symbol_select("XAUUSD", True):
            raise RuntimeError("Cannot select FTMO XAUUSD")
        try:
            rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_H1, START, END)
            if rates is None or len(rates) < 10_000:
                raise RuntimeError("FTMO H1 history unavailable or unexpectedly short")
            frame = pd.DataFrame(rates)[["time", "open", "high", "low", "close", "tick_volume", "spread"]]
        finally:
            if not was_visible and mt5.account_info() is not None and mt5.account_info().login == account.login:
                mt5.symbol_select("XAUUSD", False)
    finally:
        mt5.shutdown()

    raw = frame["time"].to_numpy(np.int64)
    if not np.all(np.diff(raw) > 0):
        raise ValueError("FTMO server H1 timestamps are not strictly increasing")
    # On this terminal, MT5 encodes broker wall time as an epoch. Never
    # interpret the returned number directly as UTC for NY session mapping.
    corrected = []
    for stamp in raw:
        wall = datetime.fromtimestamp(int(stamp), timezone.utc).replace(tzinfo=None)
        corrected.append(int(wall.replace(tzinfo=SERVER_TZ).astimezone(timezone.utc).timestamp()))
    corrected = np.array(corrected, dtype=np.int64)
    unique_utc = bool(np.all(np.diff(corrected) > 0) and np.all(corrected % 3600 == 0))
    if not unique_utc:
        raise ValueError("FTMO server-to-UTC H1 conversion is nonunique or nonhourly")
    if not ((frame["low"] <= frame[["open", "close"]].min(axis=1)).all()
            and (frame["high"] >= frame[["open", "close"]].max(axis=1)).all()
            and (frame["low"] > 0).all()):
        raise ValueError("Invalid FTMO H1 OHLC")
    frame.insert(0, "utc_epoch", corrected)
    frame.rename(columns={"time": "server_epoch"}, inplace=True)
    frame.to_csv(SOURCE, index=False, float_format="%.5f")
    info = {
        "source": "connected FTMO-Demo XAUUSD MT5 copy_rates_range H1",
        "clock_conversion": "server wall epoch localized to Europe/Athens, then UTC",
        "rows": len(frame), "first_server_epoch": int(raw[0]), "last_server_epoch": int(raw[-1]),
        "first_utc": datetime.fromtimestamp(int(corrected[0]), timezone.utc).isoformat(),
        "last_utc": datetime.fromtimestamp(int(corrected[-1]), timezone.utc).isoformat(),
        "largest_gap_hours": float(np.diff(corrected).max() / 3600),
        "strict_unique_utc_hours": unique_utc, "sha256": sha256(SOURCE),
        "non_custom": True, "contract_size": 100,
    }
    return frame, info


def build_context(bars: pd.DataFrame) -> dict:
    t = bars.utc_epoch.to_numpy(np.int64)
    slot = t // 3600
    if not np.all(np.diff(slot) > 0):
        raise ValueError("Converted H1 bars contain duplicate or reversed UTC hours")
    fields = ("open", "high", "low", "close")
    indexed = bars.set_index(pd.Index(slot))
    grid1 = np.arange(int(slot[0]), int(slot[-1]) + 1, dtype=np.int64)
    r1 = indexed.reindex(grid1).ffill()
    real1 = np.isin(grid1, slot)
    ny1 = pd.to_datetime(pd.Index(grid1 * 3600), unit="s", utc=True).tz_convert("America/New_York")
    d1 = (ny1.year * 10000 + ny1.month * 100 + ny1.day).to_numpy(np.int64)
    h1w = ny1.hour.to_numpy(np.int32)

    ny_real = pd.to_datetime(t, unit="s", utc=True).tz_convert("America/New_York")
    k4 = ((ny_real.year * 10000 + ny_real.month * 100 + ny_real.day).to_numpy(np.int64)
          * 100 + ((ny_real.hour.to_numpy(np.int32) // 4) * 4))
    g4 = (pd.DataFrame({"key": k4, "o": bars.open, "h": bars.high,
                        "l": bars.low, "c": bars.close})
          .groupby("key", sort=True)
          .agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last")))
    days = pd.date_range(ny_real[0].date(), ny_real[-1].date(), freq="D")
    grid4 = np.array([int(day.strftime("%Y%m%d")) * 100 + hour
                      for day in days for hour in (0, 4, 8, 12, 16, 20)], dtype=np.int64)
    r4 = g4.reindex(grid4).ffill()
    real4 = np.isin(grid4, g4.index.to_numpy(np.int64))
    first4 = int(np.flatnonzero(real4)[0])
    grid4, real4, r4 = grid4[first4:], real4[first4:], r4.iloc[first4:].reset_index(drop=True)

    o1, h1, l1, c1 = [r1[f].to_numpy(float) for f in fields]
    h4, l4, c4 = [r4[f].to_numpy(float) for f in ("h", "l", "c")]
    adx1 = saleh.wilder_adx(pd.Series(h1), pd.Series(l1), pd.Series(c1))
    atr1 = saleh.wilder_atr(h1, l1, c1, saleh.ATR_PERIOD)
    dcup = pd.Series(h1).rolling(saleh.DC_N, min_periods=saleh.DC_N).max().to_numpy(float)
    dclo = pd.Series(l1).rolling(saleh.DC_N, min_periods=saleh.DC_N).min().to_numpy(float)
    st_line, st_dir = saleh.supertrend(h4, l4, c4)
    containing = d1 * 100 + (h1w // 4) * 4
    mapped4 = np.searchsorted(grid4, containing, side="right") - 2
    anchor = np.zeros(len(grid1), dtype=np.int8)
    ok = mapped4 >= 0
    anchor[ok] = st_dir[mapped4[ok]]
    return {"n1": len(grid1), "grid1": grid1, "real1": real1,
            "o1": o1, "h1": h1, "l1": l1, "c1": c1,
            "d1": d1, "h1w": h1w, "grid4": grid4, "real4": real4,
            "o4": r4["o"].to_numpy(float), "h4": h4, "l4": l4, "c4": c4,
            "adx1": adx1, "atr1": atr1, "dcup": dcup, "dclo": dclo,
            "st_line": st_line, "st_dir": st_dir, "mapped4": mapped4, "anchor": anchor}


def walk_candidates(ctx: dict, candidates: list[tuple[int, int]]) -> tuple[pd.DataFrame, dict]:
    day_indices: dict[int, list[int]] = {}
    for index in np.flatnonzero(ctx["real1"] & (ctx["h1w"] <= 15)):
        day_indices.setdefault(int(ctx["d1"][index]), []).append(int(index))
    rows = []
    funnel = {"candidates": len(candidates), "skipped_entry_hour": 0,
              "shadowed_by_occupancy": 0, "unresolved_exits": 0,
              "missing_entries": 0}
    occupied_until = -1
    for i, direction in candidates:
        day = int(ctx["d1"][i])
        if day < 20240101 or day > 20260924:
            continue
        if int(ctx["h1w"][i]) > 15:
            funnel["skipped_entry_hour"] += 1
            continue
        if i <= occupied_until:
            funnel["shadowed_by_occupancy"] += 1
            continue
        if not ctx["real1"][i] or day not in day_indices:
            funnel["missing_entries"] += 1
            continue
        final_real = day_indices[day][-1]
        has_final_hour = int(ctx["h1w"][final_real]) == 15
        result = walk_same_day(i, direction, final_real, ctx)
        v = int(result["v"])
        occupied_until = v
        if not has_final_hour and result["exit_type"] == "DAY_FLAT":
            funnel["unresolved_exits"] += 1
        unresolved = not has_final_hour and result["exit_type"] == "DAY_FLAT"
        entry_bid = float(ctx["o1"][i])
        entry_ask = entry_bid + SPREAD
        risk_units = ((entry_ask - result["initial_sl"]) if direction == 1
                      else (result["initial_sl"] - entry_bid))
        if not np.isfinite(risk_units) or risk_units <= 0:
            raise ValueError("Invalid planned stop distance")
        real_path = ctx["real1"][i:v + 1]
        low = float(np.min(ctx["l1"][i:v + 1][real_path]))
        high = float(np.max(ctx["h1"][i:v + 1][real_path]))
        worst_units = low - entry_ask if direction == 1 else entry_bid - (high + SPREAD)
        net = float(result["pnl"])
        entry_utc = datetime.fromtimestamp(int(ctx["grid1"][i]) * 3600, timezone.utc)
        exit_utc = datetime.fromtimestamp((int(ctx["grid1"][v]) + 1) * 3600 - 0.001, timezone.utc)
        rows.append({"ny_date": day, "year": day // 10000,
                     "side": "BUY" if direction == 1 else "SELL",
                     "entry_utc": entry_utc.isoformat(), "exit_bar_end_utc": exit_utc.isoformat(),
                     "entry_ny_date": entry_utc.astimezone(NY).date(),
                     "entry_prague_date": entry_utc.astimezone(PRAGUE).date(),
                     "exit_prague_date": exit_utc.astimezone(PRAGUE).date(),
                     "entry_bid": entry_bid, "entry_ask": entry_ask,
                     "entry_epoch_hour": int(ctx["grid1"][i]), "exit_epoch_hour": int(ctx["grid1"][v]),
                     "initial_sl": result["initial_sl"], "final_sl": result["final_sl"],
                     "exit_type": "UNRESOLVED_MARK" if unresolved else result["exit_type"],
                     "unresolved": unresolved, "net_units": net,
                     "stress_units": net - SPREAD, "risk_units": risk_units,
                     "worst_float_units": worst_units - SPREAD,
                     "realized_r": (net - SPREAD) / risk_units,
                     "worst_float_r": (worst_units - SPREAD) / risk_units,
                     "hold_hours_upper": result["hold_hours_upper"]})
    funnel["executed_or_marked"] = len(rows)
    return pd.DataFrame(rows), funnel


def date_bootstrap(trades: pd.DataFrame) -> dict:
    grouped = trades.groupby("ny_date", sort=True)["net_units"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(int)
    if len(sums) == 0:
        return {"p_mean_le_zero": 1.0, "date_blocks": 0}
    rng = np.random.default_rng(SEED)
    means = np.empty(10_000)
    for start in range(0, 10_000, 250):
        end = min(start + 250, 10_000)
        picks = rng.integers(0, len(sums), size=(end - start, len(sums)))
        means[start:end] = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return {"p_mean_le_zero": float(np.mean(means <= 0)),
            "ci95_low": float(np.quantile(means, 0.025)),
            "ci95_high": float(np.quantile(means, 0.975)),
            "date_blocks": len(sums)}


def challenge_pace(trades: pd.DataFrame, last_source_ny_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (trades.entry_prague_date == trades.exit_prague_date).all():
        raise ValueError("A bar-model trade crossed the Europe/Prague midnight")
    starts = sorted({d for d in trades.entry_ny_date
                     if d + timedelta(days=180) <= last_source_ny_date + timedelta(days=1)})
    outcomes = []
    for risk in RISKS:
        for start in starts:
            outcomes.append({"risk_pct": 100 * risk, "start_date": start.isoformat(),
                             **replay_window(trades, risk, start, 180)})
    result = pd.DataFrame(outcomes)
    summaries = []
    for risk_pct, group in result.groupby("risk_pct", sort=True):
        passed = group[group.outcome.eq("PASS")]
        summaries.append({"risk_pct": risk_pct, "starts": len(group),
                          "pass": len(passed), "breach": int(group.outcome.eq("BREACH").sum()),
                          "timeout": int(group.outcome.eq("TIMEOUT").sum()),
                          "pass_rate": float(group.outcome.eq("PASS").mean()),
                          "breach_rate": float(group.outcome.eq("BREACH").mean()),
                          "median_days_to_pass": float(passed.days.median()) if len(passed) else None})
    return result, pd.DataFrame(summaries)


def main() -> None:
    check_predeclared()
    bars, source = export_ftmo_h1()
    ctx = build_context(bars)
    candidates, signal_funnel = saleh.detect_signals(ctx)
    trades, trade_funnel = walk_candidates(ctx, candidates)
    if trades.empty:
        raise ValueError("No H82 trades; cannot evaluate predeclared gates")
    bootstrap = date_bootstrap(trades)
    last_source = datetime.fromtimestamp(int(bars.utc_epoch.iloc[-1]), timezone.utc).astimezone(NY).date()
    outcomes, summary = challenge_pace(trades, last_source)
    years = {str(year): {"n": len(group), "net_total_units": float(group.net_units.sum()),
                         "stress_mean_units": float(group.stress_units.mean())}
             for year, group in trades.groupby("year", sort=True)}
    pace_ok = summary[(summary.starts >= 200) & (summary.pass_rate >= 0.70)
                      & (summary.breach_rate <= 0.10)]
    same_day = bool((trades.entry_ny_date.map(lambda d: int(d.strftime("%Y%m%d")))
                     == trades.ny_date).all()
                    and (trades.hold_hours_upper <= 16).all()
                    and (trades.entry_prague_date == trades.exit_prague_date).all())
    gates = {
        "G-FT1": bool(len(trades) >= 150 and trades.ny_date.nunique() >= 150
                      and trade_funnel["missing_entries"] == 0
                      and trade_funnel["unresolved_exits"] == 0),
        "G-FT2": bool(trades.stress_units.mean() > 0),
        "G-FT3": bool(bootstrap["p_mean_le_zero"] <= 0.05),
        "G-FT4": bool(years.get("2024", {}).get("net_total_units", 0) > 0
                      and years.get("2025", {}).get("net_total_units", 0) > 0),
        "G-FT5": bool(len(pace_ok) > 0),
        "G-FT6": bool(same_day and source["non_custom"] and source["strict_unique_utc_hours"]
                      and len(source["sha256"]) == 64),
    }
    trades.to_csv(OUT / "bar_trades.csv", index=False, float_format="%.8f")
    outcomes.to_csv(OUT / "rolling_outcomes.csv", index=False, float_format="%.8f")
    summary.to_csv(OUT / "pace_summary.csv", index=False, float_format="%.8f")
    net = trades.net_units.to_numpy(float)
    equity = np.r_[0.0, np.cumsum(net)]
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "spec_sha256": sha256(SPEC), "gate_sha256": sha256(GATE),
              "source": source, "signal_funnel": signal_funnel, "trade_funnel": trade_funnel,
              "test": {"n": len(trades), "dates": int(trades.ny_date.nunique()),
                       "net_mean_units": float(net.mean()), "net_total_units": float(net.sum()),
                       "stress_mean_units": float(trades.stress_units.mean()),
                       "max_drawdown_units": float(np.max(np.maximum.accumulate(equity) - equity)),
                       "win_rate": float(np.mean(net > 0)),
                       "median_hold_hours_upper": float(trades.hold_hours_upper.median()),
                       "side": trades.groupby("side").net_units.agg(["size", "sum"]).to_dict("index"),
                       "exit_type": trades.exit_type.value_counts().to_dict()},
              "years": years, "bootstrap": bootstrap,
              "pace_summary": summary.astype(object).where(pd.notna(summary), None).to_dict("records"),
              "gates": gates, "verdict": "PASS" if all(gates.values()) else "FAIL",
              "limitations": "H1 bar fills and fixed spread are optimistic; challenged-risk rows use continuous lots and omit historical EURUSD conversion, commission, margin and order execution. H70 prior FAIL unchanged."}
    (OUT / "run_meta.json").write_text(json.dumps(report, indent=2, allow_nan=False, default=str) + "\n",
                                        encoding="utf-8")
    REPORT.write_text("H82 FTMO-NATIVE GOLD H1 PHASE-1 REPORT\n"
                      + json.dumps(report, indent=2, allow_nan=False, default=str)
                      + "\nOVERALL VERDICT: " + report["verdict"] + "\n", encoding="utf-8")
    print(json.dumps({"verdict": report["verdict"], "test": report["test"],
                      "years": years, "bootstrap": bootstrap,
                      "pace_summary": report["pace_summary"], "gates": gates},
                     indent=2, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
