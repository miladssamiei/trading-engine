"""Frozen H99 EIA USOIL Phase 1: native MT5 ticks, read only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/usoil_eia_reaction_h99"
SYMBOL = "USOIL.cash"
STRESS = 0.12
SEED = 20261009


def bootstrap(values: np.ndarray) -> float:
    n = len(values)
    if n < 5:
        return float("nan")
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, n, size=(10000, int(np.ceil(n / 5))))
    ix = (starts[:, :, None] + np.arange(5)[None, None, :]) % n
    samples = values[ix.reshape(10000, -1)[:, :n]]
    return float((samples.mean(axis=1) <= 0).mean())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize: {mt5.last_error()}")
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        raise RuntimeError(f"No {SYMBOL}: {mt5.last_error()}")
    old_visible = info.visible
    if not mt5.symbol_select(SYMBOL, True):
        raise RuntimeError(f"select {SYMBOL}: {mt5.last_error()}")
    holidays = set(USFederalHolidayCalendar().holidays(
        start="2021-12-27", end="2026-09-25").date)
    coverage, trades = [], []
    try:
        for wed in pd.date_range("2022-01-05", "2026-09-23", freq="W-WED"):
            day = wed.date()
            monday = day - pd.Timedelta(days=2)
            if any((monday + pd.Timedelta(days=k)) in holidays for k in range(5)):
                coverage.append(dict(ny_date=str(day), status="FEDERAL_HOLIDAY_WEEK"))
                continue
            base = wed.tz_localize("America/New_York")
            t0 = (base + pd.Timedelta(hours=10, minutes=30)).tz_convert("UTC")
            t1 = (base + pd.Timedelta(hours=15, minutes=1)).tz_convert("UTC")
            raw = mt5.copy_ticks_range(SYMBOL, t0.to_pydatetime(),
                                       t1.to_pydatetime(), mt5.COPY_TICKS_ALL)
            if raw is None:
                coverage.append(dict(ny_date=str(day), status="MT5_NONE",
                                     error=str(mt5.last_error())))
                continue
            source_hash = hashlib.sha256(raw.tobytes()).hexdigest()
            good = raw[(raw["bid"] > 0) & (raw["ask"] >= raw["bid"])]
            ms = good["time_msc"]
            record = dict(ny_date=str(day), source_ticks=len(raw),
                          valid_ticks=len(good), source_sha256=source_hash,
                          status="")
            if len(good) == 0 or np.any(np.diff(ms) < 0):
                record["status"] = "EMPTY_OR_UNORDERED"
                coverage.append(record)
                continue
            a = int(t0.timestamp() * 1000)
            c = int((base + pd.Timedelta(hours=10, minutes=31)).tz_convert("UTC").timestamp() * 1000)
            b = int((base + pd.Timedelta(hours=11)).tz_convert("UTC").timestamp() * 1000)
            x = int((base + pd.Timedelta(hours=15)).tz_convert("UTC").timestamp() * 1000)
            i0 = int(np.searchsorted(ms, a))
            i31 = int(np.searchsorted(ms, c))
            i11 = int(np.searchsorted(ms, b))
            ix = int(np.searchsorted(ms, x))
            required = (i0 < i31 and i11 > 0 and ms[i11 - 1] >= b - 60000
                        and i11 < len(ms) and ms[i11] < b + 60000
                        and ix < len(ms) and ms[ix] < x + 60000)
            if not required:
                record["status"] = "MISSING_REQUIRED_MINUTE"
                coverage.append(record)
                continue
            observed = good[i0:i11]
            signal = float(good["bid"][i11 - 1] - good["bid"][i0])
            span = float(observed["bid"].max() - observed["bid"].min())
            if signal == 0 or span <= 0:
                record["status"] = "NO_SIGNAL"
                coverage.append(record)
                continue
            side = 1 if signal > 0 else -1
            entry = float(good["ask"][i11] if side == 1 else good["bid"][i11])
            stop = entry - side * span
            path = good[i11 + 1:ix]
            stop_hits = np.flatnonzero(path["bid"] <= stop if side == 1
                                    else path["ask"] >= stop)
            if len(stop_hits):
                j = i11 + 1 + int(stop_hits[0])
                exit_price = float(good["bid"][j] if side == 1 else good["ask"][j])
                stopped = True
            else:
                j = ix
                exit_price = float(good["bid"][j] if side == 1 else good["ask"][j])
                stopped = False
            base_net = side * (exit_price - entry)
            record["status"] = "TRADE"
            coverage.append(record)
            trades.append(dict(ny_date=str(day), year=day.year, side=side,
                               source_sha256=source_hash, signal=signal,
                               event_range=span, entry_spread=float(good["ask"][i11] - good["bid"][i11]),
                               entry_time_msc=int(ms[i11]), exit_time_msc=int(ms[j]),
                               entry=entry, stop=stop, exit=exit_price,
                               stopped=stopped, base_net=base_net,
                               stressed_net=base_net - STRESS,
                               stressed_r=(base_net - STRESS) / span))
        if not old_visible:
            mt5.symbol_select(SYMBOL, False)
    finally:
        mt5.shutdown()
    cov = pd.DataFrame(coverage)
    tr = pd.DataFrame(trades)
    cov.to_csv(OUT / "coverage.csv", index=False)
    tr.to_csv(OUT / "trades.csv", index=False)
    test = tr[tr.year >= 2023]
    by_year = test.groupby("year").agg(n=("stressed_net", "size"),
                                        stressed_sum=("stressed_net", "sum"),
                                        mean=("stressed_net", "mean")).to_dict("index")
    unresolved = int(cov[(cov.ny_date >= "2023-01-01") &
                         cov.status.isin(["MT5_NONE", "EMPTY_OR_UNORDERED",
                                           "MISSING_REQUIRED_MINUTE"])].shape[0])
    p = bootstrap(test.stressed_net.to_numpy(float))
    gates = {
        "G-OI1": bool(len(test) >= 140 and unresolved == 0),
        "G-OI2": bool(len(test) > 0 and test.stressed_net.mean() > 0),
        "G-OI3": bool(p <= 0.05),
        "G-OI4": bool(all(y in by_year and by_year[y]["stressed_sum"] > 0
                          for y in (2023, 2024, 2025, 2026))),
        "G-OI5": None,
        "G-OI6": bool(len(test) > 0 and (test.exit_time_msc > test.entry_time_msc).all()),
    }
    meta = dict(symbol=SYMBOL, contract_size=info.trade_contract_size,
                source="MT5 COPY_TICKS_ALL native", stress_price_units=STRESS,
                counts=cov.status.value_counts().to_dict(), gated_trades=len(test),
                unresolved=unresolved, annual=by_year,
                mean_stressed=float(test.stressed_net.mean()),
                median_stressed=float(test.stressed_net.median()),
                mean_r=float(test.stressed_r.mean()),
                median_entry_spread=float(test.entry_spread.median()),
                p90_entry_spread=float(test.entry_spread.quantile(.9)),
                stop_rate=float(test.stopped.mean()), bootstrap_p=p,
                gates=gates)
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
