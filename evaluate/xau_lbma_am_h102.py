"""Frozen H102 LBMA morning auction native-tick Phase 1; read only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/xau_lbma_am_h102"
SYMBOL = "XAUUSD"
SEED = 20261011
EXTRA = 0.41


def bootstrap(values: np.ndarray) -> float:
    n = len(values)
    if n < 15:
        return float("nan")
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, n, size=(10000, int(np.ceil(n / 15))))
    ix = (starts[:, :, None] + np.arange(15)[None, None, :]) % n
    return float((values[ix.reshape(10000, -1)[:, :n]].mean(axis=1) <= 0).mean())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = OUT / "REPORT_XAU_LBMA_AM_H102.txt"
    if report.exists():
        raise FileExistsError(report)
    spec = OUT / "SPEC_XAU_LBMA_AM_H102.txt"
    gate = OUT / "GATE_XAU_LBMA_AM_H102.txt"
    lines = [x for x in spec.read_text(encoding="utf-8").splitlines() if x.startswith("G-LB")]
    if lines != gate.read_text(encoding="utf-8").splitlines():
        raise ValueError("SPEC/GATE mismatch")
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init: {mt5.last_error()}")
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        raise RuntimeError(f"Missing {SYMBOL}")
    old_visible = info.visible
    if not mt5.symbol_select(SYMBOL, True):
        raise RuntimeError(f"MT5 select: {mt5.last_error()}")
    covered_ranges: list[tuple[pd.Timestamp, float]] = []
    coverage, trades = [], []
    try:
        for day in pd.date_range("2022-01-04", "2026-09-24", freq="D"):
            if day.weekday() not in (1, 2, 3):
                continue
            d = day.date()
            if (d.month == 12 and d.day >= 24) or (d.month == 1 and d.day <= 2):
                coverage.append({"london_date": str(d), "status": "YEAR_END_SKIP"})
                continue
            local = day.tz_localize("Europe/London")
            a = (local + pd.Timedelta(hours=9, minutes=30)).tz_convert("UTC")
            b = (local + pd.Timedelta(hours=10, minutes=30)).tz_convert("UTC")
            c = (local + pd.Timedelta(hours=10, minutes=31)).tz_convert("UTC")
            start_ms, exit_ms, end_ms = [int(x.timestamp() * 1000) for x in (a, b, c)]
            raw = mt5.copy_ticks_range(SYMBOL, a.to_pydatetime(),
                                       (c - pd.Timedelta(milliseconds=1)).to_pydatetime(),
                                       mt5.COPY_TICKS_ALL)
            if raw is None:
                coverage.append({"london_date": str(d), "status": "MT5_NONE",
                                 "error": str(mt5.last_error())})
                continue
            record = {"london_date": str(d), "source_ticks": len(raw),
                      "source_sha256": hashlib.sha256(raw.tobytes()).hexdigest()}
            q = raw[(raw["bid"] > 0) & (raw["ask"] >= raw["bid"])]
            record["valid_ticks"] = len(q)
            ms = q["time_msc"]
            if len(q) == 0 or np.any(np.diff(ms) < 0):
                record["status"] = "EMPTY_OR_UNORDERED"
                coverage.append(record)
                continue
            q = q[(ms >= start_ms) & (ms < end_ms)]
            ms = q["time_msc"]
            if len(q) == 0:
                record["status"] = "EMPTY_WINDOW"
                coverage.append(record)
                continue
            entry_end = start_ms + 60000
            exit_end = exit_ms + 60000
            k_entry = int(np.searchsorted(ms, start_ms))
            k_exit = int(np.searchsorted(ms, exit_ms))
            minute_slots = np.unique(ms // 60000)
            full_minutes = (len(minute_slots) == 61 and
                            minute_slots[0] == start_ms // 60000 and
                            minute_slots[-1] == exit_ms // 60000)
            if not (k_entry < len(ms) and ms[k_entry] < entry_end and
                    k_exit < len(ms) and ms[k_exit] < exit_end and full_minutes):
                record["status"] = "MISSING_REQUIRED_QUOTE_OR_MINUTE"
                coverage.append(record)
                continue
            session_range = float(q["bid"][:k_exit].max() - q["bid"][:k_exit].min())
            if session_range <= 0:
                record["status"] = "ZERO_SESSION_RANGE"
                coverage.append(record)
                continue
            prior = covered_ranges[-20:]
            prior_ready = len(prior) == 20 and (day - prior[0][0]).days <= 60
            prior_med = float(np.median([r for _, r in prior])) if prior_ready else None
            covered_ranges.append((day, session_range))
            if not prior_ready:
                record["status"] = "NO_PRIOR20_RANGE"
                coverage.append(record)
                continue
            distance = float(np.ceil(1.5 * prior_med / 0.01 - 1e-9) * 0.01)
            entry_bid = float(q["bid"][k_entry])
            entry_ask = float(q["ask"][k_entry])
            stop = entry_bid + distance
            hits = np.flatnonzero(q["ask"][k_entry + 1:k_exit] >= stop)
            if len(hits):
                j = k_entry + 1 + int(hits[0])
                reason = "STOP"
            else:
                j = k_exit
                reason = "AUCTION_START"
            exit_ask = float(q["ask"][j])
            base = entry_bid - exit_ask
            trades.append({"london_date": str(d), "year": d.year,
                           "source_sha256": record["source_sha256"],
                           "entry_time_msc": int(ms[k_entry]), "exit_time_msc": int(ms[j]),
                           "entry_bid": entry_bid, "entry_ask": entry_ask,
                           "entry_spread": entry_ask - entry_bid,
                           "prior20_median_range": prior_med, "risk_distance": distance,
                           "initial_stop": stop, "exit_ask": exit_ask,
                           "exit_bid": float(q["bid"][j]),
                           "exit_spread": float(q["ask"][j] - q["bid"][j]),
                           "exit_reason": reason, "base_net": base,
                           "stressed_net": base - EXTRA,
                           "stressed_r": (base - EXTRA) / distance})
            record["status"] = "TRADE"
            coverage.append(record)
    finally:
        if not old_visible:
            mt5.symbol_select(SYMBOL, False)
        mt5.shutdown()
    cov = pd.DataFrame(coverage)
    tr = pd.DataFrame(trades)
    cov.to_csv(OUT / "coverage.csv", index=False)
    tr.to_csv(OUT / "trades.csv", index=False)
    test = tr[tr.year >= 2024]
    annual = test.groupby("year").agg(n=("stressed_net", "size"),
                                       stressed_sum=("stressed_net", "sum"),
                                       mean=("stressed_net", "mean"),
                                       mean_r=("stressed_r", "mean")).to_dict("index")
    unresolved_status = ["MT5_NONE", "EMPTY_OR_UNORDERED", "EMPTY_WINDOW",
                         "MISSING_REQUIRED_QUOTE_OR_MINUTE", "ZERO_SESSION_RANGE"]
    unresolved = int(cov[(cov.london_date >= "2024-01-01") &
                         cov.status.isin(unresolved_status)].shape[0])
    p = bootstrap(test.stressed_net.to_numpy(float))
    gates = {"G-LB1": bool(len(test) >= 300 and unresolved == 0),
             "G-LB2": bool(len(test) and test.stressed_net.mean() > 0),
             "G-LB3": bool(p <= .05),
             "G-LB4": bool(all(y in annual and annual[y]["stressed_sum"] > 0
                               for y in (2024, 2025, 2026))),
             "G-LB5": None,
             "G-LB6": bool(len(test) and (test.exit_time_msc > test.entry_time_msc).all())}
    meta = {"source": "connected FTMO-Demo native XAUUSD COPY_TICKS_ALL",
            "source_rows": int(cov.source_ticks.fillna(0).sum()),
            "coverage": cov.status.value_counts().to_dict(),
            "gated_trades": len(test), "gated_unresolved": unresolved,
            "base_mean": float(test.base_net.mean()),
            "stressed_mean": float(test.stressed_net.mean()),
            "stressed_median": float(test.stressed_net.median()),
            "mean_r": float(test.stressed_r.mean()),
            "entry_spread_median": float(test.entry_spread.median()),
            "entry_spread_p90": float(test.entry_spread.quantile(.9)),
            "stop_rate": float(test.exit_reason.eq("STOP").mean()),
            "bootstrap_p": p, "annual": annual, "gates": gates}
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    report.write_text("H102 XAUUSD PRE-LBMA-AM SHORT PHASE-1 "
                      + ("PASS_NEEDS_PHASE2" if all(v is True for k, v in gates.items() if k != "G-LB5")
                         else "FINAL_FAIL") + "\n"
                      + json.dumps(meta, indent=2) + "\n"
                      + "Native historical quotes and exposed gold market: historical Phase-1 results are not forward validation. No account orders.\n",
                      encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
