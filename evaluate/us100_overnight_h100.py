"""Frozen H100 daily US100 overnight Phase 1; no account orders."""
from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.easter import easter

from xauusd_kc_breakout_oos_phase1 import wilder_atr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/us100_overnight_h100"
TAPE = ROOT / "data/results/virgin_assets/ustech_m1.csv"
EXPECTED = "152f9c995b370e5391cb7da99b64a8002ceee5e52cc68899ddca409ac7e15e39"
SPREAD = 1.50
EXTRA = 2.50
SEED = 20261010


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def epoch(day: pd.Timestamp, hour: int, minute: int) -> int:
    return int((day + pd.Timedelta(hours=hour, minutes=minute))
               .tz_localize("America/New_York").timestamp())


def bootstrap(values: np.ndarray) -> float:
    n = len(values)
    if n < 20:
        return float("nan")
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, n, size=(10000, int(np.ceil(n / 20))))
    ix = (starts[:, :, None] + np.arange(20)[None, None, :]) % n
    samples = values[ix.reshape(10000, -1)[:, :n]]
    return float((samples.mean(axis=1) <= 0).mean())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    actual = sha(TAPE)
    if actual != EXPECTED:
        raise ValueError(f"Unexpected BID source hash {actual}")
    tape = pd.read_csv(TAPE, usecols=["time", "open", "high", "low", "close"])
    times = tape.time.to_numpy(np.int64)
    if not np.all(np.diff(times) > 0):
        raise ValueError("Unordered or duplicate source M1")
    o, h, l, c = [tape[z].to_numpy(float) for z in ("open", "high", "low", "close")]
    wall = pd.to_datetime(times, unit="s", utc=True).tz_convert("America/New_York").tz_localize(None)
    minutes = wall.hour * 60 + wall.minute
    mask = (wall.weekday < 5) & (minutes >= 570) & (minutes < 960)
    cash = (pd.DataFrame({"date": wall[mask].normalize(), "o": o[mask],
                          "h": h[mask], "l": l[mask], "c": c[mask],
                          "minute": minutes[mask]})
            .groupby("date", sort=True)
            .agg(o=("o", "first"), h=("h", "max"), l=("l", "min"),
                 c=("c", "last"), first_minute=("minute", "first")))
    cash = cash.loc[cash.first_minute == 570]
    dates = pd.DatetimeIndex(cash.index)
    atr = wilder_atr(cash.h.to_numpy(float), cash.l.to_numpy(float),
                     cash.c.to_numpy(float), 20)
    rows, coverage = [], []
    for day in pd.date_range(wall[0].normalize(), wall[-1].normalize(), freq="D"):
        if day.weekday() >= 4:
            continue
        next_day = day + pd.Timedelta(days=1)
        label = str(day.date())
        if next_day.date() == easter(day.year) - timedelta(days=2):
            coverage.append({"ny_date": label, "status": "GOOD_FRIDAY_EVE_SKIP"})
            continue
        e0 = epoch(day, 15, 55)
        x0 = epoch(next_day, 9, 30)
        if x0 > times[-1]:
            coverage.append({"ny_date": label, "status": "INCOMPLETE_TAPE"})
            continue
        ke = int(np.searchsorted(times, e0))
        if ke >= len(times) or times[ke] != e0:
            coverage.append({"ny_date": label, "status": "NO_EXACT_ENTRY"})
            continue
        j = int(dates.searchsorted(day)) - 1
        if j < 19 or (day - dates[j]).days > 4 or not np.isfinite(atr[j]):
            coverage.append({"ny_date": label, "status": "NO_PRIOR_ATR"})
            continue
        kx = int(np.searchsorted(times, x0))
        if kx >= len(times) or times[kx] != x0:
            coverage.append({"ny_date": label, "status": "UNRESOLVED_EXIT"})
            continue
        entry = float(o[ke] + SPREAD)
        distance = float(0.5 * atr[j])
        stop = entry - distance
        path0, path1 = ke + 1, kx
        open_hits = o[path0:path1] <= stop
        low_hits = l[path0:path1] <= stop
        hits = np.flatnonzero(open_hits | low_hits)
        if len(hits):
            ky = path0 + int(hits[0])
            exit_price = float(o[ky] if o[ky] <= stop else stop)
            reason = "STOP_GAP" if o[ky] <= stop else "STOP"
        else:
            ky = kx
            exit_price = float(o[kx])
            reason = "NEXT_CASH_OPEN"
        base = exit_price - entry
        rows.append({"ny_date": label, "year": day.year, "entry_epoch": int(times[ke]),
                     "exit_epoch": int(times[ky]), "required_exit_epoch": x0,
                     "prior_cash_date": str(dates[j].date()), "prior_atr20": float(atr[j]),
                     "entry_bid": float(o[ke]), "entry_ask_proxy": entry,
                     "stop_bid": stop, "exit_bid": exit_price, "exit_reason": reason,
                     "base_net": base, "stressed_net": base - EXTRA,
                     "stressed_r": (base - EXTRA) / distance})
        coverage.append({"ny_date": label, "status": "TRADE"})
    tr = pd.DataFrame(rows)
    cov = pd.DataFrame(coverage)
    tr.to_csv(OUT / "trades.csv", index=False)
    cov.to_csv(OUT / "coverage.csv", index=False)
    test = tr[tr.year >= 2024]
    annual = test.groupby("year").agg(n=("stressed_net", "size"),
                                       sum=("stressed_net", "sum"),
                                       mean=("stressed_net", "mean")).to_dict("index")
    unresolved = int(cov[(cov.ny_date >= "2024-01-01") &
                         (cov.status == "UNRESOLVED_EXIT")].shape[0])
    p = bootstrap(test.stressed_net.to_numpy(float))
    gates = {"G-ON1": bool(len(test) >= 520 and unresolved == 0),
             "G-ON2": bool(test.stressed_net.mean() > 0),
             "G-ON3": bool(p <= .05),
             "G-ON4": bool(all(y in annual and annual[y]["sum"] > 0
                               for y in (2024, 2025, 2026))),
             "G-ON5": None,
             "G-ON6": bool(actual == EXPECTED and
                           (tr.exit_epoch > tr.entry_epoch).all() and
                           (tr.exit_epoch <= tr.required_exit_epoch).all())}
    meta = {"source_sha256": actual, "cash_dates": len(cash),
            "coverage": cov.status.value_counts().to_dict(),
            "gated_trades": len(test), "unresolved": unresolved,
            "base_mean": float(test.base_net.mean()),
            "stressed_mean": float(test.stressed_net.mean()),
            "stressed_median": float(test.stressed_net.median()),
            "mean_r": float(test.stressed_r.mean()),
            "stop_rate": float((test.exit_reason != "NEXT_CASH_OPEN").mean()),
            "bootstrap_p": p, "annual": annual, "gates": gates}
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
