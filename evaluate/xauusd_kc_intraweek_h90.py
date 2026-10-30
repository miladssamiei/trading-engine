"""H90: corrected-source Friday-flat gold KC Phase 1; no account orders."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xauusd_kc_breakout_oos_phase1 import wilder_atr


ROOT = Path(__file__).resolve().parents[1]
TAPE = ROOT / "data/results/xauusd_onboarding/xauusdtick_m1.csv"
OUT = ROOT / "data/results/xauusd_kc_intraweek_h90"
EXPECTED_TAPE_SHA = "a8c4c1d6551d27f3796f8d9b88d332172c41c76c151844e759492837152e7b1f"
P4 = 4 * 3600 * 1_000_000_000
P1D = 24 * 3600 * 1_000_000_000
O17 = 17 * 3600 * 1_000_000_000
SPREAD = 0.41
EXTRA = 0.41 + 0.25
COMMISSION_SIDE = 0.000007
SEED = 20261001


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(tape: pd.DataFrame):
    slot = tape["time"].to_numpy(np.int64) // 3600
    g = (pd.DataFrame({"slot": slot, "o": tape.open.to_numpy(float),
                       "h": tape.high.to_numpy(float),
                       "l": tape.low.to_numpy(float),
                       "c": tape.close.to_numpy(float)})
         .groupby("slot", sort=True)
         .agg(o=("o", "first"), h=("h", "max"),
              l=("l", "min"), c=("c", "last")))
    grid = np.arange(int(g.index.min()), int(g.index.max()) + 1, dtype=np.int64)
    hour = g.reindex(grid)
    missing = ~hour.index.isin(g.index)
    last = hour.c.ffill()
    for field in ("o", "h", "l", "c"):
        hour.loc[missing, field] = last.loc[missing].to_numpy()
    assert not hour[["o", "h", "l", "c"]].isna().any().any()
    assert np.all(hour.loc[missing, ["o", "h", "l", "c"]].to_numpy()
                  == last.loc[missing].to_numpy()[:, None])

    wall_1h = (pd.to_datetime(grid * 3600, unit="s", utc=True)
               .tz_convert("America/New_York").tz_localize(None))
    wns = wall_1h.values.astype("int64")
    key4 = wns - wns % P4
    g4 = (pd.DataFrame({"key": key4, "o": hour.o.to_numpy(float),
                        "h": hour.h.to_numpy(float), "l": hour.l.to_numpy(float),
                        "c": hour.c.to_numpy(float)})
          .groupby("key", sort=True)
          .agg(o=("o", "first"), h=("h", "max"),
               l=("l", "min"), c=("c", "last")))
    keys4 = g4.index.to_numpy(np.int64)
    bar_ny = pd.DatetimeIndex(pd.to_datetime(keys4, unit="ns"))
    o, h, l, c = [g4[z].to_numpy(float) for z in ("o", "h", "l", "c")]

    dkey = wns - ((wns - O17) % P1D)
    gd = pd.DataFrame({"key": dkey, "c": hour.c.to_numpy(float)}).groupby("key", sort=True).c.last()
    dkeys = gd.index.to_numpy(np.int64)
    dc = gd.to_numpy(float)
    dema = pd.Series(dc).ewm(span=50, adjust=False).mean().to_numpy()
    dema[:49] = np.nan
    prior_dkey = keys4 - ((keys4 - O17) % P1D) - P1D
    posd = np.searchsorted(dkeys, prior_dkey)
    safe_posd = np.clip(posd, 0, len(dkeys) - 1)
    valid_d = ((prior_dkey >= dkeys[0]) & (dkeys[safe_posd] == prior_dkey)
               & ~np.isnan(dema[safe_posd]))
    bull = valid_d & (dc[safe_posd] > dema[safe_posd])
    bear = valid_d & (dc[safe_posd] < dema[safe_posd])

    ema = pd.Series(c).ewm(span=29, adjust=False).mean().to_numpy()
    ema[:28] = np.nan
    atr = wilder_atr(h, l, c, 29)
    prior_c = np.roll(c, 1); prior_c[0] = np.nan
    prior_ema = np.roll(ema, 1); prior_ema[0] = np.nan
    prior_atr = np.roll(atr, 1); prior_atr[0] = np.nan
    prior_upper = np.roll(ema + 2.77 * atr, 1); prior_upper[0] = np.nan
    prior_lower = np.roll(ema - 2.77 * atr, 1); prior_lower[0] = np.nan
    armed = (valid_d & ~np.isnan(prior_c) & ~np.isnan(prior_ema)
             & ~np.isnan(prior_atr) & ~np.isnan(prior_upper)
             & ~np.isnan(prior_lower))
    signal_long = armed & bull & (prior_c > prior_upper)
    signal_short = armed & bear & (prior_c < prior_lower)

    mtime = tape.time.to_numpy(np.int64)
    mopen = tape.open.to_numpy(float)
    mhigh = tape.high.to_numpy(float)
    mlow = tape.low.to_numpy(float)
    mny = (pd.to_datetime(mtime, unit="s", utc=True)
           .tz_convert("America/New_York").tz_localize(None))
    mns = mny.values.astype("int64")
    mkey = mns - mns % P4
    mbar = np.searchsorted(keys4, mkey)
    assert np.all(keys4[np.clip(mbar, 0, len(keys4) - 1)] == mkey)
    return {"bar_ny": bar_ny, "ema": ema, "prior_ema": prior_ema,
            "prior_atr": prior_atr, "signal_long": signal_long,
            "signal_short": signal_short, "mbar": mbar,
            "mny": mny, "mtime": mtime, "mopen": mopen,
            "mhigh": mhigh, "mlow": mlow,
            "missing_hour_count": int(missing.sum()),
            "hour_count": len(hour), "bar_count": len(keys4)}


def run(z):
    bar_ny = z["bar_ny"]
    mny, mbar = z["mny"], z["mbar"]
    mtime, mopen, mhigh, mlow = [z[k] for k in
                                 ("mtime", "mopen", "mhigh", "mlow")]
    ema, prior_ema, prior_atr = z["ema"], z["prior_ema"], z["prior_atr"]
    sig_long, sig_short = z["signal_long"], z["signal_short"]
    sig = sig_long | sig_short
    trades, coverage = [], []
    i = 1
    while i < len(bar_ny):
        if not sig[i]:
            i += 1
            continue
        wall = bar_ny[i]
        if wall.weekday() >= 5 or (wall.weekday() == 4 and wall.hour >= 15):
            coverage.append({"signal_bar": str(wall), "status": "NONWEEKDAY_OR_LATE_FRIDAY"})
            i += 1
            continue
        k0 = int(np.searchsorted(mbar, i))
        if k0 >= len(mbar) or mbar[k0] != i or mny[k0] < wall or mny[k0] >= wall + pd.Timedelta(minutes=5):
            coverage.append({"signal_bar": str(wall), "status": "NO_REAL_ENTRY"})
            i += 1
            continue
        side = 1 if sig_long[i] else -1
        bid_entry = float(mopen[k0])
        entry = bid_entry + SPREAD if side == 1 else bid_entry
        dist = max(abs(bid_entry - prior_ema[i]), 1.5 * prior_atr[i])
        assert dist > 0 and np.isfinite(dist)
        stop = entry - side * dist
        cutoff = (wall.normalize() + pd.Timedelta(days=4 - wall.weekday(),
                                                  hours=15, minutes=45))
        k, prevbar = k0, i
        exit_px, reason, exit_k = None, None, None
        while k < len(mtime):
            now = mny[k]
            if now >= cutoff:
                if now < cutoff + pd.Timedelta(minutes=15):
                    exit_px = float(mopen[k] if side == 1 else mopen[k] + SPREAD)
                    exit_k, reason = k, "FRIDAY_FLAT"
                else:
                    reason = "UNRESOLVED_FRIDAY"
                break
            current_bar = int(mbar[k])
            if current_bar < prevbar:
                reason = "UNRESOLVED_BAR_ORDER"
                break
            while prevbar < current_bar:
                if np.isfinite(ema[prevbar]):
                    stop = (max(stop, float(ema[prevbar])) if side == 1
                            else min(stop, float(ema[prevbar])))
                prevbar += 1
            ask_open = mopen[k] + SPREAD
            if side == 1:
                if mopen[k] <= stop:
                    exit_px, reason, exit_k = float(mopen[k]), "STOP_GAP", k
                    break
                if mlow[k] <= stop:
                    exit_px, reason, exit_k = float(stop), "STOP", k
                    break
            else:
                if ask_open >= stop:
                    exit_px, reason, exit_k = float(ask_open), "STOP_GAP", k
                    break
                if mhigh[k] + SPREAD >= stop:
                    exit_px, reason, exit_k = float(stop), "STOP", k
                    break
            k += 1
        if exit_px is None:
            coverage.append({"signal_bar": str(wall), "status": reason or "UNRESOLVED_TAPE_END"})
            i += 1
            continue
        commission = COMMISSION_SIDE * (entry + exit_px)
        gross = side * (exit_px - entry)
        base = gross - commission
        stress = base - EXTRA
        trades.append({"block": "OOS" if wall.year >= 2024 else "CAL",
                       "signal_bar_ny": str(wall), "entry_ny": str(mny[k0]),
                       "exit_ny": str(mny[exit_k]),
                       "entry_epoch": int(mtime[k0]),
                       "exit_epoch": int(mtime[exit_k]),
                       "entry_year": int(wall.year),
                       "side": "LONG" if side == 1 else "SHORT",
                       "entry": entry, "exit": exit_px, "initial_risk": dist,
                       "exit_reason": reason, "gross": gross,
                       "commission_per_oz": commission,
                       "base_net": base, "stress_net": stress,
                       "hold_hours": float((mtime[exit_k] - mtime[k0]) / 3600),
                       "crossed_friday": False})
        coverage.append({"signal_bar": str(wall), "status": "EXECUTED"})
        i = int(mbar[exit_k]) + 1
    return pd.DataFrame(trades), pd.DataFrame(coverage)


def block_boot(values: np.ndarray):
    n = len(values)
    if n < 5:
        return None
    starts = np.arange(n - 4)
    rng = np.random.default_rng(SEED)
    positions = rng.integers(0, len(starts), size=(10000, int(np.ceil(n / 5))))
    idx = (starts[positions][:, :, None] + np.arange(5)[None, None, :])
    idx = idx.reshape(10000, -1)[:, :n]
    return float((values[idx].mean(axis=1) <= 0).mean())


def main():
    spec = OUT / "SPEC_XAUUSD_KC_INTRAWEEK_H90.txt"
    gate = OUT / "GATE_XAUUSD_KC_INTRAWEEK_H90.txt"
    report = OUT / "REPORT_XAUUSD_KC_INTRAWEEK_H90.txt"
    assert not report.exists(), "H90 REPORT immutable"
    lines = [v for v in spec.read_text(encoding="utf-8").splitlines()
             if v.startswith("G-IW")]
    assert lines == gate.read_text(encoding="utf-8").splitlines()
    assert sha(TAPE) == EXPECTED_TAPE_SHA
    tape = pd.read_csv(TAPE, usecols=["time", "open", "high", "low", "close"])
    assert tape.time.is_monotonic_increasing and not tape.time.duplicated().any()
    assert (tape.high >= tape[["open", "close", "low"]].max(axis=1)).all()
    assert (tape.low <= tape[["open", "close", "high"]].min(axis=1)).all()
    z = build(tape)
    trades, coverage = run(z)
    trades.to_csv(OUT / "trades.csv", index=False)
    coverage.to_csv(OUT / "coverage.csv", index=False)
    oos = trades[trades.block == "OOS"] if len(trades) else trades
    unresolved = int(coverage.status.str.startswith("UNRESOLVED").sum())
    nn = len(oos)
    values = oos.stress_net.to_numpy(float) if nn else np.array([])
    annual = {str(y): {"n": int((oos.entry_year == y).sum()),
                       "stress_sum": float(oos.loc[oos.entry_year == y, "stress_net"].sum())}
              for y in (2024, 2025, 2026)} if nn else {}
    boot = block_boot(values)
    gates = {
        "G-IW1": nn >= 50 and oos.entry_ny.nunique() >= 50 and unresolved == 0,
        "G-IW2": nn > 0 and float(values.mean()) > 0,
        "G-IW3": boot is not None and boot <= 0.05,
        "G-IW4": bool(annual) and all(annual[str(y)]["stress_sum"] > 0
                                     for y in (2024, 2025, 2026)),
        "G-IW5": "NOT_RUN",
        "G-IW6": unresolved == 0 and bool(len(trades))
                  and not trades.crossed_friday.any()
                  and (trades.initial_risk > 0).all(),
    }
    phase1_pass = all(gates[k] for k in ("G-IW1", "G-IW2", "G-IW3",
                                          "G-IW4", "G-IW6"))
    status = "PHASE1_PASS_NEEDS_PHASE2" if phase1_pass else "FINAL_FAIL"
    meta = {"status": status, "spec_sha256": sha(spec),
            "gate_sha256": sha(gate), "script_sha256": sha(Path(__file__)),
            "tape_sha256": sha(TAPE), "m1_rows": len(tape),
            "hour_count": z["hour_count"],
            "synthetic_flat_hours": z["missing_hour_count"],
            "bar_count": z["bar_count"],
            "all_trades": len(trades), "oos_trades": nn,
            "unresolved": unresolved,
            "coverage": coverage.status.value_counts().to_dict(),
            "oos_base_mean": float(oos.base_net.mean()) if nn else None,
            "oos_stress_mean": float(values.mean()) if nn else None,
            "oos_stress_total": float(values.sum()) if nn else None,
            "oos_boot_p_nonpositive": boot,
            "annual": annual, "gates": gates,
            "zero_swap_owner_assumption": True,
            "commission_rate_per_side": COMMISSION_SIDE}
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with report.open("x", encoding="utf-8") as f:
        f.write("H90 CORRECTED-SOURCE GOLD KC FRIDAY-FLAT - " + status + "\n")
        f.write("Dukascopy BID M1 proxy; user-stated zero swap; news holds allowed. No account orders.\n")
        f.write("Frozen source/SPEC/GATE SHA256: " + meta["tape_sha256"] + " / "
                + meta["spec_sha256"] + " / " + meta["gate_sha256"] + "\n")
        f.write("Rows/flat synthetic hours/4h bars: "
                + str((len(tape), z["missing_hour_count"], z["bar_count"])) + "\n")
        f.write("All/OOS trades/unresolved: "
                + str((len(trades), nn, unresolved)) + "\n")
        f.write("Coverage: " + str(meta["coverage"]) + "\n")
        f.write("OOS base/stress mean/stress total: "
                + str((meta["oos_base_mean"], meta["oos_stress_mean"],
                       meta["oos_stress_total"])) + "\n")
        f.write("OOS MBB P(nonpositive): " + str(boot) + "\n")
        f.write("Annual: " + json.dumps(annual) + "\n")
        f.write("Gates: " + json.dumps(gates) + "\n")
        f.write("This historical screen uses exposed data and is not independent forward confirmation.\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
