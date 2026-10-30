"""Frozen H74: PAXG weekend direction to Sunday XAUUSD intraday return."""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/xauusd_paxg_weekend_continuation_v1"
XAU = ROOT / "data/results/xauusd_onboarding/xauusdtick_m1.csv"
EXPECTED_XAU_SHA = "a8c4c1d6551d27f3796f8d9b88d332172c41c76c151844e759492837152e7b1f"
FIELDS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trade_count", "taker_base", "taker_quote", "ignore"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_paxg() -> tuple[pd.DataFrame, dict]:
    manifest_path = OUT / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = []
    for rec in manifest["records"]:
        if rec["status"] != "OK":
            continue
        path = ROOT / rec["file"]
        if not path.exists() or sha256(path) != rec["sha256"]:
            raise ValueError(f"PAXG archive hash mismatch: {path}")
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if len(names) != 1 or not names[0].endswith(".csv"):
                raise ValueError(f"PAXG archive shape invalid: {path}")
            df = pd.read_csv(z.open(names[0]), header=None, names=FIELDS)
        if len(df) != rec["rows"]:
            raise ValueError(f"PAXG archive row mismatch: {path}")
        frames.append(df)
    p = pd.concat(frames, ignore_index=True)
    raw_ts = pd.to_numeric(p.open_time, errors="raise").astype("int64")
    ms = raw_ts < 10**14
    timestamp = pd.Series(pd.NaT, index=p.index, dtype="datetime64[ns, UTC]")
    timestamp.loc[ms] = pd.to_datetime(raw_ts.loc[ms], unit="ms", utc=True)
    timestamp.loc[~ms] = pd.to_datetime(raw_ts.loc[~ms], unit="us", utc=True)
    p["utc"] = timestamp
    p["ny"] = timestamp.dt.tz_convert("America/New_York")
    if p.utc.isna().any() or p.utc.duplicated().any() or not p.utc.is_monotonic_increasing:
        raise ValueError("PAXG hourly timestamp integrity failed")
    p["date"] = p.ny.dt.date
    p["wall_hour"] = p.ny.dt.hour
    p["wall_weekday"] = p.ny.dt.weekday
    p["close"] = pd.to_numeric(p.close, errors="raise")
    p["volume"] = pd.to_numeric(p.volume, errors="raise")
    p["trade_count"] = pd.to_numeric(p.trade_count, errors="raise")
    return p, {
        "source_manifest_sha256": sha256(manifest_path),
        "source_months": len(manifest["records"]),
        "source_months_ok": sum(r["status"] == "OK" for r in manifest["records"]),
        "source_rows": len(p),
        "first_utc": str(p.utc.iloc[0]),
        "last_utc": str(p.utc.iloc[-1]),
    }


def load_xau_endpoints() -> tuple[pd.DataFrame, str]:
    digest = sha256(XAU)
    if digest != EXPECTED_XAU_SHA:
        raise ValueError("Frozen XAU BID M1 hash mismatch")
    x = pd.read_csv(XAU, usecols=["time", "open"])
    ny = pd.to_datetime(x.time, unit="s", utc=True).dt.tz_convert("America/New_York")
    keep = ny.dt.weekday.eq(6) & ny.dt.hour.isin([18, 23]) & ny.dt.minute.eq(30)
    x = x.loc[keep, ["open"]].copy()
    x["date"] = ny.loc[keep].dt.date
    x["hour"] = ny.loc[keep].dt.hour
    if x.duplicated(["date", "hour"]).any():
        raise ValueError("Duplicate real XAU M1 endpoint")
    wide = x.pivot(index="date", columns="hour", values="open").rename(columns={18: "entry_bid", 23: "exit_bid"})
    return wide, digest


def bootstrap_p(values: np.ndarray) -> float:
    rng = np.random.default_rng(20261010)
    nonpositive = 0
    for _ in range(10000):
        nonpositive += int(values[rng.integers(len(values), size=len(values))].mean() <= 0.0)
    return nonpositive / 10000.0


def stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0}
    return {
        "trades": len(t),
        "gross_mean_units": float(t.gross_units.mean()),
        "base_mean_units": float(t.base_net_units.mean()),
        "base_total_units": float(t.base_net_units.sum()),
        "stress_mean_units": float(t.stress_net_units.mean()),
        "stress_total_units": float(t.stress_net_units.sum()),
        "win_rate_base": float((t.base_net_units > 0).mean()),
    }


def main() -> None:
    report_path = OUT / "REPORT_PAXG_WEEKEND_CONTINUATION.txt"
    if report_path.exists():
        raise FileExistsError(f"Frozen report exists: {report_path}")
    p, source = load_paxg()
    endpoints, xau_sha = load_xau_endpoints()
    friday = p.loc[(p.wall_weekday == 4) & (p.wall_hour == 16), ["date", "close", "volume", "trade_count"]].set_index("date")
    sunday = p.loc[(p.wall_weekday == 6) & (p.wall_hour == 16), ["date", "close", "volume", "trade_count"]].set_index("date")
    records = []
    for sun_date, row in sunday.iterrows():
        fri_date = sun_date - timedelta(days=2)
        rec = {"ny_date": sun_date.isoformat(), "friday_date": fri_date.isoformat(), "sunday_paxg_close": float(row.close), "sunday_paxg_volume": float(row.volume), "sunday_paxg_trades": int(row.trade_count)}
        if fri_date not in friday.index:
            rec["skip_reason"] = "MISSING_FRIDAY_BAR"
        else:
            f = friday.loc[fri_date]
            rec.update(friday_paxg_close=float(f.close), friday_paxg_volume=float(f.volume), friday_paxg_trades=int(f.trade_count))
            if row.volume <= 0 or row.trade_count <= 0 or f.volume <= 0 or f.trade_count <= 0:
                rec["skip_reason"] = "NO_PAXG_TRADE"
            elif row.close == f.close:
                rec["skip_reason"] = "ZERO_SIGNAL"
            else:
                rec["direction"] = 1 if row.close > f.close else -1
                rec["paxg_move_pct"] = 100 * (row.close / f.close - 1)
                if sun_date not in endpoints.index or pd.isna(endpoints.loc[sun_date].get("entry_bid")):
                    rec["skip_reason"] = "MISSING_XAU_ENTRY"
                elif pd.isna(endpoints.loc[sun_date].get("exit_bid")):
                    rec["skip_reason"] = "MISSING_XAU_EXIT"
                else:
                    rec["entry_bid"] = float(endpoints.loc[sun_date, "entry_bid"])
                    rec["exit_bid"] = float(endpoints.loc[sun_date, "exit_bid"])
                    rec["gross_units"] = rec["direction"] * (rec["exit_bid"] - rec["entry_bid"])
                    rec["base_net_units"] = rec["gross_units"] - 0.41
                    rec["stress_net_units"] = rec["gross_units"] - 0.82
                    rec["skip_reason"] = ""
        records.append(rec)
    signals = pd.DataFrame(records)
    trades = signals.loc[signals.skip_reason.eq("")].copy()
    trades["year"] = pd.to_datetime(trades.ny_date).dt.year
    cal = trades.loc[trades.year <= 2022]
    gated = trades.loc[trades.year >= 2023]
    pval = bootstrap_p(gated.base_net_units.to_numpy(dtype=float)) if not gated.empty else 1.0
    yearly = {str(y): stats(sub) for y, sub in trades.groupby("year")}
    gates = {
        "G-PW1": len(gated) >= 150,
        "G-PW2": bool(not gated.empty and gated.stress_net_units.mean() > 0),
        "G-PW3": pval <= 0.05,
        "G-PW4": all(y in yearly and yearly[y]["base_total_units"] > 0 for y in ("2023", "2024", "2025")),
    }
    report = {
        "source": source,
        "xau_sha256": xau_sha,
        "spec_sha256": sha256(OUT / "SPEC_PAXG_WEEKEND_CONTINUATION.txt"),
        "gate_sha256": sha256(OUT / "GATE_PAXG_WEEKEND_CONTINUATION.txt"),
        "signal_sundays": len(signals),
        "executed_sundays": len(trades),
        "skip_reasons": signals.loc[signals.skip_reason.ne(""), "skip_reason"].value_counts().to_dict(),
        "calibration_through_2022": stats(cal),
        "gated_2023_plus": stats(gated),
        "bootstrap_p_nonpositive": pval,
        "yearly": yearly,
        "gates": gates,
        "verdict": "PASS" if all(gates.values()) else "FAIL",
        "caveat": "BID M1 plus fixed-spread proxy, not variable native quotes; no stop, intraday floating-risk or FTMO pace validation. Historical data previously exposed to project research.",
    }
    signals.to_csv(OUT / "weekend_signals.csv", index=False)
    trades.to_csv(OUT / "weekend_trades.csv", index=False)
    report_path.write_text("REPORT: HYPOTHESIS 74 PAXG WEEKEND GOLD CONTINUATION\n" + json.dumps(report, indent=2, allow_nan=False) + "\nFINAL PHASE-1 VERDICT: " + report["verdict"] + "\n", encoding="utf-8")
    print("FINAL PHASE-1 VERDICT", report["verdict"])
    print("GATES", gates)
    print("FUNNEL", len(signals), len(trades), report["skip_reasons"])
    print("GATED", report["gated_2023_plus"], "p", pval)
    print("YEARLY", {y: round(v["base_total_units"], 3) for y, v in yearly.items()})


if __name__ == "__main__":
    main()
