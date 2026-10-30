"""Frozen H75: PAXG weekend return less the observed XAUUSD reopening gap."""

from __future__ import annotations

import json
import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from xauusd_paxg_weekend_continuation_phase1 import EXPECTED_XAU_SHA, XAU, load_paxg, sha256


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/xauusd_paxg_gap_residual_v1"
SOURCE_MANIFEST_SHA = "b8dc6726c5686861475109bea8b9bfa82c3587da335df6a01d9cbea468c6e9fc"


def gold_prices() -> tuple[dict, dict, dict]:
    if sha256(XAU) != EXPECTED_XAU_SHA:
        raise ValueError("Frozen XAUUSD BID M1 hash mismatch")
    x = pd.read_csv(XAU, usecols=["time", "open"])
    ny = pd.to_datetime(x.time, unit="s", utc=True).dt.tz_convert("America/New_York")
    weekday = ny.dt.weekday
    hour = ny.dt.hour
    minute = ny.dt.minute
    dates = ny.dt.date
    fri_mask = weekday.eq(4) & hour.eq(16) & minute.ge(50)
    sunday_entry = weekday.eq(6) & hour.eq(18) & minute.eq(30)
    sunday_exit = weekday.eq(6) & hour.eq(23) & minute.eq(30)
    anchors = pd.DataFrame({"date": dates.loc[fri_mask].to_numpy(), "bid": x.loc[fri_mask, "open"].to_numpy()})
    friday = anchors.drop_duplicates("date", keep="last").set_index("date").bid.to_dict()
    entry = dict(zip(dates.loc[sunday_entry], x.loc[sunday_entry, "open"], strict=True))
    exit_ = dict(zip(dates.loc[sunday_exit], x.loc[sunday_exit, "open"], strict=True))
    if len(entry) != int(sunday_entry.sum()) or len(exit_) != int(sunday_exit.sum()):
        raise ValueError("Duplicate Sunday XAUUSD M1 endpoint")
    return friday, entry, exit_


def summarise(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0}
    return {
        "trades": len(trades),
        "buy_trades": int((trades.direction == 1).sum()),
        "sell_trades": int((trades.direction == -1).sum()),
        "gross_mean_units": float(trades.gross_units.mean()),
        "base_mean_units": float(trades.base_net_units.mean()),
        "base_total_units": float(trades.base_net_units.sum()),
        "stress_mean_units": float(trades.stress_net_units.mean()),
        "stress_total_units": float(trades.stress_net_units.sum()),
        "win_rate_base": float((trades.base_net_units > 0).mean()),
    }


def bootstrap_p(values: np.ndarray) -> float:
    rng = np.random.default_rng(20261011)
    n = len(values)
    return float(sum(values[rng.integers(n, size=n)].mean() <= 0.0 for _ in range(10000)) / 10000.0)


def main() -> None:
    report_path = OUT / "REPORT_PAXG_GAP_RESIDUAL.txt"
    if report_path.exists():
        raise FileExistsError(f"Frozen report exists: {report_path}")
    p, source = load_paxg()
    if source["source_manifest_sha256"] != SOURCE_MANIFEST_SHA:
        raise ValueError("Frozen Binance source manifest hash mismatch")
    gold_fri, gold_entry, gold_exit = gold_prices()
    paxg_fri = p.loc[(p.wall_weekday == 4) & (p.wall_hour == 16), ["date", "close", "volume", "trade_count"]].set_index("date")
    paxg_sun = p.loc[(p.wall_weekday == 6) & (p.wall_hour == 17), ["date", "close", "volume", "trade_count"]].set_index("date")
    records = []
    for sun_date, sunday in paxg_sun.iterrows():
        fri_date = sun_date - timedelta(days=2)
        rec = {"ny_date": sun_date.isoformat(), "friday_date": fri_date.isoformat(), "paxg_sunday_close": float(sunday.close)}
        if fri_date not in paxg_fri.index:
            rec["skip_reason"] = "MISSING_PAXG_FRIDAY"
        else:
            friday = paxg_fri.loc[fri_date]
            rec["paxg_friday_close"] = float(friday.close)
            if sunday.volume <= 0 or sunday.trade_count <= 0 or friday.volume <= 0 or friday.trade_count <= 0:
                rec["skip_reason"] = "NO_PAXG_TRADES"
            elif fri_date not in gold_fri:
                rec["skip_reason"] = "MISSING_XAU_FRIDAY_ANCHOR"
            elif sun_date not in gold_entry:
                rec["skip_reason"] = "MISSING_XAU_ENTRY"
            else:
                rec["xau_friday_bid"] = float(gold_fri[fri_date])
                rec["xau_entry_bid"] = float(gold_entry[sun_date])
                rec["paxg_weekend_log_return"] = math.log(rec["paxg_sunday_close"] / rec["paxg_friday_close"])
                rec["xau_reopen_log_return"] = math.log(rec["xau_entry_bid"] / rec["xau_friday_bid"])
                rec["residual"] = rec["paxg_weekend_log_return"] - rec["xau_reopen_log_return"]
                if rec["residual"] == 0:
                    rec["skip_reason"] = "ZERO_RESIDUAL"
                elif sun_date not in gold_exit:
                    rec["skip_reason"] = "MISSING_XAU_EXIT"
                else:
                    rec["direction"] = 1 if rec["residual"] > 0 else -1
                    rec["xau_exit_bid"] = float(gold_exit[sun_date])
                    rec["gross_units"] = rec["direction"] * (rec["xau_exit_bid"] - rec["xau_entry_bid"])
                    rec["base_net_units"] = rec["gross_units"] - 0.41
                    rec["stress_net_units"] = rec["gross_units"] - 0.82
                    rec["skip_reason"] = ""
        records.append(rec)
    signals = pd.DataFrame(records)
    trades = signals.loc[signals.skip_reason.eq("")].copy()
    trades["year"] = pd.to_datetime(trades.ny_date).dt.year
    calibration = trades.loc[trades.year <= 2022]
    gated = trades.loc[trades.year >= 2023]
    pval = bootstrap_p(gated.base_net_units.to_numpy(dtype=float)) if not gated.empty else 1.0
    yearly = {str(y): summarise(t) for y, t in trades.groupby("year")}
    gates = {
        "G-PR1": len(gated) >= 150,
        "G-PR2": bool(not gated.empty and gated.stress_net_units.mean() > 0),
        "G-PR3": bool(pval <= 0.01),
        "G-PR4": all(y in yearly and yearly[y]["base_total_units"] > 0 for y in ("2023", "2024", "2025")),
    }
    report = {
        "source": source,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA,
        "xau_sha256": EXPECTED_XAU_SHA,
        "spec_sha256": sha256(OUT / "SPEC_PAXG_GAP_RESIDUAL.txt"),
        "gate_sha256": sha256(OUT / "GATE_PAXG_GAP_RESIDUAL.txt"),
        "signal_sundays": len(signals),
        "executed_sundays": len(trades),
        "skip_reasons": signals.loc[signals.skip_reason.ne(""), "skip_reason"].value_counts().to_dict(),
        "calibration_through_2022": summarise(calibration),
        "gated_2023_plus": summarise(gated),
        "bootstrap_p_nonpositive": pval,
        "yearly": yearly,
        "gates": gates,
        "verdict": "PASS" if all(gates.values()) else "FAIL",
        "caveat": "Historical PAXG/XAUUSD source exposure; BID M1 fixed 0.41 spread proxy, no Sunday variable spread, intraday floating-loss, stop or FTMO pace test.",
    }
    report_text = "REPORT: HYPOTHESIS 75 PAXG-XAUUSD WEEKEND GAP RESIDUAL\n" + json.dumps(report, indent=2, allow_nan=False) + "\nFINAL PHASE-1 VERDICT: " + report["verdict"] + "\n"
    signals.to_csv(OUT / "residual_signals.csv", index=False)
    trades.to_csv(OUT / "residual_trades.csv", index=False)
    report_path.write_text(report_text, encoding="utf-8")
    print("FINAL PHASE-1 VERDICT", report["verdict"])
    print("GATES", gates)
    print("FUNNEL", len(signals), len(trades), report["skip_reasons"])
    print("GATED", report["gated_2023_plus"], "p", pval)
    print("YEARLY", {y: round(v["base_total_units"], 3) for y, v in yearly.items()})


if __name__ == "__main__":
    main()
