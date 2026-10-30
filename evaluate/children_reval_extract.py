# -*- coding: utf-8 -*-
"""
children_reval_extract.py - stitch the Model=4 yearly batch outputs
into per-strategy mt5_trades.csv / mt5_days.csv (original stitching
convention: tickets renumbered sequentially), copy the unchanged
Python references (and the M5 export for ORB) into the per-strategy
working dirs, and write the deliverable-named CSVs.
"""

import csv
import shutil
from pathlib import Path

BASE = Path(r"C:\Users\Milad\Desktop\RR\data\results"
            r"\tick_model_revalidation_children")
ORB_ORIG = Path(r"C:\Users\Milad\Desktop\RR\data\results\orb_2021_present")
SP2L_ORIG = Path(r"C:\Users\Milad\Desktop\RR\data\results\sp2l_2021_2026")
YEARS = ("2021", "2022", "2023", "2024", "2025", "2026")

CFG = {
    "orb": {"days": "orb_days.csv", "orig": ORB_ORIG,
            "deliverable_trades": "orb_model4_trades.csv",
            "deliverable_days": "orb_model4_days.csv"},
    "sp2l": {"days": "sp2l_days.csv", "orig": SP2L_ORIG,
             "deliverable_trades": "sp2l_model4_trades.csv",
             "deliverable_days": "sp2l_model4_days.csv"},
}


def stitch(strategy):
    cfg = CFG[strategy]
    trades, days = [], []
    for label in YEARS:
        b = BASE / strategy / "batches" / f"year_{label}"
        with open(b / "trades.csv", newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter=";"))
        rows.sort(key=lambda r: r["open_time"])
        for i, r in enumerate(rows, start=1):
            r["ticket"] = str(len(trades) + i)
        trades.extend(rows)
        with open(b / cfg["days"], newline="") as fh:
            drows = list(csv.DictReader(fh, delimiter=";"))
        drows.sort(key=lambda r: int(r["ny_date"]))
        days.extend(drows)
    wdir = BASE / strategy
    with open(wdir / "mt5_trades.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(trades[0].keys()),
                           delimiter=";")
        w.writeheader()
        w.writerows(trades)
    with open(wdir / "mt5_days.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(days[0].keys()),
                           delimiter=";")
        w.writeheader()
        w.writerows(days)
    shutil.copy2(wdir / "mt5_trades.csv", BASE / cfg["deliverable_trades"])
    shutil.copy2(wdir / "mt5_days.csv", BASE / cfg["deliverable_days"])
    # unchanged references into the working dir (for the reconcile
    # scripts, which read BASE-local filenames via --base-dir)
    for name in ("py_trades.csv", "py_days.csv"):
        shutil.copy2(cfg["orig"] / name, wdir / name)
    pnl = sum(float(t["profit"]) for t in trades)
    print(f"{strategy}: {len(trades)} trades, {len(days)} days, "
          f"total pnl {pnl:+.2f}")


def main():
    stitch("orb")
    stitch("sp2l")
    shutil.copy2(ORB_ORIG / "us500tick_m5.csv",
                 BASE / "orb" / "us500tick_m5.csv")
    print("copied py references and (orb) us500tick_m5.csv")


if __name__ == "__main__":
    main()
