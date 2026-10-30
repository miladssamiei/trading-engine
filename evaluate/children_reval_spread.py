# -*- coding: utf-8 -*-
"""
children_reval_spread.py - real-tick spread re-measured over each
strategy's own trading windows (SPEC_TICK_MODEL_REVALIDATION_CHILDREN):
ORB entry window 10:00-10:30 NY, ORB session 09:30-16:00 NY,
SP2L validated entry window 09:30-15:30 NY, from the Custom-base real
ticks via the mt5 API (method verified in the prior revalidation).
"""

import csv
import datetime as dt
import time
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np

TERMINAL = r"C:\Program Files\MetaTrader 5\terminal64.exe"
OUT = Path(r"C:\Users\Milad\Desktop\RR\data\results"
           r"\tick_model_revalidation_children")
UTC = dt.timezone.utc

WINDOWS = {
    "orb_entry_1000_1030": (600, 630),
    "orb_session_0930_1600": (570, 960),
    "sp2l_entry_0930_1530": (570, 930),
}


def main():
    if not mt5.initialize(TERMINAL):
        raise SystemExit(f"init failed: {mt5.last_error()}")
    try:
        mt5.symbol_select("US500-tick", True)
        time.sleep(3)
        agg = {k: [] for k in WINDOWS}
        per_year = {k: {} for k in WINDOWS}
        for y in range(2021, 2027):
            for m in range(1, 13):
                if y == 2026 and m > 4:
                    continue
                lo = dt.datetime(y, m, 1, tzinfo=UTC)
                hi = dt.datetime(y + (m == 12), (m % 12) + 1, 1,
                                 tzinfo=UTC)
                tk = mt5.copy_ticks_range("US500-tick", lo, hi,
                                          mt5.COPY_TICKS_ALL)
                if tk is None or not len(tk):
                    print(f"{y}{m:02d}: no ticks", flush=True)
                    continue
                ts = pd_ts = tk["time"]
                import pandas as pd
                idx = pd.DatetimeIndex(pd.to_datetime(ts, unit="s",
                                                      utc=True))
                ny = idx.tz_convert("America/New_York")
                wall = (ny.hour * 60 + ny.minute).to_numpy()
                bid = tk["bid"]
                ask = tk["ask"]
                msk = (ask > 0) & (bid > 0) & (ask > bid)
                sp = (ask - bid).astype(float)
                for name, (w0, w1) in WINDOWS.items():
                    sel = msk & (wall >= w0) & (wall < w1)
                    if sel.any():
                        agg[name].append(sp[sel])
                        per_year[name].setdefault(
                            y, []).append(sp[sel])
                print(f"{y}{m:02d}: {len(tk):,} ticks", flush=True)
        rows = [{"window": "full_tape_reference", "n": None,
                 "median_spread": 0.516, "mean_spread": 0.537,
                 "note": "prior revalidation full-tape measurement"},
                {"window": "synthetic_assumption", "n": None,
                 "median_spread": 0.41, "mean_spread": 0.41,
                 "note": "Model=0 constant (documented)"}]
        for name in WINDOWS:
            sp = np.concatenate(agg[name])
            rows.append({"window": name, "n": int(len(sp)),
                         "median_spread": round(float(np.median(sp)), 4),
                         "mean_spread": round(float(sp.mean()), 4),
                         "note": "real ticks, strategy window"})
        with open(OUT / "spread_comparison_windows.csv", "w",
                  newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["window", "n",
                                               "median_spread",
                                               "mean_spread", "note"])
            w.writeheader()
            w.writerows(rows)
        # deliverable-named copies
        orb_rows = [r for r in rows if r["window"].startswith("orb")
                    or r["window"] in ("full_tape_reference",
                                       "synthetic_assumption")]
        sp_rows = [r for r in rows if r["window"].startswith("sp2l")
                   or r["window"] in ("full_tape_reference",
                                      "synthetic_assumption")]
        for fname, rr in (("spread_comparison_orb_sessions.csv", orb_rows),
                          ("spread_comparison_sp2l_sessions.csv", sp_rows)):
            with open(OUT / fname, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["window", "n",
                                                   "median_spread",
                                                   "mean_spread", "note"])
                w.writeheader()
                w.writerows(rr)
        for r in rows:
            print(r)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
