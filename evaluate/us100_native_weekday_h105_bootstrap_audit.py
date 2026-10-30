"""Separate H105 calendar-week bootstrap implementation audit."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/us100_native_weekday_h105"
LEDGER = OUT / "trades.csv"
REPORT = OUT / "BOOTSTRAP_AUDIT_REPORT_2026-09-26.txt"
EXPECTED = "ed74f84abe8d8a974a3d0434c2b20eb55bdaa169763a2abbfb0a498a050391d3"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(REPORT)
    if sha(LEDGER) != EXPECTED:
        raise ValueError("Frozen H105 trade ledger changed")
    trades = pd.read_csv(LEDGER)
    trades = trades[trades.year >= 2024].copy()
    if len(trades) != 140 or trades.monday.duplicated().any():
        raise ValueError("H105 OOS trade count/week uniqueness changed")
    weeks = pd.date_range(trades.monday.min(), trades.monday.max(), freq="W-MON")
    by_week = {monday: float(value) for monday, value in
               zip(trades.monday, trades.stress_net_units)}
    blocks = [np.array([by_week[str(weeks[(i + j) % len(weeks)].date())]
                        for j in range(5)
                        if str(weeks[(i + j) % len(weeks)].date()) in by_week], dtype=float)
              for i in range(len(weeks))]
    rng = np.random.default_rng(20261004)
    count = 0
    for _ in range(10000):
        parts = []
        length = 0
        while length < len(trades):
            part = blocks[int(rng.integers(len(blocks)))]
            parts.append(part)
            length += len(part)
        if np.concatenate(parts)[:len(trades)].mean() <= 0:
            count += 1
    p = count / 10000
    lines = ["H105 FIVE-CALENDAR-WEEK BLOCK BOOTSTRAP AUDIT",
             "Pinned ledger SHA256: " + EXPECTED,
             f"Calendar weeks={len(weeks)}, observed trades={len(trades)}, empty weeks={len(weeks)-len(trades)}",
             "Frozen original five-TRADE block P(mean<=0)=0.0545",
             f"Corrected five-CALENDAR-WEEK block P(mean<=0)={p:.4f}",
             "Original H105 FINAL_FAIL remains immutable. No gate promotion or orders."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
