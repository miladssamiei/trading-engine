"""Apply a frozen FTMO Swing swap snapshot to exposed US100 ledgers."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/project_status/ftmo_swing_swap_transfer_2026-09-26"
INPUTS = {
    "H94": (ROOT / "data/results/us100_weekday_hold_h94/trades.csv",
            "5cf3b14bab33e0f6bfff79fbff2a5793f33822981273bb35376c1cb2a75e51e1"),
    "H95": (ROOT / "data/results/vix_weekday_h95/trades.csv",
            "dcaf647df4fec656c97f3e93fdb6f189a57a17ab370ebba891f955131297bcc3"),
}
SWAP_UNITS = {"LONG": -675.12 * 0.01, "SHORT": 20.47 * 0.01}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def rollover_weight(entry: int, exit: int) -> tuple[int, int]:
    a = pd.Timestamp(entry, unit="s", tz="UTC")
    b = pd.Timestamp(exit, unit="s", tz="UTC")
    if b <= a:
        raise ValueError("Exit is not after entry")
    local_a = a.tz_convert("Europe/Athens")
    local_b = b.tz_convert("Europe/Athens")
    count = weighted = 0
    for date in pd.date_range(local_a.date(), local_b.date(), freq="D"):
        midnight = date.tz_localize("Europe/Athens").tz_convert("UTC")
        if a < midnight < b:
            count += 1
            weighted += 3 if date.weekday() == 4 else 1
    return count, weighted


def main() -> None:
    output = OUT / "swap_scenario_trades.csv"
    report = OUT / "SWAP_TRANSFER_REPORT_2026-09-26.txt"
    if output.exists() or report.exists():
        raise FileExistsError("Swap transfer artifact already exists")
    for path, expected in INPUTS.values():
        if digest(path) != expected:
            raise ValueError(f"Frozen ledger hash mismatch: {path}")
    rows = []
    for study, (path, _) in INPUTS.items():
        trades = pd.read_csv(path)
        for index, trade in trades.iterrows():
            side = str(trade.get("side", "LONG"))
            if side not in SWAP_UNITS:
                raise ValueError(f"Invalid side: {side}")
            count, weight = rollover_weight(int(trade.entry_epoch), int(trade.exit_epoch))
            swap = SWAP_UNITS[side] * weight
            rows.append({"study": study, "row_index": index, "monday": trade.monday,
                         "year": int(trade.year), "side": side,
                         "entry_epoch": int(trade.entry_epoch), "exit_epoch": int(trade.exit_epoch),
                         "rollovers": count, "weighted_swap_days": weight,
                         "frozen_stress_net_units": float(trade.stress_net),
                         "snapshot_swap_units": swap,
                         "scenario_stress_net_units": float(trade.stress_net) + swap})
    OUT.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    frame = pd.DataFrame(rows)
    lines = ["FTMO SWING US100 CURRENT-SNAPSHOT SWAP TRANSFER",
             "Frozen H94/H95 zero-swap ledgers unchanged. Scenario only; no historical swap schedule.",
             "Current LONG/SHORT units per normal rollover: " + str(SWAP_UNITS),
             "Server midnight modeled Europe/Athens (GMT+2 plus European DST); Friday triple.",
             "Scenario CSV SHA256: " + digest(output)]
    for study in INPUTS:
        group = frame[(frame.study == study) & (frame.year >= 2024)]
        lines.extend([study + " 2024+ trades=" + str(len(group)),
                      "  mean frozen/scenario stress units=" +
                      f"{group.frozen_stress_net_units.mean():.6f}/{group.scenario_stress_net_units.mean():.6f}",
                      "  total frozen/scenario stress units=" +
                      f"{group.frozen_stress_net_units.sum():.6f}/{group.scenario_stress_net_units.sum():.6f}",
                      "  total rollovers/weighted days=" +
                      f"{group.rollovers.sum()}/{group.weighted_swap_days.sum()}",
                      "  swap total units=" + f"{group.snapshot_swap_units.sum():.6f}"])
        for year, part in group.groupby("year"):
            lines.append(f"  {year}: n={len(part)} frozen={part.frozen_stress_net_units.sum():.6f} "
                         f"scenario={part.scenario_stress_net_units.sum():.6f} "
                         f"swap={part.snapshot_swap_units.sum():.6f}")
    lines.append("H101 long-US100 child would weaken under negative swap; full prop replay not run.")
    lines.append("No native quote transfer, EUR conversion, floating-equity pass, forward validation or order.")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
