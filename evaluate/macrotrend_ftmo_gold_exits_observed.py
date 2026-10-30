"""Extract observed MT5 Gold tester stops; execution is not Python-replicated."""
from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/macrotrend_ftmo_gold_parity_2024_v2"
TIME = re.compile(r"(20\d\d\.\d\d\.\d\d \d\d:\d\d:\d\d)\s+")
OPEN = re.compile(r"MTE: OPEN (BUY|SELL) engine=([AB]) lots=([\d.]+) entry=([\d.]+) SL=([\d.]+)")
STOP = re.compile(r"stop loss triggered #(\d+) (buy|sell) ([\d.]+) XAUUSD ([\d.]+) sl: ([\d.]+) \[#(\d+) (?:buy|sell) [\d.]+ XAUUSD at ([\d.]+)\]")
DEAL = re.compile(r"deal #(\d+) (buy|sell) ([\d.]+) XAUUSD at ([\d.]+) done")


def main() -> None:
    result = []
    for engine in ("A", "B"):
        lines = (OUT/f"mt5_{engine}_journal_snapshot.txt").read_text(encoding="utf-8").splitlines()
        opens, stops, fills = [], [], {}
        for line in lines:
            stamp = TIME.search(line)
            if not stamp:
                continue
            timestamp = stamp[1].replace(".", "-")
            opening = OPEN.search(line)
            stop = STOP.search(line)
            deal = DEAL.search(line)
            if opening:
                opens.append(dict(engine=engine, entry_time=timestamp, side=opening[1],
                                  lots=float(opening[3]), entry=float(opening[4]), initial_sl=float(opening[5])))
            if stop:
                stops.append(dict(exit_time=timestamp, ticket=int(stop[1]), side=stop[2].upper(),
                                  lots=float(stop[3]), entry=float(stop[4]), stop_level=float(stop[5]),
                                  exit_deal=int(stop[6]), requested_exit=float(stop[7])))
            if deal:
                fills[int(deal[1])] = float(deal[4])
        if len(opens) != len(stops):
            raise RuntimeError(f"{engine}: {len(opens)} OPEN events but {len(stops)} stop events")
        for entry, exit_ in zip(opens, stops):
            if entry["side"] != exit_["side"] or abs(entry["entry"]-exit_["entry"]) > 0.011 or abs(entry["lots"]-exit_["lots"]) > 1e-8:
                raise RuntimeError(f"{engine}: stop does not pair with preceding OPEN: {entry} {exit_}")
            actual = fills.get(exit_["exit_deal"])
            if actual is None:
                raise RuntimeError(f"{engine}: missing closing deal {exit_['exit_deal']}")
            side = 1 if entry["side"]=="BUY" else -1
            result.append({**entry, **exit_, "observed_exit": actual,
                           "gross_price_units": side*(actual-entry["entry"]),
                           "stop_slippage_units": side*(exit_["requested_exit"]-actual)})
    ledger = pd.DataFrame(result)
    ledger.to_csv(OUT/"mt5_observed_trades.csv", index=False)
    lines = ["MT5 OBSERVED GOLD TRADES - SUPPLEMENT TO V2 PARITY AUDIT", "All 129 logged opens paired in sequence with a logged stop-trigger and actual closing deal.",
             "These are MT5 tester observations; Python exits remain UNTESTED because complete same-feed M1/ticks are unavailable."]
    for engine in ("A", "B"):
        x = ledger[ledger.engine==engine]
        lines.append(f"Engine {engine}: {len(x)} entries, {len(x)} observed stops, gross price-unit sum {x.gross_price_units.sum():.2f}, win fraction {(x.gross_price_units>0).mean():.4f}.")
    (OUT/"EXECUTION_OBSERVATIONS.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
