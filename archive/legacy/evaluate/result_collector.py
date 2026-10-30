"""
result_collector.py — Parse MT5 Strategy Tester XML reports into CAIRO-compatible CSV.

Reads MT5 XML report files from data/results/<symbol>/<fold_id>_report.xml,
extracts performance metrics, applies bad-fold propagation guards, and writes:
  - data/results/results.csv       (passing folds)
  - data/results/results_excluded.csv (flagged folds)

Bad-fold guards (from plan §8.2):
  1. Continuity check — trade count must be >= MIN_TRADES (default 5)
  2. Trade count sanity — if all folds have high trade count but one has 0,
     flag the zero one
  3. Sharpe bounds — Sharpe outside [SHARPE_MIN, SHARPE_MAX] is suspicious
     (default -5.0 to 100.0)
  4. Max DD sanity — max DD > 99.9% suggests capital wipe, flag
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


def _log(msg: str) -> None:
    print(f"[Result] {msg}", flush=True)


# ── Bad-fold guard thresholds ─────────────────────────────────────────────

MIN_TRADES    = 5
SHARPE_MIN    = -10.0
SHARPE_MAX    = 100.0
MAX_DD_WARN   = 99.9      # capital wipe indicator


# ── MT5 XML parsing ────────────────────────────────────────────────────────

def parse_report(xml_path: str | Path) -> dict:
    """Extract performance metrics from an MT5 Strategy Tester XML report.

    Returns a dict with keys matching the CAIRO results.csv columns,
    or None if the report is invalid/missing.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        _log(f"Not found: {xml_path}")
        return None

    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except ET.ParseError as e:
        _log(f"Parse error in {xml_path}: {e}")
        return None

    # Extract from <General>
    general = root.find("General") if root.tag == "StrategyTesterReport" else None
    if general is None:
        # Try flat XML structure
        general = root

    def _get(parent, tag: str, default=0.0):
        el = parent.find(tag)
        if el is not None and el.text:
            try:
                return float(el.text)
            except ValueError:
                return str(el.text)
        return default

    def _get_int(parent, tag: str, default=0):
        el = parent.find(tag)
        if el is not None and el.text:
            try:
                return int(el.text)
            except ValueError:
                return default
        return default

    analysis = root.find("Analysis")
    if analysis is None:
        analysis = root

    total_net_profit   = _get(analysis, "TotalNetProfit")
    gross_profit       = _get(analysis, "GrossProfit")
    gross_loss         = _get(analysis, "GrossLoss")
    profit_factor      = _get(analysis, "ProfitFactor")
    expected_payoff    = _get(analysis, "ExpectedPayoff")
    absolute_dd        = _get(analysis, "AbsoluteDrawdown")
    max_dd_money       = _get(analysis, "MaximalDrawdown")
    relative_dd_pct    = _get(analysis, "RelativeDrawdown")
    total_trades       = _get_int(analysis, "TotalTrades")
    short_won          = _get_int(analysis, "ShortPositionsWon")
    long_won           = _get_int(analysis, "LongPositionsWon")
    profit_trades      = _get_int(analysis, "ProfitTrades")
    loss_trades        = _get_int(analysis, "LossTrades")
    consec_wins        = _get_int(analysis, "ConsecutiveWins")
    consec_losses      = _get_int(analysis, "ConsecutiveLosses")
    max_consec_profit  = _get(analysis, "MaxConsecutiveProfit")
    max_consec_loss    = _get(analysis, "MaxConsecutiveLoss")
    sharpe_ratio       = _get(analysis, "SharpeRatio")
    recovery_factor    = _get(analysis, "RecoveryFactor")

    # Win rate calculation
    win_rate = (profit_trades / total_trades * 100.0) if total_trades > 0 else 0.0

    # Extract swap + commission totals from <Deals> if available
    deals = root.find("Deals")
    total_swap       = 0.0
    total_commission = 0.0
    if deals is not None:
        for deal in deals.findall("Deal"):
            total_swap       += _get(deal, "Swap")
            total_commission += _get(deal, "Commission")

    return {
        "total_net_profit":    total_net_profit,
        "gross_profit":        gross_profit,
        "gross_loss":          gross_loss,
        "profit_factor":       profit_factor,
        "expected_payoff":     expected_payoff,
        "absolute_drawdown":   absolute_dd,
        "max_drawdown":        max_dd_money,
        "max_drawdown_pct":    relative_dd_pct,
        "total_trades":        total_trades,
        "short_won":           short_won,
        "long_won":            long_won,
        "profit_trades":       profit_trades,
        "loss_trades":         loss_trades,
        "win_rate_pct":        round(win_rate, 2),
        "consecutive_wins":    consec_wins,
        "consecutive_losses":  consec_losses,
        "max_consec_profit":   max_consec_profit,
        "max_consec_loss":     max_consec_loss,
        "sharpe_ratio":        sharpe_ratio,
        "recovery_factor":     recovery_factor,
        "total_swap":          total_swap,
        "total_commission":    total_commission,
    }


# ── Bad-fold guards ────────────────────────────────────────────────────────

def apply_guards(
    metrics: dict,
    symbol: str,
    fold_id: str,
    all_fold_trade_counts: list[int] | None = None,
) -> tuple[bool, list[str]]:
    """Apply bad-fold propagation guards.

    Returns (ok, reasons) where reasons lists any failures.
    """
    reasons = []
    trades = metrics.get("total_trades", 0)
    sharpe = metrics.get("sharpe_ratio", 0.0)
    max_dd = metrics.get("max_drawdown_pct", 0.0)

    # 1. Trade count minimum
    if trades < MIN_TRADES:
        reasons.append(f"low_trade_count: {trades} < {MIN_TRADES}")

    # 2. Trade count sanity vs peer folds
    if all_fold_trade_counts and trades == 0:
        avg_trades = sum(all_fold_trade_counts) / max(len(all_fold_trade_counts), 1)
        if avg_trades > MIN_TRADES * 5:
            reasons.append(
                f"zero_trades_vs_peers: {trades} vs avg {avg_trades:.0f}"
            )

    # 3. Sharpe bounds
    if sharpe < SHARPE_MIN:
        reasons.append(f"sharpe_too_low: {sharpe:.2f} < {SHARPE_MIN}")
    if sharpe > SHARPE_MAX:
        reasons.append(f"sharpe_too_high: {sharpe:.2f} > {SHARPE_MAX}")

    # 4. Max drawdown — capital wipe
    if max_dd > MAX_DD_WARN:
        reasons.append(f"max_dd_capital_wipe: {max_dd:.2f}% > {MAX_DD_WARN}%")

    return (len(reasons) == 0, reasons)


# ── CSV output ─────────────────────────────────────────────────────────────

RESULTS_CSV_COLUMNS = [
    "fold_id", "symbol", "fidelity_mode", "data_source", "total_net_profit", "gross_profit", "gross_loss",
    "profit_factor", "expected_payoff", "max_drawdown", "max_drawdown_pct",
    "total_trades", "win_rate_pct", "profit_trades", "loss_trades",
    "consecutive_wins", "consecutive_losses",
    "sharpe_ratio", "recovery_factor",
    "total_swap", "total_commission",
    "flagged", "flag_reasons",
]


def write_results(
    results: list[dict],
    output_path: str | Path,
    excluded_path: str | Path = "",
) -> tuple[int, int]:
    """Write all fold results to CSV. Returns (passed_count, excluded_count)."""
    output_path = Path(output_path)
    excluded_path = Path(excluded_path) if excluded_path else output_path.with_name(
        f"{output_path.stem}_excluded.csv"
    )

    passed_rows   = []
    excluded_rows = []

    for r in results:
        if r.get("flagged"):
            excluded_rows.append(r)
        else:
            passed_rows.append(r)

    def _write(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=RESULTS_CSV_COLUMNS,
                               extrasaction="ignore")
            w.writeheader()
            for row in rows:
                row_out = {k: row.get(k, "") for k in RESULTS_CSV_COLUMNS}
                w.writerow(row_out)

    if passed_rows:
        _write(output_path, passed_rows)
        _log(f"Wrote {len(passed_rows)} rows to {output_path}")
    if excluded_rows:
        _write(excluded_path, excluded_rows)
        _log(f"Wrote {len(excluded_rows)} excluded rows to {excluded_path}")

    return len(passed_rows), len(excluded_rows)


# ── Self-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("=== result_collector.py self-test ===")

    # Generate a mock MT5 XML report
    mock_xml = """<?xml version="1.0" encoding="UTF-8"?>
<StrategyTesterReport>
<General>
  <ExpertName>MULTI_CAIRO_EA_TEST</ExpertName>
  <Symbol>EURUSD</Symbol>
  <Period>H1</Period>
</General>
<Analysis>
  <TotalNetProfit>12345.67</TotalNetProfit>
  <GrossProfit>50000.00</GrossProfit>
  <GrossLoss>-37654.33</GrossLoss>
  <ProfitFactor>1.33</ProfitFactor>
  <ExpectedPayoff>45.90</ExpectedPayoff>
  <AbsoluteDrawdown>1200.00</AbsoluteDrawdown>
  <MaximalDrawdown>3456.78</MaximalDrawdown>
  <RelativeDrawdown>12.35</RelativeDrawdown>
  <TotalTrades>269</TotalTrades>
  <ProfitTrades>134</ProfitTrades>
  <LossTrades>135</LossTrades>
  <ConsecutiveWins>8</ConsecutiveWins>
  <ConsecutiveLosses>11</ConsecutiveLosses>
  <SharpeRatio>1.85</SharpeRatio>
  <RecoveryFactor>3.57</RecoveryFactor>
</Analysis>
<Deals>
  <Deal><Swap>-23.45</Swap><Commission>-45.00</Commission></Deal>
  <Deal><Swap>12.30</Swap><Commission>-30.00</Commission></Deal>
  <Deal><Swap>-5.10</Swap><Commission>-25.00</Commission></Deal>
</Deals>
</StrategyTesterReport>"""

    # Test: parse mock report
    tmp = Path(tempfile.mkdtemp())
    xml_path = tmp / "test_report.xml"
    xml_path.write_text(mock_xml, encoding="utf-8")

    metrics = parse_report(xml_path)
    print(f"\n[1] Parsed metrics ({len(metrics)} fields):")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    # Test: apply guards — should pass
    ok, reasons = apply_guards(metrics, "EURUSD", "202606")
    print(f"\n[2] Guards: {'PASS' if ok else 'FAIL'}")
    if reasons:
        for r in reasons:
            print(f"    - {r}")

    # Test: apply guards with bad fold
    bad_metrics = dict(metrics)
    bad_metrics["total_trades"] = 2
    bad_metrics["sharpe_ratio"] = 150.0
    ok2, reasons2 = apply_guards(bad_metrics, "EURUSD", "202606")
    print(f"\n[3] Bad fold guards: {'PASS' if ok2 else 'FLAGGED'} ({len(reasons2)} reasons)")
    for r in reasons2:
        print(f"    - {r}")

    # Test: write CSV
    results = [
        {**metrics, "fold_id": "202606", "symbol": "EURUSD", "flagged": False, "flag_reasons": ""},
        {**bad_metrics, "fold_id": "202607", "symbol": "EURUSD", "flagged": True, "flag_reasons": "; ".join(reasons2)},
    ]
    passed, excluded = write_results(
        results, tmp / "results.csv", tmp / "results_excluded.csv"
    )
    print(f"\n[4] CSV output: {passed} passed, {excluded} excluded")
    print(f"    Passed CSV: {tmp}/results.csv")
    print(f"    Excluded CSV: {tmp}/results_excluded.csv")

    # Show the written CSV
    if (tmp / "results.csv").exists():
        print(f"\n    results.csv content:")
        print((tmp / "results.csv").read_text())
    if (tmp / "results_excluded.csv").exists():
        print(f"\n    results_excluded.csv content:")
        print((tmp / "results_excluded.csv").read_text())

    print("\n=== Done ===")
