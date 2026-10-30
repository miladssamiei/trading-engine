"""Frozen pooled-R Monte Carlo for the owner-specified FTMO-like scenario.

Read SPEC_MC.txt and GATE_MC.txt before changing this experiment. The script
creates a historical report once and refuses to overwrite it on later runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/results/crypto_gold_real_cost/pooled_R_curve.csv"
OUTPUT = ROOT / "data/results/ftmo_mc_crypto_gold"
REPORT = OUTPUT / "REPORT_FTMO_MC_CRYPTO_GOLD.txt"
EXPECTED_SHA256 = "d11585c778f7d638ba209cc595361fb07d15203528400da2f8c4ae2bc120e4fb"
EXPECTED_MEAN_R = 0.09691495204712176
N_PATHS = 10_000
MAX_PHASE_TRADES = 20_000
SEED = 20261031
STARTING_CASH = 10_000.0
FEE = 175.0
PROFIT_SPLIT = 0.80
TRADE_RATE_PER_MONTH = 29.1
RISK_PROFILES = (0.005, 0.010, 0.015, 0.020)


def verify_input() -> tuple[np.ndarray, str]:
    if not (OUTPUT / "SPEC_MC.txt").is_file() or not (OUTPUT / "GATE_MC.txt").is_file():
        raise RuntimeError("Frozen SPEC_MC.txt and GATE_MC.txt must exist before execution")
    digest = hashlib.sha256(INPUT.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"Pooled ledger SHA256 changed: {digest}")
    frame = pd.read_csv(INPUT)
    required = {"net_r", "cumulative_net_r"}
    if not required.issubset(frame.columns) or len(frame) != 2_275:
        raise ValueError("Pooled ledger schema or expected 2,275-trade count changed")
    net_r = frame["net_r"].to_numpy(dtype=np.float64)
    cumulative = frame["cumulative_net_r"].to_numpy(dtype=np.float64)
    if not (np.isfinite(net_r).all() and np.isfinite(cumulative).all()):
        raise ValueError("Pooled ledger has nonfinite R values")
    if abs(float(net_r.mean()) - EXPECTED_MEAN_R) > 1e-6:
        raise ValueError("Pooled mean R differs from frozen baseline")
    if not np.allclose(np.cumsum(net_r), cumulative, rtol=0, atol=1e-6):
        raise ValueError("Pooled cumulative R column does not reconcile")
    return net_r, digest


def run_evaluation_phase(
    rng: np.random.Generator, pool: np.ndarray, n: int, cash_risk: float, target: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Codes: 1 passed, -1 static loss, -2 single-trade proxy, 0 capped."""
    status = np.zeros(n, dtype=np.int8)
    count = np.zeros(n, dtype=np.int32)
    pnl = np.zeros(n, dtype=np.float64)
    for _ in range(MAX_PHASE_TRADES):
        active = np.flatnonzero(status == 0)
        if active.size == 0:
            break
        delta = cash_risk * pool[rng.integers(pool.size, size=active.size)]
        count[active] += 1
        pnl[active] += delta
        single_loss = delta <= -500.0
        static_loss = (pnl[active] <= -1_000.0) & ~single_loss
        passed = (pnl[active] >= target) & ~single_loss & ~static_loss
        status[active[single_loss]] = -2
        status[active[static_loss]] = -1
        status[active[passed]] = 1
    return status, count, pnl


def run_funded(
    rng: np.random.Generator, pool: np.ndarray, n: int, cash_risk: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Codes: 1 completed all 30 trades, -1 breached static loss."""
    status = np.zeros(n, dtype=np.int8)
    count = np.zeros(n, dtype=np.int32)
    pnl = np.zeros(n, dtype=np.float64)
    for _ in range(30):
        active = np.flatnonzero(status == 0)
        if active.size == 0:
            break
        delta = cash_risk * pool[rng.integers(pool.size, size=active.size)]
        count[active] += 1
        pnl[active] += delta
        status[active[pnl[active] <= -1_000.0]] = -1
    status[status == 0] = 1
    return status, count, pnl


def simulate_profile(pool: np.ndarray, profile: float, index: int) -> pd.DataFrame:
    cash_risk = STARTING_CASH * profile
    rng = np.random.default_rng(SEED + 1_000 * index)
    p1_status, p1_count, p1_pnl = run_evaluation_phase(
        rng, pool, N_PATHS, cash_risk, 1_000.0
    )
    p1_pass = np.flatnonzero(p1_status == 1)
    p2_status = np.full(N_PATHS, 9, dtype=np.int8)
    p2_count = np.zeros(N_PATHS, dtype=np.int32)
    p2_pnl = np.full(N_PATHS, np.nan)
    if p1_pass.size:
        s, c, p = run_evaluation_phase(rng, pool, p1_pass.size, cash_risk, 500.0)
        p2_status[p1_pass], p2_count[p1_pass], p2_pnl[p1_pass] = s, c, p
    both_pass = np.flatnonzero(p2_status == 1)
    funded_status = np.full(N_PATHS, 9, dtype=np.int8)
    funded_count = np.zeros(N_PATHS, dtype=np.int32)
    funded_pnl = np.full(N_PATHS, np.nan)
    if both_pass.size:
        s, c, p = run_funded(rng, pool, both_pass.size, cash_risk)
        funded_status[both_pass], funded_count[both_pass], funded_pnl[both_pass] = s, c, p
    payout = (funded_status == 1) & (funded_pnl > 0)
    receipt = np.zeros(N_PATHS, dtype=np.float64)
    receipt[payout] = PROFIT_SPLIT * funded_pnl[payout] + FEE
    total_eval = np.where(p2_status == 1, p1_count + p2_count, 0)

    labels = {9: "NOT_ENTERED", 1: "PASS", 0: "CAPPED", -1: "STATIC_LOSS", -2: "SINGLE_TRADE_LOSS"}
    funded_labels = {9: "NOT_ENTERED", 1: "COMPLETED", -1: "STATIC_LOSS"}
    return pd.DataFrame(
        {
            "risk_pct": np.full(N_PATHS, profile * 100.0),
            "cash_risk_usd": np.full(N_PATHS, cash_risk),
            "path_id": np.arange(1, N_PATHS + 1),
            "phase1_status": [labels[x] for x in p1_status],
            "phase1_trades": p1_count,
            "phase1_pnl_usd": p1_pnl,
            "phase2_status": [labels[x] for x in p2_status],
            "phase2_trades": p2_count,
            "phase2_pnl_usd": p2_pnl,
            "total_eval_trades_if_passed": total_eval,
            "funded_status": [funded_labels[x] for x in funded_status],
            "funded_trades": funded_count,
            "funded_pnl_usd": funded_pnl,
            "first_payout": payout,
            "gross_receipt_usd": receipt,
            "net_value_usd": receipt - FEE,
        }
    )


def audit_paths(paths: pd.DataFrame) -> None:
    if len(paths) != N_PATHS * len(RISK_PROFILES):
        raise AssertionError("Wrong number of simulated paths")
    for profile, group in paths.groupby("risk_pct", sort=True):
        if len(group) != N_PATHS or not np.array_equal(group.path_id, np.arange(1, N_PATHS + 1)):
            raise AssertionError(f"Incomplete profile {profile}")
        p1 = group.phase1_status == "PASS"
        p2 = group.phase2_status == "PASS"
        entered2 = group.phase2_status != "NOT_ENTERED"
        entered_funded = group.funded_status != "NOT_ENTERED"
        paid = group.first_payout.astype(bool)
        if not (entered2 == p1).all() or not (entered_funded == p2).all():
            raise AssertionError("Illegal phase transition")
        if not (group.phase1_trades.between(1, MAX_PHASE_TRADES)).all():
            raise AssertionError("Illegal Phase-1 trade count")
        if not (group.loc[entered2, "phase2_trades"].between(1, MAX_PHASE_TRADES)).all():
            raise AssertionError("Illegal Phase-2 trade count")
        if not (group.loc[~entered2, "phase2_trades"] == 0).all():
            raise AssertionError("Phase-2 trades on unentered paths")
        if not (group.loc[p2, "total_eval_trades_if_passed"] ==
                group.loc[p2, "phase1_trades"] + group.loc[p2, "phase2_trades"]).all():
            raise AssertionError("Combined evaluation trade count mismatch")
        if not (group.loc[~p2, "total_eval_trades_if_passed"] == 0).all():
            raise AssertionError("Combined trade count populated for a nonpasser")
        if not (group.loc[group.funded_status == "COMPLETED", "funded_trades"] == 30).all():
            raise AssertionError("Funded completion without 30 trades")
        if not (group.loc[entered_funded, "funded_trades"].between(1, 30)).all():
            raise AssertionError("Illegal funded trade count")
        if not (group.loc[~entered_funded, "funded_trades"] == 0).all():
            raise AssertionError("Funded trades on unentered paths")
        if not (group.loc[paid, "funded_status"] == "COMPLETED").all():
            raise AssertionError("Payout without funded survival")
        if not (group.loc[paid, "funded_pnl_usd"] > 0).all():
            raise AssertionError("Payout without funded profit")
        expected = np.where(paid, group.funded_pnl_usd.fillna(0) * PROFIT_SPLIT + FEE, 0)
        if not np.allclose(group.gross_receipt_usd, expected, rtol=0, atol=1e-8):
            raise AssertionError("Payout receipt mismatch")
        if not np.allclose(group.net_value_usd, group.gross_receipt_usd - FEE, rtol=0, atol=1e-8):
            raise AssertionError("Net EV identity mismatch")
        if not (group.loc[group.phase1_status == "PASS", "phase1_pnl_usd"] >= 1_000).all():
            raise AssertionError("Phase-1 pass below target")
        if not (group.loc[p2, "phase2_pnl_usd"] >= 500).all():
            raise AssertionError("Phase-2 pass below target")


def summarize(paths: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for profile, group in paths.groupby("risk_pct", sort=True):
        p1 = group.phase1_status == "PASS"
        p2 = group.phase2_status == "PASS"
        passing_trades = group.loc[p2, "total_eval_trades_if_passed"].to_numpy()
        median = float(np.quantile(passing_trades, 0.5)) if passing_trades.size else np.nan
        p95 = float(np.quantile(passing_trades, 0.95)) if passing_trades.size else np.nan
        rows.append(
            {
                "risk_pct": profile,
                "cash_risk_usd": float(group.cash_risk_usd.iloc[0]),
                "paths": len(group),
                "phase1_passes": int(p1.sum()),
                "phase1_pass_rate": float(p1.mean()),
                "phase1_static_failures": int((group.phase1_status == "STATIC_LOSS").sum()),
                "phase1_single_trade_failures": int((group.phase1_status == "SINGLE_TRADE_LOSS").sum()),
                "phase1_capped": int((group.phase1_status == "CAPPED").sum()),
                "phase2_passes": int(p2.sum()),
                "phase2_conditional_pass_rate": float(p2.sum() / p1.sum()) if p1.any() else np.nan,
                "both_phases_pass_rate": float(p2.mean()),
                "phase2_static_failures": int((group.phase2_status == "STATIC_LOSS").sum()),
                "phase2_single_trade_failures": int((group.phase2_status == "SINGLE_TRADE_LOSS").sum()),
                "phase2_capped": int((group.phase2_status == "CAPPED").sum()),
                "funded_static_failures": int((group.funded_status == "STATIC_LOSS").sum()),
                "first_payout_paths": int(group.first_payout.sum()),
                "first_payout_probability": float(group.first_payout.mean()),
                "mean_gross_receipt_usd": float(group.gross_receipt_usd.mean()),
                "net_ev_usd_per_challenge": float(group.net_value_usd.mean()),
                "median_eval_trades_given_pass": median,
                "p95_eval_trades_given_pass": p95,
                "median_months_given_pass": median / TRADE_RATE_PER_MONTH,
                "p95_months_given_pass": p95 / TRADE_RATE_PER_MONTH,
            }
        )
    return pd.DataFrame(rows)


def make_report(summary: pd.DataFrame, digest: str, pool: np.ndarray) -> str:
    lines = [
        "FTMO-LIKE CRYPTO + GOLD POOLED-R MONTE CARLO - 2026-09-29",
        "Scenario: $10,000, $175 fee, 80% first-payout split, fixed initial-account-dollar risk",
        f"Input: {len(pool):,} trades; mean net R {pool.mean():+.9f}; SHA256 {digest}",
        f"Simulation: {N_PATHS:,} IID-with-replacement paths per profile; seed {SEED}+1000*profile_index",
        "Phase targets: +$1,000 then +$500; phase static floor -$1,000; evaluation single-trade loss proxy <= -$500.",
        "Funded: 30 trades, static floor -$1,000, first receipt only if final PnL > $0.",
        "Net EV = average gross receipt across all paths minus $175 fee. Payout receipt includes the refunded fee.",
        "",
        "RESULTS MATRIX",
        "Risk | P1 pass | P2 pass conditional | Both pass | First payout | Gross receipt/path | Net EV/challenge | Median trades | P95 trades | Median months | P95 months",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.risk_pct:.1f}% | {row.phase1_pass_rate:.2%} | {row.phase2_conditional_pass_rate:.2%} | "
            f"{row.both_phases_pass_rate:.2%} | {row.first_payout_probability:.2%} | "
            f"${row.mean_gross_receipt_usd:,.2f} | ${row.net_ev_usd_per_challenge:+,.2f} | "
            f"{row.median_eval_trades_given_pass:.1f} | {row.p95_eval_trades_given_pass:.1f} | "
            f"{row.median_months_given_pass:.2f} | {row.p95_months_given_pass:.2f}"
        )
    lines += [
        "",
        "PATH COUNTS AND FAILURE BREAKDOWN",
        "Risk | P1 passes | P1 static/single/capped | P2 passes | P2 static/single/capped | Funded static | First payouts",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.risk_pct:.1f}% | {row.phase1_passes:,} | "
            f"{row.phase1_static_failures:,}/{row.phase1_single_trade_failures:,}/{row.phase1_capped:,} | "
            f"{row.phase2_passes:,} | "
            f"{row.phase2_static_failures:,}/{row.phase2_single_trade_failures:,}/{row.phase2_capped:,} | "
            f"{row.funded_static_failures:,} | {row.first_payout_paths:,}"
        )
    lines += [
        "",
        "Median/P95 trade counts include Phase 1 plus Phase 2 only among paths that passed both phases; months = trades/29.1.",
        "GATE: PASS - frozen input and path-accounting checks passed; capped paths, if any, are explicitly shown above.",
        "LIMITATIONS: The <=-$500 single-trade rule is not FTMO's full daily equity-loss calculation. IID sampling misses clustered losses, correlated BTC/ETH/gold positions, unrealized drawdown, and real phase timing. Historical net R comes from proxy fills and owner cost scenarios; months assume a constant 29.1 trades/month. Positive simulated EV is not a purchase recommendation or payout guarantee.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"Historical report exists and will not be overwritten: {REPORT}")
    pool, digest = verify_input()
    all_paths = pd.concat(
        [simulate_profile(pool, profile, i) for i, profile in enumerate(RISK_PROFILES)],
        ignore_index=True,
    )
    audit_paths(all_paths)
    summary = summarize(all_paths)
    # Recompute summaries after CSV serialization to audit the saved full ledger.
    path_file = OUTPUT / "paths.csv"
    summary_file = OUTPUT / "summary.csv"
    if path_file.exists() or summary_file.exists():
        raise FileExistsError("Historical Monte Carlo artifact already exists")
    all_paths.to_csv(path_file, index=False, float_format="%.12f")
    saved = pd.read_csv(path_file)
    audit_paths(saved)
    saved_summary = summarize(saved)
    numeric = summary.select_dtypes(include="number").columns
    if not np.allclose(summary[numeric], saved_summary[numeric], rtol=0, atol=1e-7, equal_nan=True):
        raise AssertionError("Summary changed after path-ledger serialization")
    saved_summary.to_csv(summary_file, index=False, float_format="%.12f")
    metadata = {
        "input": str(INPUT.relative_to(ROOT)),
        "input_sha256": digest,
        "input_trades": len(pool),
        "input_mean_net_r": float(pool.mean()),
        "paths_per_profile": N_PATHS,
        "risk_fractions": list(RISK_PROFILES),
        "seed_base": SEED,
        "seed_rule": "base + 1000 * zero_based_profile_index",
        "phase_trade_cap": MAX_PHASE_TRADES,
        "cash_risk_basis": "fixed fraction of initial $10,000; no compounding",
        "evaluation_single_trade_proxy_usd": -500.0,
        "funded_single_trade_proxy_applied": False,
        "trade_rate_per_month": TRADE_RATE_PER_MONTH,
        "paths_sha256": hashlib.sha256(path_file.read_bytes()).hexdigest(),
    }
    (OUTPUT / "run_meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report_text = make_report(saved_summary, digest, pool)
    with REPORT.open("x", encoding="utf-8") as handle:
        handle.write(report_text)
    print(report_text)


if __name__ == "__main__":
    main()
