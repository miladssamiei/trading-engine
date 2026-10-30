"""Engine A pooled-R $25K closed-equity Monte Carlo; see frozen SPEC."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from prop_firm_monte_carlo import load_pool, sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/futures_prop_validation"
LEDGER = OUT / "trades.csv"
RISKS = (0.0010, 0.0015, 0.0020, 0.0025)
SEEDS = (20260930, 20261001, 20261002, 20261003)
PATHS = 10_000
MAX_TRADES = 50_000
BASE_EQUITY = 25_000.0
TARGET_EQUITY = 26_500.0
TRAIL_DISTANCE = 1_000.0
DAILY_LIMIT = 500.0
ROLLING_TRADES = 4
CALENDAR_DAYS = 2458


def simulate(r_pool: np.ndarray, risk: float, seed: int) -> tuple[pd.DataFrame, int]:
    rng = np.random.default_rng(seed)
    equity = np.full(PATHS, BASE_EQUITY, float)
    peak = np.full(PATHS, BASE_EQUITY, float)
    recent_pnl = np.zeros((ROLLING_TRADES, PATHS), float)
    status = np.zeros(PATHS, np.int8)  # 0 censored, 1 pass, 2 max ruin, 3 daily ruin
    steps = np.full(PATHS, MAX_TRADES, np.int32)
    final_window = np.zeros(PATHS, float)
    alive = np.arange(PATHS, dtype=np.int32)
    simultaneous_ruin = 0
    for step in range(1, MAX_TRADES+1):
        if not len(alive):
            break
        drawn_r = r_pool[rng.integers(0, len(r_pool), size=len(alive))]
        pnl = drawn_r * (equity[alive] * risk)
        equity[alive] += pnl
        peak[alive] = np.maximum(peak[alive], equity[alive])
        recent_pnl[(step-1) % ROLLING_TRADES, alive] = pnl
        rolling_sum = recent_pnl[:, alive].sum(axis=0)
        final_window[alive] = rolling_sum
        passed = equity[alive] >= TARGET_EQUITY
        max_hit = (peak[alive] - equity[alive]) >= TRAIL_DISTANCE
        daily_hit = rolling_sum <= -DAILY_LIMIT
        simultaneous_ruin += int(np.count_nonzero(max_hit & daily_hit & ~passed))
        max_ruin = max_hit & ~passed
        daily_ruin = daily_hit & ~passed & ~max_ruin
        done = passed | max_ruin | daily_ruin
        status[alive[passed]] = 1
        status[alive[max_ruin]] = 2
        status[alive[daily_ruin]] = 3
        steps[alive[done]] = step
        alive = alive[~done]
    assert np.isfinite(equity).all() and np.isfinite(peak).all()
    assert np.all(peak >= equity)
    assert np.all(equity[status == 1] >= TARGET_EQUITY)
    assert np.all((peak[status == 2]-equity[status == 2]) >= TRAIL_DISTANCE)
    assert np.all(final_window[status == 3] <= -DAILY_LIMIT)
    assert int(np.bincount(status, minlength=4).sum()) == PATHS
    labels = np.array(["CENSORED", "PASS", "MAX_RUIN", "DAILY_RUIN"])
    frame = pd.DataFrame({"risk_fraction": risk, "path": np.arange(PATHS),
                          "outcome": labels[status], "trades": steps,
                          "final_equity": equity, "final_peak": peak,
                          "closed_drawdown": peak-equity,
                          "last_rolling_4_trade_pnl": final_window})
    return frame, simultaneous_ruin


def summarize(paths: pd.DataFrame, risk: float, seed: int,
              frequency: float, simultaneous: int) -> dict:
    counts = paths["outcome"].value_counts()
    n_pass = int(counts.get("PASS", 0))
    n_max = int(counts.get("MAX_RUIN", 0))
    n_daily = int(counts.get("DAILY_RUIN", 0))
    n_censored = int(counts.get("CENSORED", 0))
    assert n_pass+n_max+n_daily+n_censored == PATHS
    successful = paths.loc[paths.outcome == "PASS", "trades"].to_numpy()
    med = float(np.quantile(successful, .5, method="linear")) if n_pass else None
    p95 = float(np.quantile(successful, .95, method="linear")) if n_pass else None
    return dict(risk_fraction=risk, initial_dollar_risk=BASE_EQUITY*risk,
                seed=seed, pass_count=n_pass, max_ruin_count=n_max,
                daily_ruin_count=n_daily, censored_count=n_censored,
                simultaneous_ruin_count=simultaneous,
                pass_probability=n_pass/PATHS,
                max_ruin_probability=n_max/PATHS,
                daily_ruin_probability=n_daily/PATHS,
                total_ruin_probability=(n_max+n_daily)/PATHS,
                censored_probability=n_censored/PATHS,
                median_trades_to_pass=med, p95_trades_to_pass=p95,
                median_calendar_days_to_pass=med/frequency if med is not None else None,
                p95_calendar_days_to_pass=p95/frequency if p95 is not None else None)


def main() -> None:
    if not (OUT/"SPEC_MONTE_CARLO_25K.txt").exists() or not (OUT/"GATE_MONTE_CARLO_25K.txt").exists():
        raise FileNotFoundError("Frozen $25K SPEC/GATE must exist before run")
    report = OUT/"REPORT_MONTE_CARLO_25K.txt"
    if report.exists():
        raise FileExistsError("Frozen $25K report already exists; use a new study name")
    pool, r_pool = load_pool()
    frequency = len(pool)/CALENDAR_DAYS
    if len(pool) != 551 or CALENDAR_DAYS != 2458:
        raise ValueError("Source pool or declared day denominator changed; stop for review")
    worst_four_loss_bound = 4 * abs(min(0.0, float(r_pool.min()))) * TARGET_EQUITY * max(RISKS)
    if worst_four_loss_bound >= DAILY_LIMIT:
        raise ValueError("Predeclared impossible-daily-ruin bound no longer holds")
    all_paths, summaries = [], []
    for risk, seed in zip(RISKS, SEEDS):
        paths, simultaneous = simulate(r_pool, risk, seed)
        summary = summarize(paths, risk, seed, frequency, simultaneous)
        if summary["daily_ruin_count"] != 0 or simultaneous != 0:
            raise AssertionError("Daily ruin contradicts the analytic four-trade bound")
        all_paths.append(paths)
        summaries.append(summary)
        print(f"risk={risk:.2%} PASS={summary['pass_probability']:.2%} "
              f"MAX_RUIN={summary['max_ruin_probability']:.2%} "
              f"DAILY_RUIN={summary['daily_ruin_probability']:.2%} "
              f"CENSORED={summary['censored_probability']:.2%} "
              f"median_pass_trades={summary['median_trades_to_pass']}", flush=True)
    pd.concat(all_paths, ignore_index=True).to_csv(OUT/"monte_carlo_25k_paths.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT/"monte_carlo_25k_summary.csv", index=False)
    meta = dict(study="engine_a_prop_firm_monte_carlo_25k", date="2026-09-26",
                ledger=str(LEDGER.relative_to(ROOT)), ledger_sha256=sha256(LEDGER),
                script_sha256=sha256(Path(__file__)), pool_trades=len(pool),
                asset_counts={str(k):int(v) for k,v in pool.asset.value_counts().items()},
                r_mean=float(r_pool.mean()), r_min=float(r_pool.min()),
                r_max=float(r_pool.max()), calendar_days=CALENDAR_DAYS,
                average_trades_per_day=frequency, paths_per_profile=PATHS,
                max_trades_per_path=MAX_TRADES, base_equity=BASE_EQUITY,
                target_equity=TARGET_EQUITY, trailing_distance=TRAIL_DISTANCE,
                rolling_trade_count=ROLLING_TRADES, daily_proxy_limit=DAILY_LIMIT,
                worst_four_loss_bound=worst_four_loss_bound, results=summaries)
    (OUT/"monte_carlo_25k_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    def f(v):
        return "N/A" if v is None else f"{v:.2f}"
    lines = ["ENGINE A GC/BTC/ETH $25K MONTE CARLO - USER-DEFINED CLOSED-EQUITY SCENARIO",
             f"Input: {meta['ledger']} (sha256 {meta['ledger_sha256']})",
             f"Filtered A_KC pool: {len(pool)} trades; GC {meta['asset_counts'].get('GC',0)}, "
             f"BTC {meta['asset_counts'].get('BTC',0)}, ETH {meta['asset_counts'].get('ETH',0)}.",
             "R = net / abs(entry - initial_stop); all rows have finite positive initial risk.",
             f"Frequency: {len(pool)} / {CALENDAR_DAYS} = {frequency:.8f} trades per calendar day.",
             f"Base ${BASE_EQUITY:,.0f}; target ${TARGET_EQUITY:,.0f} (+6%); closed-peak "
             f"trailing distance ${TRAIL_DISTANCE:,.0f}; rolling last-{ROLLING_TRADES}-trade "
             f"loss proxy ${DAILY_LIMIT:,.0f}.",
             f"{PATHS:,} independent paths per risk; {MAX_TRADES:,}-trade cap; "
             "fractional sizing uses pre-trade closed equity.",
             "Boundary priority at a shared trade: PASS, MAX_RUIN, DAILY_RUIN.",
             f"Mathematical daily-proxy bound: four worst R losses before PASS "
             f"cannot exceed ${worst_four_loss_bound:.2f}, below ${DAILY_LIMIT:.0f}; "
             "DAILY_RUIN is unreachable on this frozen pool and sizing matrix.", ""]
    for s in summaries:
        lines += [f"Risk {s['risk_fraction']:.2%} (starting 1R ${s['initial_dollar_risk']:.2f}; seed {s['seed']}):",
                  f"  PASS: {s['pass_probability']:.2%} ({s['pass_count']}/{PATHS})",
                  f"  MAX_RUIN: {s['max_ruin_probability']:.2%} ({s['max_ruin_count']}/{PATHS})",
                  f"  DAILY_RUIN: {s['daily_ruin_probability']:.2%} ({s['daily_ruin_count']}/{PATHS})",
                  f"  Total ruin: {s['total_ruin_probability']:.2%}; "
                  f"censored: {s['censored_probability']:.2%}; "
                  f"simultaneous ruin triggers: {s['simultaneous_ruin_count']}",
                  f"  Successful paths: median {f(s['median_trades_to_pass'])} trades; "
                  f"95th percentile {f(s['p95_trades_to_pass'])} trades",
                  f"  Frequency-projected pass time: median {f(s['median_calendar_days_to_pass'])} "
                  f"calendar days; 95th percentile {f(s['p95_calendar_days_to_pass'])} days", ""]
    lines += ["LIMITS: This is an indefinite-horizon, IID pooled-R Phase-1 calculation on retrospective CFD-proxy trades, not a prospective prop-firm pass forecast.",
              "The daily rule is a rolling trade-count proxy, not calendar-session PnL. The trailing rule checks closed equity, not live unrealized account equity.",
              "Actual Apex $25K intraday evaluations publish a $1000 real-time trailing threshold and no $500 daily loss limit; the simulated rule combination is user-defined.",
              "There is no 30-day access expiry, contract-size granularity, simultaneous portfolio margin, funding stage or payout in this simulation."]
    report.write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
