"""Engine A pooled-R phase-one closed-equity Monte Carlo (see frozen SPEC)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/futures_prop_validation"
LEDGER = OUT / "trades.csv"
RISKS = (0.0025, 0.0050, 0.0075, 0.0100)
SEED = 20260926
PATHS = 10_000
MAX_TRADES = 50_000
BASE_EQUITY = 100_000.0
TARGET_EQUITY = 108_000.0
TRAIL_DISTANCE = 10_000.0
START_DATE = pd.Timestamp("2020-01-01")
END_DATE = pd.Timestamp("2026-09-23")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pool() -> tuple[pd.DataFrame, np.ndarray]:
    data = pd.read_csv(LEDGER)
    required = {"engine", "asset", "net", "entry", "initial_stop", "entry_time"}
    if not required.issubset(data.columns):
        raise ValueError(f"Ledger lacks columns: {sorted(required-set(data.columns))}")
    pool = data.loc[(data["engine"] == "A_KC") &
                    (data["asset"].isin(["GC", "BTC", "ETH"]))].copy()
    if pool.empty:
        raise ValueError("A_KC GC/BTC/ETH pool is empty")
    pool["initial_risk"] = (pool["entry"]-pool["initial_stop"]).abs()
    if not np.isfinite(pool["initial_risk"]).all() or not (pool["initial_risk"] > 0).all():
        raise ValueError("All qualifying trades need finite positive initial risk")
    if not np.isfinite(pool["net"]).all():
        raise ValueError("All qualifying trades need finite net PnL")
    pool["r_multiple"] = pool["net"] / pool["initial_risk"]
    if not np.isfinite(pool["r_multiple"]).all():
        raise ValueError("Nonfinite R multiple")
    entry_dates = pd.to_datetime(pool["entry_time"], unit="s", utc=True).dt.tz_localize(None)
    if (entry_dates < START_DATE).any() or (entry_dates >= END_DATE + pd.Timedelta(days=1)).any():
        raise ValueError("Selected trade lies outside the predeclared calendar span")
    return pool, pool["r_multiple"].to_numpy(float)


def simulate(r_pool: np.ndarray, risk: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    equity = np.full(PATHS, BASE_EQUITY, float)
    peak = np.full(PATHS, BASE_EQUITY, float)
    status = np.zeros(PATHS, np.int8)  # 0 censored, 1 pass, 2 ruin
    steps = np.full(PATHS, MAX_TRADES, np.int32)
    alive = np.arange(PATHS, dtype=np.int32)
    for step in range(1, MAX_TRADES + 1):
        if not len(alive):
            break
        sampled = r_pool[rng.integers(0, len(r_pool), size=len(alive))]
        equity[alive] += sampled * (equity[alive] * risk)
        peak[alive] = np.maximum(peak[alive], equity[alive])
        passed = equity[alive] >= TARGET_EQUITY
        ruined = ~passed & ((peak[alive] - equity[alive]) >= TRAIL_DISTANCE)
        done = passed | ruined
        status[alive[passed]] = 1
        status[alive[ruined]] = 2
        steps[alive[done]] = step
        alive = alive[~done]
    assert np.isfinite(equity).all() and np.isfinite(peak).all()
    assert np.all(peak >= equity)
    assert np.all(equity[status == 1] >= TARGET_EQUITY)
    assert np.all((peak[status == 2]-equity[status == 2]) >= TRAIL_DISTANCE)
    assert int(np.bincount(status, minlength=3).sum()) == PATHS
    return pd.DataFrame({"risk_fraction": risk, "path": np.arange(PATHS),
                         "outcome": np.array(["CENSORED", "PASS", "RUIN"])[status],
                         "trades": steps, "final_equity": equity,
                         "final_peak": peak, "closed_drawdown": peak-equity})


def row_stats(paths: pd.DataFrame, risk: float, seed: int,
              trades_per_day: float) -> dict:
    counts = paths["outcome"].value_counts()
    n_pass = int(counts.get("PASS", 0))
    n_ruin = int(counts.get("RUIN", 0))
    n_censored = int(counts.get("CENSORED", 0))
    assert n_pass+n_ruin+n_censored == PATHS
    wins = paths.loc[paths["outcome"] == "PASS", "trades"].to_numpy()
    median = float(np.quantile(wins, .5, method="linear")) if n_pass else None
    p95 = float(np.quantile(wins, .95, method="linear")) if n_pass else None
    return dict(risk_fraction=risk, seed=seed, pass_count=n_pass,
                ruin_count=n_ruin, censored_count=n_censored,
                pass_probability=n_pass/PATHS, ruin_probability=n_ruin/PATHS,
                censored_probability=n_censored/PATHS,
                median_trades_to_pass=median, p95_trades_to_pass=p95,
                median_calendar_days_to_pass=median/trades_per_day if median is not None else None,
                p95_calendar_days_to_pass=p95/trades_per_day if p95 is not None else None)


def main() -> None:
    if not (OUT / "SPEC_MONTE_CARLO.txt").exists() or not (OUT / "GATE_MONTE_CARLO.txt").exists():
        raise FileNotFoundError("Frozen Monte Carlo SPEC/GATE must exist before the run")
    report_path = OUT / "REPORT_MONTE_CARLO.txt"
    if report_path.exists():
        raise FileExistsError("Frozen Monte Carlo report already exists; use a new study name")
    pool, r_pool = load_pool()
    calendar_days = int((END_DATE - START_DATE).days + 1)
    frequency = len(pool) / calendar_days
    all_paths, summaries = [], []
    for index, risk in enumerate(RISKS):
        seed = SEED + index
        paths = simulate(r_pool, risk, seed)
        summary = row_stats(paths, risk, seed, frequency)
        all_paths.append(paths)
        summaries.append(summary)
        print(f"risk={risk:.2%} PASS={summary['pass_probability']:.4%} "
              f"RUIN={summary['ruin_probability']:.4%} "
              f"CENSORED={summary['censored_probability']:.4%} "
              f"median_pass_trades={summary['median_trades_to_pass']}", flush=True)
    pd.concat(all_paths, ignore_index=True).to_csv(OUT / "monte_carlo_paths.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "monte_carlo_summary.csv", index=False)
    meta = dict(study="engine_a_prop_firm_monte_carlo", date="2026-09-26",
                ledger=str(LEDGER.relative_to(ROOT)), ledger_sha256=sha256(LEDGER),
                script_sha256=sha256(Path(__file__)), n_pool=len(pool),
                asset_counts={k:int(v) for k,v in pool["asset"].value_counts().items()},
                r_mean=float(r_pool.mean()), r_median=float(np.median(r_pool)),
                r_min=float(r_pool.min()), r_max=float(r_pool.max()),
                calendar_start=str(START_DATE.date()), calendar_end=str(END_DATE.date()),
                calendar_days=calendar_days, average_trades_per_day=frequency,
                paths_per_profile=PATHS, max_trades_per_path=MAX_TRADES,
                base_equity=BASE_EQUITY, target_equity=TARGET_EQUITY,
                trailing_distance=TRAIL_DISTANCE, results=summaries)
    (OUT / "monte_carlo_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lines = ["ENGINE A GC/BTC/ETH PORTFOLIO MONTE CARLO - PHASE 1 CLOSED-EQUITY DIAGNOSTIC",
             "Ledger: " + meta["ledger"],
             f"Pool: {len(pool)} A_KC trades; GC {meta['asset_counts'].get('GC',0)}, "
             f"BTC {meta['asset_counts'].get('BTC',0)}, ETH {meta['asset_counts'].get('ETH',0)}.",
             "R = net / abs(entry - initial_stop), without clipping or asset reweighting.",
             f"R mean {meta['r_mean']:.6f}, median {meta['r_median']:.6f}, "
             f"range [{meta['r_min']:.6f}, {meta['r_max']:.6f}].",
             f"Historical frequency: {len(pool)} / {calendar_days} inclusive calendar days "
             f"({meta['calendar_start']}..{meta['calendar_end']}) = {frequency:.8f} trades/day.",
             f"Account ${BASE_EQUITY:,.0f}; Phase-1 target ${TARGET_EQUITY:,.0f}; "
             f"closed-peak trailing drawdown ${TRAIL_DISTANCE:,.0f}; "
             f"{PATHS:,} independent paths/risk; cap {MAX_TRADES:,} trades/path.", ""]
    for s in summaries:
        def fmt(v, digits=2):
            return "N/A" if v is None else f"{v:.{digits}f}"
        lines += [f"Risk {s['risk_fraction']:.2%} of pre-trade equity (seed {s['seed']}):",
                  f"  Phase-1 pass probability: {s['pass_probability']:.2%} ({s['pass_count']}/{PATHS})",
                  f"  Closed-trailing-drawdown ruin: {s['ruin_probability']:.2%} ({s['ruin_count']}/{PATHS})",
                  f"  Censored at cap: {s['censored_probability']:.2%} ({s['censored_count']}/{PATHS})",
                  f"  Successful paths: median trades {fmt(s['median_trades_to_pass'])}; "
                  f"95th percentile trades {fmt(s['p95_trades_to_pass'])}",
                  f"  Frequency-projected time: median {fmt(s['median_calendar_days_to_pass'])} "
                  f"calendar days; 95th percentile {fmt(s['p95_calendar_days_to_pass'])} calendar days", ""]
    lines += ["INTERPRETATION LIMITS:",
              "This models one Phase-1 target, not funding, a second phase, payout, or live firm eligibility.",
              "Only closed-equity trailing drawdown is checked here. The input ledger already has a separate M1 per-position guard; no account-wide unrealized-equity path is reconstructed.",
              "IID pooled-R replacement removes chronology, concurrency, regimes and asset-specific opportunity rates. Calendar days are a simple historical frequency conversion.",
              "The R pool comes from retrospective bid-only CFD proxy tapes with owner-supplied futures cost floors. These are conditional sample simulations, not prospective pass probabilities or a risk-profile recommendation."]
    report_path.write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
