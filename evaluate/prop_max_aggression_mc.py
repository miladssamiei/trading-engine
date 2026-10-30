"""Frozen historical random-entry experiment; see SPEC_MAX_AGGRESSION.txt.

Run with the project Python. Historical SPEC/GATE/REPORT files are immutable.
Each synthetic evaluation samples independent entries, then walks actual BID
M1 bars for each trade. This is not a chronological portfolio backtest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/results/vwap_2021_2026/us500tick_m1.csv"
OUT = ROOT / "data/results/prop_max_aggression"
REPORT = OUT / "REPORT_MAX_AGGRESSION.txt"
N_PATHS = 10_000
MAX_ATTEMPTS = 200
SEED = 20261002
FUNDED_SEED = 20261003
CASH_PER_POINT = 50.0
COST = 0.51
FEE = 80.0
PAYOUT = 1600.0
EPS = 1e-7
STATUS = {1: "PASS", -1: "FAIL", 0: "CENSORED"}
EXIT = {0: "TAPE_END_OPEN", 1: "STOP", 2: "TARGET",
        3: "ACCOUNT_STOP", 4: "ACCOUNT_TARGET"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@njit(cache=True)
def walk_trade(opens, highs, lows, entry_index, side, spread, stop, target):
    """Stops first even on collisions; ask exits for shorts, BID for longs."""
    for i in range(entry_index, len(opens)):
        adjustment = spread if side == -1 else 0.0
        quote_open = opens[i] + adjustment
        quote_high = highs[i] + adjustment
        quote_low = lows[i] + adjustment
        stop_hit = quote_low <= stop if side == 1 else quote_high >= stop
        target_hit = quote_high >= target if side == 1 else quote_low <= target
        if stop_hit:
            gap = quote_open < stop if side == 1 else quote_open > stop
            execution = quote_open if gap else stop
            return i, execution, execution - adjustment, 1, target_hit, gap
        if target_hit:
            return i, target, target - adjustment, 2, False, False
    return -1, np.nan, np.nan, 0, False, False


@njit(cache=True)
def simulate(opens, highs, lows, entries, sides, spread, intrabar, account_target):
    count = entries.shape[0]
    statuses = np.zeros(count, dtype=np.int8)
    final_equities = np.zeros(count)
    attempts = np.zeros(count, dtype=np.int32)
    records = []
    for p in range(count):
        equity = 0.0
        for step in range(entries.shape[1]):
            ei = entries[p, step]
            side = sides[p, step]
            entry_bid = opens[ei]
            entry_exec = entry_bid + (spread if side == 1 else 0.0)
            stop_distance = 40.0
            target_distance = 30.0
            if intrabar:
                stop_distance = min(stop_distance, (equity + 2000.0) / CASH_PER_POINT)
                target_distance = min(target_distance, (account_target - equity) / CASH_PER_POINT)
            stop = entry_exec - side * stop_distance
            target = entry_exec + side * target_distance
            xi, exit_exec, exit_bid, kind, collision, gap = walk_trade(
                opens, highs, lows, ei, side, spread, stop, target)
            before = equity
            net = np.nan
            gross = np.nan
            if xi >= 0:
                net = side * (exit_exec - entry_exec) * CASH_PER_POINT
                gross = side * (exit_bid - entry_bid) * CASH_PER_POINT
                equity += net
                if kind == 1 and stop_distance < 40.0 - 1e-9:
                    kind = 3
                elif kind == 2 and target_distance < 30.0 - 1e-9:
                    kind = 4
            records.append((p, step + 1, ei, xi, side, entry_exec, exit_exec,
                            exit_bid, stop, target, net, gross, before, equity,
                            kind, collision, gap))
            attempts[p] += 1
            if xi < 0:
                break
            if equity <= -2000.0 + EPS:
                statuses[p] = -1
                break
            if equity >= account_target - EPS:
                statuses[p] = 1
                break
        final_equities[p] = equity
    return statuses, final_equities, attempts, records


def markov_pass(win_probability: float) -> float:
    states = list(range(-3, 6))  # USD 500 units, absorbing <= -4 or >= 6
    matrix = np.eye(len(states))
    rhs = np.zeros(len(states))
    for i, state in enumerate(states):
        for step, probability in ((3, win_probability), (-4, 1 - win_probability)):
            nxt = state + step
            if nxt >= 6:
                rhs[i] += probability
            elif nxt > -4:
                matrix[i, states.index(nxt)] -= probability
    return float(np.linalg.solve(matrix, rhs)[states.index(0)])


def wilson(successes: int, count: int) -> tuple[float, float]:
    if count == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    proportion = successes / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    half = z * np.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count**2)) / denominator
    return float(center - half), float(center + half)


def load_tape():
    frame = pd.read_csv(SOURCE, usecols=["time", "open", "high", "low", "close"])
    epoch = frame.time.to_numpy(dtype=np.int64)
    prices = frame[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
    if len(frame) < 2 or not np.all(np.diff(epoch) > 0) or np.any(epoch % 60):
        raise ValueError("Source is not strictly increasing real minute bars")
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Nonfinite/nonpositive OHLC")
    if not (np.all(prices[:, 1] >= prices.max(axis=1)) and
            np.all(prices[:, 2] <= prices.min(axis=1))):
        raise ValueError("Invalid OHLC geometry")
    utc = pd.to_datetime(epoch, unit="s", utc=True)
    ny = utc.tz_convert("America/New_York")
    wall = ny.hour.to_numpy() * 60 + ny.minute.to_numpy()
    eligible = np.flatnonzero((wall >= 570) & (wall < 930))
    if len(eligible) == 0:
        raise ValueError("No RTH entry minutes")
    days = np.asarray(ny.strftime("%Y%m%d"), dtype=np.int32)
    return epoch, prices[:, 0].copy(), prices[:, 1].copy(), prices[:, 2].copy(), eligible, days


def schedules(eligible, seed):
    rng = np.random.default_rng(seed)
    entries = eligible[rng.integers(0, len(eligible), size=(N_PATHS, MAX_ATTEMPTS))]
    sides = (rng.integers(0, 2, size=(N_PATHS, MAX_ATTEMPTS), dtype=np.int8) * 2 - 1)
    return entries, sides


def verify_execution_examples():
    # Independently prescribed edge cases: collision, adverse gap, ask-side short.
    for side in (1, -1):
        entry = 100.0 + (COST if side == 1 else 0.0)
        stop = entry - side * 40.0
        target = entry + side * 30.0
        collision = walk_trade(np.array([100.0]), np.array([150.0]), np.array([50.0]),
                               0, side, COST, stop, target)
        assert collision[3] == 1 and collision[4] and abs(collision[1] - stop) < 1e-8
        opening = 45.0 if side == 1 else 155.0
        gap = walk_trade(np.array([opening]), np.array([opening + 1.0]),
                         np.array([opening - 1.0]), 0, side, COST, stop, target)
        assert gap[3] == 1 and gap[5]
        assert abs(gap[2] - opening) < 1e-8
        favorable = 150.0 if side == 1 else 45.0
        result = walk_trade(np.array([favorable]), np.array([favorable + 1]),
                            np.array([favorable - 1]), 0, side, COST, stop, target)
        assert result[3] == 2 and abs(side * (result[1] - entry) * 50 - 1500) < EPS


def run_arm(name, epoch, opens, highs, lows, eligible, days, entries, sides,
            spread, intrabar, target, seed):
    print(f"RUN {name}: {N_PATHS:,} paths", flush=True)
    statuses, equities, attempts, records = simulate(
        opens, highs, lows, entries, sides, spread, intrabar, target)
    columns = ["path_id", "trade_step", "entry_index", "exit_index", "side",
               "entry_exec", "exit_exec", "exit_bid", "stop_quote", "target_quote",
               "net_cash", "gross_cash", "equity_before", "equity_after",
               "exit_code", "collision", "adverse_gap"]
    ledger = pd.DataFrame(records, columns=columns)
    del records
    closed = ledger.exit_index >= 0
    done = ledger.loc[closed]
    assert len(statuses) == N_PATHS and np.isin(statuses, [-1, 0, 1]).all()
    assert np.isin(ledger.entry_index, eligible).all()
    assert np.all(done.exit_index >= done.entry_index)
    assert np.allclose(done.gross_cash - done.net_cash, spread * CASH_PER_POINT,
                       rtol=0, atol=EPS)
    assert (done.net_cash <= 1500.0 + EPS).all()
    assert np.allclose(done.equity_after, done.equity_before + done.net_cash,
                       rtol=0, atol=EPS)
    assert ledger.groupby("path_id").size().to_numpy().tolist() == attempts.tolist()
    assert np.allclose(ledger.groupby("path_id").equity_after.last().to_numpy(), equities,
                       rtol=0, atol=EPS)
    assert (equities[statuses == 1] >= target - EPS).all()
    assert (equities[statuses == -1] <= -2000.0 + EPS).all()
    # No additional trade after absorption, even when sampled dates run backward.
    not_last = ledger.trade_step.to_numpy() < attempts[ledger.path_id.to_numpy()]
    interim = ledger.loc[not_last]
    assert ((interim.equity_after > -2000.0 + EPS) &
            (interim.equity_after < target - EPS)).all()
    assert ledger.loc[ledger.collision, "exit_code"].isin([1, 3]).all()
    # Reconcile executable fills and final-bar bracket touches independently.
    ei = done.entry_index.to_numpy()
    xi = done.exit_index.to_numpy()
    direction = done.side.to_numpy()
    adjustment = np.where(direction == -1, spread, 0.0)
    assert np.allclose(done.entry_exec, opens[ei] + np.where(direction == 1, spread, 0), atol=EPS, rtol=0)
    assert np.allclose(done.exit_exec, done.exit_bid + adjustment, atol=EPS, rtol=0)
    is_stop = done.exit_code.isin([1, 3]).to_numpy()
    stops = done.stop_quote.to_numpy()
    targets = done.target_quote.to_numpy()
    stop_touch = np.where(direction == 1, lows[xi] <= stops, highs[xi] + spread >= stops)
    target_touch = np.where(direction == 1, highs[xi] >= targets, lows[xi] + spread <= targets)
    assert np.all(stop_touch[is_stop]) and np.all(target_touch[~is_stop])
    assert not np.any(stop_touch[~is_stop])
    assert np.allclose(done.exit_exec.to_numpy()[~is_stop], targets[~is_stop], rtol=0, atol=EPS)
    ledger["entry_bid"] = opens[ledger.entry_index.to_numpy()]
    ledger["entry_time_utc"] = pd.to_datetime(epoch[ledger.entry_index], unit="s", utc=True)
    safe_exit = np.maximum(ledger.exit_index.to_numpy(), 0)
    ledger["exit_time_utc"] = pd.to_datetime(epoch[safe_exit], unit="s", utc=True)
    ledger.loc[~closed, "exit_time_utc"] = pd.NaT
    ledger["overnight_ny"] = closed & (days[ledger.entry_index] != days[safe_exit])
    ledger["elapsed_minutes"] = np.where(closed, (epoch[safe_exit] - epoch[ledger.entry_index]) / 60, np.nan)
    ledger["exit_type"] = ledger.exit_code.map(EXIT)
    ledger["cost_price_units"] = spread
    ledger["cash_per_price_unit"] = CASH_PER_POINT
    ledger["seed"] = seed
    ledger.to_csv(OUT / f"trades_{name.lower()}.csv", index=False)
    paths = pd.DataFrame({"path_id": np.arange(N_PATHS), "status": [STATUS[int(x)] for x in statuses],
                          "final_equity_cash": equities, "trade_attempts": attempts, "seed": seed})
    paths["censor_reason"] = np.where(statuses != 0, "", np.where(
        ledger.groupby("path_id").exit_code.last().to_numpy() == 0, "TAPE_END", "TRADE_CAP"))
    paths.to_csv(OUT / f"paths_{name.lower()}.csv", index=False)
    passed = int((statuses == 1).sum())
    failed = int((statuses == -1).sum())
    censored = N_PATHS - passed - failed
    low, high = passed / N_PATHS, (passed + censored) / N_PATHS
    wins = int((done.net_cash > 0).sum())
    summary = {
        "arm": name, "intrabar_account_guard": intrabar, "spread": spread,
        "account_target_cash": target, "paths": N_PATHS, "passes": passed,
        "failures": failed, "censored": censored, "pass_lower": low, "pass_upper": high,
        "resolved_only_pass_rate": passed / (passed + failed) if passed + failed else None,
        "wilson_95": wilson(passed, N_PATHS), "closed_trades": len(done),
        "trade_wins": wins, "trade_win_rate": wins / len(done) if len(done) else None,
        "mean_trade_cash": float(done.net_cash.mean()), "total_trade_cash": float(done.net_cash.sum()),
        "stop_exits": int(done.exit_code.isin([1, 3]).sum()),
        "target_exits": int(done.exit_code.isin([2, 4]).sum()),
        "collisions": int(ledger.collision.sum()), "adverse_gaps": int(ledger.adverse_gap.sum()),
        "overnight_trades": int(ledger.overnight_ny.sum()),
        "max_profit_cash": float(done.net_cash.max()), "worst_loss_cash": float(done.net_cash.min()),
        "median_trades_to_pass": float(np.median(attempts[statuses == 1])) if passed else None,
        "conditional_payout_ev_lower": PAYOUT * low - FEE,
        "conditional_payout_ev_upper": PAYOUT * high - FEE,
    }
    print(f"DONE {name}: pass={passed}, fail={failed}, censored={censored}, "
          f"trade win={summary['trade_win_rate']:.4%}", flush=True)
    return summary, statuses, paths


def main():
    if REPORT.exists():
        raise FileExistsError("Historical REPORT already exists; freeze a new study for any rerun")
    spec = (OUT / "SPEC_MAX_AGGRESSION.txt").read_text(encoding="utf-8")
    gates = (OUT / "GATE_MAX_AGGRESSION.txt").read_text(encoding="utf-8").strip()
    extracted = "\n".join(line for line in spec.splitlines() if line.startswith(tuple(f"G{i}:" for i in range(1, 6))))
    if gates != extracted:
        raise ValueError("GATE is not the verbatim extraction of all frozen gates")
    verify_execution_examples()
    epoch, opens, highs, lows, eligible, days = load_tape()
    digest = sha256(SOURCE)
    print(f"SOURCE {len(epoch):,} bars; {len(eligible):,} eligible real RTH minutes; SHA256 {digest}", flush=True)
    entry_schedule, side_schedule = schedules(eligible, SEED)
    summaries, outcomes, path_frames = {}, {}, {}
    arms = (("REAL_COST_CLOSED", COST, False), ("ZERO_COST_CLOSED", 0.0, False),
            ("REAL_COST_INTRABAR", COST, True))
    for name, spread, intrabar in arms:
        summary, statuses, paths = run_arm(name, epoch, opens, highs, lows, eligible,
                                           days, entry_schedule, side_schedule,
                                           spread, intrabar, 3000.0, SEED)
        summaries[name], outcomes[name], path_frames[name] = summary, statuses, paths
    funded_entries, funded_sides = schedules(eligible, FUNDED_SEED)
    joint_summaries = {}
    for eval_name, _, intrabar in (arms[0], arms[2]):
        name = "FUNDED_" + eval_name
        summary, statuses, paths = run_arm(name, epoch, opens, highs, lows, eligible,
                                           days, funded_entries, funded_sides,
                                           COST, intrabar, 2000.0, FUNDED_SEED)
        summaries[name] = summary
        eval_status = outcomes[eval_name]
        joint_pass = (eval_status == 1) & (statuses == 1)
        possible_payout = ((eval_status == 1) & (statuses != -1)) | (eval_status == 0)
        joint_low, joint_high = joint_pass.mean(), possible_payout.mean()
        paths["evaluation_status"] = [STATUS[int(x)] for x in eval_status]
        paths["joint_payout"] = joint_pass
        paths["payout_cash"] = joint_pass.astype(float) * PAYOUT
        paths["net_challenge_ev_cash"] = paths.payout_cash - FEE
        paths.to_csv(OUT / f"paths_{name.lower()}.csv", index=False)
        ev = float(paths.net_challenge_ev_cash.mean())
        assert abs(ev - (PAYOUT * joint_low - FEE)) < EPS
        joint_summaries[eval_name] = {
            "evaluation_passes": int((eval_status == 1).sum()), "joint_payouts": int(joint_pass.sum()),
            "joint_payout_lower": float(joint_low), "joint_payout_upper": float(joint_high),
            "funded_pass_given_evaluation_pass": float(joint_pass.sum() / (eval_status == 1).sum()),
            "ev_lower": ev, "ev_upper": float(PAYOUT * joint_high - FEE),
        }
    pd.DataFrame(summaries.values()).to_csv(OUT / "summary.csv", index=False)
    theory = {"quoted_two_win_pass": (4 / 7) ** 2,
              "zero_cost_trade_win": 4 / 7,
              "matched_zero_cost_account_pass": markov_pass(4 / 7),
              "spread_only_trade_win": (40 - COST) / 70,
              "matched_spread_only_account_pass": markov_pass((40 - COST) / 70)}
    metadata = {"source": str(SOURCE), "source_sha256": digest,
                "source_bars": len(epoch), "eligible_minutes": len(eligible),
                "source_first_utc": str(pd.to_datetime(epoch[0], unit="s", utc=True)),
                "source_last_utc": str(pd.to_datetime(epoch[-1], unit="s", utc=True)),
                "seed": SEED, "funded_seed": FUNDED_SEED, "max_trade_attempts": MAX_ATTEMPTS,
                "script_sha256": sha256(Path(__file__)),
                "spec_sha256": sha256(OUT / "SPEC_MAX_AGGRESSION.txt"),
                "gate_sha256": sha256(OUT / "GATE_MAX_AGGRESSION.txt"),
                "theory": theory, "summaries": summaries, "payout_scenarios": joint_summaries,
                "gates": {f"G{i}": "PASS" for i in range(1, 6)},
                "validation_scope": "Frozen hypothetical-scenario execution integrity, not trading edge or firm payout validation"}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    cost = summaries["REAL_COST_CLOSED"]
    zero = summaries["ZERO_COST_CLOSED"]
    paired_delta = (outcomes["REAL_COST_CLOSED"] == 1).astype(float) - (outcomes["ZERO_COST_CLOSED"] == 1).astype(float)
    paired_se = float(paired_delta.std(ddof=1) / np.sqrt(N_PATHS))
    lines = [
        "ZERO-EDGE MAX AGGRESSION - US500 HISTORICAL MONTE CARLO - 2026-10-02",
        "FROZEN SCENARIO RESULT; NOT A VALIDATED TRADING EDGE OR FIRM PAYOUT FORECAST", "",
        f"Source: {SOURCE.relative_to(ROOT)}",
        f"Tape: {metadata['source_first_utc']} through {metadata['source_last_utc']}",
        f"Real M1 bars: {len(epoch):,}; eligible [09:30,15:30) NY minutes: {len(eligible):,}",
        f"Source SHA256: {digest}",
        f"Evaluation seed: {SEED}; funded seed: {FUNDED_SEED}; paths per arm: {N_PATHS:,}; attempts cap: {MAX_ATTEMPTS}",
        "Account USD50,000; pass +3000; static fail -2000; 50 lots = USD50/point.",
        "Executable-entry brackets: SL40/TP30 points; cost0.51 = USD25.50 embedded once.",
        "Nominal target gain1500; nominal stop loss2000; adverse gaps can lose more.", "",
        "THEORETICAL COMPARISON",
        f"Quoted two-consecutive-win pass: {theory['quoted_two_win_pass']:.6%} (quoted 32.65%).",
        f"Matched repeat-to-static-boundary zero-cost pass: {theory['matched_zero_cost_account_pass']:.6%}.",
        "Reason: after a win then a loss, equity is -500 and remains alive; two wins is not the only passing path.",
        f"Theoretical zero-cost trade win: {theory['zero_cost_trade_win']:.6%} (quoted 57.14%).",
        f"Idealized spread-only trade win: {theory['spread_only_trade_win']:.6%}; matched account pass: {theory['matched_spread_only_account_pass']:.6%}.",
        "Theoretical probabilities assume driftless continuous price, no gaps and no M1 collisions.", "",
        "EVALUATION MATRIX (all probabilities use 10,000 purchased evaluations)",
        "Arm                     Pass  Fail Censored   Pass lower/upper       Trade win   Closed trades   Mean/trade",
    ]
    for name, _, _ in arms:
        s = summaries[name]
        lines.append(f"{name:23s} {s['passes']:5d} {s['failures']:5d} {s['censored']:8d} "
                     f"{s['pass_lower']:8.2%}/{s['pass_upper']:.2%} "
                     f"{s['trade_win_rate']:12.4%} {s['closed_trades']:14,d} ${s['mean_trade_cash']:11.2f}")
        lines.append(f"  95% Wilson conditional MC interval: {s['wilson_95'][0]:.2%}..{s['wilson_95'][1]:.2%}; "
                     f"resolved-only pass {s['resolved_only_pass_rate']:.4%}; median trades/pass {s['median_trades_to_pass']:g}.")
        lines.append(f"  Stops {s['stop_exits']:,}; targets {s['target_exits']:,}; collisions {s['collisions']:,}; "
                     f"adverse gaps {s['adverse_gaps']:,}; NY-date-crossing trades {s['overnight_trades']:,}; "
                     f"max gain ${s['max_profit_cash']:.2f}; worst loss ${s['worst_loss_cash']:.2f}.")
    lines += ["", "PRIMARY COMPARISONS",
              f"Real-cost pass versus quoted 32.65%: {(cost['pass_lower'] - 0.3265) * 100:+.4f} percentage points.",
              f"Real-cost pass versus matched 40.83151672%: {(cost['pass_lower'] - theory['matched_zero_cost_account_pass']) * 100:+.4f} percentage points.",
              f"Real-cost trade win versus theoretical57.142857%: {(cost['trade_win_rate'] - 4/7) * 100:+.4f} percentage points.",
              f"Historical spread impact (paired schedules): {(cost['pass_lower'] - zero['pass_lower']) * 100:+.4f} percentage points in pass rate.",
              f"Paired pass-difference conditional MC standard error: {paired_se * 100:.4f} percentage points.", "",
              "ECONOMICS: USD80 fee, not refunded; payout=80% of USD2000=USD1600",
              "Arm                     EV if every pass pays       Joint first-payout probability     EV including funded target"]
    for name in ("REAL_COST_CLOSED", "REAL_COST_INTRABAR"):
        s, j = summaries[name], joint_summaries[name]
        lines.append(f"{name:23s} ${s['conditional_payout_ev_lower']:8.2f}..${s['conditional_payout_ev_upper']:.2f} "
                     f"{j['joint_payout_lower']:17.2%}..{j['joint_payout_upper']:.2%} "
                     f"${j['ev_lower']:17.2f}..${j['ev_upper']:.2f}")
        lines.append(f"  Independent funded stage: +2000/-2000, same brackets/risk model; "
                     f"funded pass among evaluation passes {j['funded_pass_given_evaluation_pass']:.2%}; "
                     f"joint payouts {j['joint_payouts']:,}/{N_PATHS:,}.")
    lines += ["", "GATE VERDICT: PASS FOR SCENARIO INTEGRITY ONLY",
              *gates.splitlines(), "",
              "LIMITS",
              "Primary checks account limits after CLOSED trades, so floating equity may breach -2000 before recovery.",
              "Intrabar arm applies remaining account stop/target distance; M1 stop-first ordering is conservative but cannot recover tick order.",
              "Historical M1 BID proxy is real recorded bars, not variable bid/ask tick validation. Spread is fixed by the request.",
              "Minute-weighted IID resampling repeats historical opportunities and destroys chronology/dependence across trades.",
              "Monte Carlo intervals measure seeded sampling precision conditional on this tape, not uncertainty about future performance.",
              "No swaps, additional commission, slippage, margin checks, trailing firm drawdown or extra payout/consistency restrictions.",
              "Some trades hold overnight. The 1500 trade cap satisfies only the supplied cap, not every possible firm's consistency calculation.",
              "The conditional EV assumes every evaluation pass pays; the funded EV models an additional target, not actual withdrawal eligibility.",
              "Censored outcomes are unresolved, not losses; lower/upper probability and EV bounds are shown even when equal.",
              "No performance-validation gate or purchase recommendation is inferred from this simulation.", "",
              "Artifacts: summary.csv, metadata.json, paths_*.csv, trades_*.csv, frozen SPEC/GATE and this REPORT."]
    with REPORT.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
