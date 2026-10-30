import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Milad\Desktop\RR")
STUDY = ROOT / "data" / "results" / "ftmo_sim_vrp_orb"
TRADES_CSV = ROOT / "data" / "results" / "orb_vrp_filter" / "orb_vrp_filtered_trades.csv"

START = 10000.0
TARGET = 10800.0
FLOOR = 9000.0
DAILY_BREACH_USD = -500.0
RISK_GRID = [0.0025, 0.0050, 0.0075, 0.0100, 0.0150, 0.0200]
N_MC = 10000
MAX_TRADES = 100000
SEED = 20260923

PASS, MAXLOSS, DAILY, UNRESOLVED = 1, 2, 3, 4
OUTCOME_NAMES = {PASS: "PASS", MAXLOSS: "MAX_LOSS", DAILY: "DAILY_BREACH", UNRESOLVED: "UNRESOLVED"}


def load_population():
    df = pd.read_csv(TRADES_CSV)
    assert "net" in df.columns, "net column missing"
    losers = df.loc[df["net"] < 0, "net"]
    assert len(losers) >= 1, "no losing trades"
    mean_loss = float(losers.abs().mean())
    assert mean_loss > 0
    r = (df["net"] / mean_loss).to_numpy(dtype=np.float64)
    stats = {
        "n_trades": int(len(df)),
        "n_losers": int(len(losers)),
        "win_rate": float((df["net"] > 0).mean()),
        "avg_net_pt": float(df["net"].mean()),
        "mean_loss_pt": mean_loss,
        "mean_R": float(r.mean()),
        "std_R": float(r.std(ddof=1)),
        "min_R": float(r.min()),
        "max_R": float(r.max()),
        "worst_net_pt": float(df["net"].min()),
        "best_net_pt": float(df["net"].max()),
        "breach_R_by_risk": {f"{p:.2%}": float(DAILY_BREACH_USD / (p * START)) for p in RISK_GRID},
    }
    return r, stats


def run_risk(r_mult, risk_pct, child):
    rng = np.random.default_rng([SEED, child])
    n = len(r_mult)
    risk_usd = risk_pct * START
    eq = np.full(N_MC, START)
    counts = np.zeros(N_MC, dtype=np.int64)
    outcome = np.zeros(N_MC, dtype=np.int8)
    active = np.arange(N_MC)
    while active.size:
        draws = r_mult[rng.integers(0, n, size=active.size)]
        pnl = draws * risk_usd
        new_eq = eq[active] + pnl
        counts[active] += 1
        o = np.zeros(active.size, dtype=np.int8)
        o[new_eq >= TARGET] = PASS
        rem = o == 0
        o[rem & (new_eq <= FLOOR)] = MAXLOSS
        rem = o == 0
        o[rem & (pnl <= DAILY_BREACH_USD)] = DAILY
        rem = o == 0
        o[rem & (counts[active] >= MAX_TRADES)] = UNRESOLVED
        eq[active] = new_eq
        outcome[active] = o
        active = active[o == 0]
    return outcome, counts


def main():
    STUDY.mkdir(parents=True, exist_ok=True)
    r_mult, pop = load_population()
    gf1 = pop["n_trades"] == 171 and pop["n_losers"] >= 1 and pop["mean_loss_pt"] > 0

    rows = []
    gf2 = True
    for i, risk in enumerate(RISK_GRID):
        outcome, counts = run_risk(r_mult, risk, i)
        n_pass = int((outcome == PASS).sum())
        n_max = int((outcome == MAXLOSS).sum())
        n_day = int((outcome == DAILY).sum())
        n_unres = int((outcome == UNRESOLVED).sum())
        if n_unres != 0 or n_pass + n_max + n_day + n_unres != N_MC:
            gf2 = False
        med_pass = float(np.median(counts[outcome == PASS])) if n_pass else None
        med_max = float(np.median(counts[outcome == MAXLOSS])) if n_max else None
        rows.append({
            "risk_pct": risk,
            "n_pass": n_pass, "n_max_loss": n_max, "n_daily": n_day, "n_unresolved": n_unres,
            "pass_rate": n_pass / N_MC, "max_loss_rate": n_max / N_MC, "daily_rate": n_day / N_MC,
            "median_trades_to_pass": med_pass, "median_trades_to_max_loss": med_max,
            "target_R": 800.0 / (risk * START), "floor_R": 1000.0 / (risk * START),
        })

    sums_ok = all(abs(row["pass_rate"] + row["max_loss_rate"] + row["daily_rate"] - 1.0) < 1e-9 for row in rows)
    gf3 = len(rows) == 6 and sums_ok and all(row["median_trades_to_pass"] is not None for row in rows)

    best_rate = max(row["pass_rate"] for row in rows)
    optimal = next(row["risk_pct"] for row in rows if row["pass_rate"] == best_rate)
    q = 1.0 - pop["win_rate"]
    longest_streak_expect = {f"{row['risk_pct']:.2%}": float(np.log(max(row["median_trades_to_pass"], 2)) / np.log(1.0 / q)) if row["median_trades_to_pass"] else None for row in rows}

    gates = {"G-F1": bool(gf1), "G-F2": bool(gf2), "G-F3": bool(gf3)}
    overall = all(gates.values())

    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("REPORT: FTMO CHALLENGE MONTE CARLO ON VRP-ORB ARM A (HYPOTHESIS #37)")
    lines.append(f"Generated: {ts}  |  script: evaluate/ftmo_sim_vrp_orb.py")
    lines.append("SPEC/GATE: data/results/ftmo_sim_vrp_orb/{SPEC,GATE}_FTMO_SIM.txt (frozen before code)")
    lines.append("")
    lines.append(f"INTEGRITY GATES: {' & '.join(f'{k}={v}' for k, v in gates.items())}  ->  OVERALL {'PASS' if overall else 'FAIL'}")
    lines.append("")
    lines.append("1. INPUT POPULATION (data/results/orb_vrp_filter/orb_vrp_filtered_trades.csv, filter_arm=A)")
    lines.append(f"   trades={pop['n_trades']}  losers={pop['n_losers']}  win_rate={pop['win_rate']*100:.1f}%")
    lines.append(f"   avg_net={pop['avg_net_pt']:.3f} pt  mean_loss={pop['mean_loss_pt']:.3f} pt")
    lines.append(f"   R distribution: mean={pop['mean_R']:.4f}  std={pop['std_R']:.4f}  min={pop['min_R']:.4f}  max={pop['max_R']:.4f}")
    lines.append(f"   worst trade {pop['worst_net_pt']:.2f} pt  best trade {pop['best_net_pt']:.2f} pt")
    lines.append("")
    lines.append("2. SIMULATION DESIGN")
    lines.append(f"   {N_MC} iid with-replacement challenges per risk level, seed {SEED}, halt priority PASS > MAX_LOSS > DAILY.")
    lines.append(f"   barriers: equity >= ${TARGET:.0f} pass, <= ${FLOOR:.0f} max loss, single-trade PnL <= ${DAILY_BREACH_USD:.0f} daily breach.")
    lines.append(f"   safety cap {MAX_TRADES} trades/challenge; unresolved must be 0 (G-F2).")
    lines.append("")
    lines.append("3. RESULTS BY RISK LEVEL")
    lines.append(f"   {'Risk %':>7} | {'Pass':>7} | {'MaxLoss':>8} | {'Daily':>7} | {'MedTrades(pass)':>15} | {'MedTrades(maxloss)':>18} | {'TargetR':>8} | {'FloorR':>7}")
    for row in rows:
        mp = f"{row['median_trades_to_pass']:.0f}" if row["median_trades_to_pass"] is not None else "n/a"
        mm = f"{row['median_trades_to_max_loss']:.0f}" if row["median_trades_to_max_loss"] is not None else "n/a"
        lines.append(f"   {row['risk_pct']*100:>6.2f}% | {row['pass_rate']*100:>6.2f}% | {row['max_loss_rate']*100:>7.2f}% | {row['daily_rate']*100:>6.2f}% | {mp:>15} | {mm:>18} | {row['target_R']:>8.1f} | {row['floor_R']:>7.1f}")
    lines.append("")
    lines.append("4. OPTIMAL RISK (frozen rule: max pass rate, ties to lower risk)")
    lines.append(f"   Optimal Risk %: {optimal*100:.2f}%  (pass rate {best_rate*100:.2f}%)")
    for row in rows:
        streak = longest_streak_expect[f"{row['risk_pct']:.2%}"]
        streak_s = f"{streak:.1f}" if streak is not None else "n/a"
        lines.append(f"   risk {row['risk_pct']*100:.2f}%: floor is {row['floor_R']:.1f}R away; expected worst losing streak within median pass length ~{streak_s} trades x ~1R each")
    lines.append("")
    lines.append("5. NOTES AND CAVEATS")
    breach_max = max(pop["breach_R_by_risk"].values())
    lines.append(f"   Daily-loss breach needs a single trade <= {breach_max:.2f}R (at 2.00% risk, the loosest threshold on the grid);")
    lines.append(f"   population min R is {pop['min_R']:.2f}R, so the single-trade daily rule {'can' if pop['min_R'] <= breach_max else 'cannot'} bind anywhere on this grid;")
    lines.append("   the binding failure mode is the $9,000 total-loss floor, not the daily rule.")
    lines.append("   Descriptive measurement only: Hypothesis #31 overall verdict remains FAIL (G-VRP3);")
    lines.append("   Arm A conditioning is in-sample selection, so these pass rates are optimistic by construction.")
    lines.append("   iid draws ignore PnL clustering/regime dependence; calendar time is not simulated")
    lines.append("   (median trades-to-pass at hundreds of trades >> the ~171 historical signals over 3.4 years).")
    report = "\n".join(lines) + "\n"
    (STUDY / "REPORT_FTMO_SIM.txt").write_text(report, encoding="utf-8")

    meta = {
        "generated": ts,
        "script": "evaluate/ftmo_sim_vrp_orb.py",
        "seed": SEED,
        "n_mc": N_MC,
        "max_trades_cap": MAX_TRADES,
        "rules": {"start": START, "target": TARGET, "floor": FLOOR, "daily_breach_usd": DAILY_BREACH_USD},
        "population": pop,
        "results": rows,
        "gates": gates,
        "overall": overall,
        "optimal_risk_pct": optimal,
    }
    (STUDY / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Gates: {gates} -> OVERALL {'PASS' if overall else 'FAIL'}")
    print(f"Population: n={pop['n_trades']} win={pop['win_rate']*100:.1f}% mean_loss={pop['mean_loss_pt']:.3f}pt mean_R={pop['mean_R']:.4f} min_R={pop['min_R']:.3f} max_R={pop['max_R']:.3f}")
    print()
    print("| Risk % | Pass Rate | Max Loss Fail | Daily Loss Fail | Median Trades to Pass |")
    print("|---|---|---|---|---|")
    for row in rows:
        mp = f"{row['median_trades_to_pass']:.0f}" if row["median_trades_to_pass"] is not None else "n/a"
        print(f"| {row['risk_pct']*100:.2f}% | {row['pass_rate']*100:.2f}% | {row['max_loss_rate']*100:.2f}% | {row['daily_rate']*100:.2f}% | {mp} |")
    print()
    print(f"Optimal Risk %: {optimal*100:.2f}% (pass rate {best_rate*100:.2f}%, max-loss fail {next(r['max_loss_rate'] for r in rows if r['risk_pct']==optimal)*100:.2f}%)")


if __name__ == "__main__":
    main()
