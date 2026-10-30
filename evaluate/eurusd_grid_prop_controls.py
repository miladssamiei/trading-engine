"""Frozen small-lot / 1% equity bailout study; see SPEC_GRID_PROP.txt."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate.eurusd_grid_bot_h1 import load_grid, digest, SOURCE, utc

OUT = ROOT / "data/results/eurusd_grid_prop"
REPORT = OUT / "REPORT_GRID_PROP.txt"
INITIAL = 10_000.0
ENVIRONMENTS = (("FLOOR_03_ZERO_SWAP", .3, False),
                ("ALLIN_07_ZERO_SWAP", .7, False),
                ("ALLIN_07_CURRENT_SWAP", .7, True))
RESTARTS = ("NEXT_DAY", "24H", "72H")
BASE_LOTS = (.01, .02, .05)
SEED = 20261005
BOOT = 10_000
CAUSE = {1: "BASKET_BAILOUT", 2: "DAILY_BAILOUT", 3: "TOTAL_BUFFER_HALT", 4: "TARGET", 0: "OPEN_AT_TAPE_END"}
BASKET_COLUMNS = ["basket_id", "entry_ix", "exit_ix", "entry_epoch", "exit_epoch", "side",
                  "depth", "total_lots", "gross_usd", "cost_usd", "swap_usd", "net_eur", "exit_bid",
                  "max_floating_loss_eur", "balance_before", "balance_after", "exit_code",
                  "adverse_gap", "eligible_after_epoch", "rollover_weight"]
ORDER_COLUMNS = ["basket_id", "entry_ix", "entry_epoch", "depth", "ideal_lots", "lots", "entry_bid", "total_lots", "equity_at_entry"]
DAY_COLUMNS = ["prague_date", "starting_balance", "end_balance", "end_equity", "min_equity",
               "max_equity", "max_daily_loss_eur", "max_equity_dd_eur", "firm_daily_breach", "firm_total_breach"]


@njit(cache=True)
def floating_eur(price, side, lots, weighted_entry, cost, swap):
    return (side * 100000.0 * (price * lots - weighted_entry) - cost + swap) / price


@njit(cache=True)
def simulate(t, o, h, l, c, first, last, signal, bucket, firm_day, next_day,
             rolls, start_ix, end_ix, base, spread, financed, cooldown_hours):
    balance = INITIAL
    day_balance = INITIAL
    peak = INITIAL
    max_dd = 0.0
    current_day = -1
    day_low = INITIAL
    day_high = INITIAL
    last_equity = INITIAL
    active = False
    halted = False
    eligible_after = 0
    side = depth = basket_id = 0
    entry_ix = entry_time = 0
    lots = weighted_entry = cost = swap = mae = rollcount = 0.0
    last_entry = 0.0
    basket_start_balance = INITIAL
    baskets, orders, days = [], [], []
    blocked_cooldown = blocked_caps = ignored_opposite = 0
    trade_days = 0
    last_trade_day = -1
    target_epoch = -1
    for i in range(start_ix, end_ix):
        if firm_day[i] != current_day:
            if current_day != -1:
                days.append((current_day, day_balance, balance, last_equity, day_low, day_high,
                             max(0.0, day_balance - day_low), max_dd,
                             day_low <= day_balance - 500.0 + 1e-6, day_low <= 9000.0 + 1e-6))
            current_day = firm_day[i]
            day_balance = balance
            day_low = day_high = last_equity
        if active and financed and rolls[i]:
            swap += lots * (-8.79 if side == 1 else .38) * rolls[i]
            rollcount += rolls[i]
        opening_net = floating_eur(o[i], side, lots, weighted_entry, cost, swap) if active else 0.0
        opening_equity = balance + opening_net
        peak = max(peak, opening_equity)
        day_low = min(day_low, opening_equity)
        day_high = max(day_high, opening_equity)
        max_dd = max(max_dd, peak - opening_equity)
        # Check the existing basket's opening gap before considering additions.
        floor = max(9100.0, day_balance - 100.0, balance - 100.0)
        opening_breach = active and opening_equity <= floor + 1e-6
        if not halted and first[i] and signal[i] != 0 and not opening_breach:
            if t[i] < eligible_after:
                blocked_cooldown += 1
            elif active and signal[i] != side:
                ignored_opposite += 1
            elif not active or side * (o[i] - last_entry) <= -.003 + 1e-12:
                ideal = base * 1.5 ** depth
                order_lots = np.floor((ideal + 1e-12) / .01) * .01
                proposed = lots + order_lots
                estimate_equity = opening_equity - spread * 10.0 * order_lots / o[i]
                margin = proposed * 100000.0 / 30.0
                if depth >= 5 or proposed > .30 + 1e-12 or margin > .15 * estimate_equity:
                    blocked_caps += 1
                else:
                    if not active:
                        active = True
                        basket_id += 1
                        side = signal[i]
                        entry_ix, entry_time = i, t[i]
                        basket_start_balance = balance
                        mae = rollcount = 0.0
                    depth += 1
                    lots = proposed
                    weighted_entry += order_lots * o[i]
                    cost += spread * 10.0 * order_lots
                    last_entry = o[i]
                    orders.append((basket_id, i, t[i], depth, ideal, order_lots, o[i], lots, estimate_equity))
                    if current_day != last_trade_day:
                        trade_days += 1
                        last_trade_day = current_day
        exit_code = 0
        gap = False
        exit_price = c[i]
        exit_time = t[i]
        if active:
            opening_net = floating_eur(o[i], side, lots, weighted_entry, cost, swap)
            opening_equity = balance + opening_net
            floor = max(9100.0, day_balance - 100.0, balance - 100.0)
            adverse = l[i] if side == 1 else h[i]
            adverse_net = floating_eur(adverse, side, lots, weighted_entry, cost, swap)
            if opening_equity <= floor + 1e-6:
                exit_price = o[i]
                gap = True
                held_low = opening_net
            elif balance + adverse_net <= floor + 1e-6:
                # F(P)=A+B/P; solve for the tightest active equity floor.
                a = side * 100000.0 * lots
                b = -side * 100000.0 * weighted_entry - cost + swap
                exit_price = b / (floor - balance - a)
                held_low = floating_eur(exit_price, side, lots, weighted_entry, cost, swap)
            else:
                held_low = min(opening_net, adverse_net)
            held_equity = balance + held_low
            mae = max(mae, -held_low)
            day_low = min(day_low, opening_equity, held_equity)
            day_high = max(day_high, opening_equity)
            peak = max(peak, opening_equity)
            max_dd = max(max_dd, peak - held_equity)
            if held_equity <= floor + 1e-6:
                if held_equity <= 9100.0 + 1e-6:
                    exit_code = 3
                    halted = True
                elif held_equity <= day_balance - 100.0 + 1e-6:
                    exit_code = 2
                else:
                    exit_code = 1
            else:
                close_net = floating_eur(c[i], side, lots, weighted_entry, cost, swap)
                last_equity = balance + close_net
                peak = max(peak, last_equity)
                day_low = min(day_low, last_equity)
                day_high = max(day_high, last_equity)
                max_dd = max(max_dd, peak - last_equity)
                net_usd = close_net * c[i]
                if last[i] and net_usd > 30.0 * base * 10.0 + 1e-9:
                    exit_code = 4
                    exit_time = bucket[i] + 3599
            if exit_code:
                gross = side * 100000.0 * (exit_price * lots - weighted_entry)
                net = (gross - cost + swap) / exit_price
                before = balance
                balance += net
                if exit_code != 4:
                    eligible_after = max(next_day[i], t[i] + cooldown_hours * 3600)
                baskets.append((basket_id, entry_ix, i, entry_time, exit_time, side, depth, lots,
                                gross, cost, swap, net, exit_price, mae, before, balance, exit_code, gap,
                                eligible_after, rollcount))
                last_equity = balance
                if balance >= 11000.0 and trade_days >= 4 and target_epoch == -1:
                    target_epoch = exit_time
                if balance <= 9100.0 + 1e-6:
                    halted = True
                active = False
                lots = weighted_entry = cost = swap = mae = rollcount = 0.0
                depth = 0
                side = 0
        else:
            last_equity = balance
        day_low = min(day_low, last_equity)
        day_high = max(day_high, last_equity)
        peak = max(peak, last_equity)
        max_dd = max(max_dd, peak - last_equity)
    if active:
        i = end_ix - 1
        gross = side * 100000.0 * (c[i] * lots - weighted_entry)
        net = (gross - cost + swap) / c[i]
        baskets.append((basket_id, entry_ix, i, entry_time, t[i] + 59, side, depth, lots,
                        gross, cost, swap, net, c[i], mae, balance, balance, 0, False,
                        eligible_after, rollcount))
    days.append((current_day, day_balance, balance, last_equity, day_low, day_high,
                 max(0.0, day_balance - day_low), max_dd,
                 day_low <= day_balance - 500.0 + 1e-6, day_low <= 9000.0 + 1e-6))
    return baskets, orders, days, halted, blocked_cooldown, blocked_caps, ignored_opposite, target_epoch


def mbb_probability(values, seed):
    values = np.asarray(values, float)
    if len(values) == 0:
        return None
    length = min(5, len(values))
    complete, remainder = divmod(len(values), length)
    sums = np.convolve(values, np.ones(length), mode="valid")
    rng = np.random.default_rng(seed)
    nonpositive = 0
    for _ in range(20):
        indices = rng.integers(len(sums), size=(500, complete + bool(remainder)))
        sampled = sums[indices[:, :complete]].sum(axis=1)
        if remainder:
            prefix = np.concatenate(([0.0], np.cumsum(values)))
            starts = indices[:, -1]
            sampled += prefix[starts + remainder] - prefix[starts]
        nonpositive += int((sampled <= 0).sum())
    return nonpositive / BOOT


def inputs(tape, grid):
    t = tape.time.to_numpy(np.int64)
    columns = [t] + [tape[k].to_numpy(float) for k in ["open", "high", "low", "close"]]
    n = len(t)
    first, last = np.zeros(n, bool), np.zeros(n, bool)
    signals = np.zeros(n, np.int64)
    real = grid.loc[grid.real]
    first[real.first_ix] = True
    last[real.last_ix] = True
    signals[real.first_ix] = real.signal
    stamp = pd.to_datetime(t, unit="s", utc=True)
    local = stamp.tz_convert("Europe/Prague")
    date_key = np.array(local.strftime("%Y%m%d"), dtype=np.int64)
    mapping = {int(d): int((pd.Timestamp(str(d), tz="Europe/Prague") + pd.DateOffset(days=1)).timestamp())
               for d in np.unique(date_key)}
    next_days = np.array([mapping[int(d)] for d in date_key], dtype=np.int64)
    dates = pd.date_range(stamp[0].date() - pd.Timedelta(days=2), stamp[-1].date() + pd.Timedelta(days=2))
    roll_epoch, roll_factor = [], []
    for date in dates:
        if date.weekday() < 5:
            roll_epoch.append(int(pd.Timestamp(f"{date.date()} 17:00", tz="America/New_York").timestamp()))
            roll_factor.append(3 if date.weekday() == 2 else 1)
    cumulative = np.r_[0, np.cumsum(roll_factor)]
    crossed = cumulative[np.searchsorted(roll_epoch, t, side="right")]
    roll_delta = np.r_[0, np.diff(crossed)].astype(np.int64)
    return (*columns, first, last, signals, t // 3600 * 3600, date_key, next_days, roll_delta)


def run_case(case_id, env, base, restart, period, bounds, data, code):
    name, spread, finance = env
    hours = (0, 24, 72)[RESTARTS.index(restart)]
    result = simulate(*data, *bounds, base, spread, finance, hours)
    raw_b, raw_o, raw_d, halted, cooldown, caps, opposite, phase1 = result
    b = pd.DataFrame(raw_b, columns=BASKET_COLUMNS)
    o = pd.DataFrame(raw_o, columns=ORDER_COLUMNS)
    d = pd.DataFrame(raw_d, columns=DAY_COLUMNS)
    b["exit_reason"] = b.exit_code.map(CAUSE)
    b["entry_time_utc"] = pd.to_datetime(b.entry_epoch, unit="s", utc=True)
    b["exit_time_utc"] = pd.to_datetime(b.exit_epoch, unit="s", utc=True)
    # End-period marks remain unrealized. Every exit retains its modeled BID.
    assert np.allclose(b.net_eur, (b.gross_usd - b.cost_usd + b.swap_usd) / b.exit_bid, rtol=0, atol=1e-6)
    closed = b.loc[b.exit_code.ne(0)]
    open_b = b.loc[b.exit_code.eq(0)]
    assert np.allclose(closed.balance_after - closed.balance_before, closed.net_eur, rtol=0, atol=1e-6)
    assert not o.entry_ix.duplicated().any()
    assert not (o.entry_epoch // 3600).duplicated().any()
    assert (o.depth <= 5).all() and (o.total_lots <= .30 + 1e-12).all()
    assert np.allclose(o.lots / .01, np.round(o.lots / .01), atol=1e-9)
    assert np.allclose(b.cost_usd, spread * 10 * b.total_lots, rtol=0, atol=1e-9)
    assert np.isclose(closed.net_eur.sum() + INITIAL, d.end_balance.iloc[-1], rtol=0, atol=1e-6)
    assert np.isclose(closed.net_eur.sum() + open_b.net_eur.sum() + INITIAL, d.end_equity.iloc[-1], rtol=0, atol=1e-6)
    for row in closed.loc[closed.exit_code.ne(4)].itertuples(index=False):
        after = o.loc[o.entry_ix.gt(row.exit_ix)]
        if len(after):
            assert after.entry_epoch.min() >= row.eligible_after_epoch
        if not row.adverse_gap:
            assert row.net_eur >= -100.0 - 1e-5
    prefix = f"{period.lower()}_{case_id:02d}"
    b.to_csv(OUT / f"baskets_{prefix}.csv", index=False)
    o.to_csv(OUT / f"orders_{prefix}.csv", index=False)
    d.to_csv(OUT / f"daily_{prefix}.csv", index=False)
    deviation = closed.net_eur.std(ddof=1)
    average = float(closed.net_eur.mean()) if len(closed) else None
    sharpe = float(average / deviation) if len(closed) > 1 and deviation > 0 else None
    probability = mbb_probability(closed.net_eur, SEED + case_id + code * 1000)
    years = pd.to_datetime(closed.exit_epoch, unit="s", utc=True).dt.tz_convert("Europe/Prague").dt.year
    annual = closed.groupby(years).net_eur.sum().to_dict()
    period_months = (data[0][bounds[1] - 1] + 60 - data[0][bounds[0]]) / (86400 * 365.2425 / 12)
    marked = float(b.net_eur.sum())
    row = {"case_id": case_id, "period": period, "environment": name, "base_lots": base,
           "restart": restart, "closed_baskets": len(closed), "open_baskets": len(open_b),
           "win_rate": float(closed.net_eur.gt(0).mean()) if len(closed) else None,
           "mean_net_eur": average, "net_sharpe": sharpe, "mbb_p": probability,
           "realized_net_eur": float(closed.net_eur.sum()), "terminal_open_eur": float(open_b.net_eur.sum()),
           "marked_net_eur": marked, "monthly_marked_eur": marked / period_months,
           "target_exits": int(closed.exit_code.eq(4).sum()), "bailouts": int(closed.exit_code.ne(4).sum()),
           "basket_bailouts": int(closed.exit_code.eq(1).sum()), "daily_bailouts": int(closed.exit_code.eq(2).sum()),
           "buffer_halts": int(halted), "firm_daily_breach_days": int(d.firm_daily_breach.sum()),
           "firm_total_breach_days": int(d.firm_total_breach.sum()),
           "max_equity_dd_pct": float(d.max_equity_dd_eur.max() / INITIAL * 100),
           "max_daily_loss_pct": float(d.max_daily_loss_eur.max() / INITIAL * 100),
           "max_basket_loss_eur": float(b.max_floating_loss_eur.max()) if len(b) else 0.0,
           "worst_closed_eur": float(closed.net_eur.min()) if len(closed) else None,
           "max_bailout_overshoot_eur": max(0.0, -float(closed.net_eur.min()) - 100) if len(closed) else 0.0,
           "adverse_gap_exits": int(closed.adverse_gap.sum()), "max_depth": int(o.depth.max()) if len(o) else 0,
           "max_total_lots": float(o.total_lots.max()) if len(o) else 0.0,
           "max_estimated_margin_eur": float(o.total_lots.max() * 100000 / 30) if len(o) else 0.0,
           "swap_usd": float(b.swap_usd.sum()), "spread_usd": float(b.cost_usd.sum()),
           "blocked_cooldown": cooldown, "blocked_caps": caps, "ignored_opposite": opposite,
           "phase1_target_epoch": int(phase1) if phase1 >= 0 else None,
           "phase1_target_days": (phase1 - data[0][bounds[0]]) / 86400 if phase1 >= 0 else None,
           "annual_exit_pnl_eur": {str(k): float(v) for k, v in annual.items()},
           "period_months": period_months}
    print(f"{period:11s} {case_id:02d} {name:22s} lot{base:.2f} {restart:8s} "
          f"n{len(closed):4d} net{marked:+9.2f} bail{row['bailouts']:3d} halt{int(halted)}", flush=True)
    return row


def gate_candidate(row):
    g4 = (row["closed_baskets"] >= 150 and row["mean_net_eur"] is not None and row["mean_net_eur"] > 0
          and row["net_sharpe"] is not None and row["net_sharpe"] > 0 and row["marked_net_eur"] > 0
          and row["mbb_p"] is not None and row["mbb_p"] <= .05)
    years = row["annual_exit_pnl_eur"]
    g5 = (row["buffer_halts"] == 0 and row["firm_daily_breach_days"] == 0 and row["firm_total_breach_days"] == 0
          and all(years.get(str(y), 0.0) >= 0 for y in (2024, 2025)))
    return bool(g4), bool(g5)


def main():
    if REPORT.exists():
        raise FileExistsError("Immutable report exists: freeze a new study for changes/rerun")
    spec = (OUT / "SPEC_GRID_PROP.txt").read_text()
    gates = (OUT / "GATE_GRID_PROP.txt").read_text().strip()
    assert gates == "\n".join(x for x in spec.splitlines() if x.startswith(tuple(f"G{i}:" for i in range(1, 6))))
    tape, grid = load_grid()
    assert digest(SOURCE) == "e578692f3b02d985a97c17d13f0fbd0f6797e47c9b31a43970667d020f4d7242"
    data = inputs(tape, grid)
    cut = int(np.searchsorted(data[0], pd.Timestamp("2024-01-01", tz="UTC").timestamp()))
    bounds = {"DEVELOPMENT": (0, cut), "VALIDATION": (cut, len(tape)), "FULL": (0, len(tape))}
    cases = [(i, env, lot, restart) for i, (env, lot, restart) in enumerate(
        [(e, b, r) for e in ENVIRONMENTS for b in BASE_LOTS for r in RESTARTS])]
    rows = []
    for case_id, env, lot, restart in cases:
        rows.append(run_case(case_id, env, lot, restart, "DEVELOPMENT", bounds["DEVELOPMENT"], data, 0))
    eligible = [r for r in rows if r["environment"] == "ALLIN_07_CURRENT_SWAP" and not r["buffer_halts"]
                and not r["firm_daily_breach_days"] and not r["firm_total_breach_days"] and r["closed_baskets"] >= 150
                and r["marked_net_eur"] > 0 and r["mean_net_eur"] is not None and r["mean_net_eur"] > 0]
    eligible.sort(key=lambda r: (-r["marked_net_eur"], r["base_lots"], RESTARTS.index(r["restart"])))
    selected = eligible[0] if eligible else None
    selection = {"selected_case_id": selected["case_id"] if selected else None,
                 "reason": "development-only highest marked net among eligible primary cases" if selected else "No primary development case meets frozen selection conditions",
                 "selected_development_metrics": selected}
    with (OUT / "selection_development.json").open("x") as stream:
        json.dump(selection, stream, indent=2, allow_nan=False)
    for code, period in [(1, "VALIDATION"), (2, "FULL")]:
        for case_id, env, lot, restart in cases:
            rows.append(run_case(case_id, env, lot, restart, period, bounds[period], data, code))
    candidate = next((r for r in rows if r["period"] == "VALIDATION" and selected and r["case_id"] == selected["case_id"]), None)
    g4, g5 = gate_candidate(candidate) if candidate else (False, False)
    status = "DEMO_CANDIDATE_PENDING_MT5_RECONCILIATION" if g4 and g5 else "NOT_ADOPTED"
    pd.DataFrame([{**r, "annual_exit_pnl_eur": json.dumps(r["annual_exit_pnl_eur"])} for r in rows]).to_csv(OUT / "case_matrix.csv", index=False)
    metadata = {"date": "2026-10-05", "status": status, "source_sha256": digest(SOURCE),
                "source_first_utc": utc(data[0][0]), "source_last_utc": utc(data[0][-1]), "m1_rows": len(tape),
                "original_evaluator_sha256": digest(ROOT / "evaluate/eurusd_grid_bot_h1.py"),
                "evaluator_sha256": digest(Path(__file__)), "spec_sha256": digest(OUT / "SPEC_GRID_PROP.txt"),
                "gate_sha256": digest(OUT / "GATE_GRID_PROP.txt"), "selection": selection,
                "selected_validation": candidate, "gates": {"G1": "PASS", "G2": "PASS", "G3": "PASS", "G4": "PASS" if g4 else "FAIL", "G5": "PASS" if g5 else "FAIL"},
                "broker_snapshot": {"currency": "EUR", "leverage": 30, "contract_size": 100000,
                                    "volume_min": .01, "volume_step": .01, "swap_mode": 1, "swap_long": -8.79, "swap_short": .38},
                "rows": rows}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    instructions = ["EURUSD GRID PROP CONTROL INSTRUCTIONS - 2026-10-05", f"STATUS: {status}",
                    "Account model EUR10,000; basket bailout EUR100; daily equity allowance EUR100 from Prague-midnight balance.",
                    "Panic-close all strategy positions when the tightest basket/daily/internal-total equity floor is touched.",
                    "Daily loss uses equity, including all open positions, costs and swap. Never reset it after each basket.",
                    "After bailout: persist the latch through next Prague midnight and any configured UTC cooldown; start next basket at base lots.",
                    "Static internal floor EUR9,100 permanently disables entries; model firm floor EUR9,000/dailyEUR500.",
                    "Maximum5 positions, total0.30lots; round ideal geometric volumes down to broker0.01 step; max estimated margin15% equity.",
                    "No order before old-position risk check; adverse gaps fill at actual open and can exceed nominalEUR100.",
                    "Gap, variable-spread, financing and execution-parity tests are required before live/prop-ready use."]
    if candidate:
        instructions += [f"Development-selected base lots {candidate['base_lots']:.2f}; restart {candidate['restart']}; primary costs0.7pips plus frozen current swap sensitivity.",
                         f"Validation G4={g4}, G5={g5}; no post-validation replacement is adopted."]
    else:
        instructions += ["No development candidate qualifies. Risk controls are a research template; do not adopt a validation winner after inspecting these rows."]
    (OUT / "PROP_CONTROL_INSTRUCTIONS.txt").write_text("\n".join(instructions) + "\n")
    lines = ["EURUSD H1 GRID PROP CONTROLS V1 - REPORT - 2026-10-05", f"VERDICT: {status}",
             "EUR10,000, 1% basket/daily equity bailout,9% internal static loss guard, fixed5-level/0.30-lot caps.",
             "DEVELOPMENT2020-2023 selects ONE primary case; VALIDATION2024-2026 tests it on a fresh account.",
             f"Source: {utc(data[0][0])} to {utc(data[0][-1])}; {len(tape):,} real M1 bars.",
             "All profit exits are H1-close proxies; all risk exits use M1 adverse quotes and actual opening gaps.", "",
             "PRIMARY ENVIRONMENT: 0.7pips all-inclusive plus frozen current swap sensitivity",
             "Period       Base  Restart   Closed Wins%    MarkedEUR  Bail  Halt EquityDD% DailyLoss%  MBBp"]
    for r in rows:
        if r["environment"] == "ALLIN_07_CURRENT_SWAP":
            lines.append(f"{r['period']:11s} {r['base_lots']:.2f} {r['restart']:8s} {r['closed_baskets']:6d} "
                         f"{100*r['win_rate']:6.2f} {r['marked_net_eur']:12.2f} {r['bailouts']:5d} {r['buffer_halts']:5d} "
                         f"{r['max_equity_dd_pct']:9.3f} {r['max_daily_loss_pct']:10.3f} {r['mbb_p']:7.4f}")
    lines += ["", "DEVELOPMENT SELECTION", json.dumps(selection, indent=2), "",
              "SELECTED VALIDATION", json.dumps(candidate, indent=2), "", "GATES", json.dumps(metadata["gates"], indent=2), "",
              "LIMITS: Current swaps are applied across history as a sensitivity, not dated historical financing.",
              "Lot/contract/leverage snapshot and NY17 rollover are assumptions; account conversion uses BID EURUSD.",
              "Virtual intraminute stop fills exclude execution latency/slippage; gaps and stop overshoots remain in ledgers.",
              "Full replay stops permanently at the internal total buffer. Flat periods after that halt are not a recovered account.",
              "MBB is conditional basket-mean uncertainty, not a prop pass-rate guarantee. Historical control selection remains exploratory.",
              "Daily equity DD is measured at M1 opens/adverse held marks/closes, not all actual tick peaks.",
              "Other environments/all case matrices and yearly exits are in case_matrix.csv and metadata.json.",
              "Public2-Step limits checked2026-10-05: https://ftmo.com/en/trading-objectives/",
              "Swing specs: https://ftmo.com/en/faq/what-are-the-account-specifications/",
              "No live orders, activation, account purchase or payout economics are included."]
    with REPORT.open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
