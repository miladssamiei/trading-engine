"""H50/H51 and H57/H58 gold engines at old and all-inclusive CFD costs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from futures_prop_validation_phase1 import engine_a, engine_b, grids, load, sha

ROOT = Path(__file__).resolve().parents[1]
TAPE = ROOT / 'data/results/xauusd_onboarding/xauusdtick_m1.csv'
OUT = ROOT / 'data/results/gold_ensemble_real_cost'
COSTS = {'old': 0.410, 'strict': 0.585}
LOT_DOLLARS_PER_UNIT = 100.0
BOOT = 10_000
BLOCK = 5
SEED = 20260928


def mbb_p(values):
    x = np.asarray(values, float)
    n = len(x)
    if n < BLOCK:
        return float('nan')
    rng = np.random.default_rng(SEED)
    block_sums = np.convolve(x, np.ones(BLOCK), mode='valid')
    count, extra = divmod(n, BLOCK)
    tail_sums = np.convolve(x, np.ones(extra), mode='valid') if extra else None
    bad = 0
    for offset in range(0, BOOT, 250):
        k = min(250, BOOT-offset)
        starts = rng.integers(0, n-BLOCK+1, size=(k, count+int(extra > 0)))
        sums = block_sums[starts[:, :count]].sum(axis=1)
        if extra:
            sums += tail_sums[starts[:, count]]
        bad += int(np.count_nonzero(sums <= 0))
    return bad/BOOT


def replay(m, engine, cost):
    t, o, h, l, c = (m[k] for k in ('time', 'open', 'high', 'low', 'close'))
    left = np.searchsorted(t, engine['start'])
    right = np.searchsorted(t, engine['end'])
    rows = []
    occupied_until = -1
    shadowed = 0
    no_real = 0
    for i in np.flatnonzero(engine['side']):
        if i <= occupied_until:
            shadowed += 1
            continue
        if left[i] >= right[i]:
            no_real += 1
            continue
        side = int(engine['side'][i])
        entry_ix = int(left[i])
        entry_bid = float(o[entry_ix])
        atr = float(engine['atr'][i-1])
        if not np.isfinite(atr) or atr <= 0:
            raise AssertionError('Invalid prior ATR at an armed entry')
        if engine['name'] == 'A_KC':
            distance = max(abs(entry_bid-float(engine['ema'][i-1])), 1.5*atr)
            stop = entry_bid-side*distance  # BID-coordinate stop for both sides
        else:
            distance = 4.0*atr
            stop = entry_bid-side*distance  # long BID stop / short ASK stop
        initial_stop = float(stop)
        entry_fill = entry_bid+cost if side == 1 else entry_bid
        exit_bid = None
        exit_fill = None
        exit_ix = None
        exit_bar = None
        reason = None
        for j in range(i, len(engine['side'])):
            a, z = int(left[j]), int(right[j])
            if j == i:
                a = entry_ix
            if a < z:
                if side == 1:
                    hits = np.flatnonzero(l[a:z] <= stop)
                else:
                    ask_stop = stop+cost if engine['name'] == 'A_KC' else stop
                    hits = np.flatnonzero(h[a:z]+cost >= ask_stop)
                if len(hits):
                    exit_ix = a+int(hits[0])
                    exit_bar = j
                    if side == 1:
                        exit_bid = min(float(stop), float(o[exit_ix]))
                        exit_fill = exit_bid
                    else:
                        exit_fill = max(float(ask_stop), float(o[exit_ix])+cost)
                        exit_bid = exit_fill-cost
                    reason = ('SL_TRAIL' if side*(stop-initial_stop) > 1e-9
                              else 'SL_INIT')
                    break
            if engine['name'] == 'A_KC':
                if np.isfinite(engine['ema'][j]):
                    candidate = float(engine['ema'][j])
                    stop = max(stop, candidate) if side == 1 else min(stop, candidate)
            elif a < z and engine['real'][j] and np.isfinite(engine['atr'][j]):
                candidate = float(engine['close'][j]-side*4.0*engine['atr'][j])
                stop = max(stop, candidate) if side == 1 else min(stop, candidate)
        if exit_ix is None:
            exit_ix = len(t)-1
            exit_bar = len(engine['side'])-1
            exit_bid = float(c[-1])
            exit_fill = exit_bid if side == 1 else exit_bid+cost
            reason = 'END'
        occupied_until = exit_bar
        gross = side*(exit_bid-entry_bid)
        net = side*(exit_fill-entry_fill)
        if abs(net-(gross-cost)) > 1e-8:
            raise AssertionError('Cost embedded more or less than once')
        entry_ny = pd.Timestamp(int(t[entry_ix]), unit='s', tz='UTC').tz_convert('America/New_York')
        exit_ny = pd.Timestamp(int(t[exit_ix]), unit='s', tz='UTC').tz_convert('America/New_York')
        rows.append({'engine': engine['name'], 'side': side,
                     'entry_bar': int(i), 'exit_bar': int(exit_bar),
                     'entry_time': int(t[entry_ix]), 'exit_time': int(t[exit_ix]),
                     'entry_bid': entry_bid, 'entry_fill': entry_fill,
                     'exit_bid': exit_bid, 'exit_fill': exit_fill,
                     'initial_stop': initial_stop, 'final_stop': float(stop),
                     'cost': cost, 'gross': gross, 'net': net,
                     'exit_type': reason, 'entry_ny_year': int(entry_ny.year),
                     'ny_overnight': entry_ny.date() != exit_ny.date(),
                     'hold_hours': float((t[exit_ix]-t[entry_ix])/3600)})
    if rows:
        en = np.array([r['entry_time'] for r in rows], np.int64)
        ex = np.array([r['exit_time'] for r in rows], np.int64)
        if np.any(ex < en) or np.any(en[1:] <= ex[:-1]):
            raise AssertionError('Overlapping trade times')
        if np.any(np.array([r['entry_bar'] for r in rows[1:]]) <=
                  np.array([r['exit_bar'] for r in rows[:-1]])):
            raise AssertionError('Signals were not shadowed through exit bar')
    return rows, {'candidates': int(np.count_nonzero(engine['side'])),
                  'shadowed': shadowed, 'no_real_entry_bar': no_real}


def summarize(rows):
    x = np.array([r['net'] for r in rows], float)
    if len(x) < 2:
        raise AssertionError('Too few trades for stress statistics')
    annual = {}
    for r in rows:
        y = str(r['entry_ny_year'])
        v = annual.setdefault(y, {'trades': 0, 'net_units': 0.0})
        v['trades'] += 1
        v['net_units'] += r['net']
    mean = float(x.mean())
    p = mbb_p(x)
    year_sign = all(annual.get(str(y), {}).get('net_units', 0) > 0
                    for y in range(2021, 2026))
    return {'trades': len(rows), 'win_rate': float((x > 0).mean()),
            'net_expectancy_units': mean, 'net_total_units': float(x.sum()),
            'gross_total_units': float(sum(r['gross'] for r in rows)),
            'net_sharpe_per_trade': float(mean/x.std(ddof=1)),
            'mbb_p': p, 'year_sign_2021_2025': year_sign,
            'robustness_gate_pass': bool(len(rows) >= 150 and mean > 0 and
                                         p <= .05 and year_sign),
            'overnight_trades': sum(bool(r['ny_overnight']) for r in rows),
            'exit_counts': {name: sum(r['exit_type'] == name for r in rows)
                            for name in ('SL_INIT', 'SL_TRAIL', 'END')},
            'annual': annual}


def main():
    spec = OUT/'SPEC_GOLD_REAL_COST.txt'
    gate = OUT/'GATE_GOLD_REAL_COST.txt'
    report = OUT/'REPORT_GOLD_REAL_COST.txt'
    if not spec.exists() or not gate.exists():
        raise FileNotFoundError('Frozen SPEC and GATE must predate this run')
    if report.exists():
        raise FileExistsError('Historical REPORT is immutable; use a new study name')
    m = load(TAPE)
    t, o, h, l, c = (m[k] for k in ('time', 'open', 'high', 'low', 'close'))
    if (len(t) < 2_000_000 or not np.all(t[1:] > t[:-1]) or
            not np.all(t % 60 == 0) or not np.isfinite(np.column_stack((o,h,l,c))).all() or
            np.any(o <= 0) or np.any(h < np.maximum(o,c)) or
            np.any(l > np.minimum(o,c))):
        raise AssertionError('XAUUSD M1 source integrity failed')
    g = grids(m)
    engines = (engine_a(g), engine_b(g))
    results = []
    for e in engines:
        by_case = {}
        for case, cost in COSTS.items():
            rows, funnel = replay(m, e, cost)
            stats = summarize(rows)
            pd.DataFrame(rows).to_csv(OUT/f'trades_{e["name"]}_{case}.csv', index=False)
            by_case[case] = {'cost': cost, 'stats': stats, 'funnel': funnel}
            print(f"{e['name']} {case}: {stats['trades']} trades, win {stats['win_rate']:.2%}, "
                  f"mean {stats['net_expectancy_units']:.4f}, total {stats['net_total_units']:.2f}, "
                  f"MBB p {stats['mbb_p']:.4f}", flush=True)
        old, new = by_case['old'], by_case['strict']
        so, sn = old['stats'], new['stats']
        full_dollars = LOT_DOLLARS_PER_UNIT*(so['net_total_units']-sn['net_total_units'])
        posting_dollars = LOT_DOLLARS_PER_UNIT*(new['cost']*sn['trades']-
                                                old['cost']*so['trades'])
        gross_path_dollars = LOT_DOLLARS_PER_UNIT*(so['gross_total_units']-
                                                   sn['gross_total_units'])
        if abs(full_dollars-posting_dollars-gross_path_dollars) > 1e-6:
            raise AssertionError('Dollar decomposition does not balance')
        results.append({'engine': e['name'], 'scenarios': by_case,
                        'full_replay_old_minus_strict_dollars_per_lot': full_dollars,
                        'cost_posting_component_dollars_per_lot': posting_dollars,
                        'gross_path_component_dollars_per_lot': gross_path_dollars})
    meta = {'study': 'gold_ensemble_real_cost', 'tape': str(TAPE),
            'tape_sha256': sha(TAPE), 'm1_rows': len(t),
            'first_utc': pd.Timestamp(int(t[0]),unit='s',tz='UTC').isoformat(),
            'last_utc': pd.Timestamp(int(t[-1]),unit='s',tz='UTC').isoformat(),
            'script_sha256': sha(Path(__file__)), 'spec_sha256': sha(spec),
            'gate_sha256': sha(gate), 'usd_per_unit_per_standard_lot': LOT_DOLLARS_PER_UNIT,
            'results': results}
    (OUT/'run_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    overall = all(r['scenarios']['strict']['stats']['robustness_gate_pass'] for r in results)
    lines = ['GOLD MACRO ENGINE ALL-INCLUSIVE COST STRESS - REPORT',
             '2020-2026 full BID M1 proxy tape; old 0.410 vs strict 0.585 units round trip.',
             f'Tape {TAPE}; SHA256 {meta["tape_sha256"]}; {len(t):,} M1 bars; {meta["first_utc"]} to {meta["last_utc"]}.',
             'MBB: 10,000 moving-block resamples, 5 trades/block, seed 20260928.',
             'Dollar conversion: 1 standard XAUUSD lot = 100 oz = $100 per gold price unit.',
             f'OVERALL EXPLORATORY ROBUSTNESS GATE: {"PASS" if overall else "FAIL"}', '']
    for r in results:
        lines.append(r['engine'])
        for case in ('old', 'strict'):
            z = r['scenarios'][case]
            s = z['stats']
            lines += [f"  {case.upper()} cost {z['cost']:.3f} | trades {s['trades']} | win {s['win_rate']:.2%} | net expectancy {s['net_expectancy_units']:.6f} gold units/trade | total net {s['net_total_units']:.4f} gold units | Net Sharpe {s['net_sharpe_per_trade']:.5f} | MBB p {s['mbb_p']:.4f} | gate {'PASS' if s['robustness_gate_pass'] else 'FAIL'}",
                      f"    Exit counts {s['exit_counts']}; NY overnight {s['overnight_trades']}; 2021-2025 year signs {'PASS' if s['year_sign_2021_2025'] else 'FAIL'}; annual entry-cohort PnL {s['annual']}; funnel {z['funnel']}"]
        lines += [f"  EXACT FULL-REPLAY LOSS FROM HIGHER COST: ${r['full_replay_old_minus_strict_dollars_per_lot']:,.2f} per 1.00 standard lot (old net minus strict net).",
                  f"    Cost-posting component ${r['cost_posting_component_dollars_per_lot']:,.2f}; cost-induced gross/exit-path component ${r['gross_path_component_dollars_per_lot']:,.2f}; components sum to full difference.", '']
    lines += ['Costs are embedded once in Ask/Bid fills. Engine A keeps its BID-coordinate stop path; Engine B retains its original ASK-trigger short stop, which can change exits when cost changes.',
              'The 0.410 baseline here is a fresh M1-stop replay. It is not the previously published 4H/1H OHLC-run total.',
              'Fixed cost is a scenario floor, not measured time-varying spread or commission. No swap, slippage, MT5 fill fidelity, lot sizing, or FTMO account drawdown is modeled. Retrospective prior-researched proxy performance is not independent validation.']
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Wrote {report}', flush=True)


if __name__ == '__main__':
    main()
