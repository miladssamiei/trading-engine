"""Gold-only SP2L remeasurement; frozen protocol in corresponding SPEC/GATE."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import futures_prop_validation_intraday as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/results/sp2l_gold_only_retest_2026_09_26_v1'
TAPE = ROOT / 'data/results/xauusd_onboarding/xauusdtick_m1.csv'


def detail(trades):
    x = np.array([r['net'] for r in trades], dtype=float)
    curve = np.r_[0.0, np.cumsum(x)]
    drawdown = float(np.max(np.maximum.accumulate(curve) - curve))
    sides = {}
    for side, name in ((1, 'long'), (-1, 'short')):
        v = np.array([r['net'] for r in trades if r['side'] == side], dtype=float)
        sides[name] = {'n': len(v), 'mean': float(v.mean()) if len(v) else None,
                       'net': float(v.sum())}
    return {'max_drawdown_gold_units': drawdown, 'sides': sides}


def main():
    if not (OUT / 'SPEC_SP2L_GOLD_ONLY_RETEST_V1.txt').exists() or not (OUT / 'GATE_SP2L_GOLD_ONLY_RETEST_V1.txt').exists():
        raise RuntimeError('SPEC and GATE must predate execution')
    report = OUT / 'REPORT_SP2L_GOLD_ONLY_RETEST_V1.txt'
    if report.exists():
        raise FileExistsError(report)
    df = base.load(TAPE)
    results = []
    trade_frames = []
    for label, cost in (('XAUUSD_CFD_0.41', 0.41), ('GC_PROXY_0.146', 0.146)):
        trades, funnel = base.run_c(df, 'GOLD', cost)
        stats = base.stats(trades)
        extra = detail(trades)
        years = stats['yearly']
        passed = bool(stats['n'] >= 150 and stats['mean'] > 0 and stats['sharpe'] > 0
                      and stats['mbb_p'] <= 0.05
                      and all(years.get(y, {}).get('net', 0) > 0 for y in ('2024', '2025')))
        results.append({'scenario': label, 'cost': cost, 'funnel': funnel,
                        'stats': stats, 'detail': extra, 'gate_pass': passed})
        frame = pd.DataFrame(trades)
        frame.insert(0, 'scenario', label)
        trade_frames.append(frame)
        print(label, stats['n'], stats['mean'], stats['mbb_p'], passed, flush=True)
    pd.concat(trade_frames, ignore_index=True).to_csv(OUT / 'trades.csv', index=False)
    meta = {'study': 'sp2l_gold_only_retest_2026_09_26_v1',
            'tape': str(TAPE.relative_to(ROOT)), 'tape_sha256': base.digest(TAPE),
            'engine_sha256': base.digest(Path(base.__file__)),
            'script_sha256': base.digest(Path(__file__)),
            'rows': len(df),
            'first_utc': pd.Timestamp(df.time.iloc[0], unit='s', tz='UTC').isoformat(),
            'last_utc': pd.Timestamp(df.time.iloc[-1], unit='s', tz='UTC').isoformat(),
            'bootstrap': {'replicates': base.BOOT, 'block': base.BLOCK, 'seed': base.SEED},
            'results': results}
    (OUT / 'run_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    lines = ['SP2L GOLD ONLY RETEST V1 - REPORT',
             f"Tape: {meta['tape']} | {meta['rows']} M1 rows | {meta['first_utc']} through {meta['last_utc']}",
             f"Tape SHA256: {meta['tape_sha256']}",
             'Bid-only XAUUSD proxy; corrected H56 transfer engine; full-tape exploratory test.',
             'MBB uses 10000 trade-order moving-block samples, block length 5, seed 20260926.', '']
    for r in results:
        s, d = r['stats'], r['detail']
        lines += [f"{r['scenario']} | cost {r['cost']:.3f} | gate {'PASS' if r['gate_pass'] else 'FAIL'}",
                  f"Trades {s['n']}; net mean {s['mean']:.6f}; net sum {s['total']:.3f} gold units; per-trade mean/std {s['sharpe']:.5f}; MBB P(mean<=0) {s['mbb_p']:.4f}",
                  f"Exits {s['exits']}; same-minute {s['same_minute_share']:.2%}; max equity DD {d['max_drawdown_gold_units']:.3f} gold units",
                  f"Sides {d['sides']}",
                  f"Annual NY entry cohorts {s['yearly']}",
                  f"Funnel {r['funnel']}", '']
    lines += ['Limitations: Constant cost floor and bid-only M1 OHLC; same-minute order path is unknown and extreme-entry fills may be optimistic.',
              'The short entry-minute stop bug in the original H56 tester is corrected in this transfer engine. This is not a direct rerun of its original implementation.',
              'The full tape has informed prior research; passes are exploratory, not walk-forward confirmation.',
              'GC proxy is not native CME GC data or Lucid execution; actual spreads, market depth, fees, slippage, fills and prop-rule compliance remain unmeasured.']
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
