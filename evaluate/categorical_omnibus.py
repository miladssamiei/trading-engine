"""Six fixed categorical regime transfers; see frozen SPEC_CATEGORICAL.txt."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from unseen_asset_saleh_donchian_phase1 import wilder_atr, wilder_adx

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/results/categorical_omnibus'
ASSETS = (
    ('GC', 'data/results/xauusd_onboarding/xauusdtick_m1.csv', 0.41, 0.13),
    ('NQ', 'data/results/virgin_assets/ustech_m1.csv', 1.00, 0.39),
    ('ES', 'data/results/vwap_2021_2026/us500tick_m1.csv', 0.51, 0.30),
)
SEED, BOOT, BLOCK = 20260927, 10000, 5


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_tape(path):
    df = pd.read_csv(path, usecols=['time', 'open', 'high', 'low', 'close'])
    if not df.time.is_monotonic_increasing or not df.time.is_unique:
        raise AssertionError('M1 time must be sorted and unique')
    if not np.isfinite(df[['open', 'high', 'low', 'close']].to_numpy()).all():
        raise AssertionError('Non-finite M1 OHLC')
    return {k: df[k].to_numpy(np.int64 if k == 'time' else float)
            for k in ('time', 'open', 'high', 'low', 'close')}


def continuous_h1(m):
    slot = m['time']//3600
    frame = pd.DataFrame({'key': slot, 'o': m['open'], 'h': m['high'],
                          'l': m['low'], 'c': m['close']})
    grouped = frame.groupby('key', sort=True).agg(
        o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    keys = np.arange(int(slot[0]), int(slot[-1])+1, dtype=np.int64)
    q = grouped.reindex(keys).ffill()
    if not np.isfinite(q.to_numpy()).all():
        raise AssertionError('Unfilled H1 grid')
    real = np.isin(keys, grouped.index.to_numpy(np.int64))
    return keys, q, real


def signal_table(grid):
    keys, q, real = grid
    h, l, c = (q[k].to_numpy(float) for k in ('h', 'l', 'c'))
    atr = wilder_atr(h, l, c, 14)
    adx = wilder_adx(pd.Series(h), pd.Series(l), pd.Series(c), 14)
    sma = pd.Series(c).rolling(20, min_periods=20).mean().to_numpy(float)
    std = pd.Series(c).rolling(20, min_periods=20).std(ddof=0).to_numpy(float)
    upper, lower = sma+2*std, sma-2*std
    dc_high = pd.Series(h).rolling(20, min_periods=20).max().to_numpy(float)
    dc_low = pd.Series(l).rolling(20, min_periods=20).min().to_numpy(float)
    regime = np.zeros(len(keys), np.int8)  # 1 consolidation, 2 directional
    side = np.zeros(len(keys), np.int8)
    i = np.arange(21, len(keys))
    p = i-1
    valid = real[i] & np.isfinite(atr[p]) & (atr[p] > 0) & np.isfinite(adx[p])
    consolidate = valid & (adx[p] < 25) & np.isfinite(sma[p]) & np.isfinite(std[p])
    directional = valid & (adx[p] >= 25) & np.isfinite(dc_high[i-2]) & np.isfinite(dc_low[i-2])
    long_mr = consolidate & (c[p] < lower[p])
    short_mr = consolidate & (c[p] > upper[p])
    long_mo = directional & (c[p] > dc_high[i-2])
    short_mo = directional & (c[p] < dc_low[i-2])
    side[i] = np.where(long_mr | long_mo, 1,
                       np.where(short_mr | short_mo, -1, 0))
    regime[i] = np.where(long_mr | short_mr, 1,
                         np.where(long_mo | short_mo, 2, 0))
    if np.any((regime == 0) & (side != 0)):
        raise AssertionError('Unclassified signal')
    return {'side': side, 'regime': regime, 'atr': atr, 'adx': adx,
            'sma': sma, 'signal_on_filled_prior': ~np.r_[False, real[:-1]]}


def walk_trade(m, left, right, start_hour, side, regime, atr, target_mean,
               cost, prop):
    t, o, h, l, c = (m[k] for k in ('time', 'open', 'high', 'low', 'close'))
    entry_ix = int(left[start_hour])
    entry_bid = float(o[entry_ix])
    entry_exec = entry_bid+cost if side == 1 else entry_bid
    stop = entry_exec-side*atr
    if regime == 1:
        target = float(target_mean)
    else:
        target = entry_exec+side*1.5*atr
    if side*(target-entry_exec) <= 0:
        return None
    peak = 0.0
    peak_at_exit = 0.0
    reason = ''
    exit_bid = float('nan')
    exit_ix = -1
    tie = gap = False
    for j in range(start_hour, len(left)):
        a, z = int(left[j]), int(right[j])
        for x in range(a, z):
            if side == 1:
                stop_hit = l[x] <= stop
                target_hit = h[x] >= target
                adverse_net = float(l[x]-entry_exec)
                favorable_net = float(h[x]-entry_exec)
            else:
                stop_hit = h[x]+cost >= stop
                target_hit = l[x]+cost <= target
                adverse_net = float(entry_exec-h[x]-cost)
                favorable_net = float(entry_exec-l[x]-cost)
            guard_hit = bool(prop and adverse_net <= peak-6.0*atr)
            if stop_hit:
                if side == 1:
                    exit_bid = min(stop, float(o[x]))
                    gap = bool(o[x] < stop)
                else:
                    exit_bid = max(stop, float(o[x])+cost)-cost
                    gap = bool(o[x]+cost > stop)
                reason = 'STOP_LOSS'
                tie = bool(target_hit or guard_hit)
            elif guard_hit:
                threshold_net = peak-6.0*atr
                if side == 1:
                    level = entry_exec+threshold_net
                    exit_bid = min(level, float(o[x]))
                    gap = bool(o[x] < level)
                else:
                    ask_level = entry_exec-threshold_net
                    exit_bid = max(ask_level, float(o[x])+cost)-cost
                    gap = bool(o[x]+cost > ask_level)
                reason = 'PROP_LIQUIDATION'
                tie = bool(target_hit)
            elif target_hit:
                if side == 1:
                    exit_bid = max(target, float(o[x]))
                else:
                    exit_bid = min(target, float(o[x])+cost)-cost
                reason = 'TARGET'
            if reason:
                exit_ix = x
                peak_at_exit = peak
                break
            peak = max(peak, favorable_net)
        if reason:
            break
    if exit_ix < 0:
        exit_ix = len(t)-1
        exit_bid = float(c[-1])
        reason = 'OPEN_END'
        j = len(left)-1
        peak_at_exit = peak
    entry_utc = pd.Timestamp(int(t[entry_ix]), unit='s', tz='UTC')
    exit_utc = pd.Timestamp(int(t[exit_ix]), unit='s', tz='UTC')
    entry_ny = entry_utc.tz_convert('America/New_York')
    exit_ny = exit_utc.tz_convert('America/New_York')
    net = side*(exit_bid-entry_bid)-cost
    return {'regime': 'CONSOLIDATION' if regime == 1 else 'DIRECTIONAL',
            'side': side, 'entry_time': int(t[entry_ix]),
            'exit_time': int(t[exit_ix]), 'entry_bid': entry_bid,
            'entry_exec': entry_exec, 'entry_atr14': atr,
            'initial_stop_exec': stop, 'target_exec': target,
            'exit_bid': exit_bid, 'exit_type': reason,
            'cost': cost, 'net': net if reason != 'OPEN_END' else np.nan,
            'open_end_mark_net': net if reason == 'OPEN_END' else np.nan,
            'peak_open_equity': peak_at_exit, 'guard_distance': 6.0*atr if prop else np.nan,
            'same_m1_exit': bool(entry_ix == exit_ix), 'm1_tie': tie,
            'adverse_exit_gap': gap, 'entry_ny_year': int(entry_ny.year),
            'ny_overnight': bool(entry_ny.date() != exit_ny.date())}


def replay(m, grid, signals, asset, environment, cost):
    keys, q, real = grid
    t = m['time']
    left = np.searchsorted(t, keys*3600)
    right = np.searchsorted(t, (keys+1)*3600)
    rows = []
    occupied_until = -1
    funnel = {'candidates': int(np.count_nonzero(signals['side'])),
              'busy': 0, 'no_real': 0, 'invalid_target_after_gap': 0,
              'entered': 0, 'filled_prior_signals': 0}
    prop = environment == 'FUTURES'
    for i in np.flatnonzero(signals['side']):
        if i <= occupied_until:
            funnel['busy'] += 1
            continue
        if left[i] == right[i]:
            funnel['no_real'] += 1
            continue
        side = int(signals['side'][i])
        regime = int(signals['regime'][i])
        atr = float(signals['atr'][i-1])
        target_mean = float(signals['sma'][i-1])
        row = walk_trade(m, left, right, i, side, regime, atr,
                         target_mean, cost, prop)
        if row is None:
            funnel['invalid_target_after_gap'] += 1
            continue
        row['asset'] = asset
        row['environment'] = environment
        row['filled_prior_signal'] = bool(signals['signal_on_filled_prior'][i])
        funnel['filled_prior_signals'] += int(row['filled_prior_signal'])
        rows.append(row)
        occupied_until = int(row['exit_time']//3600-keys[0])
        funnel['entered'] += 1
    return rows, funnel


def mbb_p(x):
    n = len(x)
    if n < BLOCK:
        return float('nan')
    rng = np.random.default_rng(SEED)
    block_sums = np.convolve(x, np.ones(BLOCK), mode='valid')
    full, extra = divmod(n, BLOCK)
    partial_sums = (np.convolve(x, np.ones(extra), mode='valid')
                    if extra else None)
    bad = 0
    for offset in range(0, BOOT, 250):
        count = min(250, BOOT-offset)
        starts = rng.integers(0, n-BLOCK+1,
                              size=(count, full+int(extra > 0)))
        totals = block_sums[starts[:, :full]].sum(axis=1)
        if extra:
            totals += partial_sums[starts[:, full]]
        bad += int(np.count_nonzero(totals <= 0))
    return bad/BOOT


def summarize(rows):
    closed = [r for r in rows if r['exit_type'] != 'OPEN_END']
    x = np.array([r['net'] for r in closed], float)
    n = len(x)
    mean = float(x.mean()) if n else float('nan')
    sd = float(x.std(ddof=1)) if n > 1 else float('nan')
    sharpe = mean/sd if sd > 0 else float('nan')
    regimes = {}
    for name in ('CONSOLIDATION', 'DIRECTIONAL'):
        values = np.array([r['net'] for r in closed if r['regime'] == name], float)
        regimes[name] = {'trades': len(values), 'net_total': float(values.sum()),
                         'net_expectancy': float(values.mean()) if len(values) else None}
    yearly = {}
    for r in closed:
        year = str(r['entry_ny_year'])
        v = yearly.setdefault(year, {'trades': 0, 'net_total': 0.0})
        v['trades'] += 1
        v['net_total'] += r['net']
    return {'entries': len(rows), 'closed_trades': n,
            'open_end': len(rows)-n,
            'open_end_mark_net': sum(r['open_end_mark_net'] for r in rows
                                     if r['exit_type'] == 'OPEN_END'),
            'net_expectancy': mean, 'net_total': float(x.sum()),
            'net_sharpe': sharpe, 'mbb_p': mbb_p(x),
            'gate_pass': bool(n >= 30 and sharpe > 0),
            'regimes': regimes, 'yearly': yearly,
            'exits': {name: sum(r['exit_type'] == name for r in rows)
                      for name in ('STOP_LOSS', 'TARGET', 'PROP_LIQUIDATION', 'OPEN_END')},
            'm1_ties': sum(r['m1_tie'] for r in rows),
            'adverse_exit_gaps': sum(r['adverse_exit_gap'] for r in rows),
            'same_m1_exits': sum(r['same_m1_exit'] for r in rows),
            'ny_overnight_entries': sum(r['ny_overnight'] for r in rows)}


def main():
    if not (OUT/'SPEC_CATEGORICAL.txt').exists() or not (OUT/'GATE_CATEGORICAL.txt').exists():
        raise RuntimeError('SPEC and GATE must predate execution')
    report = OUT/'REPORT_CATEGORICAL.txt'
    if report.exists():
        raise FileExistsError('Frozen REPORT exists; use a new named study')
    results = []
    tapes = []
    for asset, rel, cfd_cost, futures_cost in ASSETS:
        path = ROOT/rel
        m = load_tape(path)
        grid = continuous_h1(m)
        table = signal_table(grid)
        first = pd.Timestamp(int(m['time'][0]), unit='s', tz='UTC')
        last = pd.Timestamp(int(m['time'][-1]), unit='s', tz='UTC')
        tapes.append({'asset': asset, 'path': rel, 'sha256': digest(path),
                      'm1_rows': len(m['time']), 'first_utc': first.isoformat(),
                      'last_utc': last.isoformat(), 'h1_bars': len(grid[0]),
                      'h1_real': int(grid[2].sum())})
        for environment, cost in (('CFD', cfd_cost), ('FUTURES', futures_cost)):
            rows, funnel = replay(m, grid, table, asset, environment, cost)
            stats = summarize(rows)
            pd.DataFrame(rows).to_csv(OUT/f'trades_{environment}_{asset}.csv', index=False)
            results.append({'asset': asset, 'environment': environment,
                            'cost': cost, 'funnel': funnel, 'stats': stats})
            print(f"{asset} {environment}: n={stats['closed_trades']} "
                  f"mean={stats['net_expectancy']:.5f} sharpe={stats['net_sharpe']:.5f} "
                  f"p={stats['mbb_p']:.4f} prop={stats['exits']['PROP_LIQUIDATION']} "
                  f"gate={stats['gate_pass']}", flush=True)
    meta = {'study': 'categorical_omnibus', 'date': '2026-09-27',
            'script_sha256': digest(Path(__file__)),
            'bootstrap': {'replicates': BOOT, 'block': BLOCK, 'seed': SEED},
            'tapes': tapes, 'overall_gate_pass': all(r['stats']['gate_pass'] for r in results),
            'results': results}
    (OUT/'run_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    lines = ['CATEGORICAL REGIME-SWITCHING OMNIBUS - REPORT',
             'OVERALL GATE: '+('PASS' if meta['overall_gate_pass'] else 'FAIL'),
             'Six independent bid-only M1 proxy runs; static structural stops in CFD, separate per-position 6xH1 ATR open-equity guard in futures.',
             'Net Sharpe is unannualized per trade. MBB p-value uses 10000 five-trade block resamples, seed 20260927.',
             'Gate per row: >=30 closed trades and Net Sharpe >0; MBB p is descriptive.', '']
    for tape in tapes:
        lines += [f"TAPE {tape['asset']}: {tape['path']} | {tape['m1_rows']} M1 rows | {tape['first_utc']} through {tape['last_utc']} | H1 {tape['h1_real']}/{tape['h1_bars']} real",
                  f"SHA256 {tape['sha256']}"]
    lines.append('')
    for result in results:
        s = result['stats']
        lines += [f"{result['asset']} / {result['environment']} | cost {result['cost']:.2f} | GATE {'PASS' if s['gate_pass'] else 'FAIL'}",
                  f"Closed trades {s['closed_trades']}; entries {s['entries']}; net expectancy {s['net_expectancy']:.6f} price units/trade; net total {s['net_total']:.3f}; Net Sharpe {s['net_sharpe']:.6f}; MBB p {s['mbb_p']:.4f}",
                  f"Consolidation: {s['regimes']['CONSOLIDATION']['trades']} trades, PnL {s['regimes']['CONSOLIDATION']['net_total']:.3f}, mean {s['regimes']['CONSOLIDATION']['net_expectancy']}",
                  f"Directional: {s['regimes']['DIRECTIONAL']['trades']} trades, PnL {s['regimes']['DIRECTIONAL']['net_total']:.3f}, mean {s['regimes']['DIRECTIONAL']['net_expectancy']}",
                  f"Exits {s['exits']}; exact PROP_LIQUIDATION count {s['exits']['PROP_LIQUIDATION']}",
                  f"M1 ties {s['m1_ties']}; adverse exit gaps {s['adverse_exit_gaps']}; same-M1 exits {s['same_m1_exits']}; NY overnight entries {s['ny_overnight_entries']}",
                  f"OPEN_END last-close mark excluded from stats {s['open_end_mark_net']:.3f}; funnel {result['funnel']}",
                  'Annual NY entry cohorts: '+json.dumps(s['yearly'], sort_keys=True), '']
    lines += ['Limitations: The futures costs/guard are applied to CFD-derived proxy M1 bars, not native CME execution. The guard is per position rather than account-wide, and M1 extrema have unknown ordering.',
              'Variable CFD spread, futures quotes/queue/rolls, swap/financing, sizing and full prop-account drawdown are unmeasured. Full-tape findings are exploratory, not walk-forward confirmation.']
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
