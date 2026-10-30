"""US500 M15 Williams %R long-only backtest; see frozen SPEC_WILLIAMS.txt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from unseen_asset_saleh_donchian_phase1 import supertrend

ROOT = Path(__file__).resolve().parents[1]
TAPE = ROOT / 'data/results/vwap_2021_2026/us500tick_m1.csv'
OUT = ROOT / 'data/results/youtube_williams_us500'
COST, STOP_DISTANCE, TARGET_DISTANCE = 0.51, 10.0, 50.0


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_tape():
    frame = pd.read_csv(TAPE, usecols=['time', 'open', 'high', 'low', 'close'])
    if not frame.time.is_monotonic_increasing or not frame.time.is_unique:
        raise AssertionError('Tape timestamps must be sorted and unique')
    if not np.isfinite(frame[['open', 'high', 'low', 'close']].to_numpy()).all():
        raise AssertionError('Non-finite tape OHLC')
    return {k: frame[k].to_numpy(np.int64 if k == 'time' else float)
            for k in ('time', 'open', 'high', 'low', 'close')}


def continuous_m15(m):
    slots = m['time']//900
    frame = pd.DataFrame({'key': slots, 'o': m['open'], 'h': m['high'],
                          'l': m['low'], 'c': m['close']})
    grouped = frame.groupby('key', sort=True).agg(
        o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    keys = np.arange(int(slots[0]), int(slots[-1])+1, dtype=np.int64)
    q = grouped.reindex(keys).ffill()
    if not np.isfinite(q.to_numpy()).all():
        raise AssertionError('Continuous grid has non-finite OHLC')
    real = np.isin(keys, grouped.index.to_numpy(np.int64))
    return keys, q, real


def signals(grid):
    keys, q, real = grid
    high, low, close = (q[k].to_numpy(float) for k in ('h', 'l', 'c'))
    st_line, _ = supertrend(high, low, close, 10, 3.0)
    hh = pd.Series(high).rolling(10, min_periods=10).max().to_numpy(float)
    ll = pd.Series(low).rolling(10, min_periods=10).min().to_numpy(float)
    width = hh-ll
    wr = np.full(len(keys), np.nan)
    valid = np.isfinite(width) & (width > 0)
    wr[valid] = -100.0*(hh[valid]-close[valid])/width[valid]
    candidate = np.zeros(len(keys), bool)
    i = np.arange(10, len(keys))
    candidate[i] = (real[i] & (close[i-1] > st_line[i-1]) &
                    np.isfinite(wr[i-1]) & (wr[i-1] <= -90.0))
    return candidate, wr, st_line


def replay(m, grid, candidate):
    t, o, h, l, c = (m[k] for k in ('time', 'open', 'high', 'low', 'close'))
    keys, q, real = grid
    left = np.searchsorted(t, keys*900)
    right = np.searchsorted(t, (keys+1)*900)
    rows = []
    busy = no_real = same_m1_ties = 0
    occupied_until = -1
    for i in np.flatnonzero(candidate):
        if i <= occupied_until:
            busy += 1
            continue
        if left[i] == right[i]:
            no_real += 1
            continue
        entry_ix = int(left[i])
        entry_bid = float(o[entry_ix])
        entry_ask = entry_bid+COST
        stop_bid = entry_ask-STOP_DISTANCE
        target_bid = entry_ask+TARGET_DISTANCE
        exit_ix = -1
        exit_bid = float('nan')
        reason = ''
        gap = tie = False
        for j in range(i, len(keys)):
            a, z = int(left[j]), int(right[j])
            if a == z:
                continue
            stop_at = np.flatnonzero(l[a:z] <= stop_bid)
            target_at = np.flatnonzero(h[a:z] >= target_bid)
            s = int(stop_at[0]) if len(stop_at) else z-a
            u = int(target_at[0]) if len(target_at) else z-a
            if s == z-a and u == z-a:
                continue
            if s <= u:
                exit_ix = a+s
                exit_bid = min(stop_bid, float(o[exit_ix]))
                reason = 'STOP_LOSS'
                gap = bool(o[exit_ix] < stop_bid)
                tie = bool(s == u)
                same_m1_ties += int(tie)
            else:
                exit_ix = a+u
                exit_bid = max(target_bid, float(o[exit_ix]))
                reason = 'TAKE_PROFIT'
            break
        if exit_ix < 0:
            exit_ix = len(t)-1
            exit_bid = float(c[-1])
            reason = 'OPEN_END'
            j = len(keys)-1
        occupied_until = j
        entry_utc = pd.Timestamp(int(t[entry_ix]), unit='s', tz='UTC')
        exit_utc = pd.Timestamp(int(t[exit_ix]), unit='s', tz='UTC')
        entry_ny = entry_utc.tz_convert('America/New_York')
        exit_ny = exit_utc.tz_convert('America/New_York')
        marked_net = exit_bid-entry_ask
        rows.append({'entry_time': int(t[entry_ix]), 'exit_time': int(t[exit_ix]),
                     'entry_bid': entry_bid, 'entry_ask': entry_ask,
                     'initial_stop_bid': stop_bid, 'target_bid': target_bid,
                     'exit_bid': exit_bid, 'exit_type': reason,
                     'cost': COST, 'net': marked_net if reason != 'OPEN_END' else np.nan,
                     'open_end_mark_net': marked_net if reason == 'OPEN_END' else np.nan,
                     'adverse_stop_gap': gap, 'same_m1_tie': tie,
                     'same_m1_exit': bool(entry_ix == exit_ix),
                     'filled_signal_bar': bool(not real[i-1]),
                     'ny_entry_year': int(entry_ny.year),
                     'ny_overnight': bool(entry_ny.date() != exit_ny.date()),
                     'hold_hours': float((t[exit_ix]-t[entry_ix])/3600)})
    return rows, {'candidates': int(candidate.sum()), 'busy': busy,
                  'no_real': no_real, 'same_m1_ties': same_m1_ties}


def summarize(rows):
    closed = [r for r in rows if r['exit_type'] != 'OPEN_END']
    x = np.array([r['net'] for r in closed], float)
    n = len(x)
    mean = float(x.mean()) if n else float('nan')
    sd = float(x.std(ddof=1)) if n > 1 else float('nan')
    sharpe = mean/sd if sd > 0 else float('nan')
    yearly = {}
    for r in closed:
        y = str(r['ny_entry_year'])
        v = yearly.setdefault(y, {'trades': 0, 'net': 0.0})
        v['trades'] += 1
        v['net'] += r['net']
    curve = np.r_[0.0, np.cumsum(x)]
    return {'entries': len(rows), 'closed_trades': n,
            'open_end': len(rows)-n,
            'open_end_mark_net': sum(r['open_end_mark_net'] for r in rows
                                     if r['exit_type'] == 'OPEN_END'),
            'net_expectancy': mean, 'net_total': float(x.sum()),
            'net_sharpe': sharpe, 'gate_pass': bool(n >= 50 and sharpe > 0),
            'win_rate': float((x > 0).mean()) if n else float('nan'),
            'stop_losses': sum(r['exit_type'] == 'STOP_LOSS' for r in rows),
            'take_profits': sum(r['exit_type'] == 'TAKE_PROFIT' for r in rows),
            'adverse_stop_gaps': sum(r['adverse_stop_gap'] for r in rows),
            'same_m1_exits': sum(r['same_m1_exit'] for r in rows),
            'filled_signal_bar_entries': sum(r['filled_signal_bar'] for r in rows),
            'ny_overnight_entries': sum(r['ny_overnight'] for r in rows),
            'max_closed_trade_drawdown_points': float(
                np.max(np.maximum.accumulate(curve)-curve)),
            'yearly': yearly}


def main():
    if not (OUT/'SPEC_WILLIAMS.txt').exists() or not (OUT/'GATE_WILLIAMS.txt').exists():
        raise RuntimeError('Predeclared SPEC and GATE must exist before execution')
    report = OUT/'REPORT_WILLIAMS.txt'
    if report.exists():
        raise FileExistsError('Historical report is immutable; use a named correction')
    m = load_tape()
    grid = continuous_m15(m)
    candidate, wr, st_line = signals(grid)
    rows, funnel = replay(m, grid, candidate)
    stats = summarize(rows)
    pd.DataFrame(rows).to_csv(OUT/'trades.csv', index=False)
    first = pd.Timestamp(int(m['time'][0]), unit='s', tz='UTC')
    last = pd.Timestamp(int(m['time'][-1]), unit='s', tz='UTC')
    meta = {'study': 'youtube_williams_us500',
            'tape': str(TAPE.relative_to(ROOT)), 'tape_sha256': digest(TAPE),
            'script_sha256': digest(Path(__file__)),
            'm1_rows': len(m['time']), 'first_utc': first.isoformat(),
            'last_utc': last.isoformat(), 'm15_bars': len(grid[0]),
            'm15_real': int(grid[2].sum()), 'cost': COST,
            'stop_distance': STOP_DISTANCE, 'target_distance': TARGET_DISTANCE,
            'funnel': funnel, 'stats': stats}
    (OUT/'run_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    lines = ['US500 M15 WILLIAMS %R MEAN-REVERSION - REPORT',
             f"GATE: {'PASS' if stats['gate_pass'] else 'FAIL'} (>=50 closed trades and Net Sharpe >0)",
             f"Tape: {meta['tape']}; {len(m['time'])} M1 rows; {first.isoformat()} through {last.isoformat()}",
             f"Tape SHA256: {meta['tape_sha256']}",
             f"Continuous M15 grid: {meta['m15_bars']} bars, {meta['m15_real']} real",
             'Long ask entry=bid open+0.51; stop=ask entry-10.0; TP=ask entry+50.0; M1 stop priority; no time limit.', '',
             f"Trade Count (closed): {stats['closed_trades']}; total entries {stats['entries']}; OPEN_END {stats['open_end']}",
             f"Net Expectancy: {stats['net_expectancy']:.6f} index points/trade; Net Total: {stats['net_total']:.3f} index points",
             f"Net Sharpe: {stats['net_sharpe']:.6f} (unannualized per trade); Win Rate: {stats['win_rate']:.2%}",
             f"Stop Losses: {stats['stop_losses']}; Take Profits: {stats['take_profits']}",
             f"Adverse stop gaps: {stats['adverse_stop_gaps']}; same-M1 exits: {stats['same_m1_exits']}; M1 stop/TP ties: {funnel['same_m1_ties']}",
             f"Entries following filled M15 signal bars: {stats['filled_signal_bar_entries']}; NY overnight entries: {stats['ny_overnight_entries']}",
             f"OPEN_END last-close mark (excluded from closed stats): {stats['open_end_mark_net']:.3f} points; max closed-trade drawdown: {stats['max_closed_trade_drawdown_points']:.3f} points",
             f"Funnel: {funnel}",
             'Annual NY entry cohorts: '+json.dumps(stats['yearly'], sort_keys=True), '',
             'Limits: Full-tape retrospective bid-only proxy with a fixed 0.51 CFD spread. M1 OHLC cannot establish exact bid/ask fills or intraminute order; stop wins ties.',
             'No time limit permits overnight/weekend holdings, but swap was not measured and is excluded. No FTMO account-dollar sizing, equity-limit simulation, or walk-forward confirmation.']
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f"closed={stats['closed_trades']} mean={stats['net_expectancy']:.5f} "
          f"sharpe={stats['net_sharpe']:.5f} win={stats['win_rate']:.2%} "
          f"stop={stats['stop_losses']} tp={stats['take_profits']} "
          f"open_end={stats['open_end']} gate={stats['gate_pass']}", flush=True)


if __name__ == '__main__':
    main()
