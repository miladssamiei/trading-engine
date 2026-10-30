"""Gold 1H bias / 1M continuation backtest; frozen SPEC_YOUTUBE.txt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from unseen_asset_saleh_donchian_phase1 import wilder_atr

ROOT = Path(__file__).resolve().parents[1]
TAPE = ROOT / 'data/results/xauusd_onboarding/xauusdtick_m1.csv'
OUT = ROOT / 'data/results/youtube_1h_1m'
COST = 0.41


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_tape():
    df = pd.read_csv(TAPE, usecols=['time', 'open', 'high', 'low', 'close'])
    if not df.time.is_monotonic_increasing or not df.time.is_unique:
        raise AssertionError('M1 time must be unique and sorted')
    if not np.isfinite(df[['open', 'high', 'low', 'close']].to_numpy()).all():
        raise AssertionError('Non-finite OHLC')
    return {k: df[k].to_numpy(np.int64 if k == 'time' else float)
            for k in ('time', 'open', 'high', 'low', 'close')}


def hour_grid(m):
    slots = m['time']//3600
    raw = pd.DataFrame({'key': slots, 'o': m['open'], 'h': m['high'],
                        'l': m['low'], 'c': m['close']})
    grouped = raw.groupby('key', sort=True).agg(
        o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    keys = np.arange(int(slots[0]), int(slots[-1])+1, dtype=np.int64)
    q = grouped.reindex(keys).ffill()
    if not np.isfinite(q.to_numpy()).all():
        raise AssertionError('Hourly grid has an unfilled leading bar')
    real = np.isin(keys, grouped.index.to_numpy(np.int64))
    return keys, q, real


def bias_from_prior_hour(grid):
    keys, q, real = grid
    o, h, l, c = (q[k].to_numpy(float) for k in ('o', 'h', 'l', 'c'))
    atr = wilder_atr(h, l, c, 14)
    body = np.abs(c-o)
    strong = (body >= 0.5*atr) & (body <= 2.5*atr)
    bull = (c > o) & strong & (h-np.maximum(o, c) < 0.5*body)
    bear = (c < o) & strong & (np.minimum(o, c)-l < 0.5*body)
    side = np.zeros(len(keys), np.int8)
    i = np.arange(14, len(keys))
    valid = real[i] & real[i-1] & np.isfinite(atr[i-1])
    side[i] = np.where(valid & bull[i-1], 1,
                       np.where(valid & bear[i-1], -1, 0))
    return side, o, atr


def ny_clock(t):
    ny = pd.to_datetime(t, unit='s', utc=True).tz_convert('America/New_York')
    wall = np.asarray(ny.hour*60+ny.minute, dtype=np.int16)
    day = np.asarray(ny.year*10000+ny.month*100+ny.day, dtype=np.int32)
    return ny, wall, day


def confirmed_pivots(m, hour, day):
    h, l = m['high'], m['low']
    n = len(h)
    high = np.zeros(n, bool)
    low = np.zeros(n, bool)
    p = slice(2, n-2)
    same = (hour[:-4] == hour[4:]) & (day[:-4] == day[4:])
    high[p] = (same & (h[p] > h[:-4]) & (h[p] > h[1:-3]) &
               (h[p] > h[3:-1]) & (h[p] > h[4:]))
    low[p] = (same & (l[p] < l[:-4]) & (l[p] < l[1:-3]) &
              (l[p] < l[3:-1]) & (l[p] < l[4:]))
    both = high & low
    high[both] = False
    low[both] = False
    return high, low


def signal_candidates(m, grid, bias, hourly_open, wall, day):
    t, h, l, c = (m[k] for k in ('time', 'high', 'low', 'close'))
    hour = t//3600
    piv_high, piv_low = confirmed_pivots(m, hour, day)
    candidates = []
    funnel = {'confirmed_highs': int(piv_high.sum()),
              'confirmed_lows': int(piv_low.sum()),
              'cross_signals': 0, 'next_bar_unavailable': 0}
    # Generate all candidates before occupancy and daily-limit filtering.
    first_hour = int(grid[0][0])
    last_hour = -1
    low_first = high_after_low = high_first = low_after_high = -1
    used_buy = used_sell = None
    for j in np.flatnonzero((wall >= 120) & (wall < 660)):
        q = int(hour[j])-first_hour
        if q < 0 or q >= len(bias):
            continue
        if hour[j] != last_hour:
            last_hour = int(hour[j])
            low_first = high_after_low = high_first = low_after_high = -1
            used_buy = used_sell = None
        p = j-2
        if p >= 2 and piv_low[p]:
            low_first, high_after_low = p, -1
            if high_first >= 0 and p > high_first:
                low_after_high = p
        if p >= 2 and piv_high[p]:
            high_first, low_after_high = p, -1
            if low_first >= 0 and p > low_first:
                high_after_low = p
        if j == 0 or bias[q] == 0:
            continue
        side, stop_level, pair = 0, 0.0, None
        if (bias[q] == 1 and low_first >= 0 and high_after_low > low_first and
                c[j] > hourly_open[q] and h[j] > h[high_after_low] and
                h[j-1] <= h[high_after_low]):
            pair = (low_first, high_after_low)
            if pair != used_buy:
                side, stop_level = 1, float(l[low_first])
                used_buy = pair
        elif (bias[q] == -1 and high_first >= 0 and low_after_high > high_first and
              c[j] < hourly_open[q] and l[j] < l[low_after_high] and
              l[j-1] >= l[low_after_high]):
            pair = (high_first, low_after_high)
            if pair != used_sell:
                side, stop_level = -1, float(h[high_first])
                used_sell = pair
        if side == 0:
            continue
        funnel['cross_signals'] += 1
        k = j+1
        if (k >= len(t) or t[k]-t[j] != 60 or hour[k] != hour[j] or
                day[k] != day[j] or wall[k] < 120 or wall[k] >= 660):
            funnel['next_bar_unavailable'] += 1
            continue
        candidates.append((k, side, stop_level, j, pair))
    return candidates, funnel


def session_exit_indices(day, wall):
    changes = np.r_[0, np.flatnonzero(day[1:] != day[:-1])+1, len(day)]
    result = np.full(len(day), -1, dtype=np.int64)
    for a, z in zip(changes[:-1], changes[1:]):
        valid = np.flatnonzero(wall[a:z] < 960)
        if len(valid):
            result[a:z] = a+valid[-1]
    return result


def walk_trade(m, wall, day, close_index, entry_ix, side, structural_stop):
    t, o, h, l, c = (m[k] for k in ('time', 'open', 'high', 'low', 'close'))
    entry_bid = float(o[entry_ix])
    structural_distance = side*(entry_bid-structural_stop)
    if structural_distance < 0.50:
        return None
    entry_exec = entry_bid+COST if side == 1 else entry_bid
    stop = structural_stop if side == 1 else structural_stop+COST
    risk = entry_exec-stop if side == 1 else stop-entry_exec
    if risk <= 0:
        raise AssertionError('Non-positive all-in risk')
    target = entry_exec+2*risk if side == 1 else entry_exec-2*risk
    threshold = entry_exec+1.5*risk if side == 1 else entry_exec-1.5*risk
    last = int(close_index[entry_ix])
    if last < entry_ix:
        return None
    break_even = False
    for x in range(entry_ix, last+1):
        if side == 1:
            stop_hit = l[x] <= stop
            target_hit = h[x] >= target
            threshold_hit = h[x] >= threshold
        else:
            stop_hit = h[x]+COST >= stop
            target_hit = l[x]+COST <= target
            threshold_hit = l[x]+COST <= threshold
        if stop_hit:
            if side == 1:
                exit_bid = min(stop, float(o[x]))
            else:
                exit_bid = max(stop, float(o[x])+COST)-COST
            reason = 'BREAK_EVEN' if break_even else 'STOP_LOSS'
            break
        if target_hit:
            if side == 1:
                exit_bid = max(target, float(o[x]))
            else:
                exit_bid = min(target, float(o[x])+COST)-COST
            reason = 'TARGET_2R'
            break
        if x == last:
            exit_bid = float(c[x])
            reason = 'TIME_EXIT'
            break
        if threshold_hit and not break_even:
            stop = entry_exec
            break_even = True
    net = side*(exit_bid-entry_bid)-COST
    entry_utc = pd.Timestamp(int(t[entry_ix]), unit='s', tz='UTC')
    return {'side': side, 'entry_time': int(t[entry_ix]), 'exit_time': int(t[x]),
            'ny_day': int(day[entry_ix]), 'entry_bid': entry_bid,
            'entry_exec': entry_exec, 'exit_bid': exit_bid,
            'structural_stop': structural_stop, 'final_stop': stop,
            'initial_risk_all_in': risk, 'target_exec': target,
            'exit_type': reason, 'net': net, 'net_r': net/risk,
            'entry_month_utc': entry_utc.strftime('%Y-%m'),
            'entry_year': int(entry_utc.year),
            'exit_at_1559': bool(wall[x] == 959 and reason == 'TIME_EXIT'),
            'time_exit_before_1559': bool(wall[x] < 959 and reason == 'TIME_EXIT'),
            'same_m1_exit': bool(x == entry_ix)}


def replay(m, wall, day, candidates):
    close_index = session_exit_indices(day, wall)
    rows = []
    entered_per_day = {}
    occupied_through = -1
    funnel = {'busy': 0, 'daily_cap': 0, 'risk_below_0_50': 0,
              'no_session_close': 0, 'entered': 0}
    t = m['time']
    for k, side, stop_level, signal_ix, pair in candidates:
        if k <= occupied_through:
            funnel['busy'] += 1
            continue
        d = int(day[k])
        if entered_per_day.get(d, 0) >= 2:
            funnel['daily_cap'] += 1
            continue
        trade = walk_trade(m, wall, day, close_index, k, side, stop_level)
        if trade is None:
            if close_index[k] < k:
                funnel['no_session_close'] += 1
            else:
                funnel['risk_below_0_50'] += 1
            continue
        trade['signal_time'] = int(t[signal_ix])
        trade['pivot_pair'] = f'{pair[0]}:{pair[1]}'
        rows.append(trade)
        occupied_through = int(np.searchsorted(t, trade['exit_time']))
        entered_per_day[d] = entered_per_day.get(d, 0)+1
        funnel['entered'] += 1
    return rows, funnel


def summarize(rows, months):
    x = np.array([r['net'] for r in rows], float)
    rr = np.array([r['net_r'] for r in rows], float)
    n = len(rows)
    mean = float(x.mean()) if n else float('nan')
    sd = float(x.std(ddof=1)) if n > 1 else float('nan')
    sharpe = mean/sd if sd > 0 else float('nan')
    counts = {name: sum(r['exit_type'] == name for r in rows)
              for name in ('TARGET_2R', 'BREAK_EVEN', 'STOP_LOSS', 'TIME_EXIT')}
    annual = {}
    monthly = {}
    for r in rows:
        year = str(r['entry_year'])
        a = annual.setdefault(year, {'trades': 0, 'net': 0.0, 'net_r': 0.0})
        a['trades'] += 1
        a['net'] += r['net']
        a['net_r'] += r['net_r']
        month = r['entry_month_utc']
        monthly[month] = monthly.get(month, 0.0)+r['net_r']
    curve = np.r_[0.0, np.cumsum(rr)]
    return {'trade_count': n, 'net_expectancy': mean, 'net_total': float(x.sum()),
            'net_sharpe': sharpe, 'gate_pass': bool(n >= 50 and sharpe > 0),
            'win_rate': float((x > 0).mean()) if n else float('nan'),
            'exit_counts': counts,
            'exit_ratios': {k: v/n if n else None for k, v in counts.items()},
            'total_r': float(rr.sum()), 'total_tape_months': months,
            'monthly_r_velocity': float(rr.sum()/months),
            'exceeds_5_33r_month': bool(rr.sum()/months > 5.33),
            'r_max_drawdown': float(np.max(np.maximum.accumulate(curve)-curve)),
            'annual': annual, 'monthly_r': dict(sorted(monthly.items())),
            'time_exit_before_1559': sum(r['time_exit_before_1559'] for r in rows),
            'same_m1_exit': sum(r['same_m1_exit'] for r in rows)}


def main():
    if not (OUT/'SPEC_YOUTUBE.txt').exists() or not (OUT/'GATE_YOUTUBE.txt').exists():
        raise RuntimeError('SPEC and GATE must precede the evaluator')
    report = OUT/'REPORT_YOUTUBE.txt'
    if report.exists():
        raise FileExistsError('Frozen REPORT exists; create a named correction study')
    m = load_tape()
    t = m['time']
    first = pd.Timestamp(int(t[0]), unit='s', tz='UTC')
    last = pd.Timestamp(int(t[-1]), unit='s', tz='UTC')
    months = (last.year-first.year)*12+last.month-first.month+1
    hourly = hour_grid(m)
    bias, hourly_open, atr = bias_from_prior_hour(hourly)
    ny, wall, day = ny_clock(t)
    candidates, signal_funnel = signal_candidates(m, hourly, bias, hourly_open, wall, day)
    rows, replay_funnel = replay(m, wall, day, candidates)
    stats = summarize(rows, months)
    pd.DataFrame(rows).to_csv(OUT/'trades.csv', index=False)
    meta = {'study': 'youtube_1h_1m', 'tape': str(TAPE.relative_to(ROOT)),
            'tape_sha256': digest(TAPE), 'script_sha256': digest(Path(__file__)),
            'm1_rows': len(t), 'first_utc': first.isoformat(),
            'last_utc': last.isoformat(), 'h1_bars': len(hourly[0]),
            'h1_real': int(hourly[2].sum()), 'cfd_round_trip_cost': COST,
            'swap': 0.0, 'signal_funnel': signal_funnel,
            'execution_funnel': replay_funnel, 'stats': stats}
    (OUT/'run_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    c = stats['exit_counts']
    v = stats['exit_ratios']
    lines = ['GOLD 1H/1M CONTINUATION - REPORT',
             f"GATE: {'PASS' if stats['gate_pass'] else 'FAIL'} (>=50 trades and Net Sharpe >0)",
             f"+5.33R/month pace: {'EXCEEDED' if stats['exceeds_5_33r_month'] else 'NOT MET'}",
             f"Tape {meta['tape']}; {len(t)} M1 rows; {first.isoformat()} through {last.isoformat()}; {months} inclusive UTC calendar months",
             f"Tape SHA256 {meta['tape_sha256']}",
             f"Continuous H1 grid {len(hourly[0])} bars, {meta['h1_real']} real; NY entry [02:00,11:00); last session bar exit before 16:00 ET",
             f"CFD round-trip cost {COST:.2f} embedded once in fills; swaps {meta['swap']:.1f}.", '',
             f"Trade Count: {stats['trade_count']}",
             f"Net Expectancy: {stats['net_expectancy']:.6f} gold units/trade; Net Total: {stats['net_total']:.3f} gold units",
             f"Net Sharpe: {stats['net_sharpe']:.6f} (unannualized per trade); Win Rate: {stats['win_rate']:.2%}",
             f"Exit ratio target/break-even/stop/time: {c['TARGET_2R']}/{c['BREAK_EVEN']}/{c['STOP_LOSS']}/{c['TIME_EXIT']} counts; {v['TARGET_2R']:.2%}/{v['BREAK_EVEN']:.2%}/{v['STOP_LOSS']:.2%}/{v['TIME_EXIT']:.2%}",
             f"Total net R: {stats['total_r']:.3f}; Monthly R-Velocity: {stats['monthly_r_velocity']:.4f} R/month; R max closed-trade drawdown: {stats['r_max_drawdown']:.3f}",
             f"Time exits before 15:59 due missing exact bar: {stats['time_exit_before_1559']}; same-M1 exits: {stats['same_m1_exit']}",
             f"Signal funnel: {signal_funnel}",
             f"Execution funnel: {replay_funnel}",
             'Annual entry cohorts: '+json.dumps(stats['annual'], sort_keys=True),
             'UTC calendar-month entry-cohort R: '+json.dumps(stats['monthly_r'], sort_keys=True), '',
             'Limits: Full-tape retrospective research on bid-only M1 bars; 5-bar pivots confirmed two bars late and orders enter next M1 open. Stop takes priority on M1 ties. Fixed 0.41 spread excludes variable quotes/slippage.',
             'Session-flat exits justify zero assumed swap here; this does not establish actual broker fills, account sizing, FTMO daily/overall equity compliance, or a 45-day pass probability.']
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f"trades={stats['trade_count']} mean={stats['net_expectancy']:.5f} "
          f"sharpe={stats['net_sharpe']:.5f} win={stats['win_rate']:.2%} "
          f"R/month={stats['monthly_r_velocity']:.4f} gate={stats['gate_pass']}", flush=True)


if __name__ == '__main__':
    main()
