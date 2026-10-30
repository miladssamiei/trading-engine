"""Fixed M15/H1 gold breakout transfers; see frozen SPEC_GOLD_15M.txt."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from unseen_asset_kc_replication_phase1 import wilder_atr as kc_atr
from unseen_asset_saleh_donchian_phase1 import wilder_atr, wilder_adx, supertrend

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/results/gold_15m_acceleration'
TAPE = ROOT / 'data/results/xauusd_onboarding/xauusdtick_m1.csv'
COST = 0.41
SEED, BOOT, BLOCK = 20260926, 10000, 5


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def grid(m: dict, seconds: int):
    slots = m['time'] // seconds
    frame = pd.DataFrame({'slot': slots, 'o': m['open'], 'h': m['high'],
                          'l': m['low'], 'c': m['close']})
    grouped = frame.groupby('slot', sort=True).agg(
        o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    keys = np.arange(int(slots[0]), int(slots[-1]) + 1, dtype=np.int64)
    filled = grouped.reindex(keys).ffill()
    assert np.isfinite(filled.to_numpy()).all()
    real = np.isin(keys, grouped.index.to_numpy(np.int64))
    return keys, filled, real


def engines(g15, g1):
    keys, q, real = g15
    h, l, c = (q[k].to_numpy(float) for k in ('h', 'l', 'c'))
    ema29 = pd.Series(c).ewm(span=29, adjust=False).mean().to_numpy(float)
    ema29[:28] = np.nan
    atr29 = kc_atr(h, l, c, 29)
    atr14 = wilder_atr(h, l, c, 14)
    adx14 = wilder_adx(pd.Series(h), pd.Series(l), pd.Series(c), 14)
    hi20 = pd.Series(h).rolling(20, min_periods=20).max().to_numpy(float)
    lo20 = pd.Series(l).rolling(20, min_periods=20).min().to_numpy(float)

    hkeys, hourly, _ = g1
    hc = hourly.c.to_numpy(float)
    h_ema = pd.Series(hc).ewm(span=50, adjust=False).mean().to_numpy(float)
    h_ema[:49] = np.nan
    _, h_st = supertrend(*(hourly[k].to_numpy(float) for k in ('h', 'l', 'c')), 10, 3.0)
    completed_hour = np.searchsorted(hkeys, (keys * 900) // 3600 - 1)
    valid_hour = ((completed_hour >= 0) & (completed_hour < len(hkeys)))
    safe = np.clip(completed_hour, 0, len(hkeys)-1)
    valid_hour &= hkeys[safe] == (keys * 900) // 3600 - 1
    e_side = np.zeros(len(keys), np.int8)
    f_side = np.zeros(len(keys), np.int8)
    i = np.arange(2, len(keys))
    e_ok = (real[i] & valid_hour[i] & np.isfinite(ema29[i-1]) &
            np.isfinite(atr29[i-1]) & np.isfinite(h_ema[safe[i]]))
    e_side[i] = np.where(e_ok & (c[i-1] > ema29[i-1] + 2.77*atr29[i-1]) &
                         (hc[safe[i]] > h_ema[safe[i]]), 1,
                         np.where(e_ok & (c[i-1] < ema29[i-1] - 2.77*atr29[i-1]) &
                                  (hc[safe[i]] < h_ema[safe[i]]), -1, 0))
    f_ok = (real[i] & valid_hour[i] & np.isfinite(atr14[i-1]) &
            np.isfinite(adx14[i-1]) & (adx14[i-1] >= 20))
    f_side[i] = np.where(f_ok & (c[i-1] > hi20[i-2]) & (h_st[safe[i]] == 1), 1,
                         np.where(f_ok & (c[i-1] < lo20[i-2]) & (h_st[safe[i]] == -1), -1, 0))
    common = {'start': keys*900, 'end': (keys+1)*900, 'real': real, 'close': c}
    return [dict(common, name='E_KELTNER', side=e_side, atr=atr29, ema=ema29),
            dict(common, name='F_DONCHIAN', side=f_side, atr=atr14)]


def replay(m: dict, engine: dict):
    t, o, h, l, c = (m[k] for k in ('time', 'open', 'high', 'low', 'close'))
    starts, ends = engine['start'], engine['end']
    left, right = np.searchsorted(t, starts), np.searchsorted(t, ends)
    rows = []
    busy = no_real = 0
    occupied_until = -1
    for i in np.flatnonzero(engine['side']):
        if i <= occupied_until:
            busy += 1
            continue
        if left[i] == right[i]:
            no_real += 1
            continue
        side = int(engine['side'][i])
        entry_ix = int(left[i])
        entry_bid = float(o[entry_ix])
        prior_atr = float(engine['atr'][i-1])
        if not np.isfinite(prior_atr) or prior_atr <= 0:
            continue
        if engine['name'] == 'E_KELTNER':
            distance = max(abs(entry_bid - float(engine['ema'][i-1])), 1.5*prior_atr)
        else:
            distance = 4.0*prior_atr
        # E's short stop is bid-reference; F's short stop is ask-reference.
        stop = entry_bid - side*distance
        initial = stop
        initial_risk = distance+COST if side == 1 or engine['name'] == 'E_KELTNER' else distance
        if initial_risk <= 0:
            raise AssertionError('Initial all-in risk must be positive')
        found = False
        for j in range(i, len(starts)):
            a, z = int(left[j]), int(right[j])
            if a < z:
                if side == 1:
                    touches = np.flatnonzero(l[a:z] <= stop)
                elif engine['name'] == 'E_KELTNER':
                    touches = np.flatnonzero(h[a:z] >= stop)
                else:
                    touches = np.flatnonzero(h[a:z]+COST >= stop)
                if len(touches):
                    x = a+int(touches[0])
                    if side == 1:
                        exit_bid = min(stop, float(o[x]))
                    elif engine['name'] == 'E_KELTNER':
                        exit_bid = max(stop, float(o[x]))
                    else:
                        exit_bid = max(stop, float(o[x])+COST)-COST
                    reason = 'SL_TRAIL' if side*(stop-initial) > 1e-9 else 'SL_INIT'
                    found = True
                    break
            if engine['real'][j]:
                if engine['name'] == 'E_KELTNER':
                    candidate = engine['ema'][j]
                else:
                    candidate = engine['close'][j]-side*4.0*engine['atr'][j]
                if np.isfinite(candidate):
                    stop = max(stop, candidate) if side == 1 else min(stop, candidate)
        if not found:
            x, exit_bid, reason = len(t)-1, float(c[-1]), 'END'
            j = len(starts)-1
        occupied_until = j
        gross = side*(exit_bid-entry_bid)
        net = gross-COST
        entry_utc = pd.Timestamp(int(t[entry_ix]), unit='s', tz='UTC')
        exit_utc = pd.Timestamp(int(t[x]), unit='s', tz='UTC')
        rows.append({'engine': engine['name'], 'side': side,
                     'entry_time': int(t[entry_ix]), 'exit_time': int(t[x]),
                     'entry_bid': entry_bid, 'exit_bid': exit_bid,
                     'initial_stop': initial, 'final_stop': stop,
                     'initial_risk_all_in': initial_risk, 'exit_type': reason,
                     'gross': gross, 'cost': COST, 'net': net,
                     'net_r': net/initial_risk,
                     'entry_year': int(entry_utc.year),
                     'entry_month_utc': entry_utc.strftime('%Y-%m'),
                     'overnight_utc': bool(entry_utc.date() != exit_utc.date()),
                     'same_m1_exit': bool(entry_ix == x)})
    return rows, {'candidates': int(np.count_nonzero(engine['side'])),
                  'busy': busy, 'no_real': no_real}


def mbb(x: np.ndarray) -> float:
    if len(x) < BLOCK:
        return float('nan')
    rng = np.random.default_rng(SEED)
    bad = 0
    for _ in range(BOOT):
        starts = rng.integers(0, len(x)-BLOCK+1, size=math.ceil(len(x)/BLOCK))
        sample = np.concatenate([x[s:s+BLOCK] for s in starts])[:len(x)]
        bad += sample.mean() <= 0
    return bad/BOOT


def summarize(rows, months: int):
    x = np.array([r['net'] for r in rows], float)
    rr = np.array([r['net_r'] for r in rows], float)
    n = len(x)
    mean = float(x.mean()) if n else float('nan')
    std = float(x.std(ddof=1)) if n > 1 else float('nan')
    sharpe = mean/std if std > 0 else float('nan')
    p = mbb(x)
    monthly = {}
    for r in rows:
        monthly[r['entry_month_utc']] = monthly.get(r['entry_month_utc'], 0.0)+r['net_r']
    yearly = {}
    for r in rows:
        y = str(r['entry_year'])
        v = yearly.setdefault(y, {'trades': 0, 'net': 0.0, 'net_r': 0.0})
        v['trades'] += 1
        v['net'] += r['net']
        v['net_r'] += r['net_r']
    curve = np.r_[0.0, np.cumsum(rr)]
    return {'trades': n, 'net_expectancy': mean, 'net_total': float(x.sum()),
            'net_sharpe': sharpe, 'mbb_p': p,
            'gate_pass': bool(n >= 50 and sharpe > 0 and p <= 0.20),
            'total_r': float(rr.sum()), 'tape_months': months,
            'monthly_r_velocity': float(rr.sum()/months),
            'exceeds_5_33r': bool(rr.sum()/months > 5.33),
            'r_max_drawdown': float(np.max(np.maximum.accumulate(curve)-curve)),
            'monthly_r': dict(sorted(monthly.items())), 'yearly': yearly,
            'r_velocity_2024_25': sum(r['net_r'] for r in rows if r['entry_year'] in (2024, 2025))/24,
            'exits': {k: sum(r['exit_type'] == k for r in rows)
                      for k in ('SL_INIT', 'SL_TRAIL', 'END')},
            'overnight_utc_trades': sum(r['overnight_utc'] for r in rows),
            'same_m1_exits': sum(r['same_m1_exit'] for r in rows)}


def main():
    if not (OUT/'SPEC_GOLD_15M.txt').exists() or not (OUT/'GATE_GOLD_15M.txt').exists():
        raise RuntimeError('SPEC and GATE must exist before execution')
    report = OUT/'REPORT_GOLD_15M.txt'
    if report.exists():
        raise FileExistsError('Historical REPORT is immutable; use a new study name')
    df = pd.read_csv(TAPE, usecols=['time', 'open', 'high', 'low', 'close'])
    if not df.time.is_monotonic_increasing or not df.time.is_unique or df.isna().any().any():
        raise AssertionError('Tape must be sorted, unique and finite')
    m = {k: df[k].to_numpy(np.int64 if k == 'time' else float) for k in df.columns}
    first = pd.Timestamp(int(m['time'][0]), unit='s', tz='UTC')
    last = pd.Timestamp(int(m['time'][-1]), unit='s', tz='UTC')
    months = (last.year-first.year)*12+last.month-first.month+1
    g15, g1 = grid(m, 900), grid(m, 3600)
    results, all_rows = [], []
    for engine in engines(g15, g1):
        rows, funnel = replay(m, engine)
        stats = summarize(rows, months)
        results.append({'engine': engine['name'], 'funnel': funnel, 'stats': stats})
        all_rows.extend(rows)
        print(f"{engine['name']}: n={stats['trades']} mean={stats['net_expectancy']:.5f} "
              f"sharpe={stats['net_sharpe']:.4f} p={stats['mbb_p']:.4f} "
              f"R/month={stats['monthly_r_velocity']:.4f} gate={stats['gate_pass']}", flush=True)
    pd.DataFrame(all_rows).to_csv(OUT/'trades.csv', index=False)
    meta = {'tape': str(TAPE.relative_to(ROOT)), 'tape_sha256': digest(TAPE),
            'script_sha256': digest(Path(__file__)), 'm1_rows': len(df),
            'first_utc': first.isoformat(), 'last_utc': last.isoformat(),
            'm15_bars': len(g15[0]), 'm15_real': int(g15[2].sum()),
            'h1_bars': len(g1[0]), 'h1_real': int(g1[2].sum()),
            'cost': COST, 'swaps': 0.0,
            'bootstrap': {'replicates': BOOT, 'block': BLOCK, 'seed': SEED},
            'results': results}
    (OUT/'run_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    lines = ['GOLD 15M ACCELERATION - REPORT',
             'VERDICT: ' + ('BOTH GATES PASS' if all(r['stats']['gate_pass'] for r in results)
                           else 'AT LEAST ONE GATE FAILS'),
             f"Tape: {meta['tape']} | {len(df)} M1 rows | {first.isoformat()} through {last.isoformat()} | {months} inclusive UTC calendar months",
             f"Source SHA256: {meta['tape_sha256']}",
             f"Grid: M15 {meta['m15_real']}/{meta['m15_bars']} real; H1 {meta['h1_real']}/{meta['h1_bars']} real",
             'Bid-only gold CFD proxy; round-trip cost 0.41 embedded in fills; swap 0.0.',
             'Net Sharpe is unannualized per trade. MBB: 10000 resamples, five-trade blocks, seed 20260926.', '']
    for result in results:
        s = result['stats']
        lines += [f"{result['engine']} | gate {'PASS' if s['gate_pass'] else 'FAIL'} | 5.33R/month target {'EXCEEDED' if s['exceeds_5_33r'] else 'NOT MET'}",
                  f"Trade Count: {s['trades']}; Net Expectancy: {s['net_expectancy']:.6f} gold units/trade; Net Total: {s['net_total']:.3f} gold units",
                  f"Net Sharpe: {s['net_sharpe']:.6f}; MBB p-value: {s['mbb_p']:.4f}",
                  f"Total net R: {s['total_r']:.3f}; Total tape months: {months}; Monthly R-Velocity: {s['monthly_r_velocity']:.4f} R/month; 2024-25: {s['r_velocity_2024_25']:.4f} R/month",
                  f"Max trade-sequence R drawdown: {s['r_max_drawdown']:.3f}; exits: {s['exits']}; UTC overnight trades: {s['overnight_utc_trades']}; same-M1 exits: {s['same_m1_exits']}",
                  f"Funnel: {result['funnel']}",
                  'Annual entry cohorts: '+json.dumps(s['yearly'], sort_keys=True),
                  'Monthly entry-cohort R: '+json.dumps(s['monthly_r'], sort_keys=True), '']
    lines += ['Interpretation: Velocity is a full-tape average, not the probability of passing within any specific 45-day window.',
              'FTMO 2-Step static 10% overall loss and separate 5% daily equity limit are not simulated here; this run has no lot sizing, open-equity account path, or challenge-pass estimate.',
              'Zero swaps are assumed even if a trade crosses rollover; see UTC overnight counts. Fixed cost does not measure variable FTMO spread or slippage.',
              'M1 bid-only OHLC cannot prove executable ask fills or intraminute paths. This previously researched tape offers exploratory evidence only, without walk-forward validation.']
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
