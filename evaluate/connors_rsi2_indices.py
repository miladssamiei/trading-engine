"""Frozen two-index RSI(2) and Bollinger Phase 1; see SPEC_RSI2.txt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/results/connors_rsi2'
ASSETS = (
    ('US500', ROOT / 'data/results/vwap_2021_2026/us500tick_m1.csv', 0.55),
    ('USTECH', ROOT / 'data/results/virgin_assets/ustech_m1.csv', 1.66),
)
BOOT = 10_000
BLOCK = 5
SEED = 20261010
LEDGER_COLUMNS = (
    'asset', 'side', 'signal_bar', 'signal_time_utc', 'entry_ix', 'exit_ix',
    'entry_time_utc', 'exit_time_utc', 'entry_bid', 'entry_exec', 'exit_bid',
    'exit_exec', 'target_exec', 'stop_exec', 'atr14', 'rsi2', 'sma20',
    'h1_close', 'h1_ema200', 'initial_risk', 'cost', 'gross', 'net',
    'exit_type', 'hold_wall_minutes', 'time_stop_delayed', 'entry_ny_year',
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for part in iter(lambda: fh.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def load(path: Path) -> dict[str, np.ndarray]:
    q = pd.read_csv(path, usecols=['time', 'open', 'high', 'low', 'close'])
    m = {k: q[k].to_numpy(np.int64 if k == 'time' else np.float64)
         for k in ('time', 'open', 'high', 'low', 'close')}
    t, o, h, l, c = (m[k] for k in ('time', 'open', 'high', 'low', 'close'))
    if (len(t) < 100_000 or np.any(np.diff(t) <= 0) or np.any(t % 60 != 0)
            or not np.isfinite(np.column_stack((o, h, l, c))).all()
            or np.any(o <= 0) or np.any(h < np.maximum(o, c))
            or np.any(l > np.minimum(o, c))):
        raise AssertionError(f'Bad M1 source: {path}')
    return m


def continuous(m: dict, seconds: int) -> dict[str, np.ndarray]:
    key = m['time'] // seconds
    frame = pd.DataFrame({'key': key, 'o': m['open'], 'h': m['high'],
                          'l': m['low'], 'c': m['close']})
    real = frame.groupby('key', sort=True).agg(
        o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'))
    keys = np.arange(int(key[0]), int(key[-1]) + 1, dtype=np.int64)
    full = real.reindex(keys).ffill()
    if not np.isfinite(full.to_numpy()).all():
        raise AssertionError('Non-finite continuous grid')
    observed = np.isin(keys, real.index.to_numpy(np.int64), assume_unique=True)
    return {'start': keys * seconds, 'end': (keys + 1) * seconds,
            'real': observed,
            **{k: full[k].to_numpy(float) for k in ('o', 'h', 'l', 'c')}}


def wilder_rsi2(close: np.ndarray) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan)
    if n < 3:
        return out
    delta = np.diff(close)
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    ag = float(np.mean(gain[:2]))
    al = float(np.mean(loss[:2]))
    for i in range(2, n):
        if i > 2:
            ag = (ag + gain[i - 1]) / 2.0
            al = (al + loss[i - 1]) / 2.0
        if al == 0.0:
            out[i] = 100.0 if ag > 0.0 else 50.0
        elif ag == 0.0:
            out[i] = 0.0
        else:
            out[i] = 100.0 - 100.0/(1.0 + ag/al)
    return out


def wilder_atr14(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    prev = np.r_[c[0], c[:-1]]
    tr = np.maximum(h-l, np.maximum(np.abs(h-prev), np.abs(l-prev)))
    out = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        out[13] = tr[:14].mean()
        for i in range(14, len(tr)):
            out[i] = (13.0*out[i-1]+tr[i])/14.0
    return out


def signal_table(q15: dict, q1h: dict) -> dict:
    close = q15['c']
    rsi = wilder_rsi2(close)
    cs = pd.Series(close)
    middle = cs.rolling(20, min_periods=20).mean().to_numpy(float)
    std = cs.rolling(20, min_periods=20).std(ddof=0).to_numpy(float)
    atr = wilder_atr14(q15['h'], q15['l'], close)
    ema = pd.Series(q1h['c']).ewm(span=200, adjust=False).mean().to_numpy(float)
    ema[:199] = np.nan
    # Completed H1 candle at or before the completed M15 signal timestamp.
    hidx = np.searchsorted(q1h['end'], q15['end'], side='right') - 1
    hvalid = hidx >= 0
    safe = np.maximum(hidx, 0)
    hclose = q1h['c'][safe]
    hema = ema[safe]
    hclose[~hvalid] = np.nan
    hema[~hvalid] = np.nan
    ny = pd.to_datetime(q15['end'], unit='s', utc=True).tz_convert('America/New_York')
    minute = ny.hour.to_numpy()*60 + ny.minute.to_numpy()
    session = (minute >= 180) & (minute < 960)
    valid = (q15['real'] & session & np.isfinite(rsi) & np.isfinite(middle)
             & np.isfinite(std) & np.isfinite(atr) & (atr > 0)
             & np.isfinite(hclose) & np.isfinite(hema))
    long = valid & (hclose > hema) & (rsi < 10.0) & (close <= middle-2.0*std)
    short = valid & (hclose < hema) & (rsi > 90.0) & (close >= middle+2.0*std)
    if np.any(long & short):
        raise AssertionError('Both directions on one bar')
    return {'side': np.where(long, 1, np.where(short, -1, 0)).astype(np.int8),
            'rsi': rsi, 'middle': middle, 'atr': atr,
            'hclose': hclose, 'hema': hema, 'signal_session': session}


def walk(m: dict, q15: dict, signal: dict, asset: str, cost: float):
    t, o, h, l, c = (m[k] for k in ('time', 'open', 'high', 'low', 'close'))
    rows = []
    counters = {'raw_signals': int(np.count_nonzero(signal['side'])),
                'no_exact_next_m1': 0, 'shadowed': 0,
                'target_not_favorable': 0, 'invalid_risk': 0,
                'time_stop_delayed': 0}
    occupied_until = -1
    for bar in np.flatnonzero(signal['side']):
        side = int(signal['side'][bar])
        signal_end = int(q15['end'][bar])
        entry_ix = int(np.searchsorted(t, signal_end))
        if entry_ix == len(t) or int(t[entry_ix]) != signal_end:
            counters['no_exact_next_m1'] += 1
            continue
        if entry_ix <= occupied_until:
            counters['shadowed'] += 1
            continue
        entry_bid = float(o[entry_ix])
        entry_exec = entry_bid+cost if side == 1 else entry_bid
        target = float(signal['middle'][bar])
        atr = float(signal['atr'][bar])
        stop = entry_exec-side*1.5*atr
        risk = side*(entry_exec-stop)
        if risk <= 0 or not np.isfinite(risk):
            counters['invalid_risk'] += 1
            continue
        if side*(target-entry_exec) <= 0:
            counters['target_not_favorable'] += 1
            continue
        deadline_time = int(t[entry_ix])+120*60
        deadline_ix = int(np.searchsorted(t, deadline_time))
        exit_ix = -1
        exit_exec = np.nan
        reason = ''
        for j in range(entry_ix, min(deadline_ix, len(t))):
            if side == 1:
                stop_hit = l[j] <= stop
                target_hit = h[j] >= target
                if stop_hit:
                    exit_exec = min(stop, float(o[j]))
                    reason = 'SL'
                elif target_hit:
                    exit_exec = max(target, float(o[j]))
                    reason = 'TP'
            else:
                stop_hit = h[j]+cost >= stop
                target_hit = l[j]+cost <= target
                if stop_hit:
                    exit_exec = max(stop, float(o[j])+cost)
                    reason = 'SL'
                elif target_hit:
                    exit_exec = min(target, float(o[j])+cost)
                    reason = 'TP'
            if reason:
                exit_ix = j
                break
        if exit_ix < 0:
            if deadline_ix < len(t):
                exit_ix = deadline_ix
                exit_exec = float(o[exit_ix]) if side == 1 else float(o[exit_ix])+cost
                reason = 'TIME'
            else:
                exit_ix = len(t)-1
                exit_exec = float(c[-1]) if side == 1 else float(c[-1])+cost
                reason = 'OPEN_END'
        occupied_until = exit_ix
        exit_bid = exit_exec if side == 1 else exit_exec-cost
        gross = side*(exit_bid-entry_bid)
        net = side*(exit_exec-entry_exec)
        if abs(net-(gross-cost)) > max(1e-9, abs(net)*1e-10):
            raise AssertionError('Cost not embedded exactly once')
        hold_min = (int(t[exit_ix])-int(t[entry_ix]))/60.0
        delayed = bool(reason == 'TIME' and hold_min > 120.0)
        counters['time_stop_delayed'] += int(delayed)
        year = pd.Timestamp(int(t[entry_ix]), unit='s', tz='UTC').tz_convert('America/New_York').year
        rows.append({'asset': asset, 'side': side, 'signal_bar': int(bar),
                     'signal_time_utc': signal_end, 'entry_ix': entry_ix,
                     'exit_ix': exit_ix, 'entry_time_utc': int(t[entry_ix]),
                     'exit_time_utc': int(t[exit_ix]), 'entry_bid': entry_bid,
                     'entry_exec': entry_exec, 'exit_bid': exit_bid,
                     'exit_exec': exit_exec, 'target_exec': target,
                     'stop_exec': stop, 'atr14': atr, 'rsi2': float(signal['rsi'][bar]),
                     'sma20': target, 'h1_close': float(signal['hclose'][bar]),
                     'h1_ema200': float(signal['hema'][bar]),
                     'initial_risk': risk, 'cost': cost, 'gross': gross,
                     'net': net, 'exit_type': reason,
                     'hold_wall_minutes': hold_min, 'time_stop_delayed': delayed,
                     'entry_ny_year': int(year)})
    return rows, counters


def mbb_p(values: np.ndarray) -> float:
    n = len(values)
    if n < BLOCK:
        return float('nan')
    rng = np.random.default_rng(SEED)
    block_sums = np.convolve(values, np.ones(BLOCK), mode='valid')
    blocks, remainder = divmod(n, BLOCK)
    tail_sums = np.convolve(values, np.ones(remainder), mode='valid') if remainder else None
    bad = 0
    for offset in range(0, BOOT, 250):
        count = min(250, BOOT-offset)
        start = rng.integers(0, n-BLOCK+1, size=(count, blocks+int(remainder > 0)))
        totals = block_sums[start[:, :blocks]].sum(axis=1)
        if remainder:
            totals += tail_sums[start[:, blocks]]
        bad += int(np.count_nonzero(totals <= 0))
    return bad/BOOT


def metrics(rows: list[dict]) -> dict:
    closed = [r for r in rows if r['exit_type'] != 'OPEN_END']
    x = np.array([r['net'] for r in closed], dtype=float)
    gross = np.array([r['gross'] for r in closed], dtype=float)
    mean = float(np.mean(x)) if len(x) else float('nan')
    sd = float(np.std(x, ddof=1)) if len(x) >= 2 else float('nan')
    sharpe = mean/sd if np.isfinite(sd) and sd > 0 else float('nan')
    p = mbb_p(x)
    gates = {'G1': len(x) >= 150, 'G2': bool(mean > 0),
             'G3': bool(sharpe > 0), 'G4': bool(p <= .05)}
    annual = {}
    for r in closed:
        y = str(r['entry_ny_year'])
        z = annual.setdefault(y, {'trades': 0, 'net': 0.0})
        z['trades'] += 1
        z['net'] += r['net']
    return {'closed_trades': len(x), 'open_end': len(rows)-len(x),
            'win_rate': float(np.mean(x > 0)) if len(x) else float('nan'),
            'exit_counts': {label: sum(r['exit_type'] == label for r in closed)
                            for label in ('TP', 'SL', 'TIME')},
            'gross_expectancy': float(np.mean(gross)) if len(gross) else float('nan'),
            'net_expectancy': mean, 'net_total': float(np.sum(x)),
            'net_sharpe_per_trade': float(sharpe), 'mbb_p': p,
            'annual': annual, 'gates': gates, 'pass': all(gates.values())}


def main():
    spec, gate, report = (OUT/name for name in ('SPEC_RSI2.txt', 'GATE_RSI2.txt',
                                               'REPORT_RSI2.txt'))
    if not spec.exists() or not gate.exists():
        raise FileNotFoundError('Frozen SPEC and GATE required before execution')
    if report.exists():
        raise FileExistsError('Historical REPORT is immutable; use a new study')
    extracted = '\n'.join(line for line in spec.read_text(encoding='utf-8').splitlines()
                          if line.startswith('G') and len(line) > 2 and line[1].isdigit())+'\n'
    if extracted != gate.read_text(encoding='utf-8'):
        raise AssertionError('GATE differs from frozen SPEC extraction')
    results = []
    for asset, path, cost in ASSETS:
        m = load(path)
        q15 = continuous(m, 900)
        q1h = continuous(m, 3600)
        signals = signal_table(q15, q1h)
        rows, funnel = walk(m, q15, signals, asset, cost)
        ledger = OUT/f'trades_{asset}.csv'
        pd.DataFrame(rows, columns=LEDGER_COLUMNS).to_csv(ledger, index=False)
        stats = metrics(rows)
        result = {'asset': asset, 'input': str(path), 'input_sha256': digest(path),
                  'm1_rows': len(m['time']), 'm15_bars': len(q15['c']),
                  'real_m15_bars': int(np.count_nonzero(q15['real'])),
                  'h1_bars': len(q1h['c']), 'first_utc': int(m['time'][0]),
                  'last_utc': int(m['time'][-1]), 'cost': cost,
                  'funnel': funnel, 'ledger': str(ledger), **stats}
        results.append(result)
        print(asset, 'trades', stats['closed_trades'], 'net mean',
              f"{stats['net_expectancy']:.6f}", 'Sharpe',
              f"{stats['net_sharpe_per_trade']:.5f}", 'MBB p',
              f"{stats['mbb_p']:.4f}", 'gate',
              'PASS' if stats['pass'] else 'FAIL', flush=True)
    meta = {'study': 'connors_rsi2', 'spec_sha256': digest(spec),
            'gate_sha256': digest(gate), 'script_sha256': digest(Path(__file__)),
            'bootstrap': {'draws': BOOT, 'block': BLOCK, 'seed': SEED},
            'overall_pass': all(r['pass'] for r in results), 'results': results}
    (OUT/'run_meta.json').write_text(json.dumps(meta, indent=2, allow_nan=True), encoding='utf-8')
    lines = ['CONNORS RSI(2) PLUS BOLLINGER INDICES - PHASE 1 REPORT',
             'Frozen two-asset M15 signal / completed H1 trend / exact next M1 entry proxy.',
             'Retrospective BID-only tapes, fixed round-trip cost floors, no swap/slippage.',
             f'MBB P(mean net <= 0): {BOOT} moving-block draws, block {BLOCK}, seed {SEED}.',
             'Unannualized Sharpe is mean net trade PnL / sample SD of net trade PnL.',
             f"OVERALL G5: {'PASS' if meta['overall_pass'] else 'FAIL'}", '',
             'ASSET   COST TRADES WIN%  TP  SL TIME GROSS/TRADE NET/TRADE NET TOTAL SHARPE MBB P GATE']
    for r in results:
        e = r['exit_counts']
        lines.append(f"{r['asset']:<8}{r['cost']:<5.2f}{r['closed_trades']:<7}"
                     f"{100*r['win_rate']:<6.2f}{e['TP']:<4}{e['SL']:<4}{e['TIME']:<5}"
                     f"{r['gross_expectancy']:<12.6f}{r['net_expectancy']:<11.6f}"
                     f"{r['net_total']:<10.3f}{r['net_sharpe_per_trade']:<7.4f}"
                     f"{r['mbb_p']:<6.4f}{'PASS' if r['pass'] else 'FAIL'}")
    lines += ['', 'DETAILS']
    for r in results:
        lines += [f"{r['asset']}: {r['m1_rows']:,} M1 rows from "
                  f"{pd.Timestamp(r['first_utc'], unit='s', tz='UTC')} through "
                  f"{pd.Timestamp(r['last_utc'], unit='s', tz='UTC')}; "
                  f"{r['real_m15_bars']:,} observed M15 buckets; input SHA256 {r['input_sha256']}",
                  f"  Gate {r['gates']}; signal funnel {r['funnel']}; open_end {r['open_end']}",
                  f"  Entry-cohort annual count/net {r['annual']}",
                  f"  Ledger {r['ledger']}"]
    lines += ['', 'INTERPRETATION LIMITS',
              'USTECH tick_volume=0 on all rows is an onboarding schema placeholder, not proof that all its recorded OHLC bars are empty.',
              'The time stop uses the first observed M1 open at or after the 120-minute deadline; market closures can delay it. Delays are counted above.',
              'BID M1 OHLC cannot measure actual ASK paths, intraminute ordering or broker fill priority. Fixed cost and zero swap are scenario assumptions.',
              'These retrospective proxy histories have been researched previously; a Phase 1 pass would need separately predeclared execution/forward validation.']
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Wrote', report, flush=True)


if __name__ == '__main__':
    main()
