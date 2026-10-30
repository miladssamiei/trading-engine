"""Frozen TJR sweep/BoS/trail proxy study; see SPEC_TJR.txt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/results/tjr_smc_improved'
ASSETS = (
    ('XAUUSD', ROOT/'data/results/xauusd_onboarding/xauusdtick_m1.csv', 300, .585),
    ('US500', ROOT/'data/results/vwap_2021_2026/us500tick_m1.csv', 900, .51),
)
BOOT, BLOCK, SEED = 10_000, 5, 20261012


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for part in iter(lambda: fh.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def load(path: Path) -> dict:
    d = pd.read_csv(path, usecols=['time', 'open', 'high', 'low', 'close'])
    m = {k: d[k].to_numpy(np.int64 if k == 'time' else float)
         for k in ('time', 'open', 'high', 'low', 'close')}
    t, o, h, l, c = (m[k] for k in ('time', 'open', 'high', 'low', 'close'))
    if (len(t) < 100_000 or np.any(np.diff(t) <= 0) or np.any(t % 60 != 0)
            or not np.isfinite(np.column_stack((o, h, l, c))).all()
            or np.any(o <= 0) or np.any(h < np.maximum(o, c))
            or np.any(l > np.minimum(o, c))):
        raise AssertionError(f'Bad M1 source {path}')
    return m


def continuous(m: dict, seconds: int) -> dict:
    bucket = m['time']//seconds
    d = pd.DataFrame({'key': bucket, 'o': m['open'], 'h': m['high'],
                      'l': m['low'], 'c': m['close']})
    grouped = d.groupby('key', sort=True).agg(o=('o', 'first'), h=('h', 'max'),
                                              l=('l', 'min'), c=('c', 'last'))
    keys = np.arange(int(bucket[0]), int(bucket[-1])+1, dtype=np.int64)
    q = grouped.reindex(keys).ffill()
    if not np.isfinite(q.to_numpy()).all():
        raise AssertionError('Filled grid contains missing OHLC')
    real = np.isin(keys, grouped.index.to_numpy(np.int64), assume_unique=True)
    return {'start': keys*seconds, 'end': (keys+1)*seconds, 'real': real,
            **{k: q[k].to_numpy(float) for k in ('o', 'h', 'l', 'c')}}


def pivots(q: dict) -> tuple[np.ndarray, np.ndarray]:
    h, l, real = q['h'], q['l'], q['real']
    n = len(h)
    hi = np.zeros(n, bool)
    lo = np.zeros(n, bool)
    if n < 9:
        return hi, lo
    indices = np.arange(6, n-2)
    valid = real[indices].copy()
    hi_values = np.ones(len(indices), bool)
    lo_values = np.ones(len(indices), bool)
    for delta in (-6, -5, -4, -3, -2, -1, 1, 2):
        j = indices+delta
        valid &= real[j]
        hi_values &= h[indices] > h[j]
        lo_values &= l[indices] < l[j]
    hi[indices] = valid & hi_values
    lo[indices] = valid & lo_values
    return hi, lo


def cash_mask(q: dict) -> tuple[np.ndarray, np.ndarray]:
    start = pd.to_datetime(q['start'], unit='s', utc=True).tz_convert('America/New_York')
    end = pd.to_datetime(q['end'], unit='s', utc=True).tz_convert('America/New_York')
    sm = start.hour.to_numpy()*60+start.minute.to_numpy()
    em = end.hour.to_numpy()*60+end.minute.to_numpy()
    inside = ((start.date == end.date) & (sm >= 570) & (sm < 960)
              & (em > 570) & (em <= 960))
    entry_inside = inside & (em < 960)
    return inside, entry_inside


def signals(q: dict, asset: str) -> tuple[list[dict], dict]:
    ph, pl = pivots(q)
    h, l, c, real = (q[k] for k in ('h', 'l', 'c', 'real'))
    if asset == 'US500':
        allowed, entry_allowed = cash_mask(q)
    else:
        allowed = entry_allowed = np.ones(len(c), bool)
    latest_high = latest_low = np.nan
    latest_high_ix = latest_low_ix = -1
    pending = None
    out = []
    counts = {'pivot_highs': int(ph.sum()), 'pivot_lows': int(pl.sum()),
              'bull_sweeps': 0, 'bear_sweeps': 0, 'dual_sweeps': 0,
              'no_opposing_pivot': 0, 'invalidated': 0, 'replaced_or_cancelled': 0,
              'confirmed_bos': 0, 'no_cash_entry': 0}
    for i in range(len(c)):
        p = i-3  # at this bar's open, p has two COMPLETED following bars
        if p >= 0:
            if ph[p]:
                latest_high, latest_high_ix = float(h[p]), p
            if pl[p]:
                latest_low, latest_low_ix = float(l[p]), p
        if not real[i] or not allowed[i]:
            if asset == 'US500' and pending is not None:
                counts['replaced_or_cancelled'] += 1
                pending = None
            continue
        bull = bool(np.isfinite(latest_low) and l[i] < latest_low and c[i] > latest_low)
        bear = bool(np.isfinite(latest_high) and h[i] > latest_high and c[i] < latest_high)
        if bull and bear:
            counts['dual_sweeps'] += 1
            pending = None
            continue
        if bull or bear:
            side = 1 if bull else -1
            counts['bull_sweeps' if bull else 'bear_sweeps'] += 1
            if pending is not None:
                counts['replaced_or_cancelled'] += 1
            pending = None
            if side == -1 and asset == 'US500':
                continue
            opposing = latest_high if side == 1 else latest_low
            opposing_ix = latest_high_ix if side == 1 else latest_low_ix
            if not np.isfinite(opposing):
                counts['no_opposing_pivot'] += 1
                continue
            pending = {'side': side, 'sweep_bar': i,
                       'sweep_time': int(q['end'][i]),
                       'sweep_wick': float(l[i] if side == 1 else h[i]),
                       'swept_pivot': float(latest_low if side == 1 else latest_high),
                       'opposing_pivot': float(opposing),
                       'opposing_pivot_bar': int(opposing_ix)}
            continue  # BoS must occur AFTER the sweep bar
        if pending is None:
            continue
        side = pending['side']
        invalid = (l[i] < pending['sweep_wick'] if side == 1
                   else h[i] > pending['sweep_wick'])
        if invalid:
            counts['invalidated'] += 1
            pending = None
            continue
        bos = c[i] > pending['opposing_pivot'] if side == 1 else c[i] < pending['opposing_pivot']
        if bos:
            counts['confirmed_bos'] += 1
            if entry_allowed[i]:
                out.append({**pending, 'bos_bar': i, 'bos_time': int(q['end'][i]),
                            'bos_close': float(c[i])})
            else:
                counts['no_cash_entry'] += 1
            pending = None
    return out, counts


def walk(m: dict, setup: dict, entry_ix: int, cost: float) -> dict | None:
    t, o, h, l, c = (m[k] for k in ('time', 'open', 'high', 'low', 'close'))
    side = setup['side']
    entry_bid = float(o[entry_ix])
    entry_fill = entry_bid+cost if side == 1 else entry_bid
    initial_stop = float(setup['sweep_wick'])
    risk = side*(entry_fill-initial_stop)
    if not np.isfinite(risk) or risk <= 0:
        return None
    stop = initial_stop
    mfe = 0.0
    max_step = 0
    exit_ix = -1
    exit_fill = np.nan
    reason = ''
    for j in range(entry_ix, len(t)):
        if side == 1:
            if l[j] <= stop:
                exit_ix = j
                exit_fill = min(float(stop), float(o[j]))
                reason = 'INITIAL_STOP' if max_step == 0 else 'TRAIL_STOP'
                break
            favorable = float(h[j]-entry_fill)
        else:
            if h[j]+cost >= stop:
                exit_ix = j
                exit_fill = max(float(stop), float(o[j])+cost)
                reason = 'INITIAL_STOP' if max_step == 0 else 'TRAIL_STOP'
                break
            favorable = float(entry_fill-l[j]-cost)
        mfe = max(mfe, favorable)
        step = int(np.floor((mfe+1e-12*risk)/risk))
        if step > max_step:
            proposed = entry_fill+side*(step-1)*risk
            stop = max(stop, proposed) if side == 1 else min(stop, proposed)
            max_step = step
    if exit_ix < 0:
        exit_ix = len(t)-1
        exit_fill = float(c[-1]) if side == 1 else float(c[-1])+cost
        reason = 'OPEN_END'
    exit_bid = exit_fill if side == 1 else exit_fill-cost
    gross = side*(exit_bid-entry_bid)
    net = side*(exit_fill-entry_fill)
    if abs(net-(gross-cost)) > 1e-8:
        raise AssertionError('Cost was not embedded exactly once')
    entry_ny = pd.Timestamp(int(t[entry_ix]), unit='s', tz='UTC').tz_convert('America/New_York')
    exit_ny = pd.Timestamp(int(t[exit_ix]), unit='s', tz='UTC').tz_convert('America/New_York')
    return {**setup, 'entry_ix': int(entry_ix), 'exit_ix': int(exit_ix),
            'entry_time_utc': int(t[entry_ix]), 'exit_time_utc': int(t[exit_ix]),
            'entry_bid': entry_bid, 'entry_fill': entry_fill,
            'initial_stop': initial_stop, 'final_stop': float(stop),
            'initial_risk': risk, 'mfe_units': mfe, 'mfe_r': mfe/risk,
            'max_ratchet_step': max_step, 'exit_bid': exit_bid,
            'exit_fill': exit_fill, 'cost': cost, 'gross': gross, 'net': net,
            'net_r': net/risk, 'exit_type': reason,
            'hold_minutes': (int(t[exit_ix])-int(t[entry_ix]))/60,
            'ny_overnight': bool(entry_ny.date() != exit_ny.date()),
            'entry_ny_year': int(entry_ny.year)}


def replay(m: dict, setups: list[dict], cost: float):
    t = m['time']
    occupied_until = -1
    rows = []
    funnel = {'bos_candidates': len(setups), 'no_exact_m1': 0,
              'shadowed': 0, 'invalid_initial_risk': 0}
    for setup in setups:
        ix = int(np.searchsorted(t, setup['bos_time']))
        if ix >= len(t) or int(t[ix]) != setup['bos_time']:
            funnel['no_exact_m1'] += 1
            continue
        if ix <= occupied_until:
            funnel['shadowed'] += 1
            continue
        row = walk(m, setup, ix, cost)
        if row is None:
            funnel['invalid_initial_risk'] += 1
            continue
        occupied_until = row['exit_ix']
        rows.append(row)
    return rows, funnel


def mbb_p(values: np.ndarray) -> float:
    n = len(values)
    if n < BLOCK:
        return np.nan
    rng = np.random.default_rng(SEED)
    block_sums = np.convolve(values, np.ones(BLOCK), mode='valid')
    blocks, extra = divmod(n, BLOCK)
    tail = np.convolve(values, np.ones(extra), mode='valid') if extra else None
    bad = 0
    for offset in range(0, BOOT, 250):
        count = min(250, BOOT-offset)
        starts = rng.integers(0, n-BLOCK+1, (count, blocks+int(extra > 0)))
        totals = block_sums[starts[:, :blocks]].sum(axis=1)
        if extra:
            totals += tail[starts[:, blocks]]
        bad += int(np.count_nonzero(totals <= 0))
    return bad/BOOT


def metrics(rows: list[dict]) -> dict:
    closed = [r for r in rows if r['exit_type'] != 'OPEN_END']
    net = np.array([r['net'] for r in closed], float)
    gross = np.array([r['gross'] for r in closed], float)
    mean = float(net.mean()) if len(net) else np.nan
    sd = float(net.std(ddof=1)) if len(net) >= 2 else np.nan
    sharpe = mean/sd if np.isfinite(sd) and sd > 0 else np.nan
    p = mbb_p(net)
    annual = {}
    for row in closed:
        y = str(row['entry_ny_year'])
        item = annual.setdefault(y, {'trades': 0, 'net': 0.0})
        item['trades'] += 1
        item['net'] += row['net']
    gates = {'G1': len(net) >= 150, 'G2': bool(mean > 0), 'G3': bool(p <= .05)}
    return {'closed_trades': len(net), 'open_end': len(rows)-len(net),
            'win_rate': float(np.mean(net > 0)) if len(net) else np.nan,
            'gross_expectancy': float(gross.mean()) if len(gross) else np.nan,
            'net_expectancy': mean, 'net_total': float(net.sum()),
            'net_sharpe_per_trade': float(sharpe), 'mbb_p': p,
            'mean_net_r': float(np.mean([r['net_r'] for r in closed])) if closed else np.nan,
            'mean_mfe_r': float(np.mean([r['mfe_r'] for r in closed])) if closed else np.nan,
            'max_ratchet_step': max((r['max_ratchet_step'] for r in closed), default=0),
            'ny_overnight_trades': sum(r['ny_overnight'] for r in closed),
            'exit_counts': {reason: sum(r['exit_type'] == reason for r in closed)
                            for reason in ('INITIAL_STOP', 'TRAIL_STOP')},
            'annual': annual, 'gates': gates, 'pass': all(gates.values())}


def main():
    spec, gate, report = (OUT/name for name in ('SPEC_TJR.txt', 'GATE_TJR.txt',
                                               'REPORT_TJR.txt'))
    if not spec.exists() or not gate.exists():
        raise FileNotFoundError('Frozen SPEC and GATE must precede this run')
    if report.exists():
        raise FileExistsError('Historical REPORT is immutable; use new study name')
    extracted = '\n'.join(line for line in spec.read_text(encoding='utf-8').splitlines()
                          if line.startswith('G') and len(line) > 2 and line[1].isdigit())+'\n'
    if extracted != gate.read_text(encoding='utf-8'):
        raise AssertionError('GATE is not the verbatim SPEC extraction')
    results = []
    for asset, path, period, cost in ASSETS:
        m = load(path)
        q = continuous(m, period)
        setup_rows, state_funnel = signals(q, asset)
        rows, replay_funnel = replay(m, setup_rows, cost)
        pd.DataFrame(rows).to_csv(OUT/f'trades_{asset}.csv', index=False)
        stats = metrics(rows)
        record = {'asset': asset, 'source': str(path), 'source_sha256': digest(path),
                  'm1_rows': len(m['time']), 'grid_bars': len(q['c']),
                  'real_grid_bars': int(np.count_nonzero(q['real'])),
                  'first_utc': int(m['time'][0]), 'last_utc': int(m['time'][-1]),
                  'cost': cost, 'grid_seconds': period,
                  'state_funnel': state_funnel, 'replay_funnel': replay_funnel,
                  **stats}
        results.append(record)
        print(asset, 'closed', stats['closed_trades'],
              'gross', f"{stats['gross_expectancy']:.5f}",
              'net', f"{stats['net_expectancy']:.5f}",
              'Sharpe', f"{stats['net_sharpe_per_trade']:.4f}",
              'p', f"{stats['mbb_p']:.4f}",
              'gate', 'PASS' if stats['pass'] else 'FAIL', flush=True)
    overall = all(r['pass'] for r in results)
    meta = {'study': 'tjr_smc_improved', 'requested_label': 'Hypothesis #95',
            'label_collision': 'us100_vix_weekday_h95 already uses #95',
            'spec_sha256': digest(spec), 'gate_sha256': digest(gate),
            'script_sha256': digest(Path(__file__)),
            'overall_pass': overall, 'results': results}
    (OUT/'run_meta.json').write_text(json.dumps(meta, indent=2, allow_nan=True),
                                     encoding='utf-8')
    lines = ['TJR V1 IMPROVED SWEEP + BOS + 1R TRAIL - PHASE 1 REPORT',
             'Requested Hypothesis #95 label collides with existing us100_vix_weekday_h95; unique study identity tjr_smc_improved.',
             'Observed BID M1 proxy tapes; fixed round-trip cost floors XAUUSD 0.585, US500 0.51.',
             'Swing pivots require 6 past and 2 future real bars, with 2-bar confirmation lag.',
             'M1 stop has priority; favorable ratchet becomes effective on the following observed M1.',
             f'MBB: {BOOT} moving-block resamples, block {BLOCK}, seed {SEED}.',
             f"OVERALL TWO-ASSET PHASE 1 GATE: {'PASS' if overall else 'FAIL'}", '',
             'ASSET | CLOSED | WIN% | GROSS/TRADE | NET/TRADE | NET TOTAL | NET SHARPE | MBB P | GATE']
    for r in results:
        lines.append(' | '.join((r['asset'], str(r['closed_trades']),
            f"{100*r['win_rate']:.2f}", f"{r['gross_expectancy']:.6f}",
            f"{r['net_expectancy']:.6f}", f"{r['net_total']:.4f}",
            f"{r['net_sharpe_per_trade']:.5f}", f"{r['mbb_p']:.4f}",
            'PASS' if r['pass'] else 'FAIL')))
    lines += ['', 'DETAILS']
    for r in results:
        lines += [f"{r['asset']}: {r['m1_rows']:,} observed M1 rows, "
                  f"{r['real_grid_bars']:,} real M{r['grid_seconds']//60} grid bars, "
                  f"cost {r['cost']:.3f}; source SHA256 {r['source_sha256']}",
                  f"  G1-G3 {r['gates']}; exit {r['exit_counts']}; OPEN_END {r['open_end']}; "
                  f"NY overnight {r['ny_overnight_trades']}; mean net R {r['mean_net_r']:.5f}; "
                  f"mean MFE R {r['mean_mfe_r']:.5f}; largest ratchet step {r['max_ratchet_step']}",
                  f"  Pivot/sweep/BoS funnel {r['state_funnel']}; fill/occupancy funnel {r['replay_funnel']}",
                  f"  Annual entry-cohort count/net {r['annual']}",
                  f"  Ledger {OUT/('trades_'+r['asset']+'.csv')}"]
    lines += ['', 'LIMITS',
              'Short initial stop is the sweep high as an ASK-trigger price; the fixed BID/ASK floor can cause an immediate short stop.',
              'No fixed target, time exit or forced cash-session exit. Open-end positions are excluded from closed statistics.',
              'Fixed cost, BID-only OHLC, same-M1 uncertainty, zero financing and previously inspected tapes prevent broker/account validation.']
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Wrote', report, flush=True)


if __name__ == '__main__':
    main()
