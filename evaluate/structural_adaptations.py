"""Frozen RTH entry and forced-flat Engine A/B study; see SPEC_STRUCTURAL.txt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from futures_prop_validation_phase1 import engine_a, engine_b, grids
from usoil_macro_engines import rollover_schedule, swap_for


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/results/structural_adaptations'
ASSETS = (
    ('US500', ROOT/'data/results/vwap_2021_2026/us500tick_m1.csv', .51, 570, 960, True, 2021, False),
    ('USTECH', ROOT/'data/results/virgin_assets/ustech_m1.csv', 1.66, 570, 960, True, 2020, False),
    ('USOIL', Path(r'C:\Users\Milad\download\lightcmdusd-m1-bid-2015-01-01-2026-09-27.csv'), .03, 540, 870, False, 2020, True),
    ('GBPJPY', Path(r'C:\Users\Milad\download\download\gbpjpy-m1-bid-2010-01-01-2026-09-27.csv'), .025, 180, 960, False, 2020, True),
)
BOOT, BLOCK, SEED = 10_000, 5, 20261030


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for part in iter(lambda: fh.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def load_source(asset: str, path: Path, raw: bool) -> tuple[dict, dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    time_col = 'timestamp' if raw else 'time'
    d = pd.read_csv(path, usecols=[time_col, 'open', 'high', 'low', 'close'])
    raw_time = d[time_col].to_numpy(np.int64)
    if raw and np.any(raw_time % 60000):
        raise ValueError('Raw timestamp is not a UTC M1 millisecond open')
    t = raw_time//1000 if raw else raw_time
    if (len(t) < 1_000_000 or np.any(np.diff(t) <= 0)
            or np.any(t % 60)):
        raise ValueError('Invalid, duplicated or insufficient UTC M1 timestamps')
    o, h, l, c = (d[k].to_numpy(float) for k in ('open', 'high', 'low', 'close'))
    if (not np.isfinite(np.column_stack((o, h, l, c))).all()
            or np.any(np.minimum.reduce((o, h, l, c)) <= 0)):
        raise ValueError('Nonfinite or nonpositive OHLC')
    error = np.maximum.reduce((o-h, c-h, l-o, l-c, l-h,
                               np.zeros(len(o))))
    bad = error > 1e-10
    repairs = int(bad.sum())
    if repairs:
        months = pd.to_datetime(t[bad], unit='s', utc=True).strftime('%Y-%m')
        if (asset != 'GBPJPY' or np.any(months != '2024-10')
                or float(error.max()) > .0031):
            raise ValueError(f'Unapproved OHLC envelope error {error.max():.8f}')
        h = np.maximum.reduce((h, o, c))
        l = np.minimum.reduce((l, o, c))
    carried = ((o == h) & (h == l) & (l == c)
               & np.r_[False, c[1:] == c[:-1]]) if asset != 'US500' else np.zeros(len(t), bool)
    keep = ~carried
    m = {'time': t[keep], 'open': o[keep], 'high': h[keep],
         'low': l[keep], 'close': c[keep]}
    first_limit = '2021-01-07' if asset == 'US500' else '2020-01-03'
    if (len(m['time']) < 1_000_000
            or m['time'][0] > int(pd.Timestamp(first_limit, tz='UTC').timestamp())
            or m['time'][-1] < int(pd.Timestamp('2026-01-01', tz='UTC').timestamp())):
        raise ValueError('Insufficient requested source coverage')
    gaps = np.diff(m['time'])
    meta = {'path': str(path), 'sha256': digest(path),
            'raw_rows': int(len(t)), 'observed_m1': int(len(m['time'])),
            'carried_removed': int(carried.sum()),
            'known_2024_10_ohlc_repaired': repairs,
            'first_utc': int(m['time'][0]), 'last_utc': int(m['time'][-1]),
            'gaps_over_3h': int(np.count_nonzero(gaps > 3*3600)),
            'max_gap_hours': float(gaps.max()/3600)}
    return m, meta


def rth_context(t: np.ndarray, start_min: int, end_min: int) -> dict:
    ny = pd.to_datetime(t, unit='s', utc=True).tz_convert('America/New_York')
    minute = ny.hour.to_numpy()*60+ny.minute.to_numpy()
    day_key = ny.tz_localize(None).normalize().asi8//86_400_000_000_000
    mask = (minute >= start_min) & (minute < end_min)
    rth_ix = np.flatnonzero(mask)
    if not len(rth_ix):
        raise ValueError('No real M1 inside RTH window')
    unique_days, first = np.unique(day_key[rth_ix], return_index=True)
    last_pos = np.r_[first[1:]-1, len(rth_ix)-1]
    last_ix = rth_ix[last_pos]
    last_by_day = dict(zip(unique_days.tolist(), last_ix.tolist()))
    return {'minute': minute, 'day_key': day_key, 'mask': mask,
            'last_by_day': last_by_day, 'end_min': end_min}


def replay_arm(m: dict, engine: dict, cost: float, ctx: dict,
               long_only: bool, arm: str) -> tuple[list[dict], dict]:
    t, o, h, l, c = (m[k] for k in ('time', 'open', 'high', 'low', 'close'))
    left = np.searchsorted(t, engine['start'])
    right = np.searchsorted(t, engine['end'])
    rows = []
    occupied_until = -1
    funnel = {'candidates': int(np.count_nonzero(engine['side'])),
              'shadowed': 0, 'no_real_entry_bucket': 0,
              'side_filtered': 0, 'outside_rth': 0,
              'forced_nonfinal_minute': 0}
    for i in np.flatnonzero(engine['side']):
        if i <= occupied_until:
            funnel['shadowed'] += 1
            continue
        if left[i] >= right[i]:
            funnel['no_real_entry_bucket'] += 1
            continue
        side = int(engine['side'][i])
        if long_only and side == -1:
            funnel['side_filtered'] += 1
            continue
        entry_ix = int(left[i])
        if not ctx['mask'][entry_ix]:
            funnel['outside_rth'] += 1
            continue
        entry_bid = float(o[entry_ix])
        entry_fill = entry_bid+cost if side == 1 else entry_bid
        atr = float(engine['atr'][i-1])
        if not np.isfinite(atr) or atr <= 0:
            raise AssertionError('Armed entry has invalid prior ATR')
        if engine['name'] == 'A_KC':
            distance = max(abs(entry_bid-float(engine['ema'][i-1])), 1.5*atr)
        else:
            distance = 4.0*atr
        stop = entry_bid-side*distance
        initial_stop = float(stop)
        force_ix = (ctx['last_by_day'][int(ctx['day_key'][entry_ix])]
                    if arm == 'FLAT' else None)
        if force_ix is not None and force_ix < entry_ix:
            raise AssertionError('Forced-flat minute precedes entry')
        exit_ix = exit_bar = None
        exit_bid = exit_fill = None
        reason = None
        for j in range(i, len(engine['side'])):
            a, z = int(left[j]), int(right[j])
            if j == i:
                a = entry_ix
            if force_ix is not None:
                z = min(z, force_ix+1)
            if a < z:
                if side == 1:
                    hits = np.flatnonzero(l[a:z] <= stop)
                else:
                    ask_stop = stop+cost if engine['name'] == 'A_KC' else stop
                    hits = np.flatnonzero(h[a:z]+cost >= ask_stop)
                if len(hits):
                    exit_ix = a+int(hits[0]); exit_bar = j
                    if side == 1:
                        exit_bid = min(float(stop), float(o[exit_ix]))
                        exit_fill = exit_bid
                    else:
                        exit_fill = max(float(ask_stop), float(o[exit_ix])+cost)
                        exit_bid = exit_fill-cost
                    reason = ('SL_TRAIL' if side*(stop-initial_stop) > 1e-9
                              else 'SL_INIT')
                    break
            if force_ix is not None and a <= force_ix < int(right[j]):
                exit_ix = force_ix; exit_bar = j
                exit_bid = float(c[force_ix])
                exit_fill = exit_bid if side == 1 else exit_bid+cost
                reason = 'FORCED_RTH'
                if ctx['minute'][force_ix] != ctx['end_min']-1:
                    funnel['forced_nonfinal_minute'] += 1
                break
            if engine['name'] == 'A_KC':
                if np.isfinite(engine['ema'][j]):
                    candidate = float(engine['ema'][j])
                    stop = max(stop, candidate) if side == 1 else min(stop, candidate)
            elif a < z and engine['real'][j] and np.isfinite(engine['atr'][j]):
                candidate = float(engine['close'][j]-side*4.0*engine['atr'][j])
                stop = max(stop, candidate) if side == 1 else min(stop, candidate)
        if exit_ix is None:
            if force_ix is not None:
                raise AssertionError('RTH forced-flat trade not closed in entry session')
            exit_ix = len(t)-1
            exit_bar = len(engine['side'])-1
            exit_bid = float(c[-1])
            exit_fill = exit_bid if side == 1 else exit_bid+cost
            reason = 'OPEN_END'
        occupied_until = int(exit_bar)
        gross = side*(exit_bid-entry_bid)
        fill = side*(exit_fill-entry_fill)
        if abs(fill-(gross-cost)) > 1e-8:
            raise AssertionError('Spread applied other than once')
        rows.append({'asset': '', 'engine': engine['name'], 'arm': arm,
                     'side': side, 'entry_bar': int(i), 'exit_bar': int(exit_bar),
                     'entry_time': int(t[entry_ix]), 'exit_time': int(t[exit_ix]),
                     'entry_bid': entry_bid, 'entry_fill': entry_fill,
                     'exit_bid': float(exit_bid) if reason != 'OPEN_END' else np.nan,
                     'exit_fill': float(exit_fill) if reason != 'OPEN_END' else np.nan,
                     'initial_stop': initial_stop, 'final_stop': float(stop),
                     'gross': gross if reason != 'OPEN_END' else np.nan,
                     'fill_pnl': fill if reason != 'OPEN_END' else np.nan,
                     'exit_type': reason,
                     'entry_ny_date': str(pd.Timestamp(int(t[entry_ix]), unit='s',
                           tz='UTC').tz_convert('America/New_York').date()),
                     'hold_hours': float((t[exit_ix]-t[entry_ix])/3600)})
    if rows:
        en = np.array([r['entry_time'] for r in rows], np.int64)
        ex = np.array([r['exit_time'] for r in rows], np.int64)
        if np.any(ex < en) or np.any(en[1:] <= ex[:-1]):
            raise AssertionError('Overlapping trade times')
    return rows, funnel


def decorate(asset: str, rows: list[dict], schedule: dict,
             cost: float, score_start: int) -> list[dict]:
    out = []
    for r in rows:
        if r['entry_time'] < score_start:
            continue
        swap, count, weighted, stale = swap_for(
            schedule, r['entry_time'], r['exit_time'])
        closed = r['exit_type'] != 'OPEN_END'
        net = r['fill_pnl']+swap if closed else np.nan
        if closed and abs(net-(r['gross']-cost+swap)) > 1e-8:
            raise AssertionError('Net PnL cost/financing identity failed')
        row = {**r, 'asset': asset, 'cost': cost,
               'swap': swap, 'rollover_count': count,
               'weighted_rollover_count': weighted,
               'max_swap_mark_stale_hours': stale,
               'net': net}
        out.append(row)
    return out


def mbb_p(x: np.ndarray) -> float:
    n = len(x)
    if n < BLOCK:
        return np.nan
    rng = np.random.default_rng(SEED)
    blocks = (n+BLOCK-1)//BLOCK
    bad = 0
    for base in range(0, BOOT, 250):
        k = min(250, BOOT-base)
        starts = rng.integers(0, n-BLOCK+1, (k, blocks))
        ix = (starts[:, :, None]+np.arange(BLOCK)).reshape(k, -1)[:, :n]
        bad += int(np.count_nonzero(x[ix].mean(axis=1) <= 0))
    return bad/BOOT


def summarize(asset: str, engine: str, arm: str, rows: list[dict],
              funnel: dict) -> dict:
    closed = [r for r in rows if r['exit_type'] != 'OPEN_END']
    x = np.array([r['net'] for r in closed], float)
    mean = float(x.mean()) if len(x) else np.nan
    sd = float(x.std(ddof=1)) if len(x) >= 2 else np.nan
    sharpe = mean/sd if np.isfinite(sd) and sd > 0 else np.nan
    p = mbb_p(x)
    annual = {str(y): {'trades': 0, 'net': 0.0} for y in range(2020, 2027)}
    for r in closed:
        y = r['entry_ny_date'][:4]
        if y in annual:
            annual[y]['trades'] += 1
            annual[y]['net'] += r['net']
    gates = {'G1': len(x) >= 50, 'G2': bool(mean > 0),
             'G3': bool(sharpe > 0), 'G4': bool(p <= .05), 'G5': True}
    return {'asset': asset, 'engine': engine, 'arm': arm,
            'closed_trades': len(x), 'scored_entries': len(rows),
            'open_end': len(rows)-len(x),
            'win_rate': float(np.mean(x > 0)) if len(x) else np.nan,
            'net_expectancy': mean, 'total_net': float(x.sum()),
            'net_sharpe_per_trade': float(sharpe), 'mbb_p': p,
            'swap_paid_closed': -sum(r['swap'] for r in closed),
            'swap_accrued_open_end': -sum(r['swap'] for r in rows
                                         if r['exit_type'] == 'OPEN_END'),
            'rollovers_all': sum(r['rollover_count'] for r in rows),
            'max_swap_mark_stale_hours': max((r['max_swap_mark_stale_hours']
                                             for r in rows), default=0.),
            'exit_mix': {name: sum(r['exit_type'] == name for r in closed)
                         for name in ('SL_INIT', 'SL_TRAIL', 'FORCED_RTH')},
            'annual': annual, 'funnel': funnel, 'gates': gates,
            'pass': all(gates.values())}


def fmt(value, digits=5):
    return 'NA' if not np.isfinite(value) else f'{value:.{digits}f}'


def main():
    spec, gate, report = (OUT/n for n in ('SPEC_STRUCTURAL.txt',
                                         'GATE_STRUCTURAL.txt',
                                         'REPORT_STRUCTURAL.txt'))
    if not spec.exists() or not gate.exists() or report.exists():
        raise FileExistsError('Frozen SPEC/GATE missing or historical REPORT exists')
    extracted = '\n'.join(s for s in spec.read_text(encoding='utf-8').splitlines()
                          if len(s) > 2 and s[0] == 'G' and s[1].isdigit())+'\n'
    if extracted != gate.read_text(encoding='utf-8'):
        raise AssertionError('GATE is not verbatim SPEC extraction')
    source_meta = {}
    results = []
    errors = {}
    for asset, path, cost, start_min, end_min, long_only, start_year, raw in ASSETS:
        print('Evaluating', asset, flush=True)
        try:
            m, meta = load_source(asset, path, raw)
            source_meta[asset] = meta
            ctx = rth_context(m['time'], start_min, end_min)
            schedule = rollover_schedule(m)
            g = grids(m)
            engines = (engine_a(g), engine_b(g))
            score_start = int(pd.Timestamp(f'{start_year}-01-01',
                                           tz='UTC').timestamp())
            for e in engines:
                for arm in ('SWING', 'FLAT'):
                    raw_rows, funnel = replay_arm(m, e, cost, ctx,
                                                  long_only, arm)
                    rows = decorate(asset, raw_rows, schedule, cost,
                                    score_start)
                    pd.DataFrame(rows).to_csv(
                        OUT/f'trades_{asset}_{e["name"]}_{arm}.csv', index=False)
                    stats = summarize(asset, e['name'], arm, rows, funnel)
                    results.append(stats)
                    print(asset, e['name'], arm, stats['closed_trades'],
                          'closed, net mean', fmt(stats['net_expectancy']),
                          'p', fmt(stats['mbb_p'], 4),
                          'PASS' if stats['pass'] else 'FAIL', flush=True)
            del m, ctx, g, engines
        except (FileNotFoundError, ValueError, AssertionError, MemoryError) as exc:
            errors[asset] = str(exc)
            print(asset, 'DATA BLOCKED:', exc, flush=True)
    complete = len(results) == 16 and len(source_meta) == 4 and not errors
    verdict = ('PASS' if complete and all(r['pass'] for r in results)
               else 'FAIL' if complete else 'DATA BLOCKED')
    pd.DataFrame([{k: v for k, v in r.items()
                   if k not in ('annual', 'funnel', 'gates', 'exit_mix')}
                  for r in results]).to_csv(OUT/'summary.csv', index=False)
    pd.DataFrame([{'asset': r['asset'], 'engine': r['engine'],
                   'arm': r['arm'],
                   **{str(y): r['annual'][str(y)]['net']
                      for y in range(2020, 2027)}}
                  for r in results]).to_csv(OUT/'annual_matrix.csv', index=False,
                                           float_format='%.6f')
    meta = {'study': 'structural_adaptations',
            'spec_sha256': digest(spec), 'gate_sha256': digest(gate),
            'script_sha256': digest(Path(__file__)),
            'sources': source_meta, 'results': results,
            'errors': errors, 'overall': verdict,
            'bootstrap': {'resamples': BOOT, 'block': BLOCK, 'seed': SEED}}
    (OUT/'run_meta.json').write_text(json.dumps(meta, indent=2, allow_nan=True),
                                     encoding='utf-8')
    lines = ['ENGINE A/B STRUCTURAL ADAPTATIONS - REPORT',
             f'OVERALL SIXTEEN-CELL GATE: {verdict}',
             'RTH entry in NY wall time; equities long only; owner -5% annual weekday swap.',
             'SWING keeps normal stops; FLAT closes at last real RTH M1 close after stop checks.',
             '', 'ASSET | ENGINE | ARM | CLOSED | WIN% | NET/TRADE | TOTAL NET | SHARPE | MBB P | SWAP PAID | GATE']
    lookup = {(r['asset'], r['engine'], r['arm']): r for r in results}
    for asset, *_ in ASSETS:
        for engine in ('A_KC', 'B_DONCHIAN'):
            for arm in ('SWING', 'FLAT'):
                r = lookup.get((asset, engine, arm))
                if r is None:
                    lines.append(f'{asset} | {engine} | {arm} | DATA BLOCKED')
                    continue
                lines.append(' | '.join((asset, engine, arm,
                    str(r['closed_trades']), fmt(100*r['win_rate'], 2),
                    fmt(r['net_expectancy'], 6), fmt(r['total_net'], 4),
                    fmt(r['net_sharpe_per_trade'], 5), fmt(r['mbb_p'], 4),
                    fmt(r['swap_paid_closed']+r['swap_accrued_open_end'], 4),
                    'PASS' if r['pass'] else 'FAIL')))
    lines += ['', 'ANNUAL NET PNL MATRIX (NATIVE PRICE UNITS; 2026 PARTIAL)',
              'ASSET | ENGINE | ARM | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026']
    for asset, *_ in ASSETS:
        for engine in ('A_KC', 'B_DONCHIAN'):
            for arm in ('SWING', 'FLAT'):
                r = lookup.get((asset, engine, arm))
                vals = [fmt(r['annual'][str(y)]['net'], 4) if r else 'NA'
                        for y in range(2020, 2027)]
                lines.append(' | '.join((asset, engine, arm, *vals)))
    lines += ['', 'SOURCE AND EXECUTION AUDIT']
    for asset, *_ in ASSETS:
        lines.append(f'{asset}: source {source_meta.get(asset)}, error {errors.get(asset)}')
    for r in results:
        lines.append(f"{r['asset']} {r['engine']} {r['arm']}: exits {r['exit_mix']}, "
                     f"open_end {r['open_end']}, closed swap "
                     f"{r['swap_paid_closed']:.6f}, open accrued "
                     f"{r['swap_accrued_open_end']:.6f}, rollovers "
                     f"{r['rollovers_all']}, max mark stale "
                     f"{r['max_swap_mark_stale_hours']:.2f}h, "
                     f"gates {r['gates']}, funnel {r['funnel']}")
    lines += ['', 'LIMITATIONS',
              'Price units differ by asset and are not summed as cash portfolio equity.',
              'The fixed round-turn floors and -5% annualized both-side swap are owner scenarios, not verified historical FTMO charges.',
              'M1 BID OHLC does not reproduce broker ticks, variable spread or all intraminute event order; adverse stop has priority.',
              'The two arms and engines are independent counterfactuals. No capital sizing, FTMO loss guard or challenge EV is modeled.']
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Wrote', report, flush=True)


if __name__ == '__main__':
    main()
