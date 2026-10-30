"""Frozen daily Turtle Donchian portfolio test; see SPEC_D1_TURTLE.txt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/results/d1_turtle_portfolio'
RAW = Path(r'C:\Users\Milad\download\download')
ASSETS = (
    ('USDJPY', RAW/'usdjpy-m1-bid-2010-01-01-2026-09-27.csv', .009, .01),
    ('GBPJPY', RAW/'gbpjpy-m1-bid-2010-01-01-2026-09-27.csv', .025, .01),
    ('EURJPY', RAW/'eurjpy-m1-bid-2010-01-01-2026-09-27.csv', .015, .01),
    ('EURUSD', ROOT/'data/results/eurusd_onboarding/eurusdtick_m1.csv', .00003, .0001),
)
BOOT = 10_000
BLOCK = 5
SEED = 20261020


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for part in iter(lambda: fh.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def load_source(asset: str, path: Path) -> tuple[dict, dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    time_col = 'time' if asset == 'EURUSD' else 'timestamp'
    d = pd.read_csv(path, usecols=[time_col, 'open', 'high', 'low', 'close'])
    raw_t = d[time_col].to_numpy(np.int64)
    if asset != 'EURUSD':
        if np.any(raw_t % 60000):
            raise ValueError('Dukascopy timestamps are not M1 millisecond opens')
        t = raw_t // 1000
    else:
        t = raw_t
    if len(t) < 1_000_000 or np.any(np.diff(t) <= 0) or np.any(t % 60):
        raise ValueError('Source has insufficient, duplicated or nonminute timestamps')
    o, h, l, c = (d[k].to_numpy(float) for k in ('open', 'high', 'low', 'close'))
    if (not np.isfinite(np.column_stack((o, h, l, c))).all()
            or np.any(np.minimum.reduce((o, h, l, c)) <= 0)):
        raise ValueError('Nonfinite or nonpositive OHLC')
    envelope_error = np.maximum.reduce((o-h, c-h, l-o, l-c, l-h,
                                        np.zeros(len(o))))
    bad = envelope_error > 1e-10
    repairs = int(bad.sum())
    if repairs:
        bad_month = pd.to_datetime(t[bad], unit='s', utc=True).strftime('%Y-%m')
        if (asset == 'EURUSD' or np.any(bad_month != '2024-10')
                or float(envelope_error.max()) > .0031):
            raise ValueError(f'Unapproved OHLC envelope error {envelope_error.max():.8f}')
        h = np.maximum.reduce((h, o, c))
        l = np.minimum.reduce((l, o, c))
    carried = ((o == h) & (h == l) & (l == c)
               & np.r_[False, c[1:] == c[:-1]]) if asset != 'EURUSD' else np.zeros(len(t), bool)
    keep = ~carried
    m = {'time': t[keep], 'open': o[keep], 'high': h[keep],
         'low': l[keep], 'close': c[keep]}
    if asset != 'EURUSD':
        if (m['time'][0] > int(pd.Timestamp('2010-01-04', tz='UTC').timestamp())
                or m['time'][-1] < int(pd.Timestamp('2026-09-25', tz='UTC').timestamp())):
            raise ValueError('JPY source does not cover declared 2010-2026 span')
    else:
        if (m['time'][0] > int(pd.Timestamp('2020-01-03', tz='UTC').timestamp())
                or m['time'][-1] < int(pd.Timestamp('2026-09-18', tz='UTC').timestamp())):
            raise ValueError('Verified EURUSD source does not cover declared span')
    meta = {'source': str(path), 'sha256': digest(path), 'raw_rows': len(t),
            'observed_m1': len(m['time']), 'carried_removed': int(carried.sum()),
            'known_2024_10_ohlc_repaired': repairs,
            'first_utc': int(m['time'][0]), 'last_utc': int(m['time'][-1])}
    return m, meta


def daily_grid(m: dict) -> dict:
    t = m['time']
    wall = pd.to_datetime(t, unit='s', utc=True).tz_convert(
        'America/New_York').tz_localize(None)
    session_day = (wall-pd.Timedelta(hours=17)).normalize()+pd.Timedelta(days=1)
    frame = pd.DataFrame({'day': session_day, 'o': m['open'], 'h': m['high'],
                          'l': m['low'], 'c': m['close'], 'ix': np.arange(len(t))})
    a = frame.groupby('day', sort=True).agg(
        o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'),
        n=('o', 'size'), first_ix=('ix', 'first'), last_ix=('ix', 'last'))
    a = a[(a.index.dayofweek < 5) & (a.n >= 300)]
    if len(a) < 1000 or not a.index.is_monotonic_increasing:
        raise ValueError('Insufficient accepted daily sessions')
    first, last = (a[k].to_numpy(np.int64) for k in ('first_ix', 'last_ix'))
    if np.any(first[1:] <= last[:-1]):
        raise AssertionError('Daily sessions overlap in M1')
    return {'date': a.index, 'open': a.o.to_numpy(float),
            'high': a.h.to_numpy(float), 'low': a.l.to_numpy(float),
            'close': a.c.to_numpy(float), 'n': a.n.to_numpy(int),
            'first_ix': first, 'last_ix': last}


def wilder_atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period=20) -> np.ndarray:
    tr = np.empty(len(c), float)
    tr[0] = h[0]-l[0]
    tr[1:] = np.maximum.reduce((h[1:]-l[1:], np.abs(h[1:]-c[:-1]),
                                np.abs(l[1:]-c[:-1])))
    out = np.full(len(c), np.nan)
    if len(c) >= period:
        out[period-1] = tr[:period].mean()
        for i in range(period, len(c)):
            out[i] = ((period-1)*out[i-1]+tr[i])/period
    return out


def indicators(d: dict) -> dict:
    h, l, c = (d[k] for k in ('high', 'low', 'close'))
    ema = pd.Series(c).ewm(span=50, adjust=False).mean().to_numpy()
    ema[:49] = np.nan
    hi20 = pd.Series(h).rolling(20, min_periods=20).max().shift(1).to_numpy()
    lo20 = pd.Series(l).rolling(20, min_periods=20).min().shift(1).to_numpy()
    hi10 = pd.Series(h).rolling(10, min_periods=10).max().shift(1).to_numpy()
    lo10 = pd.Series(l).rolling(10, min_periods=10).min().shift(1).to_numpy()
    atr = wilder_atr(h, l, c)
    side = np.where((c > hi20) & (c > ema) & np.isfinite(atr), 1,
                    np.where((c < lo20) & (c < ema) & np.isfinite(atr), -1, 0))
    return {'ema50': ema, 'atr20': atr, 'hi20': hi20, 'lo20': lo20,
            'hi10': hi10, 'lo10': lo10, 'side': side}


def replay(asset: str, m: dict, d: dict, ind: dict, cost: float,
           pip: float) -> tuple[list[dict], dict]:
    t, o, h, l, c = (m[k] for k in ('time', 'open', 'high', 'low', 'close'))
    n = len(d['date'])
    rows = []
    funnel = {'signals': int(np.count_nonzero(ind['side'])),
              'shadowed': 0, 'no_next_daily_open': 0}
    i = 0
    while i < n:
        side = int(ind['side'][i])
        if side == 0:
            i += 1
            continue
        if i+1 >= n:
            funnel['no_next_daily_open'] += 1
            break
        e = i+1
        entry_ix = int(d['first_ix'][e])
        entry_bid = float(o[entry_ix])
        entry_fill = entry_bid+cost if side == 1 else entry_bid
        atr = float(ind['atr20'][i])
        if not np.isfinite(atr) or atr <= 0:
            raise AssertionError('Invalid setup ATR')
        stop = entry_fill-side*2*atr
        pending_trail = False
        exit_ix = exit_day = None
        exit_bid = exit_fill = None
        reason = 'OPEN_END'
        for j in range(e, n):
            start = int(d['first_ix'][j])
            end = int(d['last_ix'][j])+1
            if pending_trail:
                # A market exit at this open precedes the minute's high/low.
                if side == 1 and o[start] <= stop:
                    exit_bid = float(o[start]); exit_fill = exit_bid
                    reason = 'INITIAL_STOP_GAP'
                elif side == -1 and o[start]+cost >= stop:
                    exit_bid = float(o[start]); exit_fill = exit_bid+cost
                    reason = 'INITIAL_STOP_GAP'
                else:
                    exit_bid = float(o[start])
                    exit_fill = exit_bid if side == 1 else exit_bid+cost
                    reason = 'DONCHIAN10_CLOSE'
                exit_ix, exit_day = start, j
                break
            if side == 1:
                hit = np.flatnonzero(l[start:end] <= stop)
            else:
                hit = np.flatnonzero(h[start:end]+cost >= stop)
            if len(hit):
                k = start+int(hit[0])
                if side == 1:
                    exit_bid = min(float(stop), float(o[k]))
                    exit_fill = exit_bid
                    gap = o[k] <= stop
                else:
                    exit_fill = max(float(stop), float(o[k])+cost)
                    exit_bid = exit_fill-cost
                    gap = o[k]+cost >= stop
                exit_ix, exit_day = k, j
                reason = 'INITIAL_STOP_GAP' if gap else 'INITIAL_STOP'
                break
            if j+1 < n:
                pending_trail = bool((side == 1 and d['close'][j] < ind['lo10'][j])
                                     or (side == -1 and d['close'][j] > ind['hi10'][j]))
        if exit_ix is None:
            exit_day = n-1
        for k in range(e, int(exit_day)+1):
            if k < len(ind['side']):
                funnel['shadowed'] += int(ind['side'][k] != 0 and k < exit_day)
        entry_day = d['date'][e]
        scored = entry_day.year >= 2020
        gross = (side*(exit_bid-entry_bid)/pip if exit_ix is not None else np.nan)
        net = (side*(exit_fill-entry_fill)/pip if exit_ix is not None else np.nan)
        if exit_ix is not None and abs(net-(gross-cost/pip)) > 1e-6:
            raise AssertionError('Cost was not embedded exactly once')
        row = {'asset': asset, 'signal_day': str(d['date'][i].date()),
               'entry_day': str(entry_day.date()), 'entry_time': int(t[entry_ix]),
               'side': side, 'entry_bid': entry_bid, 'entry_fill': entry_fill,
               'setup_atr20': atr, 'initial_stop_fill_coordinate': stop,
               'exit_day': str(d['date'][exit_day].date()) if exit_ix is not None else '',
               'exit_time': int(t[exit_ix]) if exit_ix is not None else np.nan,
               'exit_bid': exit_bid if exit_ix is not None else np.nan,
               'exit_fill': exit_fill if exit_ix is not None else np.nan,
               'exit_type': reason, 'gross_pips': gross, 'net_pips': net,
               'cost_pips': cost/pip, 'hold_daily_sessions': int(exit_day-e+1),
               'entry_year': int(entry_day.year)}
        if scored:
            rows.append(row)
        # A close signal on the stop day may enter next session. A signal on
        # the day before a next-open Turtle exit occurred while occupied.
        i = int(exit_day)
        if exit_ix is None:
            break
    return rows, funnel


def mbb_p(x: np.ndarray) -> float:
    n = len(x)
    if n < BLOCK:
        return np.nan
    rng = np.random.default_rng(SEED)
    starts_max = n-BLOCK+1
    nblocks = (n+BLOCK-1)//BLOCK
    bad = 0
    for offset in range(0, BOOT, 250):
        k = min(250, BOOT-offset)
        starts = rng.integers(0, starts_max, (k, nblocks))
        ix = (starts[:, :, None]+np.arange(BLOCK)).reshape(k, -1)[:, :n]
        bad += int(np.count_nonzero(x[ix].mean(axis=1) <= 0))
    return bad/BOOT


def summarize(asset: str, rows: list[dict], source: dict, d: dict,
              funnel: dict) -> dict:
    closed = [r for r in rows if r['exit_type'] != 'OPEN_END']
    x = np.array([r['net_pips'] for r in closed], float)
    mean = float(x.mean()) if len(x) else np.nan
    sd = float(x.std(ddof=1)) if len(x) >= 2 else np.nan
    sharpe = mean/sd if np.isfinite(sd) and sd > 0 else np.nan
    p = mbb_p(x)
    annual = {str(y): {'trades': 0, 'net_pips': 0.0} for y in range(2020, 2027)}
    for r in closed:
        y = str(r['entry_year'])
        if y in annual:
            annual[y]['trades'] += 1
            annual[y]['net_pips'] += r['net_pips']
    gates = {'G1': len(x) >= 30, 'G2': bool(mean > 0),
             'G3': bool(sharpe > 0), 'G4': bool(p <= .05),
             'G5': all(annual[str(y)]['net_pips'] >= -1e-8
                       for y in range(2020, 2026))}
    return {'asset': asset, 'closed_trades': len(x),
            'open_end': len(rows)-len(x),
            'win_rate': float(np.mean(x > 0)) if len(x) else np.nan,
            'net_expectancy_pips': mean, 'total_net_pips': float(x.sum()),
            'net_sharpe_per_trade': float(sharpe), 'mbb_p': p,
            'exit_mix': {'INITIAL_STOP': sum(r['exit_type'] == 'INITIAL_STOP'
                                           for r in closed),
                         'INITIAL_STOP_GAP': sum(r['exit_type'] == 'INITIAL_STOP_GAP'
                                               for r in closed),
                         'DONCHIAN10_CLOSE': sum(r['exit_type'] == 'DONCHIAN10_CLOSE'
                                               for r in closed)},
            'annual': annual, 'gates': gates, 'pass': all(gates.values()),
            'scored_entries': len(rows), 'daily_bars': len(d['date']),
            'first_daily_day': str(d['date'][0].date()),
            'last_daily_day': str(d['date'][-1].date()),
            'source': source, 'funnel': funnel}


def fmt(x, n=3):
    return 'NA' if not np.isfinite(x) else f'{x:.{n}f}'


def main():
    spec = OUT/'SPEC_D1_TURTLE.txt'
    gate = OUT/'GATE_D1_TURTLE.txt'
    report = OUT/'REPORT_D1_TURTLE.txt'
    if not spec.exists() or not gate.exists() or report.exists():
        raise FileExistsError('SPEC/GATE must be frozen and historical REPORT absent')
    extracted = '\n'.join(s for s in spec.read_text(encoding='utf-8').splitlines()
                          if len(s) > 2 and s[0] == 'G' and s[1].isdigit())+'\n'
    if extracted != gate.read_text(encoding='utf-8'):
        raise AssertionError('GATE differs from frozen SPEC extraction')
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    errors = {}
    for asset, path, cost, pip in ASSETS:
        print('Evaluating', asset, flush=True)
        try:
            m, source = load_source(asset, path)
            d = daily_grid(m)
            ind = indicators(d)
            rows, funnel = replay(asset, m, d, ind, cost, pip)
            pd.DataFrame(rows).to_csv(OUT/f'trades_{asset}.csv', index=False)
            result = summarize(asset, rows, source, d, funnel)
            results.append(result)
            print(asset, result['closed_trades'], 'closed, net pips/trade',
                  fmt(result['net_expectancy_pips']), 'p', fmt(result['mbb_p'], 4),
                  'PASS' if result['pass'] else 'FAIL', flush=True)
            del m, d, ind
        except (FileNotFoundError, ValueError, AssertionError, MemoryError) as exc:
            errors[asset] = str(exc)
            print(asset, 'DATA BLOCKED:', exc, flush=True)
    complete = len(results) == len(ASSETS)
    verdict = ('PASS' if complete and all(r['pass'] for r in results)
               else 'FAIL' if complete else 'DATA BLOCKED')
    lookup = {r['asset']: r for r in results}
    annual_rows = []
    summary_rows = []
    for asset, _, cost, pip in ASSETS:
        r = lookup.get(asset)
        if r is None:
            continue
        annual_rows.append({'asset': asset,
                            **{str(y): r['annual'][str(y)]['net_pips']
                               for y in range(2020, 2027)}})
        summary_rows.append({'asset': asset, 'cost_pips': cost/pip,
                             **{k: r[k] for k in ('closed_trades', 'win_rate',
                               'net_expectancy_pips', 'total_net_pips',
                               'net_sharpe_per_trade', 'mbb_p', 'pass')}})
    pd.DataFrame(annual_rows).to_csv(OUT/'annual_matrix.csv', index=False,
                                     float_format='%.6f')
    pd.DataFrame(summary_rows).to_csv(OUT/'summary.csv', index=False,
                                      float_format='%.8f')
    meta = {'study': 'd1_turtle_portfolio', 'spec_sha256': digest(spec),
            'gate_sha256': digest(gate), 'script_sha256': digest(Path(__file__)),
            'overall': verdict, 'results': results, 'errors': errors,
            'bootstrap': {'resamples': BOOT, 'block': BLOCK, 'seed': SEED}}
    (OUT/'run_meta.json').write_text(json.dumps(meta, indent=2, allow_nan=True),
                                     encoding='utf-8')
    lines = ['DAILY TURTLE DONCHIAN FOUR-ASSET TRANSFER - REPORT',
             f'OVERALL FOUR-ASSET GATE: {verdict}',
             '17:00 ET daily M1-open buckets; 2020-2026 entry-cohort scoring.',
             'Net metrics are in pips after one fixed round-turn cost per trade.',
             '', 'ASSET | CLOSED | WIN% | NET PIPS/TRADE | TOTAL NET PIPS | SHARPE | MBB P | STOP | STOP GAP | DONCHIAN10 | GATE']
    for asset, _, _, _ in ASSETS:
        r = lookup.get(asset)
        if r is None:
            lines.append(f'{asset} | NA | NA | NA | NA | NA | NA | NA | NA | NA | DATA BLOCKED')
            continue
        mix = r['exit_mix']
        lines.append(' | '.join((asset, str(r['closed_trades']),
            fmt(100*r['win_rate'], 2), fmt(r['net_expectancy_pips']),
            fmt(r['total_net_pips'], 2), fmt(r['net_sharpe_per_trade'], 4),
            fmt(r['mbb_p'], 4), str(mix['INITIAL_STOP']),
            str(mix['INITIAL_STOP_GAP']), str(mix['DONCHIAN10_CLOSE']),
            'PASS' if r['pass'] else 'FAIL')))
    lines += ['', 'YEAR-BY-YEAR NET PNL MATRIX (PIPS; 2026 PARTIAL)',
              'ASSET | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026']
    for asset, _, _, _ in ASSETS:
        r = lookup.get(asset)
        vals = [fmt(r['annual'][str(y)]['net_pips'], 2) if r else 'NA'
                for y in range(2020, 2027)]
        lines.append(' | '.join([asset, *vals]))
    lines += ['', 'SOURCE AND GATE DETAILS']
    for asset, _, cost, pip in ASSETS:
        r = lookup.get(asset)
        if r is None:
            lines.append(f'{asset}: DATA BLOCKED {errors.get(asset)}')
        else:
            lines.append(f"{asset}: D1 {r['first_daily_day']}..{r['last_daily_day']} "
                         f"({r['daily_bars']} bars); source {r['source']['source']}; "
                         f"SHA256 {r['source']['sha256']}; raw M1 "
                         f"{r['source']['raw_rows']}, observed M1 "
                         f"{r['source']['observed_m1']}, carried removed "
                         f"{r['source']['carried_removed']}, 2024-10 repairs "
                         f"{r['source']['known_2024_10_ohlc_repaired']}; "
                         f"cost {cost/pip:g} pips; open_end {r['open_end']}; "
                         f"gates {r['gates']}; annual counts "
                         f"{ {y: v['trades'] for y, v in r['annual'].items()} }; "
                         f"funnel {r['funnel']}")
    lines += ['', 'LIMITATIONS',
              'BID proxy plus fixed cost floor omits variable spread, slippage, swaps and account-level FTMO loss limits.',
              'Hard stops use chronological M1 extrema with adverse open gap fills. Daily close exits fill at the next real daily open.',
              'Open-end trades are not included in closed PnL, Sharpe or MBB. Different source end dates make 2026 partial across assets.',
              'Price-unit PnL is converted to pips per asset; pips are not currency-adjusted or risk-weighted portfolio returns.']
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Wrote', report, flush=True)


if __name__ == '__main__':
    main()
