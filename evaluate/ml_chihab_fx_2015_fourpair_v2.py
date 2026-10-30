"""Frozen V2 four-pair hourly RF+Probit FX measurement from 2015 BID M1."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from categorical_omnibus import continuous_h1, digest, mbb_p
from ml_chihab_replication import accuracy_summary, execute, feature_table, walk_forward

ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD = Path(r'C:\Users\Milad\download')
OUT = ROOT / 'data/results/ml_chihab_fx_2015_fourpair_v2'
START = int(pd.Timestamp('2015-01-01', tz='UTC').timestamp())
END = int(pd.Timestamp('2026-09-20', tz='UTC').timestamp())
SOURCES = {
    'EURUSD': (DOWNLOAD / 'eurusd-m1-bid-2015-01-01-2019-12-31.csv',
               ROOT / 'data/results/eurusd_onboarding/eurusdtick_m1.csv'),
    'GBPUSD': (DOWNLOAD / 'gbpusd-m1-bid-2015-01-01-2026-09-27.csv',),
    'USDJPY': (DOWNLOAD / 'usdjpy-m1-bid-2015-01-01-2026-09-27.csv',),
    'EURGBP': (DOWNLOAD / 'eurgbp-m1-bid-2015-01-01-2026-09-27.csv',),
}
PIP = {'EURUSD': .0001, 'GBPUSD': .0001, 'USDJPY': .01, 'EURGBP': .0001}
COST = {
    'EURUSD': {'upper': .7, 'lower': .6},
    'GBPUSD': {'upper': .9, 'lower': .7},
    'USDJPY': {'upper': .9, 'lower': .7},
    'EURGBP': {'upper': 1.19, 'lower': .79},
}


def _annual_counts(t):
    years = pd.to_datetime(t, unit='s', utc=True).year.to_numpy()
    return {str(y): int((years == y).sum()) for y in range(2015, 2027)}


def _read_user_bid(path, pair):
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, usecols=['timestamp', 'open', 'high', 'low', 'close'])
    stamp = df.timestamp.to_numpy(np.int64)
    if len(stamp) < 1_000_000 or not np.all(stamp[1:] > stamp[:-1]):
        raise AssertionError(f'{pair}: sparse, duplicate or unsorted source {path}')
    if not np.all(stamp % 60000 == 0):
        raise AssertionError(f'{pair}: non-minute timestamp in {path}')
    time = stamp // 1000
    o, h, l, c = (df[k].to_numpy(float) for k in ('open', 'high', 'low', 'close'))
    if not np.isfinite(np.column_stack((o, h, l, c))).all() or min(
            float(o.min()), float(h.min()), float(l.min()), float(c.min())) <= 0:
        raise AssertionError(f'{pair}: nonfinite or nonpositive OHLC in {path}')
    error = np.maximum.reduce((o-h, c-h, l-o, l-c, l-h, np.zeros(len(o))))
    bad = error > 1e-10
    if bad.any():
        months = pd.to_datetime(time[bad], unit='s', utc=True).strftime('%Y-%m')
        if np.any(months != '2024-10') or float(error.max()) > .31*PIP[pair]:
            raise AssertionError(f'{pair}: unapproved OHLC envelope error in {path}')
    h = np.maximum.reduce((h, o, c))
    l = np.minimum.reduce((l, o, c))
    carried = ((o == h) & (h == l) & (l == c) &
               np.r_[False, c[1:] == c[:-1]])
    keep = ~carried & (time >= START) & (time < END)
    m = {'time': time[keep], 'open': o[keep], 'high': h[keep],
         'low': l[keep], 'close': c[keep]}
    meta = {'path': str(path), 'sha256': digest(path), 'raw_rows': len(df),
            'carried_flat_removed': int(carried.sum()),
            'ohlc_repaired_rows': int(bad.sum()),
            'max_ohlc_error_pips': float(error.max()/PIP[pair]),
            'retained_in_common_period': len(m['time'])}
    return m, meta


def _read_existing_eurusd(path):
    df = pd.read_csv(path, usecols=['time', 'open', 'high', 'low', 'close', 'tick_volume'])
    if not (df.tick_volume.to_numpy() > 0).all():
        raise AssertionError('Existing EURUSD tick-derived tape has nonpositive volume')
    time = df.time.to_numpy(np.int64)
    if not np.all(time[1:] > time[:-1]) or not np.all(time % 60 == 0):
        raise AssertionError('Existing EURUSD M1 timestamps invalid')
    keep = (time >= START) & (time < END)
    m = {k: df[k].to_numpy(np.int64 if k == 'time' else float)[keep]
         for k in ('time', 'open', 'high', 'low', 'close')}
    o, h, l, c = (m[k] for k in ('open', 'high', 'low', 'close'))
    if not np.isfinite(np.column_stack((o, h, l, c))).all() or np.any(
            h < np.maximum(o, c)) or np.any(l > np.minimum(o, c)):
        raise AssertionError('Existing EURUSD OHLC invalid')
    return m, {'path': str(path), 'sha256': digest(path), 'raw_rows': len(df),
               'carried_flat_removed': 0, 'ohlc_repaired_rows': 0,
               'max_ohlc_error_pips': 0.0,
               'retained_in_common_period': len(time[keep])}


def load_pair(pair):
    paths = SOURCES[pair]
    blocks, sources = [], []
    for i, path in enumerate(paths):
        m, meta = (_read_existing_eurusd(path) if pair == 'EURUSD' and i == 1
                   else _read_user_bid(path, pair))
        blocks.append(m)
        sources.append(meta)
    m = {k: np.concatenate([b[k] for b in blocks])
         for k in ('time', 'open', 'high', 'low', 'close')}
    t = m['time']
    if (len(t) < 2_500_000 or not np.all(t[1:] > t[:-1]) or
            int(t[0]) < START or int(t[-1]) >= END or
            int(t[-1]) < int(pd.Timestamp('2026-09-18', tz='UTC').timestamp())):
        raise AssertionError(f'{pair}: invalid stitched/common-period coverage')
    annual = _annual_counts(t)
    if any(count < 100_000 for count in annual.values()):
        raise AssertionError(f'{pair}: insufficient year coverage: {annual}')
    return m, {'pair': pair, 'sources': sources, 'kept_m1': len(t),
               'first_utc': pd.Timestamp(int(t[0]), unit='s', tz='UTC').isoformat(),
               'last_utc': pd.Timestamp(int(t[-1]), unit='s', tz='UTC').isoformat(),
               'annual_kept_m1': annual}


def summarize(rows, accuracy, pip, cost_pips):
    if rows:
        entries = np.array([r['entry_time'] for r in rows], np.int64)
        exits = np.array([r['exit_time'] for r in rows], np.int64)
        if np.any(exits < entries) or np.any(entries[1:] < exits[:-1]):
            raise AssertionError('Overlapping or reversed trades')
        error = max(abs(r['net']/pip -
                        (r['side']*(r['exit_bid']-r['entry_bid'])/pip-cost_pips))
                    for r in rows if r['exit_type'] != 'OPEN_END')
        if error > 1e-8:
            raise AssertionError(f'Net-pip cost identity error {error}')
    else:
        error = None
    closed = [r for r in rows if r['exit_type'] != 'OPEN_END']
    x = np.array([r['net']/pip for r in closed], float)
    if len(x) < 2:
        raise AssertionError('Insufficient closed trades for summary')
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    sharpe = mean/sd if sd > 0 else float('nan')
    win = float((x > 0).mean())
    p = mbb_p(x)
    annual = {}
    for r in closed:
        y = str(r['entry_ny_year'])
        v = annual.setdefault(y, {'trades': 0, 'net_pips': 0.0})
        v['trades'] += 1
        v['net_pips'] += r['net']/pip
    years_positive = all(annual.get(str(y), {}).get('net_pips', 0) > 0
                         for y in range(2016, 2026))
    return {'closed_trades': len(x), 'open_end': len(rows)-len(x),
            'mean_net_pips': mean, 'total_net_pips': float(x.sum()),
            'net_sharpe': sharpe, 'mbb_p': p, 'win_rate': win,
            'accuracy_minus_win_pp': 100*(accuracy['agreement_accuracy']-win),
            'numerical_gate_pass': bool(len(x) >= 150 and sharpe > 0 and p <= .05),
            'research_year_sign_pass': years_positive,
            'ny_overnight_entries': sum(bool(r['ny_overnight']) for r in rows),
            'max_signal_to_open_gap_hours': max(r['entry_signal_gap_hours'] for r in rows),
            'max_cost_identity_error_pips': error, 'annual': annual}


def main():
    spec = OUT/'SPEC_ML_FX_2015_4PAIR_V2.txt'
    gate = OUT/'GATE_ML_FX_2015_4PAIR_V2.txt'
    report = OUT/'REPORT_ML_FX_2015_4PAIR_V2.txt'
    if not spec.exists() or not gate.exists():
        raise FileNotFoundError('Frozen V2 SPEC/GATE must predate the run')
    if report.exists():
        raise FileExistsError('Historical REPORT is immutable; create a named correction')
    results, sources, accuracies, folds_all = [], [], {}, []
    for pair in SOURCES:
        m, source = load_pair(pair)
        grid = continuous_h1(m)
        source['h1_total'] = len(grid[0])
        source['h1_real'] = int(grid[2].sum())
        source['h1_filled'] = source['h1_total']-source['h1_real']
        sources.append(source)
        print(f"{pair}: {source['kept_m1']:,} cleaned M1; {source['h1_real']:,} real H1", flush=True)
        features, valid, atr = feature_table(grid)
        pred, folds = walk_forward(pair, grid, features, valid)
        accuracy = accuracy_summary(pred)
        accuracies[pair] = accuracy
        pred.to_csv(OUT/f'predictions_{pair}.csv', index=False)
        folds_all.append(folds)
        for scenario in ('upper', 'lower'):
            cost_pips = COST[pair][scenario]
            rows, funnel = execute(m, pred, atr, pair, 'CFD', cost_pips*PIP[pair])
            for r in rows:
                r['scenario'] = scenario
                r['cost_pips'] = cost_pips
                r['net_pips'] = (r['net']/PIP[pair] if r['exit_type'] != 'OPEN_END'
                                 else np.nan)
            pd.DataFrame(rows).to_csv(OUT/f'trades_{pair}_{scenario}.csv', index=False)
            stats = summarize(rows, accuracy, PIP[pair], cost_pips)
            results.append({'pair': pair, 'scenario': scenario,
                            'cost_pips': cost_pips, 'stats': stats, 'funnel': funnel})
            print(f"{pair} {scenario} {cost_pips:.2f}p: n={stats['closed_trades']} "
                  f"mean={stats['mean_net_pips']:.4f}p total={stats['total_net_pips']:.1f}p "
                  f"p={stats['mbb_p']:.4f} gate={stats['numerical_gate_pass']}", flush=True)
        del m, grid, features, pred
    pd.concat(folds_all, ignore_index=True).to_csv(OUT/'folds.csv', index=False)
    primary = [r for r in results if r['scenario'] == 'upper']
    overall_pass = all(r['stats']['numerical_gate_pass'] for r in primary)
    meta = {'study': 'ml_chihab_fx_2015_fourpair_v2', 'source_common_period_utc':
            '[2015-01-01, 2026-09-20)', 'script_sha256': digest(Path(__file__)),
            'spec_sha256': digest(spec), 'gate_sha256': digest(gate),
            'source': sources, 'accuracy': accuracies, 'results': results,
            'overall_upper_cost_numerical_gate_pass': overall_pass}
    (OUT/'run_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    lines = ['CHIHAB-STYLE HOURLY FX FOUR-PAIR V2 - PHASE 1 REPORT',
             'PRIMARY UPPER-COST FOUR-PAIR NUMERICAL GATE: '+('PASS' if overall_pass else 'FAIL'),
             'This user-specified 2015-2026 four-pair H1 adaptation is not an exact replication of Chihab et al. (2019).',
             'Owner-provided all-inclusive cost ranges are scenario inputs, not measured historical fills.',
             'Common UTC sample [2015-01-01, 2026-09-20); continuous H1 grid; six-month train/one-month OOS test.',
             'Pips: EURUSD/GBPUSD/EURGBP=0.0001; USDJPY=0.01; cost embedded once in BID proxy fills.', '']
    for source in sources:
        pair = source['pair']
        a = accuracies[pair]
        lines += [f'{pair}: {source["kept_m1"]:,} cleaned M1; real/filled H1 {source["h1_real"]:,}/{source["h1_filled"]:,}; {source["first_utc"]} to {source["last_utc"]}',
                  f'Annual cleaned M1 {source["annual_kept_m1"]}',
                  'Sources/cleaning '+str(source['sources']),
                  f"OOS scored H1 {a['scored_hours']:,}; RF {a['rf_accuracy']:.4%}; Probit {a['probit_accuracy']:.4%}; both-agree {a['agreement_accuracy']:.4%}; agreement coverage {a['agreement_coverage']:.4%}; train-majority baseline {a['train_majority_baseline_accuracy']:.4%}."]
        for scenario in ('upper', 'lower'):
            r = next(x for x in results if x['pair'] == pair and x['scenario'] == scenario)
            s = r['stats']
            lines += [f"{scenario.upper()} COST {r['cost_pips']:.2f} pips | closed {s['closed_trades']} | win {s['win_rate']:.4%} | mean {s['mean_net_pips']:.4f} pips/trade | total {s['total_net_pips']:.2f} pips | Net Sharpe {s['net_sharpe']:.6f} | MBB p {s['mbb_p']:.4f} | numerical gate {'PASS' if s['numerical_gate_pass'] else 'FAIL'}",
                      f"Accuracy minus trade win {s['accuracy_minus_win_pp']:.2f} percentage points (different denominators); NY overnight {s['ny_overnight_entries']}; OPEN_END {s['open_end']}; max signal-to-open gap {s['max_signal_to_open_gap_hours']:.1f} h; cost identity max error {s['max_cost_identity_error_pips']:.2g} pips.",
                      f"Full-year 2016-2025 positive PnL {'YES' if s['research_year_sign_pass'] else 'NO'}; annual entry-cohort PnL {s['annual']}"]
        lines.append('')
    lines += ['Primary gate uses only upper-cost rows; lower-cost rows are a predeclared sensitivity.',
              'Source flat-minute filtering can discard genuine unchanged-price minutes; small October 2024 OHLC envelopes were repaired outward as predeclared.',
              'The user-provided BID M1 data lack contemporaneous ASK quotes; fixed spreads, slippage and zero swap may understate account costs, especially for overnight holds. No live or FTMO eligibility claim is made.']
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Wrote {report}', flush=True)


if __name__ == '__main__':
    main()
