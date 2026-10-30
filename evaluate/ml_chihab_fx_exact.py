"""User-specified hourly Chihab-style FX adaptation; frozen SPEC_ML_FX.txt."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from categorical_omnibus import continuous_h1, digest, mbb_p
from ml_chihab_replication import feature_table, walk_forward, accuracy_summary, execute

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/results/ml_chihab_fx'
RAW = ROOT / 'data/raw/dukascopy_fx'
EURUSD = ROOT / 'data/results/eurusd_onboarding/eurusdtick_m1.csv'
START = int(pd.Timestamp('2020-01-01', tz='UTC').timestamp())
END = int(pd.Timestamp('2026-09-02', tz='UTC').timestamp())
PIP = 0.0001
PAIRS = (('EURUSD', .3), ('GBPUSD', .5), ('EURGBP', .8))


def download_if_missing(pair):
    RAW.mkdir(parents=True, exist_ok=True)
    stem = f'{pair.lower()}_m1_bid_20200101_20260902'
    path = RAW / f'{stem}.csv'
    if path.exists() and path.stat().st_size > 0:
        with path.open('rb') as fh:
            first = fh.readline()
            second = fh.readline()
        with path.open('rb') as fh:
            fh.seek(max(0, path.stat().st_size-256))
            tail = fh.read().splitlines()[-1]
        if (first.startswith(b'timestamp,') and second and tail and
                int(second.split(b',', 1)[0]) // 1000 >= START and
                int(tail.split(b',', 1)[0]) // 1000 >= int(pd.Timestamp('2026-08-31', tz='UTC').timestamp())):
            print(f'{pair}: using existing complete raw M1 BID {path}', flush=True)
            return path, 'existing'
        failed = RAW / f'{stem}.failed_bulk.partial.csv'
        path.replace(failed)
        print(f'{pair}: preserved incomplete bulk download as {failed.name}', flush=True)
    npx = shutil.which('npx.cmd') or r'C:\Program Files\nodejs\npx.cmd'
    chunks = RAW / f'{pair.lower()}_weekly'
    chunks.mkdir(exist_ok=True)
    bounds = pd.date_range('2020-01-01', '2026-09-02', freq='7D', tz='UTC').tolist()
    if bounds[-1] != pd.Timestamp('2026-09-02', tz='UTC'):
        bounds.append(pd.Timestamp('2026-09-02', tz='UTC'))
    log = OUT / f'download_{pair}_weekly.log'
    print(f'{pair}: downloading {len(bounds)-1} observed-BID weekly chunks', flush=True)
    for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
        chunk = chunks / f'{start:%Y%m%d}_{end:%Y%m%d}.csv'
        if chunk.exists() and chunk.stat().st_size > 40:
            continue
        base = [npx, '--no-install', 'dukascopy-node', '-i', pair.lower(),
                '-from', start.strftime('%Y-%m-%d'), '-to', end.strftime('%Y-%m-%d'),
                '-t', 'm1', '-p', 'bid', '-f', 'csv', '-v', '-vu', 'units',
                '-dir', str(chunks), '-fn', chunk.stem]
        for attempt in range(4):
            throttle = (['-bs', '10', '-bp', '1000', '-r', '3', '-rp', '1000']
                        if attempt == 0 else
                        ['-bs', '2', '-bp', '2000', '-r', '5', '-rp', '5000'])
            argv = base + throttle
            run = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                                 timeout=900, check=False)
            with log.open('a', encoding='utf-8') as fh:
                fh.write(f'week {i}/{len(bounds)-1} attempt {attempt+1}: '
                         + ' '.join(argv) + '\n' + run.stdout + run.stderr + '\n')
            if run.returncode == 0 and chunk.exists() and chunk.stat().st_size > 40:
                break
            if chunk.exists():
                chunk.unlink()
            time.sleep(15 * (attempt+1))
        else:
            raise RuntimeError(f'{pair}: Dukascopy HTTP failure at {start:%Y-%m-%d} '
                               f'through {end:%Y-%m-%d}; completed chunks preserved; see {log}')
        if i % 20 == 0:
            print(f'{pair}: downloaded {i}/{len(bounds)-1} weekly chunks', flush=True)
    tmp = path.with_suffix('.assembling.csv')
    with tmp.open('wb') as out:
        for i, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
            chunk = chunks / f'{start:%Y%m%d}_{end:%Y%m%d}.csv'
            with chunk.open('rb') as src:
                header = src.readline()
                if header != b'timestamp,open,high,low,close,volume\n':
                    raise RuntimeError(f'{pair}: unexpected chunk header: {chunk}')
                if i == 0:
                    out.write(header)
                shutil.copyfileobj(src, out)
    tmp.replace(path)
    print(f'{pair}: assembled {path.stat().st_size:,} bytes', flush=True)
    return path, 'downloaded'


def validate_m1(m, pair, source):
    t = m['time']
    if len(t) < 1_500_000 or not np.all(t[1:] > t[:-1]):
        raise AssertionError(f'{pair}: insufficient or nonunique M1 observations')
    if int(t[0]) < START or int(t[-1]) >= END:
        raise AssertionError(f'{pair}: source outside frozen date range')
    if int(t[-1]) < int(pd.Timestamp('2026-08-31', tz='UTC').timestamp()):
        raise AssertionError(f'{pair}: source does not reach Aug 31, 2026')
    year = pd.to_datetime(t, unit='s', utc=True).year.to_numpy()
    annual = {str(y): int((year == y).sum()) for y in range(2020, 2027)}
    if any(v < 100_000 for v in annual.values()):
        raise AssertionError(f'{pair}: one calendar year has fewer than 100k M1 observations')
    arr = np.column_stack([m[k] for k in ('open', 'high', 'low', 'close')])
    if not np.isfinite(arr).all() or np.any(arr <= 0):
        raise AssertionError(f'{pair}: invalid M1 OHLC')
    if np.any(m['high'] < np.maximum(m['open'], m['close'])) or np.any(m['low'] > np.minimum(m['open'], m['close'])):
        raise AssertionError(f'{pair}: inconsistent OHLC')
    return {'pair': pair, 'source': str(source), 'sha256': digest(source),
            'm1_rows': len(t), 'first_utc': pd.Timestamp(int(t[0]),unit='s',tz='UTC').isoformat(),
            'last_utc': pd.Timestamp(int(t[-1]),unit='s',tz='UTC').isoformat(),
            'annual_m1': annual}


def load_pair(pair):
    if pair == 'EURUSD':
        path = EURUSD
        df = pd.read_csv(path, usecols=['time', 'open', 'high', 'low', 'close', 'tick_volume'])
        if not np.all(df.tick_volume.to_numpy() > 0):
            raise AssertionError('EURUSD tick-derived source contains zero-volume M1 bars')
        status = 'existing_tick_derived'
    else:
        path, status = download_if_missing(pair)
        df = pd.read_csv(path, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if not np.all(df.volume.to_numpy(float) > 0):
            raise AssertionError(f'{pair}: Dukascopy CLI returned zero-volume flat bars')
        stamp = df.timestamp.to_numpy(np.int64)
        if not np.all(stamp % 60000 == 0):
            raise AssertionError(f'{pair}: non-M1-aligned CLI timestamps')
        df['time'] = stamp // 1000
    df = df.loc[(df.time >= START) & (df.time < END)]
    m = {k: df[k].to_numpy(np.int64 if k == 'time' else float)
         for k in ('time', 'open', 'high', 'low', 'close')}
    source = validate_m1(m, pair, path)
    source['acquisition_status'] = status
    return m, source


def fx_summary(rows, accuracy):
    closed = [r for r in rows if r['exit_type'] != 'OPEN_END']
    pips = np.array([r['net']/PIP for r in closed], float)
    n = len(pips)
    mean = float(pips.mean()) if n else float('nan')
    sd = float(pips.std(ddof=1)) if n > 1 else float('nan')
    sharpe = mean/sd if sd > 0 else float('nan')
    win = float((pips > 0).mean()) if n else float('nan')
    yearly = {}
    for r in closed:
        y = str(r['entry_ny_year'])
        v = yearly.setdefault(y, {'trades': 0, 'net_pips': 0.0})
        v['trades'] += 1
        v['net_pips'] += r['net']/PIP
    year_sign = all(yearly.get(str(y), {}).get('net_pips', 0) > 0 for y in range(2021, 2026))
    p = mbb_p(pips)
    return {'entries': len(rows), 'closed_trades': n, 'open_end': len(rows)-n,
            'open_end_mark_pips': sum(r['open_end_mark_net']/PIP for r in rows
                                      if r['exit_type'] == 'OPEN_END'),
            'net_expectancy_pips': mean, 'total_net_pips': float(pips.sum()),
            'net_sharpe': sharpe, 'mbb_p': p, 'trade_win_rate': win,
            'accuracy_minus_trade_win_pp': 100*(accuracy['agreement_accuracy']-win),
            'gate_pass': bool(n >= 150 and sharpe > 0 and p <= .05),
            'research_year_sign_pass': year_sign,
            'warning_exits': sum(r['exit_type'] == 'MODEL_WARNING' for r in rows),
            'ny_overnight_entries': sum(r['ny_overnight'] for r in rows),
            'max_signal_to_open_gap_hours': max((r['entry_signal_gap_hours'] for r in rows), default=None),
            'yearly': yearly}


def main():
    if not (OUT/'SPEC_ML_FX.txt').exists() or not (OUT/'GATE_ML_FX.txt').exists() or not (OUT/'SPEC_ADDENDUM_SOURCE_VOLUME_ML_FX.txt').exists():
        raise RuntimeError('Frozen SPEC/GATE/source addendum must predate execution')
    report = OUT/'REPORT_ML_FX.txt'
    if report.exists():
        raise FileExistsError('Historical report is immutable; use a named correction')
    sources, results, accuracies, folds_all = [], [], {}, []
    for pair, cost_pips in PAIRS:
        m, source = load_pair(pair)
        grid = continuous_h1(m)
        source['h1_total'] = len(grid[0])
        source['h1_real'] = int(grid[2].sum())
        source['h1_filled'] = len(grid[0])-source['h1_real']
        sources.append(source)
        print(f"{pair}: {source['m1_rows']:,} observed M1 bars; H1 real {source['h1_real']:,}", flush=True)
        x, valid, atr = feature_table(grid)
        pred, folds = walk_forward(pair, grid, x, valid)
        accuracy = accuracy_summary(pred)
        accuracies[pair] = accuracy
        pred.to_csv(OUT/f'predictions_{pair}.csv', index=False)
        folds_all.append(folds)
        rows, funnel = execute(m, pred, atr, pair, 'CFD', cost_pips*PIP)
        for r in rows:
            r['cost_pips'] = cost_pips
            r['net_pips'] = r['net']/PIP if r['exit_type'] != 'OPEN_END' else np.nan
            r['open_end_mark_pips'] = r['open_end_mark_net']/PIP if r['exit_type'] == 'OPEN_END' else np.nan
        pd.DataFrame(rows).to_csv(OUT/f'trades_{pair}.csv', index=False)
        stats = fx_summary(rows, accuracy)
        results.append({'pair': pair, 'cost_pips': cost_pips, 'funnel': funnel,
                        'stats': stats})
        print(f"{pair}: OOS agree accuracy {accuracy['agreement_accuracy']:.4f}; "
              f"closed {stats['closed_trades']}; mean {stats['net_expectancy_pips']:.4f} pips; "
              f"total {stats['total_net_pips']:.2f} pips; gate {stats['gate_pass']}", flush=True)
        del m, grid, x, pred
    pd.concat(folds_all, ignore_index=True).to_csv(OUT/'folds.csv', index=False)
    meta = {'study': 'ml_chihab_fx_hourly_adaptation', 'date': '2026-09-27',
            'paper': 'https://onlinelibrary.wiley.com/doi/10.1155/2019/8342461',
            'script_sha256': digest(Path(__file__)), 'pip_size': PIP,
            'sources': sources, 'accuracy': accuracies, 'results': results,
            'overall_gate_pass': all(r['stats']['gate_pass'] for r in results)}
    (OUT/'run_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    lines = ['CHIHAB-STYLE RF + PROBIT HOURLY FX ADAPTATION - REPORT',
             'OVERALL GATE: '+('PASS' if meta['overall_gate_pass'] else 'FAIL'),
             'The user-specified EURUSD/GBPUSD/EURGBP H1 test is NOT an exact replication of the 2019 intraweek USD/EUR, USD/JPY, USD/GBP study.',
             meta['paper'],
             'Seven frozen V2 features; rolling six-month train/one-month strictly OOS test; real H1 labels and next-real-M1 executions.',
             'Pip=0.0001 quote units; fixed bid/ask round-trip cost embedded once; zero swap assumed though overnight positions are counted.', '']
    for src in sources:
        a = accuracies[src['pair']]
        r = next(v for v in results if v['pair'] == src['pair'])
        s = r['stats']
        lines += [f"{src['pair']} | CFD cost {r['cost_pips']:.1f} pips | GATE {'PASS' if s['gate_pass'] else 'FAIL'}",
                  f"Source {src['source']} ({src['acquisition_status']}); SHA256 {src['sha256']}",
                  f"Observed M1 {src['m1_rows']}; {src['first_utc']} through {src['last_utc']}; H1 real/filled {src['h1_real']}/{src['h1_filled']}; annual M1 {src['annual_m1']}",
                  f"OOS classification: scored {a['scored_hours']}; RF accuracy {a['rf_accuracy']:.4f}; Probit accuracy {a['probit_accuracy']:.4f}; both-agree accuracy {a['agreement_accuracy']:.4f}; agreement coverage {a['agreement_coverage']:.4f}; train-majority baseline {a['train_majority_baseline_accuracy']:.4f}",
                  f"Closed trades {s['closed_trades']}; realized win rate {s['trade_win_rate']:.2%}; net expectancy {s['net_expectancy_pips']:.4f} pips/trade; total net PnL {s['total_net_pips']:.2f} pips; Net Sharpe {s['net_sharpe']:.5f}; MBB p {s['mbb_p']:.4f}",
                  f"Both-agree accuracy minus realized win rate {s['accuracy_minus_trade_win_pp']:.2f} percentage points (different denominators)",
                  f"Warning exits {s['warning_exits']}; OPEN_END {s['open_end']} marked {s['open_end_mark_pips']:.2f} pips outside closed stats; NY overnight entries {s['ny_overnight_entries']}; max signal-to-open gap {s['max_signal_to_open_gap_hours']} h",
                  f"Full-year sign qualification {'PASS' if s['research_year_sign_pass'] else 'FAIL'}; annual entry-cohort PnL {s['yearly']}",
                  f"Execution funnel {r['funnel']}", '']
    lines += ['Interpretation: Classification accuracy measures next-H1 direction, whereas trade win rate and PnL reflect model-warning holding time and bid/ask cost; their difference is diagnostic, not directly a profit conversion.',
              'The original paper used different pairs, years and forecast horizon. BID M1 proxy fills, fixed spread and zero swap cannot establish account-specific executable profitability or FTMO compliance.']
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def finalize_blocked_acquisition():
    """Record the observed external-data failure without inventing pair results."""
    report = OUT/'REPORT_ML_FX.txt'
    if report.exists():
        raise FileExistsError('Historical report is immutable')
    trade_path = OUT/'trades_EURUSD.csv'
    pred_path = OUT/'predictions_EURUSD.csv'
    if not trade_path.exists() or not pred_path.exists():
        raise RuntimeError('EURUSD partial result has not been executed')
    trades = pd.read_csv(trade_path)
    prediction = pd.read_csv(pred_path)
    a = accuracy_summary(prediction)
    closed = trades.loc[trades.exit_type != 'OPEN_END']
    x = closed.net_pips.to_numpy(float)
    if len(x) < 150 or not np.isfinite(x).all():
        raise AssertionError('EURUSD partial ledger invalid')
    mean = float(x.mean())
    sharpe = mean / float(x.std(ddof=1))
    win = float((x > 0).mean())
    p = mbb_p(x)
    annual = closed.groupby('entry_ny_year').net_pips.agg(['count', 'sum'])
    lines = [
        'CHIHAB-STYLE RF + PROBIT HOURLY FX ADAPTATION - DATA-BLOCKED REPORT',
        'OVERALL THREE-PAIR GATE: NOT EVALUABLE; GBPUSD AND EURGBP FULL TAPES ARE UNAVAILABLE.',
        'The requested EURUSD/GBPUSD/EURGBP H1 test is an adaptation, not an exact replication of the 2019 paper.',
        'The script was executed. GBPUSD bulk retrieval failed with HTTP 429 after a partial 2020 file; weekly checkpoint retrieval repeatedly failed for 2020-04-08 through 2020-04-15, and the 2020-04-08 daily request also failed with HTTP 429. A full-period EURGBP bulk request likewise failed with HTTP 429.',
        'No incomplete GBPUSD/EURGBP file was admitted into model fitting or execution. GBPUSD successful weekly chunks are preserved for a later independent recovery study.',
        '',
        'EURUSD ONLY - completed out-of-sample evaluation on the existing documented Dukascopy BID tick-derived M1 tape:',
        f'Source {EURUSD}; SHA256 {digest(EURUSD)}',
        f"Scored H1 predictions {a['scored_hours']}; RF accuracy {a['rf_accuracy']:.4%}; Probit accuracy {a['probit_accuracy']:.4%}; both-agree accuracy {a['agreement_accuracy']:.4%}; agreement coverage {a['agreement_coverage']:.4%}; train-majority baseline {a['train_majority_baseline_accuracy']:.4%}.",
        f'Closed trades {len(x)}; realized win rate {win:.4%}; net expectancy {mean:.4f} pips/trade; total net PnL {float(x.sum()):.2f} pips; Net Sharpe {sharpe:.6f}; 10,000-resample 5-trade MBB p-value {p:.4f}.',
        f'Both-agree accuracy minus executed trade win rate {100*(a["agreement_accuracy"]-win):.2f} percentage points; these use different denominators.',
        f"EURUSD predeclared numerical gate {'PASS' if len(x)>=150 and sharpe>0 and p<=.05 else 'FAIL'}; full-year sign qualification FAIL; NY overnight entries {int(trades.ny_overnight.sum())}; OPEN_END {int((trades.exit_type=='OPEN_END').sum())} excluded from closed statistics.",
        'Annual entry-cohort net pips: '+str({int(y): {'trades': int(v['count']), 'net_pips': round(float(v['sum']), 2)} for y, v in annual.iterrows()}),
        '',
        'GBPUSD: NOT TESTED. EURGBP: NOT TESTED. No three-pair performance or gate verdict can be inferred.',
        'Zero swaps and fixed spread remain optimistic for positions held overnight; this proxy test does not establish executable account profitability.'
    ]
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (OUT/'run_meta.json').write_text(json.dumps({
        'status': 'DATA_BLOCKED', 'full_three_pair_test_complete': False,
        'script_sha256': digest(Path(__file__)),
        'eurusd': {'closed_trades': len(x), 'agreement_accuracy': a['agreement_accuracy'],
                   'win_rate': win, 'net_expectancy_pips': mean,
                   'total_net_pips': float(x.sum()), 'net_sharpe': sharpe, 'mbb_p': p},
        'unavailable': ['GBPUSD', 'EURGBP']}, indent=2), encoding='utf-8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--finalize-blocked':
        finalize_blocked_acquisition()
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit('usage: ml_chihab_fx_exact.py [--finalize-blocked]')
