"""Frozen four-asset Engine A/B transfer; see SPEC_MACRO_PORTFOLIO.txt."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from futures_prop_validation_phase1 import engine_a, engine_b, grids
from gold_ensemble_real_cost import replay


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD = Path(r'C:\Users\Milad\download')
OUT = ROOT/'data/results/macro_portfolio'
ASSETS = (
    ('XAGUSD', 'xagusd', .025),
    ('GBPJPY', 'gbpjpy', .025),
    ('USDJPY', 'usdjpy', .009),
    ('EURJPY', 'eurjpy', .015),
)
START = '2010-01-01'
END = '2026-09-27'
SCORE_START = int(pd.Timestamp('2020-01-01', tz='UTC').timestamp())
SCORE_END = int(pd.Timestamp('2026-09-27', tz='UTC').timestamp())
BOOT, BLOCK, SEED = 10_000, 5, 20261015


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for part in iter(lambda: fh.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def command(slug: str) -> list[str]:
    return ['npx', 'dukascopy-node', '-i', slug, '-from', START,
            '-to', END, '-t', 'm1', '-f', 'csv', '--cache']


def expected_file(slug: str) -> Path:
    # dukascopy-node creates a "download" subdirectory relative to its cwd.
    return DOWNLOAD/'download'/f'{slug}-m1-bid-{START}-{END}.csv'


def complete_file(path: Path) -> bool:
    """Reject a partially written file left by an interrupted download."""
    if not path.exists() or path.stat().st_size <= 1_000_000:
        return False
    with path.open('rb') as fh:
        if fh.readline().strip() != b'timestamp,open,high,low,close':
            return False
        first = fh.readline().split(b',', 1)[0]
        fh.seek(max(0, path.stat().st_size - 1024))
        last = fh.read().splitlines()[-1].split(b',', 1)[0]
    try:
        return (int(first) <= 1262563200000 and
                int(last) >= 1790294400000)
    except ValueError:
        return False


def acquire(slug: str) -> tuple[Path | None, dict]:
    """Run the owner's exact npx argument sequence if a full-span file is absent."""
    target = expected_file(slug)
    args = command(slug)
    provenance = {'command': ' '.join(args), 'cwd': str(DOWNLOAD),
                  'expected_file': str(target), 'attempted': False}
    for candidate in (target, DOWNLOAD/target.name):
        if complete_file(candidate):
            provenance.update(status='EXISTING_FILE', bytes=candidate.stat().st_size,
                              produced_file=str(candidate))
            return candidate, provenance
    exe = shutil.which('npx.cmd') or shutil.which('npx')
    if exe is None:
        provenance.update(status='NPX_NOT_FOUND')
        return None, provenance
    print('Downloading', slug, 'with', ' '.join(args), flush=True)
    provenance['attempted'] = True
    try:
        run = subprocess.run([exe, *args[1:]], cwd=DOWNLOAD, capture_output=True,
                             text=True, errors='replace', timeout=1800, check=False)
        log = (f"COMMAND: {' '.join(args)}\nEXECUTABLE: {exe}\n"
               f"RETURN CODE: {run.returncode}\nSTDOUT:\n{run.stdout}\nSTDERR:\n{run.stderr}\n")
        (OUT/f'download_{slug}.log').write_text(log, encoding='utf-8')
        provenance.update(return_code=run.returncode,
                          stdout_tail=run.stdout[-1200:], stderr_tail=run.stderr[-1200:])
        if run.returncode != 0:
            provenance['status'] = 'COMMAND_FAILED'
        elif not target.exists():
            provenance['status'] = 'OUTPUT_MISSING'
        elif not complete_file(target):
            provenance['status'] = 'OUTPUT_INCOMPLETE'
        else:
            provenance.update(status='DOWNLOADED', bytes=target.stat().st_size,
                              produced_file=str(target))
            return target, provenance
    except subprocess.TimeoutExpired as exc:
        (OUT/f'download_{slug}.log').write_text(
            f"COMMAND: {' '.join(args)}\nTIMEOUT AFTER {exc.timeout} SECONDS\n",
            encoding='utf-8')
        provenance.update(status='TIMEOUT', timeout_seconds=exc.timeout)
    except OSError as exc:
        (OUT/f'download_{slug}.log').write_text(
            f"COMMAND: {' '.join(args)}\nOS ERROR: {exc}\n", encoding='utf-8')
        provenance.update(status='OS_ERROR', error=str(exc))
    return None, provenance


def load_source(path: Path, asset: str) -> tuple[dict, dict]:
    d = pd.read_csv(path, usecols=['timestamp', 'open', 'high', 'low', 'close'])
    stamp = d['timestamp'].to_numpy(np.int64)
    if (len(stamp) < 2_000_000 or np.any(np.diff(stamp) <= 0)
            or np.any(stamp % 60000 != 0)):
        raise ValueError('Incomplete, duplicated or non-minute M1 timestamp series')
    t = stamp//1000
    o, h, l, c = (d[k].to_numpy(float) for k in ('open', 'high', 'low', 'close'))
    if not np.isfinite(np.column_stack((o, h, l, c))).all() or np.any(
            np.minimum.reduce((o, h, l, c)) <= 0):
        raise ValueError('Nonfinite or nonpositive OHLC')
    error = np.maximum.reduce((o-h, c-h, l-o, l-c, l-h, np.zeros(len(o))))
    bad = error > 1e-10
    repair = 0
    if bad.any():
        # Known Dukascopy October-2024 JPY export envelope errors, previously audited.
        months = pd.to_datetime(t[bad], unit='s', utc=True).strftime('%Y-%m')
        if (asset not in ('GBPJPY', 'USDJPY', 'EURJPY')
                or np.any(months != '2024-10') or float(error.max()) > .0031):
            raise ValueError(f'Unapproved OHLC envelope error max={error.max():.8f}')
        repair = int(bad.sum())
        h = np.maximum.reduce((h, o, c))
        l = np.minimum.reduce((l, o, c))
    carried = ((o == h) & (h == l) & (l == c)
               & np.r_[False, c[1:] == c[:-1]])
    keep = ~carried
    m = {'time': t[keep], 'open': o[keep], 'high': h[keep],
         'low': l[keep], 'close': c[keep]}
    if (m['time'][0] > int(pd.Timestamp('2010-01-04', tz='UTC').timestamp())
            or m['time'][-1] < int(pd.Timestamp('2026-09-25', tz='UTC').timestamp())
            or len(m['time']) < 2_000_000):
        raise ValueError('Full 2010-2026 observed M1 coverage not present')
    years = pd.to_datetime(m['time'], unit='s', utc=True).year.to_numpy()
    annual = {str(y): int(np.count_nonzero(years == y)) for y in range(2010, 2027)}
    if any(annual[str(y)] < 100_000 for y in range(2011, 2026)):
        raise ValueError(f'Insufficient full-year observed M1 coverage {annual}')
    meta = {'path': str(path), 'sha256': digest(path), 'raw_rows': len(d),
            'cleaned_observed_m1': len(m['time']), 'carried_flat_removed': int(carried.sum()),
            'known_2024_10_ohlc_repaired': repair,
            'max_ohlc_envelope_error': float(error.max()),
            'first_utc': int(m['time'][0]), 'last_utc': int(m['time'][-1]),
            'annual_observed_m1': annual}
    return m, meta


def mbb_p(x: np.ndarray) -> float:
    n = len(x)
    if n < BLOCK:
        return np.nan
    rng = np.random.default_rng(SEED)
    block_sums = np.convolve(x, np.ones(BLOCK), mode='valid')
    count, extra = divmod(n, BLOCK)
    tail = np.convolve(x, np.ones(extra), mode='valid') if extra else None
    bad = 0
    for offset in range(0, BOOT, 250):
        k = min(250, BOOT-offset)
        starts = rng.integers(0, n-BLOCK+1, (k, count+int(extra > 0)))
        totals = block_sums[starts[:, :count]].sum(axis=1)
        if extra:
            totals += tail[starts[:, count]]
        bad += int(np.count_nonzero(totals <= 0))
    return bad/BOOT


def summarize(rows: list[dict]) -> dict:
    closed = [r for r in rows if r['exit_type'] != 'END']
    x = np.array([r['net'] for r in closed], float)
    gross = np.array([r['gross'] for r in closed], float)
    mean = float(x.mean()) if len(x) else np.nan
    sd = float(x.std(ddof=1)) if len(x) >= 2 else np.nan
    sharpe = mean/sd if np.isfinite(sd) and sd > 0 else np.nan
    p = mbb_p(x)
    annual = {str(y): {'trades': 0, 'net': 0.0} for y in range(2020, 2027)}
    for r in closed:
        y = str(r['entry_ny_year'])
        if y not in annual:
            annual[y] = {'trades': 0, 'net': 0.0}
        annual[y]['trades'] += 1
        annual[y]['net'] += r['net']
    gates = {'G1': len(x) >= 150, 'G2': bool(mean > 0),
             'G3': bool(sharpe > 0), 'G4': bool(p <= .05),
             'G5': all(annual[str(y)]['net'] >= -1e-10 for y in range(2020, 2026))}
    return {'closed_trades': len(x), 'open_end': len(rows)-len(x),
            'win_rate': float(np.mean(x > 0)) if len(x) else np.nan,
            'gross_expectancy': float(gross.mean()) if len(gross) else np.nan,
            'net_expectancy': mean, 'net_total': float(x.sum()),
            'net_sharpe_per_trade': float(sharpe), 'mbb_p': p,
            'annual': annual, 'overnight_trades': sum(r['ny_overnight'] for r in closed),
            'exit_counts': {name: sum(r['exit_type'] == name for r in closed)
                            for name in ('SL_INIT', 'SL_TRAIL')},
            'gates': gates, 'pass': all(gates.values())}


def evaluate_asset(asset: str, m: dict, cost: float) -> tuple[list[dict], list[dict]]:
    g = grids(m)
    engines = (engine_a(g), engine_b(g))
    results = []
    monthly = []
    for e in engines:
        all_rows, funnel = replay(m, e, cost)
        rows = [r for r in all_rows if SCORE_START <= r['entry_time'] < SCORE_END]
        name = e['name']
        pd.DataFrame(rows).to_csv(OUT/f'trades_{asset}_{name}.csv', index=False)
        stats = summarize(rows)
        result = {'asset': asset, 'engine': name, 'cost': cost,
                  'full_history_replay_entries': len(all_rows),
                  'scored_entries': len(rows), 'funnel': funnel, **stats}
        results.append(result)
        for row in rows:
            month = pd.Timestamp(int(row['entry_time']), unit='s', tz='UTC').tz_convert(
                'America/New_York').strftime('%Y-%m')
            monthly.append({'month': month, 'asset': asset, 'engine': name,
                            'entry_time': int(row['entry_time'])})
        print(asset, name, stats['closed_trades'], 'closed, net mean',
              f"{stats['net_expectancy']:.6f}", 'p', f"{stats['mbb_p']:.4f}",
              'gate', 'PASS' if stats['pass'] else 'FAIL', flush=True)
    return results, monthly


def fmt(v, precision=5):
    return 'NA' if not np.isfinite(v) else f'{v:.{precision}f}'


def main():
    spec, gate, report = (OUT/n for n in ('SPEC_MACRO_PORTFOLIO.txt',
                                         'GATE_MACRO_PORTFOLIO.txt',
                                         'REPORT_MACRO_PORTFOLIO.txt'))
    if not spec.exists() or not gate.exists():
        raise FileNotFoundError('Frozen SPEC and GATE must precede this run')
    if report.exists():
        raise FileExistsError('Historical REPORT is immutable; use a new study')
    extracted = '\n'.join(line for line in spec.read_text(encoding='utf-8').splitlines()
                          if line.startswith('G') and len(line) > 2 and line[1].isdigit())+'\n'
    if extracted != gate.read_text(encoding='utf-8'):
        raise AssertionError('GATE is not the verbatim SPEC extraction')
    acquisition = {}
    files = {}
    for asset, slug, _ in ASSETS:
        path, info = acquire(slug)
        acquisition[asset] = info
        files[asset] = path
        print('Acquisition', asset, info['status'], flush=True)
    results = []
    source_meta = {}
    monthly = []
    for asset, _, cost in ASSETS:
        path = files[asset]
        if path is None:
            continue
        try:
            m, meta = load_source(path, asset)
            source_meta[asset] = meta
            asset_results, asset_monthly = evaluate_asset(asset, m, cost)
            results.extend(asset_results)
            monthly.extend(asset_monthly)
            del m
        except (ValueError, KeyError, AssertionError, MemoryError) as exc:
            acquisition[asset]['status'] = 'SOURCE_OR_EVALUATION_BLOCKED'
            acquisition[asset]['reason'] = str(exc)
            print(asset, 'blocked:', exc, flush=True)
    month_index = pd.period_range('2020-01', '2026-09', freq='M').astype(str)
    monthly_frame = pd.DataFrame({'month': month_index})
    if monthly:
        counts = pd.DataFrame(monthly).groupby('month').size().rename('available_asset_new_trades')
        monthly_frame = monthly_frame.join(counts, on='month')
    monthly_frame['available_asset_new_trades'] = monthly_frame[
        'available_asset_new_trades'].fillna(0).astype(int)
    complete = len(results) == 8 and all(a in source_meta for a, _, _ in ASSETS)
    monthly_frame['four_asset_total_evaluable'] = complete
    monthly_frame.to_csv(OUT/'monthly_portfolio.csv', index=False)
    overall = ('PASS' if complete and all(r['pass'] for r in results)
               else 'FAIL' if complete else 'DATA BLOCKED')
    metadata = {'study': 'macro_portfolio', 'spec_sha256': digest(spec),
                'gate_sha256': digest(gate), 'script_sha256': digest(Path(__file__)),
                'acquisition': acquisition, 'sources': source_meta,
                'results': results, 'overall': overall,
                'partial_monthly_new_trades': int(len(monthly)),
                'partial_monthly_mean': float(monthly_frame.available_asset_new_trades.mean())}
    (OUT/'run_meta.json').write_text(json.dumps(metadata, indent=2, allow_nan=True),
                                     encoding='utf-8')
    lines = ['MACRO ENGINE A/B FOUR-ASSET PORTFOLIO TRANSFER - REPORT',
             'Requested full 2010-2026 Dukascopy M1 BID sources, 2020-2026 entry-cohort scoring.',
             'A 4H Keltner; B 1H Saleh Donchian. Fixed owner retail cost floors embedded once.',
             f'OVERALL FOUR-ASSET GATE: {overall}', '', 'ACQUISITION STATUS']
    for asset, _, _ in ASSETS:
        lines.append(f'{asset}: {acquisition[asset]}')
    lines += ['', 'ASSET | ENGINE | CLOSED | WIN% | GROSS/TRADE | NET/TRADE | TOTAL NET | SHARPE | MBB P | GATE']
    lookup = {(r['asset'], r['engine']): r for r in results}
    for asset, _, _ in ASSETS:
        for engine in ('A_KC', 'B_DONCHIAN'):
            r = lookup.get((asset, engine))
            if r is None:
                lines.append(f'{asset} | {engine} | NA | NA | NA | NA | NA | NA | NA | DATA BLOCKED')
            else:
                lines.append(' | '.join((asset, engine, str(r['closed_trades']),
                    fmt(100*r['win_rate'], 2), fmt(r['gross_expectancy'], 6),
                    fmt(r['net_expectancy'], 6), fmt(r['net_total'], 4),
                    fmt(r['net_sharpe_per_trade'], 5), fmt(r['mbb_p'], 4),
                    'PASS' if r['pass'] else 'FAIL')))
    lines += ['', 'ANNUAL ENTRY COHORTS 2020-2026']
    for r in results:
        lines.append(f"{r['asset']} {r['engine']}: annual {r['annual']} | "
                     f"overnight {r['overnight_trades']} | exits {r['exit_counts']} | "
                     f"open_end {r['open_end']} | G1-G5 {r['gates']} | funnel {r['funnel']}")
    lines += ['', 'PORTFOLIO NEW TRADES PER MONTH',
              f'Four-asset total evaluable: {complete}. Available-asset 2020-2026 executed entries: {len(monthly)}; monthly mean {monthly_frame.available_asset_new_trades.mean():.3f}.',
              'Month | Available-asset new trades | Four-asset total evaluable']
    for row in monthly_frame.itertuples(index=False):
        lines.append(f'{row.month} | {row.available_asset_new_trades} | {row.four_asset_total_evaluable}')
    lines += ['', 'LIMITS',
              'DATA BLOCKED means a required 2010-2026 source could not be acquired/qualified; NA is not a zero-return result.',
              'Unlike asset price units are never summed as portfolio PnL. Same-month entry counts can be added, with cross-engine overlap allowed.',
              'Fixed BID/ASK proxy costs, omitted swaps and tick-path uncertainty prevent FTMO-account conclusions.']
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Wrote', report, flush=True)


if __name__ == '__main__':
    main()
