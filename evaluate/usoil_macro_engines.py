"""Frozen US Oil Engine A/B transfer with 17:00 ET financing; see SPEC_USOIL_MACRO.txt."""
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
from macro_portfolio_expansion import load_source


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD = Path(r'C:\Users\Milad\download')
OUT = ROOT/'data/results/usoil_macro_engines'
NAME = 'lightcmdusd-m1-bid-2010-01-01-2026-09-27.csv'
COMMAND = ['npx', 'dukascopy-node', '-i', 'lightcmdusd', '-from',
           '2010-01-01', '-to', '2026-09-27', '-t', 'm1', '-f', 'csv', '--cache']
SPREAD = .03
ANNUAL_SWAP = .05
SCORE_START = int(pd.Timestamp('2020-01-01', tz='UTC').timestamp())
SCORE_END = int(pd.Timestamp('2026-09-27', tz='UTC').timestamp())
BOOT, BLOCK, SEED = 10_000, 5, 20261028


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for part in iter(lambda: fh.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def complete_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1_000_000:
        return False
    with path.open('rb') as fh:
        if fh.readline().strip() != b'timestamp,open,high,low,close':
            return False
        first = fh.readline().split(b',', 1)[0]
        fh.seek(max(0, path.stat().st_size-1024))
        last = fh.read().splitlines()[-1].split(b',', 1)[0]
    try:
        return int(first) <= 1262563200000 and int(last) >= 1790294400000
    except ValueError:
        return False


def acquire() -> tuple[Path | None, dict]:
    provenance = {'command': ' '.join(COMMAND), 'cwd': str(DOWNLOAD),
                  'attempted': False}
    for candidate in (DOWNLOAD/'download'/NAME, DOWNLOAD/NAME):
        if complete_file(candidate):
            provenance.update(status='EXISTING_FILE', path=str(candidate),
                              bytes=candidate.stat().st_size)
            return candidate, provenance
    exe = shutil.which('npx.cmd') or shutil.which('npx')
    if exe is None:
        provenance['status'] = 'NPX_NOT_FOUND'
        return None, provenance
    provenance['attempted'] = True
    print('Downloading:', ' '.join(COMMAND), flush=True)
    try:
        run = subprocess.run([exe, *COMMAND[1:]], cwd=DOWNLOAD,
                             capture_output=True, text=True, errors='replace',
                             timeout=1800, check=False)
        (OUT/'download_lightcmdusd.log').write_text(
            f"COMMAND: {' '.join(COMMAND)}\nEXECUTABLE: {exe}\n"
            f"RETURN CODE: {run.returncode}\nSTDOUT:\n{run.stdout}\n"
            f"STDERR:\n{run.stderr}\n", encoding='utf-8')
        provenance.update(return_code=run.returncode,
                          stdout_tail=run.stdout[-1200:],
                          stderr_tail=run.stderr[-1200:])
        if run.returncode != 0:
            provenance['status'] = 'COMMAND_FAILED'
            return None, provenance
        for candidate in (DOWNLOAD/'download'/NAME, DOWNLOAD/NAME):
            if complete_file(candidate):
                provenance.update(status='DOWNLOADED', path=str(candidate),
                                  bytes=candidate.stat().st_size)
                return candidate, provenance
        provenance['status'] = 'OUTPUT_MISSING_OR_INCOMPLETE'
    except subprocess.TimeoutExpired as exc:
        (OUT/'download_lightcmdusd.log').write_text(
            f"COMMAND: {' '.join(COMMAND)}\nTIMEOUT AFTER {exc.timeout} SECONDS\n",
            encoding='utf-8')
        provenance.update(status='TIMEOUT', timeout_seconds=exc.timeout)
    except OSError as exc:
        (OUT/'download_lightcmdusd.log').write_text(
            f"COMMAND: {' '.join(COMMAND)}\nOS ERROR: {exc}\n",
            encoding='utf-8')
        provenance.update(status='OS_ERROR', error=str(exc))
    return None, provenance


def rollover_schedule(m: dict) -> dict:
    t = m['time']
    first = pd.Timestamp(int(t[0]), unit='s', tz='UTC').tz_convert(
        'America/New_York').date()
    last = pd.Timestamp(int(t[-1]), unit='s', tz='UTC').tz_convert(
        'America/New_York').date()
    days = pd.date_range(first, last, freq='D')
    days = days[days.dayofweek < 5]
    wall = (days+pd.Timedelta(hours=17)).tz_localize('America/New_York')
    boundary = (wall.tz_convert('UTC').asi8//1_000_000_000).astype(np.int64)
    valid = (boundary > t[0]) & (boundary <= t[-1])
    boundary = boundary[valid]
    weight = np.where(days.dayofweek.to_numpy()[valid] == 2, 3, 1).astype(np.int8)
    ix = np.searchsorted(t, boundary, side='left')-1
    if np.any(ix < 0):
        raise AssertionError('Rollover has no strictly prior quote')
    mark = m['close'][ix]
    if np.any(mark <= 0):
        raise AssertionError('Nonpositive swap mark')
    return {'time': boundary, 'weight': weight,
            'charge': np.abs(mark)*(ANNUAL_SWAP/365.)*weight,
            'stale_hours': (boundary-t[ix])/3600}


def swap_for(schedule: dict, entry: int, exit_: int) -> tuple[float, int, int, float]:
    b = schedule['time']
    lo = int(np.searchsorted(b, entry, side='right'))
    hi = int(np.searchsorted(b, exit_, side='right'))
    return (-float(schedule['charge'][lo:hi].sum()), hi-lo,
            int(schedule['weight'][lo:hi].sum()),
            float(schedule['stale_hours'][lo:hi].max()) if hi > lo else 0.)


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


def decorate(raw: list[dict], engine: str, schedule: dict) -> list[dict]:
    rows = []
    for r in raw:
        if not SCORE_START <= r['entry_time'] < SCORE_END:
            continue
        swap, roll_count, weighted_count, staleness = swap_for(
            schedule, int(r['entry_time']), int(r['exit_time']))
        closed = r['exit_type'] != 'END'
        gross = float(r['gross'])
        fill = float(r['net'])
        if abs(fill-(gross-SPREAD)) > 1e-8:
            raise AssertionError('Spread embedded other than once')
        net = fill+swap
        rows.append({'engine': engine, 'side': int(r['side']),
                     'entry_time': int(r['entry_time']),
                     'exit_time': int(r['exit_time']),
                     'entry_ny_year': int(r['entry_ny_year']),
                     'entry_bid': float(r['entry_bid']),
                     'entry_fill': float(r['entry_fill']),
                     'exit_bid': float(r['exit_bid']) if closed else np.nan,
                     'exit_fill': float(r['exit_fill']) if closed else np.nan,
                     'initial_stop': float(r['initial_stop']),
                     'final_stop': float(r['final_stop']),
                     'spread_price_units': SPREAD,
                     'swap_price_units_per_barrel': swap,
                     'rollover_count': roll_count,
                     'weighted_rollover_count': weighted_count,
                     'max_mark_stale_hours': staleness,
                     'gross_price_units': gross if closed else np.nan,
                     'fill_price_units': fill if closed else np.nan,
                     'net_price_units': net if closed else np.nan,
                     'exit_type': r['exit_type'] if closed else 'OPEN_END',
                     'hold_hours': float(r['hold_hours'])})
        if closed and abs(net-(gross-SPREAD+swap)) > 1e-8:
            raise AssertionError('Net cost identity failed')
    return rows


def summarize(engine: str, rows: list[dict], funnel: dict) -> dict:
    closed = [r for r in rows if r['exit_type'] != 'OPEN_END']
    x = np.array([r['net_price_units'] for r in closed], float)
    gross = np.array([r['gross_price_units'] for r in closed], float)
    mean = float(x.mean()) if len(x) else np.nan
    sd = float(x.std(ddof=1)) if len(x) >= 2 else np.nan
    sharpe = mean/sd if np.isfinite(sd) and sd > 0 else np.nan
    p = mbb_p(x)
    if closed:
        en = np.array([r['entry_time'] for r in closed], np.int64)
        ex = np.array([r['exit_time'] for r in closed], np.int64)
        if np.any(ex < en) or np.any(en[1:] <= ex[:-1]):
            raise AssertionError('Overlapping or reversed trade chronology')
    annual = {str(y): {'trades': 0, 'net': 0.0} for y in range(2020, 2027)}
    for r in closed:
        y = str(r['entry_ny_year'])
        if y in annual:
            annual[y]['trades'] += 1
            annual[y]['net'] += r['net_price_units']
    gates = {'G1': len(x) >= 150, 'G2': bool(mean > 0),
             'G3': bool(sharpe > 0), 'G4': bool(p <= .05),
             'G5': all(annual[str(y)]['net'] >= -1e-10
                       for y in range(2020, 2026))}
    return {'engine': engine, 'closed_trades': len(x),
            'scored_entries': len(rows), 'open_end': len(rows)-len(x),
            'win_rate': float(np.mean(x > 0)) if len(x) else np.nan,
            'gross_expectancy': float(gross.mean()) if len(gross) else np.nan,
            'net_expectancy': mean, 'total_net': float(x.sum()),
            'net_sharpe_per_trade': float(sharpe), 'mbb_p': p,
            'swap_paid_closed': -sum(r['swap_price_units_per_barrel']
                                     for r in closed),
            'swap_accrued_open_end': -sum(r['swap_price_units_per_barrel']
                                         for r in rows
                                         if r['exit_type'] == 'OPEN_END'),
            'rollovers_all': sum(r['rollover_count'] for r in rows),
            'weighted_rollovers_all': sum(r['weighted_rollover_count']
                                          for r in rows),
            'max_swap_mark_stale_hours': max((r['max_mark_stale_hours']
                                             for r in rows), default=0.),
            'exit_mix': {name: sum(r['exit_type'] == name for r in closed)
                         for name in ('SL_INIT', 'SL_TRAIL')},
            'annual': annual, 'funnel': funnel, 'gates': gates,
            'pass': all(gates.values())}


def fmt(value, digits=5):
    return 'NA' if not np.isfinite(value) else f'{value:.{digits}f}'


def main():
    spec, gate, report = (OUT/n for n in ('SPEC_USOIL_MACRO.txt',
                                         'GATE_USOIL_MACRO.txt',
                                         'REPORT_USOIL_MACRO.txt'))
    if not spec.exists() or not gate.exists() or report.exists():
        raise FileExistsError('Frozen SPEC/GATE missing or historical REPORT exists')
    extracted = '\n'.join(s for s in spec.read_text(encoding='utf-8').splitlines()
                          if len(s) > 2 and s[0] == 'G' and s[1].isdigit())+'\n'
    if extracted != gate.read_text(encoding='utf-8'):
        raise AssertionError('GATE is not verbatim SPEC extraction')
    path, acquisition = acquire()
    results = []
    source = None
    if path is not None:
        try:
            m, source = load_source(path, 'USOIL')
            gaps = np.diff(m['time'])
            source['gaps_over_3h'] = int(np.count_nonzero(gaps > 3*3600))
            source['max_gap_hours'] = float(gaps.max()/3600)
            schedule = rollover_schedule(m)
            g = grids(m)
            for e in (engine_a(g), engine_b(g)):
                raw, funnel = replay(m, e, SPREAD)
                rows = decorate(raw, e['name'], schedule)
                pd.DataFrame(rows).to_csv(
                    OUT/f"trades_{e['name']}.csv", index=False)
                stats = summarize(e['name'], rows, funnel)
                results.append(stats)
                print(e['name'], stats['closed_trades'], 'closed, net mean',
                      fmt(stats['net_expectancy']), 'p', fmt(stats['mbb_p'], 4),
                      'PASS' if stats['pass'] else 'FAIL', flush=True)
        except (ValueError, AssertionError, MemoryError) as exc:
            acquisition['status'] = 'SOURCE_OR_EVALUATION_BLOCKED'
            acquisition['reason'] = str(exc)
            print('USOIL DATA BLOCKED:', exc, flush=True)
    complete = len(results) == 2 and source is not None
    verdict = ('PASS' if complete and all(r['pass'] for r in results)
               else 'FAIL' if complete else 'DATA BLOCKED')
    pd.DataFrame([{k: v for k, v in r.items()
                   if k not in ('annual', 'funnel', 'gates', 'exit_mix')}
                  for r in results]).to_csv(OUT/'summary.csv', index=False)
    pd.DataFrame([{'engine': r['engine'],
                   **{str(y): r['annual'][str(y)]['net']
                      for y in range(2020, 2027)}}
                  for r in results]).to_csv(OUT/'annual_matrix.csv', index=False,
                                           float_format='%.6f')
    meta = {'study': 'usoil_macro_engines', 'spec_sha256': digest(spec),
            'gate_sha256': digest(gate), 'script_sha256': digest(Path(__file__)),
            'acquisition': acquisition, 'source': source, 'results': results,
            'overall': verdict, 'bootstrap': {'resamples': BOOT,
            'block': BLOCK, 'seed': SEED}}
    (OUT/'run_meta.json').write_text(json.dumps(meta, indent=2, allow_nan=True),
                                     encoding='utf-8')
    lines = ['US OIL ENGINE A/B MACRO TRANSFER - REPORT',
             f'OVERALL PHASE-1 GATE: {verdict}',
             'Full 2010-2026 Dukascopy LIGHTCMDUSD BID M1; scored entries 2020-2026.',
             'Net price units = BID-coordinate gross -0.03 floor - modeled swap.',
             '', f'ACQUISITION: {acquisition}',
             f'SOURCE: {source}', '',
             'ENGINE | CLOSED | WIN% | GROSS/TRADE | NET/TRADE | TOTAL NET | SWAP PAID ALL | SHARPE | MBB P | GATE']
    for name in ('A_KC', 'B_DONCHIAN'):
        r = next((v for v in results if v['engine'] == name), None)
        if r is None:
            lines.append(f'{name} | NA | NA | NA | NA | NA | NA | NA | NA | DATA BLOCKED')
            continue
        lines.append(' | '.join((name, str(r['closed_trades']),
            fmt(100*r['win_rate'], 2), fmt(r['gross_expectancy'], 6),
            fmt(r['net_expectancy'], 6), fmt(r['total_net'], 4),
            fmt(r['swap_paid_closed']+r['swap_accrued_open_end'], 4),
            fmt(r['net_sharpe_per_trade'], 5), fmt(r['mbb_p'], 4),
            'PASS' if r['pass'] else 'FAIL')))
    lines += ['', 'YEAR-BY-YEAR NET PNL MATRIX (USD PRICE UNITS PER ONE BARREL; 2026 PARTIAL)',
              'ENGINE | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026']
    for name in ('A_KC', 'B_DONCHIAN'):
        r = next((v for v in results if v['engine'] == name), None)
        vals = [fmt(r['annual'][str(y)]['net'], 4) if r else 'NA'
                for y in range(2020, 2027)]
        lines.append(' | '.join([name, *vals]))
    lines += ['', 'ENGINE AUDIT']
    for r in results:
        lines.append(f"{r['engine']}: closed swap paid {r['swap_paid_closed']:.6f}, "
                     f"OPEN_END accrued {r['swap_accrued_open_end']:.6f}, "
                     f"rollovers {r['rollovers_all']}, Wednesday-weighted "
                     f"rollovers {r['weighted_rollovers_all']}, "
                     f"max stale mark {r['max_swap_mark_stale_hours']:.2f}h, "
                     f"open_end {r['open_end']}, exits {r['exit_mix']}, "
                     f"gates {r['gates']}, annual counts "
                     f"{ {y: a['trades'] for y, a in r['annual'].items()} }, "
                     f"funnel {r['funnel']}")
    lines += ['', 'LIMITATIONS',
              'The 0.03 spread and -5% annual swap are owner assumptions applied to all years, not verified historical FTMO USOIL charges.',
              'Swap USD values are per one barrel, not per broker lot; absolute cash needs verified contract size and volume.',
              'Dukascopy LIGHTCMDUSD is a BID proxy and can have roll/quote gaps; no broker ticks, variable spread, slippage or account guard is modeled.',
              'A and B may overlap. Their price-unit totals are not a sized account equity curve. OPEN_END financing is included in swap paid but not closed-return statistics.']
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Wrote', report, flush=True)


if __name__ == '__main__':
    main()
