"""2015-source correction to blocked US Oil full-span run; see SPEC_USOIL_MACRO_V2.txt."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import usoil_macro_engines as base
from futures_prop_validation_phase1 import engine_a, engine_b, grids
from gold_ensemble_real_cost import replay


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r'C:\Users\Milad\download\lightcmdusd-m1-bid-2015-01-01-2026-09-27.csv')
OUT = ROOT/'data/results/usoil_macro_engines_v2'


def load_source() -> tuple[dict, dict]:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    d = pd.read_csv(SOURCE, usecols=['timestamp', 'open', 'high', 'low', 'close'])
    stamp = d.timestamp.to_numpy(np.int64)
    if (len(stamp) < 4_000_000 or np.any(np.diff(stamp) <= 0)
            or np.any(stamp % 60000)):
        raise ValueError('Invalid, nonunique or insufficient M1 timestamps')
    t = stamp//1000
    o, h, l, c = (d[k].to_numpy(float) for k in ('open', 'high', 'low', 'close'))
    if (not np.isfinite(np.column_stack((o, h, l, c))).all()
            or np.any(np.minimum.reduce((o, h, l, c)) <= 0)
            or np.any(h < np.maximum(o, c))
            or np.any(l > np.minimum(o, c))):
        raise ValueError('Nonfinite, nonpositive or malformed OHLC')
    carried = ((o == h) & (h == l) & (l == c)
               & np.r_[False, c[1:] == c[:-1]])
    keep = ~carried
    m = {'time': t[keep], 'open': o[keep], 'high': h[keep],
         'low': l[keep], 'close': c[keep]}
    if (len(m['time']) < 3_000_000
            or m['time'][0] > int(pd.Timestamp('2015-01-03', tz='UTC').timestamp())
            or m['time'][-1] < int(pd.Timestamp('2026-09-25', tz='UTC').timestamp())):
        raise ValueError('2015-2026 coverage not adequate')
    years = pd.to_datetime(m['time'], unit='s', utc=True).year.to_numpy()
    annual = {str(y): int(np.count_nonzero(years == y))
              for y in range(2015, 2027)}
    if any(annual[str(y)] < 100_000 for y in range(2016, 2026)):
        raise ValueError(f'Insufficient full-year observed M1 coverage {annual}')
    gaps = np.diff(m['time'])
    meta = {'source': str(SOURCE), 'sha256': base.digest(SOURCE),
            'raw_rows': int(len(t)), 'observed_m1': int(len(m['time'])),
            'carried_removed': int(carried.sum()),
            'first_utc': int(m['time'][0]), 'last_utc': int(m['time'][-1]),
            'annual_observed_m1': annual,
            'gaps_over_3h': int(np.count_nonzero(gaps > 3*3600)),
            'max_gap_hours': float(gaps.max()/3600)}
    return m, meta


def main():
    spec, gate, report = (OUT/n for n in ('SPEC_USOIL_MACRO_V2.txt',
                                         'GATE_USOIL_MACRO_V2.txt',
                                         'REPORT_USOIL_MACRO_V2.txt'))
    if not spec.exists() or not gate.exists() or report.exists():
        raise FileExistsError('Frozen V2 SPEC/GATE missing or historical REPORT exists')
    extracted = '\n'.join(s for s in spec.read_text(encoding='utf-8').splitlines()
                          if len(s) > 2 and s[0] == 'G' and s[1].isdigit())+'\n'
    if extracted != gate.read_text(encoding='utf-8'):
        raise AssertionError('V2 GATE is not verbatim SPEC extraction')
    results = []
    source_meta = None
    error = None
    try:
        m, source_meta = load_source()
        schedule = base.rollover_schedule(m)
        g = grids(m)
        for e in (engine_a(g), engine_b(g)):
            raw, funnel = replay(m, e, base.SPREAD)
            rows = base.decorate(raw, e['name'], schedule)
            pd.DataFrame(rows).to_csv(OUT/f"trades_{e['name']}.csv", index=False)
            stats = base.summarize(e['name'], rows, funnel)
            results.append(stats)
            print(e['name'], stats['closed_trades'], 'closed, net mean',
                  base.fmt(stats['net_expectancy']), 'p',
                  base.fmt(stats['mbb_p'], 4),
                  'PASS' if stats['pass'] else 'FAIL', flush=True)
    except (FileNotFoundError, ValueError, AssertionError, MemoryError) as exc:
        error = str(exc)
        print('V2 DATA BLOCKED:', error, flush=True)
    complete = len(results) == 2 and source_meta is not None
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
    meta = {'study': 'usoil_macro_engines_v2',
            'v1_report': str(ROOT/'data/results/usoil_macro_engines/REPORT_USOIL_MACRO.txt'),
            'spec_sha256': base.digest(spec), 'gate_sha256': base.digest(gate),
            'script_sha256': base.digest(Path(__file__)), 'source': source_meta,
            'results': results, 'overall': verdict, 'error': error}
    (OUT/'run_meta.json').write_text(json.dumps(meta, indent=2, allow_nan=True),
                                     encoding='utf-8')
    lines = ['US OIL ENGINE A/B 2015-SOURCE CORRECTION - REPORT',
             f'OVERALL PHASE-1 GATE: {verdict}',
             'V1 exact 2010-start download returned fetch failed; earliest Dukascopy-node LIGHTCMDUSD M1 is 2011-09-23.',
             'V2 uses local 2015-2026 BID M1 with five pre-2020 warmup years; scoring 2020-2026 is unchanged.',
             'Net price units = BID-coordinate gross -0.03 floor - modeled swap per one barrel.',
             f'SOURCE: {source_meta}',
             f'ERROR: {error}', '',
             'ENGINE | CLOSED | WIN% | GROSS/TRADE | NET/TRADE | TOTAL NET | SWAP PAID ALL | SHARPE | MBB P | GATE']
    for name in ('A_KC', 'B_DONCHIAN'):
        r = next((v for v in results if v['engine'] == name), None)
        if r is None:
            lines.append(f'{name} | NA | NA | NA | NA | NA | NA | NA | NA | DATA BLOCKED')
        else:
            lines.append(' | '.join((name, str(r['closed_trades']),
                base.fmt(100*r['win_rate'], 2),
                base.fmt(r['gross_expectancy'], 6),
                base.fmt(r['net_expectancy'], 6),
                base.fmt(r['total_net'], 4),
                base.fmt(r['swap_paid_closed']+r['swap_accrued_open_end'], 4),
                base.fmt(r['net_sharpe_per_trade'], 5),
                base.fmt(r['mbb_p'], 4),
                'PASS' if r['pass'] else 'FAIL')))
    lines += ['', 'YEAR-BY-YEAR NET PNL MATRIX (PRICE UNITS; 2026 PARTIAL)',
              'ENGINE | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026']
    for name in ('A_KC', 'B_DONCHIAN'):
        r = next((v for v in results if v['engine'] == name), None)
        vals = [base.fmt(r['annual'][str(y)]['net'], 4) if r else 'NA'
                for y in range(2020, 2027)]
        lines.append(' | '.join([name, *vals]))
    lines += ['', 'ENGINE AUDIT']
    for r in results:
        lines.append(f"{r['engine']}: closed swap paid {r['swap_paid_closed']:.6f}, "
                     f"OPEN_END accrued {r['swap_accrued_open_end']:.6f}, "
                     f"rollovers {r['rollovers_all']}, weighted rollovers "
                     f"{r['weighted_rollovers_all']}, max stale mark "
                     f"{r['max_swap_mark_stale_hours']:.2f}h, "
                     f"open_end {r['open_end']}, exits {r['exit_mix']}, "
                     f"gates {r['gates']}, annual counts "
                     f"{ {y: a['trades'] for y, a in r['annual'].items()} }, "
                     f"funnel {r['funnel']}")
    lines += ['', 'LIMITATIONS',
              'The original 2010-start source requirement is unavailable; this V2 does not claim a 2010-2026 tape.',
              'The fixed 0.03 spread and -5% annualized swap are owner assumptions, not verified historical FTMO terms.',
              'Swap dollars are per one barrel, not a broker lot; quote gaps can leave stale rollover marks.',
              'BID M1 OHLC cannot reproduce broker ticks, variable spread, futures rolls, slippage or FTMO account rules.']
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Wrote', report, flush=True)


if __name__ == '__main__':
    main()
