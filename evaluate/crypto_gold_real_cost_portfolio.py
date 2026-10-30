"""Frozen Engine A/B crypto-gold cost and financing test; see SPEC_REAL_COST.txt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from futures_prop_validation_phase1 import engine_a, engine_b, grids
from gold_ensemble_real_cost import replay


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/results/crypto_gold_real_cost'
SOURCES = (
    ('BTCUSD', ROOT/'data/results/virgin_assets/btc_m1.csv', 3.0, .00065, .15),
    ('ETHUSD', ROOT/'data/results/virgin_assets/eth_m1.csv', .60, .00065, .15),
    ('XAUUSD', ROOT/'data/results/xauusd_onboarding/xauusdtick_m1.csv', .585, 0., .035),
)
SCORE_START = int(pd.Timestamp('2020-03-01', tz='UTC').timestamp())
BOOT, BLOCK, SEED = 10_000, 10, 20261025


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for part in iter(lambda: fh.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def load_source(path: Path, cutoff: int | None) -> tuple[dict, dict]:
    d = pd.read_csv(path, usecols=['time', 'open', 'high', 'low', 'close'])
    t = d.time.to_numpy(np.int64)
    if (len(t) < 1_000_000 or np.any(np.diff(t) <= 0)
            or np.any(t % 60)):
        raise ValueError('M1 chronology is invalid or source is too short')
    o, h, l, c = (d[k].to_numpy(float) for k in ('open', 'high', 'low', 'close'))
    if (not np.isfinite(np.column_stack((o, h, l, c))).all()
            or np.any(np.minimum.reduce((o, h, l, c)) <= 0)
            or np.any(h < np.maximum(o, c)) or np.any(l > np.minimum(o, c))):
        raise ValueError('M1 OHLC is invalid')
    carried = ((o == h) & (h == l) & (l == c)
               & np.r_[False, c[1:] == c[:-1]])
    keep = ~carried
    if cutoff is not None:
        keep &= t <= cutoff
    m = {'time': t[keep], 'open': o[keep], 'high': h[keep],
         'low': l[keep], 'close': c[keep]}
    if (len(m['time']) < 1_000_000
            or m['time'][0] > int(pd.Timestamp('2020-01-03', tz='UTC').timestamp())
            or m['time'][-1] < int(pd.Timestamp('2026-09-01', tz='UTC').timestamp())):
        raise ValueError('Common 2020-2026 observation window is not covered')
    gaps = np.diff(m['time'])
    meta = {'path': str(path), 'sha256': digest(path),
            'raw_rows': int(len(t)), 'retained_m1': int(len(m['time'])),
            'carried_flat_removed': int(carried.sum()),
            'first_utc': int(m['time'][0]), 'last_utc': int(m['time'][-1]),
            'gaps_over_3h': int(np.count_nonzero(gaps > 3*3600)),
            'max_gap_hours': float(gaps.max()/3600)}
    return m, meta


def rollovers(asset: str, m: dict, annual_rate: float) -> dict:
    t = m['time']
    if asset in ('BTCUSD', 'ETHUSD'):
        times = np.arange((t[0]//86400+1)*86400, t[-1]+1, 86400,
                          dtype=np.int64)
        weights = np.ones(len(times), np.int8)
        convention = '00:00 UTC, all calendar days'
    else:
        first = pd.Timestamp(int(t[0]), unit='s', tz='UTC').tz_convert(
            'America/New_York').date()
        last = pd.Timestamp(int(t[-1]), unit='s', tz='UTC').tz_convert(
            'America/New_York').date()
        days = pd.date_range(first, last, freq='D')
        days = days[days.dayofweek < 5]
        wall = (days+pd.Timedelta(hours=17)).tz_localize(
            'America/New_York')
        times = (wall.tz_convert('UTC').asi8//1_000_000_000).astype(np.int64)
        valid = (times > t[0]) & (times <= t[-1])
        times = times[valid]
        weights = np.where(days.dayofweek.to_numpy()[valid] == 2, 3, 1).astype(np.int8)
        convention = '17:00 America/New_York, weekdays, Wednesday x3'
    prior_ix = np.searchsorted(t, times, side='left')-1
    if np.any(prior_ix < 0):
        raise AssertionError('Rollover has no strictly prior quote')
    marks = m['close'][prior_ix]
    charge_usd = marks*(annual_rate/365.0)*weights
    stale_hours = (times-t[prior_ix])/3600
    return {'time': times, 'weight': weights, 'mark': marks,
            'charge_usd': charge_usd, 'stale_hours': stale_hours,
            'convention': convention}


def swap_for_trade(schedule: dict, entry_time: int, exit_time: int) -> dict:
    times = schedule['time']
    left = int(np.searchsorted(times, entry_time, side='right'))
    right = int(np.searchsorted(times, exit_time, side='right'))
    return {'swap_usd': -float(schedule['charge_usd'][left:right].sum()),
            'rollover_count': right-left,
            'weighted_rollover_count': int(schedule['weight'][left:right].sum()),
            'max_mark_staleness_hours': float(schedule['stale_hours'][left:right].max())
            if right > left else 0.0}


def decorate(asset: str, engine: str, raw: list[dict], spread: float,
             commission_rate: float, schedule: dict) -> list[dict]:
    rows = []
    for r in raw:
        if r['entry_time'] < SCORE_START:
            continue
        side = int(r['side'])
        # Stop coordinates follow the frozen replay. A shorts stop in ASK is
        # BID-coordinate stop + spread; B shorts stop is already ASK.
        if side == 1:
            risk = float(r['entry_fill']-r['initial_stop'])
        else:
            stop_ask = float(r['initial_stop']+(spread if engine == 'A_KC' else 0.))
            risk = stop_ask-float(r['entry_fill'])
        if not np.isfinite(risk) or risk <= 0:
            raise AssertionError('Initial executable risk is nonpositive')
        entry_bid = float(r['entry_bid'])
        commission = commission_rate*entry_bid
        swap = swap_for_trade(schedule, int(r['entry_time']), int(r['exit_time']))
        closed = r['exit_type'] != 'END'
        gross = float(r['gross'])
        fill = float(r['net'])
        if abs(fill-(gross-spread)) > 1e-7:
            raise AssertionError('Spread not applied exactly once')
        if abs(fill-side*(r['exit_fill']-r['entry_fill'])) > 1e-7:
            raise AssertionError('Fill PnL identity failed')
        net = fill-commission+swap['swap_usd']
        row = {'asset': asset, 'engine': engine, 'side': side,
               'entry_time': int(r['entry_time']), 'exit_time': int(r['exit_time']),
               'entry_bid': entry_bid, 'entry_fill': float(r['entry_fill']),
               'exit_bid': float(r['exit_bid']) if closed else np.nan,
               'exit_fill': float(r['exit_fill']) if closed else np.nan,
               'initial_stop': float(r['initial_stop']),
               'initial_risk_usd_per_unit': risk,
               'spread_usd_per_unit': spread,
               'commission_usd_per_unit': commission,
               'swap_usd_per_unit': swap['swap_usd'],
               'rollover_count': swap['rollover_count'],
               'weighted_rollover_count': swap['weighted_rollover_count'],
               'max_mark_staleness_hours': swap['max_mark_staleness_hours'],
               'exit_type': 'OPEN_END' if not closed else r['exit_type'],
               'gross_usd_per_unit': gross if closed else np.nan,
               'fill_usd_per_unit': fill if closed else np.nan,
               'net_usd_per_unit': net if closed else np.nan,
               'gross_r': gross/risk if closed else np.nan,
               'net_r': net/risk if closed else np.nan,
               'hold_hours': float(r['hold_hours'])}
        if closed and abs(row['net_usd_per_unit']-(gross-spread-commission+
                                                swap['swap_usd'])) > 1e-7:
            raise AssertionError('Total cost identity failed')
        rows.append(row)
    return rows


def summarize(asset: str, engine: str, rows: list[dict], source: dict,
              funnel: dict) -> dict:
    closed = [r for r in rows if r['exit_type'] != 'OPEN_END']
    x = np.array([r['net_r'] for r in closed], float)
    gross = np.array([r['gross_usd_per_unit'] for r in closed], float)
    net = np.array([r['net_usd_per_unit'] for r in closed], float)
    mean = float(x.mean()) if len(x) else np.nan
    sd = float(x.std(ddof=1)) if len(x) >= 2 else np.nan
    sharpe = mean/sd if np.isfinite(sd) and sd > 0 else np.nan
    if closed:
        entry = np.array([r['entry_time'] for r in closed], np.int64)
        exit_ = np.array([r['exit_time'] for r in closed], np.int64)
        if np.any(exit_ < entry) or np.any(entry[1:] <= exit_[:-1]):
            raise AssertionError('Asset-engine ledger overlaps or reverses time')
    return {'asset': asset, 'engine': engine, 'closed_trades': len(closed),
            'scored_entries': len(rows), 'open_end': len(rows)-len(closed),
            'win_rate': float(np.mean(net > 0)) if len(net) else np.nan,
            'gross_usd_expectancy_per_unit': float(gross.mean()) if len(gross) else np.nan,
            'net_usd_expectancy_per_unit': float(net.mean()) if len(net) else np.nan,
            'net_usd_total_per_unit': float(net.sum()),
            'net_r_expectancy': mean, 'net_r_total': float(x.sum()),
            'net_r_sharpe_per_trade': float(sharpe),
            'swap_paid_closed_usd_per_unit': -sum(r['swap_usd_per_unit'] for r in closed),
            'swap_paid_open_usd_per_unit': -sum(r['swap_usd_per_unit'] for r in rows
                                                 if r['exit_type'] == 'OPEN_END'),
            'commission_paid_all_usd_per_unit': sum(r['commission_usd_per_unit']
                                                     for r in rows),
            'rollover_count_all': sum(r['rollover_count'] for r in rows),
            'weighted_rollover_count_all': sum(r['weighted_rollover_count']
                                               for r in rows),
            'max_mark_staleness_hours': max((r['max_mark_staleness_hours']
                                             for r in rows), default=0.),
            'source': source, 'funnel': funnel}


def mbb_p(x: np.ndarray) -> float:
    n = len(x)
    if n < BLOCK:
        return np.nan
    rng = np.random.default_rng(SEED)
    nblocks = (n+BLOCK-1)//BLOCK
    bad = 0
    for base in range(0, BOOT, 250):
        k = min(250, BOOT-base)
        starts = rng.integers(0, n-BLOCK+1, (k, nblocks))
        ix = (starts[:, :, None]+np.arange(BLOCK)).reshape(k, -1)[:, :n]
        bad += int(np.count_nonzero(x[ix].mean(axis=1) <= 0))
    return bad/BOOT


def concurrency(rows: list[dict]) -> int:
    events = []
    for r in rows:
        events.append((int(r['entry_time']), 1))
        if r['exit_type'] != 'OPEN_END':
            events.append((int(r['exit_time']), -1))
    active = peak = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    return peak


def fmt(x, digits=4):
    return 'NA' if not np.isfinite(x) else f'{x:.{digits}f}'


def main():
    spec, gate, report = (OUT/n for n in ('SPEC_REAL_COST.txt',
                                         'GATE_REAL_COST.txt',
                                         'REPORT_REAL_COST.txt'))
    if not spec.exists() or not gate.exists() or report.exists():
        raise FileExistsError('Frozen SPEC/GATE missing or historical REPORT exists')
    extracted = '\n'.join(s for s in spec.read_text(encoding='utf-8').splitlines()
                          if len(s) > 2 and s[0] == 'G' and s[1].isdigit())+'\n'
    if extracted != gate.read_text(encoding='utf-8'):
        raise AssertionError('GATE is not verbatim SPEC extraction')
    gold_m, gold_meta = load_source(SOURCES[2][1], None)
    cutoff = int(gold_m['time'][-1])
    if cutoff != 1788296340:
        raise AssertionError('XAUUSD common cutoff differs from predeclared tape')
    all_rows = []
    results = []
    errors = {}
    for asset, path, spread, commission_rate, annual_rate in SOURCES:
        print('Evaluating', asset, flush=True)
        try:
            if asset == 'XAUUSD':
                m, source = gold_m, gold_meta
            else:
                m, source = load_source(path, cutoff)
            schedule = rollovers(asset, m, annual_rate)
            g = grids(m)
            for engine in (engine_a(g), engine_b(g)):
                raw, funnel = replay(m, engine, spread)
                rows = decorate(asset, engine['name'], raw, spread,
                                commission_rate, schedule)
                pd.DataFrame(rows).to_csv(
                    OUT/f"trades_{asset}_{engine['name']}.csv", index=False)
                result = summarize(asset, engine['name'], rows, source, funnel)
                result['spread_usd_per_unit'] = spread
                result['commission_rate_entry'] = commission_rate
                result['swap_annual_rate'] = annual_rate
                result['rollover_convention'] = schedule['convention']
                results.append(result)
                all_rows.extend(rows)
                print(asset, engine['name'], result['closed_trades'],
                      'net R/trade', fmt(result['net_r_expectancy']), flush=True)
            del g
            if asset != 'XAUUSD':
                del m
        except (FileNotFoundError, ValueError, AssertionError, MemoryError) as exc:
            errors[asset] = str(exc)
            print(asset, 'DATA BLOCKED:', exc, flush=True)
    closed = sorted((r for r in all_rows if r['exit_type'] != 'OPEN_END'),
                    key=lambda r: (r['exit_time'], r['asset'], r['engine']))
    curve = pd.DataFrame([{'asset': r['asset'], 'engine': r['engine'],
                           'entry_time': r['entry_time'], 'exit_time': r['exit_time'],
                           'net_r': r['net_r']} for r in closed])
    if len(curve):
        curve['cumulative_net_r'] = curve.net_r.cumsum()
    curve.to_csv(OUT/'pooled_R_curve.csv', index=False)
    x = np.array([r['net_r'] for r in closed], float)
    mean = float(x.mean()) if len(x) else np.nan
    sd = float(x.std(ddof=1)) if len(x) >= 2 else np.nan
    sharpe = mean/sd if np.isfinite(sd) and sd > 0 else np.nan
    p = mbb_p(x)
    complete = len(results) == 6 and not errors
    gates = {'G1': complete and all(r['closed_trades'] >= 50 for r in results),
             'G2': complete and all(r['net_r_expectancy'] > 0 for r in results),
             'G3': bool(mean > 0 and sharpe > 0),
             'G4': bool(p <= .05),
             'G5': complete}
    verdict = ('PASS' if complete and all(gates.values())
               else 'FAIL' if complete else 'DATA BLOCKED')
    pooled = {'closed_trades': len(closed), 'scored_entries': len(all_rows),
              'mean_net_r': mean, 'total_net_r': float(x.sum()),
              'net_r_sharpe_per_trade': float(sharpe), 'mbb_p': p,
              'max_concurrent_positions': concurrency(all_rows),
              'gates': gates, 'verdict': verdict}
    summary = pd.DataFrame([{k: v for k, v in r.items()
                             if k not in ('source', 'funnel')}
                            for r in results])
    summary.to_csv(OUT/'summary.csv', index=False)
    meta = {'study': 'crypto_gold_real_cost', 'spec_sha256': digest(spec),
            'gate_sha256': digest(gate), 'script_sha256': digest(Path(__file__)),
            'common_cutoff_utc': cutoff, 'bootstrap': {'resamples': BOOT,
            'block': BLOCK, 'seed': SEED}, 'results': results, 'pooled': pooled,
            'errors': errors}
    (OUT/'run_meta.json').write_text(json.dumps(meta, indent=2, allow_nan=True),
                                     encoding='utf-8')
    lines = ['CRYPTO/GOLD ENGINE A/B REAL-COST PORTFOLIO - REPORT',
             f'OVERALL PHASE-1 GATE: {verdict}',
             'Owner-specified fixed historical friction and annual financing scenario.',
             'Common source end: 2026-09-01 20:59 UTC; scored entries from 2020-03-01 UTC.',
             'USD is per one BTC, ETH or troy ounce; R is per executable initial-stop risk.',
             '', 'ASSET | ENGINE | CLOSED | WIN% | GROSS USD/TR | NET USD/TR | SWAP PAID USD ALL | NET R/TR | R SHARPE']
    for asset, _, _, _, _ in SOURCES:
        for engine in ('A_KC', 'B_DONCHIAN'):
            r = next((v for v in results if v['asset'] == asset and
                      v['engine'] == engine), None)
            if r is None:
                lines.append(f'{asset} | {engine} | DATA BLOCKED')
                continue
            lines.append(' | '.join((asset, engine, str(r['closed_trades']),
                fmt(100*r['win_rate'], 2),
                fmt(r['gross_usd_expectancy_per_unit'], 2),
                fmt(r['net_usd_expectancy_per_unit'], 2),
                fmt(r['swap_paid_closed_usd_per_unit']+
                    r['swap_paid_open_usd_per_unit'], 2),
                fmt(r['net_r_expectancy'], 4),
                fmt(r['net_r_sharpe_per_trade'], 4))))
    lines += ['', 'SWAP TOTALS BY ASSET (USD PER ONE BASE ASSET UNIT; BOTH ENGINES)',
              'ASSET | CLOSED SWAP PAID | OPEN-END ACCRUED | TOTAL PAID | ROLLOVERS | WEIGHTED ROLLOVERS']
    for asset, _, _, _, _ in SOURCES:
        rs = [r for r in results if r['asset'] == asset]
        if not rs:
            lines.append(f'{asset} | DATA BLOCKED')
        else:
            closed_swap = sum(r['swap_paid_closed_usd_per_unit'] for r in rs)
            open_swap = sum(r['swap_paid_open_usd_per_unit'] for r in rs)
            lines.append(f"{asset} | {closed_swap:.2f} | {open_swap:.2f} | "
                         f"{closed_swap+open_swap:.2f} | "
                         f"{sum(r['rollover_count_all'] for r in rs)} | "
                         f"{sum(r['weighted_rollover_count_all'] for r in rs)}")
    lines += ['', 'POOLED ONE-R-PER-ENTRY SERIES',
              f"Closed {pooled['closed_trades']}; scored entries {pooled['scored_entries']}; "
              f"mean net R {fmt(mean, 5)}; total net R {fmt(pooled['total_net_r'], 3)}; "
              f"per-trade R Sharpe {fmt(sharpe, 5)}; MBB p {fmt(p, 4)}; "
              f"peak simultaneous positions {pooled['max_concurrent_positions']}.",
              f'Gates: {gates}', '', 'SOURCE AND FINANCING AUDIT']
    for r in results:
        s = r['source']
        lines.append(f"{r['asset']} {r['engine']}: {s['retained_m1']} observed M1, "
                     f"{s['carried_flat_removed']} carried removed, "
                     f"UTC {pd.Timestamp(s['first_utc'], unit='s', tz='UTC')}.."
                     f"{pd.Timestamp(s['last_utc'], unit='s', tz='UTC')}, "
                     f"gaps>3h {s['gaps_over_3h']} (max {s['max_gap_hours']:.1f}h), "
                     f"SHA256 {s['sha256']}; open_end {r['open_end']}, "
                     f"rollover max stale mark {r['max_mark_staleness_hours']:.1f}h, "
                     f"funnel {r['funnel']}")
    for asset, error in errors.items():
        lines.append(f'{asset}: DATA BLOCKED {error}')
    lines += ['', 'LIMITATIONS',
              'FTMO announced 0.0325% crypto commission per side from 2025-07-28; the requested 0.065% of entry price is applied counterfactually to all years and differs from exit-notional billing.',
              'The specified annual swap rates and spread floors are owner scenarios, not verified historical broker charges. Both sides pay negative swaps here.',
              'Swap marks use the strictly prior available BID close; gaps can make marks stale. M1 OHLC cannot order all intraminute events.',
              'The cumulative R curve assumes one initial-risk unit per trade despite overlapping positions, and is not cash account equity or FTMO challenge EV.',
              'No lot sizing, contract conversion, variable spread, slippage, broker ticks or live account drawdown checks are modeled.']
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Wrote', report, flush=True)


if __name__ == '__main__':
    main()
