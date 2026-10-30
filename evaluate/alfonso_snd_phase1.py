"""Frozen first-touch supply/demand proxy study; see SPEC_ALFONSO_SND.txt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/results/alfonso_snd'
ASSETS = {
    'XAUUSD': ('data/results/xauusd_onboarding/xauusdtick_m1.csv', 0.585, 2020),
    'US500': ('data/results/vwap_2021_2026/us500tick_m1.csv', 0.51, 2021),
    'EURUSD': ('data/results/eurusd_onboarding/eurusdtick_m1.csv', 0.00007, 2020),
}
PERIODS = {'M5': 300, 'M15': 900, 'H1': 3600, 'H4': 14400, 'D1': 86400}
SEQUENCES = {
    'SWING': ('H4', 'H1', 'M15'),
    'INTRADAY': ('H1', 'M15', 'M5'),
    'H1_BASE': ('H4', 'H4', 'H1'),
    'H4_BASE': ('D1', 'D1', 'H4'),
}
RATIOS = (1, 2, 3, 4, 5)
BOOT, BLOCK, SEED = 10_000, 5, 20260928


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path) -> dict:
    df = pd.read_csv(path, usecols=['time', 'open', 'high', 'low', 'close'])
    m = {k: df[k].to_numpy(np.int64 if k == 'time' else np.float64)
         for k in ('time', 'open', 'high', 'low', 'close')}
    t, o, h, l, c = (m[k] for k in ('time', 'open', 'high', 'low', 'close'))
    if (len(t) < 100_000 or np.any(np.diff(t) <= 0) or np.any(t % 60 != 0)
            or not np.isfinite(np.column_stack((o, h, l, c))).all()
            or np.any(o <= 0) or np.any(h < np.maximum(o, c))
            or np.any(l > np.minimum(o, c))):
        raise AssertionError(f'Invalid M1 source {path}')
    return m


def grid(m: dict, seconds: int) -> dict:
    bucket = m['time'] // seconds
    q = pd.DataFrame({'key': bucket, 'o': m['open'], 'h': m['high'],
                      'l': m['low'], 'c': m['close']})
    grouped = q.groupby('key', sort=True).agg(o=('o', 'first'), h=('h', 'max'),
                                              l=('l', 'min'), c=('c', 'last'))
    keys = np.arange(int(bucket[0]), int(bucket[-1])+1, dtype=np.int64)
    full = grouped.reindex(keys).ffill()
    real = np.isin(keys, grouped.index.to_numpy(np.int64), assume_unique=True)
    if not np.isfinite(full.to_numpy()).all():
        raise AssertionError('Grid has non-finite filled fields')
    return {'start': keys*seconds, 'end': (keys+1)*seconds,
            'real': real, **{k: full[k].to_numpy(float) for k in ('o', 'h', 'l', 'c')}}


def atr14(q: dict) -> np.ndarray:
    h, l, c = q['h'], q['l'], q['c']
    prev = np.r_[c[0], c[:-1]]
    tr = np.maximum(h-l, np.maximum(np.abs(h-prev), np.abs(l-prev)))
    out = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        out[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            out[i] = (out[i-1]*13+tr[i])/14
    return out


@njit(cache=True)
def detect(o, h, l, c, real, atr):
    """Return causal zones and completed-bar trend states on one grid."""
    n = len(c)
    side = np.zeros(n, np.int8)
    prox = np.empty(n, np.float64)
    distal = np.empty(n, np.float64)
    born = np.empty(n, np.int64)
    dead = np.full(n, n, np.int64)
    z_atr = np.empty(n, np.float64)
    active = np.empty(n, np.int64)
    active_n = 0
    nz = 0
    trend = np.zeros(n, np.int8)
    last_high, last_low = np.nan, np.nan
    last_dr = last_sb = last_sr = last_db = -1
    base_run = 0
    for i in range(n):
        if i >= 4:
            p = i-2
            if real[p-2] and real[p-1] and real[p] and real[p+1] and real[p+2]:
                if (h[p] > h[p-2] and h[p] > h[p-1]
                        and h[p] > h[p+1] and h[p] > h[p+2]):
                    last_high = h[p]
                if (l[p] < l[p-2] and l[p] < l[p-1]
                        and l[p] < l[p+1] and l[p] < l[p+2]):
                    last_low = l[p]
        opp_up = opp_down = False
        j = 0
        if real[i]:
            while j < active_n:
                z = active[j]
                if side[z] == 1:
                    if c[i] < distal[z]:
                        opp_down = True
                    touched = l[i] <= prox[z]
                    broken = c[i] < distal[z]
                    if touched or broken:
                        dead[z] = i
                        if broken:
                            last_db = i
                        elif c[i] > prox[z]:
                            last_dr = i
                        active_n -= 1
                        active[j] = active[active_n]
                        continue
                else:
                    if c[i] > distal[z]:
                        opp_up = True
                    touched = h[i] >= prox[z]
                    broken = c[i] > distal[z]
                    if touched or broken:
                        dead[z] = i
                        if broken:
                            last_sb = i
                        elif c[i] < prox[z]:
                            last_sr = i
                        active_n -= 1
                        active[j] = active[active_n]
                        continue
                j += 1
        bearish = max(last_sr, last_db)
        bullish = max(last_dr, last_sb)
        if last_dr >= 0 and last_sb >= 0 and min(last_dr, last_sb) > bearish:
            trend[i] = 1
        elif last_sr >= 0 and last_db >= 0 and min(last_sr, last_db) > bullish:
            trend[i] = -1
        if not real[i]:
            base_run = 0
            continue
        rng = h[i]-l[i]
        body = abs(c[i]-o[i])
        is_base = rng > 0 and body <= 0.5*rng
        if is_base:
            base_run += 1
            continue
        if (rng > 0 and body >= 0.8*rng and 1 <= base_run <= 6
                and np.isfinite(atr[i]) and atr[i] > 0):
            first = i-base_run
            base_hi = -np.inf
            base_lo = np.inf
            demand_prox = -np.inf
            supply_prox = np.inf
            for k in range(first, i):
                base_hi = max(base_hi, h[k])
                base_lo = min(base_lo, l[k])
                demand_prox = max(demand_prox, max(o[k], c[k]))
                supply_prox = min(supply_prox, min(o[k], c[k]))
            if (c[i] > o[i] and c[i] > base_hi
                    and ((np.isfinite(last_high) and c[i] > last_high) or opp_up)
                    and demand_prox > base_lo):
                side[nz] = 1
                prox[nz], distal[nz] = demand_prox, base_lo
                born[nz], z_atr[nz] = i, atr[i]
                active[active_n] = nz
                active_n += 1
                nz += 1
            elif (c[i] < o[i] and c[i] < base_lo
                    and ((np.isfinite(last_low) and c[i] < last_low) or opp_down)
                    and supply_prox < base_hi):
                side[nz] = -1
                prox[nz], distal[nz] = supply_prox, base_hi
                born[nz], z_atr[nz] = i, atr[i]
                active[active_n] = nz
                active_n += 1
                nz += 1
        base_run = 0
    return side[:nz], prox[:nz], distal[:nz], born[:nz], dead[:nz], z_atr[:nz], trend


def context(q: dict, zones: dict, bar: int, price: float, direction: int,
            cost: float, execution: bool = False) -> tuple[bool, float]:
    """Find nearest live enclosing zones and opposing target cap at a completed bar."""
    live = (zones['born'] <= bar) & (zones['dead'] > bar)
    demand = zones['prox'][live & (zones['side'] == 1) & (zones['prox'] < price)]
    supply = zones['prox'][live & (zones['side'] == -1) & (zones['prox'] > price)]
    if execution:
        if direction == 1:
            return True, float(np.min(supply)) if len(supply) else np.inf
        return True, float(np.max(demand)) if len(demand) else -np.inf
    if not len(demand) or not len(supply):
        return False, np.nan
    d, s = float(np.max(demand)), float(np.min(supply))
    if s <= d:
        return False, np.nan
    fraction = (price-d)/(s-d)
    return (fraction <= .33 if direction == 1 else fraction >= .67), fraction


@njit(cache=True)
def first_touch(m1_start, m1_low, m1_high, bmin, bmax, threshold, side, block):
    n = len(m1_low)
    k = m1_start
    while k < n:
        b = k//block
        stop = min((b+1)*block, n)
        if k % block == 0 and (bmin[b] > threshold if side == 1 else bmax[b] < threshold):
            k = stop
            continue
        while k < stop:
            if (m1_low[k] <= threshold if side == 1 else m1_high[k] >= threshold):
                return k
            k += 1
    return -1


@njit(cache=True)
def bracket(m1_open, m1_high, m1_low, m1_close, bmin, bmax,
            entry_ix, side, stop, target, cost, block):
    n = len(m1_open)
    k = entry_ix
    while k < n:
        b = k//block
        z = min((b+1)*block, n)
        if k % block == 0 and k > entry_ix:
            if side == 1 and bmin[b] > stop and bmax[b] < target:
                k = z
                continue
            if side == -1 and bmax[b]+cost < stop and bmin[b]+cost > target:
                k = z
                continue
        while k < z:
            if side == 1:
                if m1_low[k] <= stop:
                    return k, min(stop, m1_open[k]), 1
                if k > entry_ix and m1_high[k] >= target:
                    return k, max(target, m1_open[k]), 2
            else:
                if m1_high[k]+cost >= stop:
                    return k, max(stop, m1_open[k]+cost), 1
                if k > entry_ix and m1_low[k]+cost <= target:
                    return k, min(target, m1_open[k]+cost), 2
            k += 1
    return n-1, m1_close[-1] if side == 1 else m1_close[-1]+cost, 3


def mbb_p(x: np.ndarray) -> float:
    n = len(x)
    if n < BLOCK:
        return np.nan
    rng = np.random.default_rng(SEED)
    sums = np.convolve(x, np.ones(BLOCK), 'valid')
    blocks, extra = divmod(n, BLOCK)
    tail = np.convolve(x, np.ones(extra), 'valid') if extra else None
    bad = 0
    for start in range(0, BOOT, 200):
        count = min(200, BOOT-start)
        ix = rng.integers(0, n-BLOCK+1, (count, blocks+int(extra > 0)))
        totals = sums[ix[:, :blocks]].sum(axis=1)
        if extra:
            totals += tail[ix[:, blocks]]
        bad += int(np.count_nonzero(totals <= 0))
    return bad/BOOT


def summarize(rows: list[dict], first_year: int) -> dict:
    closed = [r for r in rows if r['exit_type'] != 'OPEN_END']
    x = np.array([r['net'] for r in closed], float)
    gross = np.array([r['gross'] for r in closed], float)
    gains = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    annual = {str(y): {'trades': 0, 'net': 0.0} for y in range(first_year, 2027)}
    for r in closed:
        y = str(r['entry_ny_year'])
        if y not in annual:
            annual[y] = {'trades': 0, 'net': 0.0}
        annual[y]['trades'] += 1
        annual[y]['net'] += r['net']
    mean = float(x.mean()) if len(x) else np.nan
    sharpe = float(mean/x.std(ddof=1)) if len(x) > 1 and x.std(ddof=1) > 0 else np.nan
    p = mbb_p(x)
    year_ok = all(annual[str(y)]['net'] >= -1e-10 for y in range(first_year, 2026))
    gates = {'G1': len(x) >= 150, 'G2': bool(mean > 0),
             'G3': bool(sharpe > 0), 'G4': bool(p <= .05), 'G5': year_ok}
    return {'closed_trades': len(x), 'open_end': len(rows)-len(x),
            'win_rate': float(np.mean(x > 0)) if len(x) else np.nan,
            'profit_factor': gains/losses if losses > 0 else (np.inf if gains > 0 else np.nan),
            'gross_expectancy': float(gross.mean()) if len(gross) else np.nan,
            'net_expectancy': mean, 'net_total': float(x.sum()),
            'net_sharpe_per_trade': sharpe, 'mbb_p': p,
            'annual': annual, 'gates': gates, 'pass': all(gates.values())}


def prepare_zones(q):
    q['atr'] = atr14(q)
    side, prox, distal, born, dead, z_atr, trend = detect(
        q['o'], q['h'], q['l'], q['c'], q['real'], q['atr'])
    return {'side': side, 'prox': prox, 'distal': distal, 'born': born,
            'dead': dead, 'atr': z_atr, 'trend': trend}


def candidates(m, grids, zones, sequence, cost, block_low, block_high, block):
    curve_name, trend_name, exec_name = SEQUENCES[sequence]
    curve_q, trend_q, exec_q = (grids[n] for n in (curve_name, trend_name, exec_name))
    curve_z, trend_z, exec_z = (zones[n] for n in (curve_name, trend_name, exec_name))
    t, o, lo, hi = (m[k] for k in ('time', 'open', 'low', 'high'))
    orders = []
    funnel = {'zones': len(exec_z['side']), 'curve_rejected': 0, 'trend_rejected': 0,
              'mitigated_before_fill': 0, 'filled_touch': 0}
    for z in range(len(exec_z['side'])):
        eb = int(exec_z['born'][z])
        validation = int(exec_q['end'][eb])
        cb = int(np.searchsorted(curve_q['end'], validation, side='right')-1)
        tb = int(np.searchsorted(trend_q['end'], validation, side='right')-1)
        if cb < 0 or tb < 0:
            funnel['curve_rejected'] += 1
            continue
        side = int(exec_z['side'][z])
        price = float(exec_q['c'][eb])
        allowed, fraction = context(curve_q, curve_z, cb, price, side, cost)
        if not allowed:
            funnel['curve_rejected'] += 1
            continue
        if int(trend_z['trend'][tb]) != side:
            funnel['trend_rejected'] += 1
            continue
        start = int(np.searchsorted(t, validation))
        if start >= len(t):
            continue
        proximal = float(exec_z['prox'][z])
        distal = float(exec_z['distal'][z])
        ix = int(first_touch(start, lo, hi, block_low, block_high,
                             proximal, side, block))
        if ix < 0:
            continue
        if side == 1 and lo[ix]+cost > proximal:
            funnel['mitigated_before_fill'] += 1
            continue
        entry = min(proximal, float(o[ix])+cost) if side == 1 else max(proximal, float(o[ix]))
        stop = distal-side*0.20*float(exec_z['atr'][z])
        risk = side*(entry-stop)
        if risk <= 0:
            continue
        _, cap = context(exec_q, exec_z, eb, price, side, cost, execution=True)
        funnel['filled_touch'] += 1
        orders.append({'zone_id': z, 'side': side, 'validation': validation,
                       'touch_ix': ix, 'entry': entry, 'stop': stop, 'risk': risk,
                       'proximal': proximal, 'distal': distal, 'atr': float(exec_z['atr'][z]),
                       'curve_fraction': fraction, 'cap': cap})
    orders.sort(key=lambda x: (x['touch_ix'], x['validation'], x['zone_id']))
    return orders, funnel


def replay(m, orders, cost, ratio, block_low, block_high, block):
    t, o, h, l, c = (m[k] for k in ('time', 'open', 'high', 'low', 'close'))
    rows = []
    occupied_until = -1
    shadowed = cap_rejected = 0
    for order in orders:
        ix = order['touch_ix']
        if ix <= occupied_until:
            shadowed += 1
            continue
        side, entry, stop, risk = (order[k] for k in ('side', 'entry', 'stop', 'risk'))
        target = entry+side*ratio*risk
        cap = order['cap']
        if ((side == 1 and target >= cap) or (side == -1 and target <= cap)):
            cap_rejected += 1
            continue
        ex_ix, exit_exec, code = bracket(o, h, l, c, block_low, block_high,
                                         ix, side, stop, target, cost, block)
        occupied_until = ex_ix
        exit_bid = exit_exec if side == 1 else exit_exec-cost
        entry_bid = entry-cost if side == 1 else entry
        gross = side*(exit_bid-entry_bid)
        net = side*(exit_exec-entry)
        if abs(net-(gross-cost)) > max(1e-9, abs(net)*1e-9):
            raise AssertionError('Round-trip cost posted incorrectly')
        year = pd.Timestamp(int(t[ix]), unit='s', tz='UTC').tz_convert('America/New_York').year
        rows.append({'zone_id': order['zone_id'], 'side': side, 'ratio': ratio,
                     'validation_time_utc': order['validation'],
                     'entry_time_utc': int(t[ix]), 'exit_time_utc': int(t[ex_ix]),
                     'entry_bid': entry_bid, 'entry_exec': entry,
                     'stop_exec': stop, 'target_exec': target,
                     'exit_bid': exit_bid, 'exit_exec': exit_exec,
                     'initial_risk': risk, 'cost': cost, 'gross': gross,
                     'net': net, 'exit_type': {1: 'STOP', 2: 'TARGET', 3: 'OPEN_END'}[code],
                     'entry_ny_year': int(year), 'curve_fraction': order['curve_fraction']})
    return rows, {'shadowed': shadowed, 'opposing_target_cap_rejected': cap_rejected}


def fmt(x, digits=5):
    return 'NA' if not np.isfinite(x) else f'{x:.{digits}f}'


def main():
    spec, gate, report = (OUT/n for n in ('SPEC_ALFONSO_SND.txt',
                                         'GATE_ALFONSO_SND.txt',
                                         'REPORT_ALFONSO_SND.txt'))
    if not spec.exists() or not gate.exists():
        raise FileNotFoundError('Frozen SPEC and GATE required before run')
    if report.exists():
        raise FileExistsError('Historical REPORT is immutable; create a new study')
    gate_text = '\n'.join(line for line in spec.read_text(encoding='utf-8').splitlines()
                          if line.startswith('G') and len(line) > 2 and line[1].isdigit())+'\n'
    if gate_text != gate.read_text(encoding='utf-8'):
        raise AssertionError('GATE is not the verbatim SPEC extraction')
    all_results = []
    input_meta = {}
    block = 1024
    for asset, (relative, cost, first_year) in ASSETS.items():
        path = ROOT/relative
        m = load(path)
        input_meta[asset] = {'path': relative, 'sha256': sha(path),
                             'm1_rows': len(m['time']),
                             'first_utc': int(m['time'][0]), 'last_utc': int(m['time'][-1])}
        grids = {name: grid(m, seconds) for name, seconds in PERIODS.items()}
        zones = {name: prepare_zones(q) for name, q in grids.items()}
        print(asset, 'M1', len(m['time']), 'zones',
              {name: len(z['side']) for name, z in zones.items()}, flush=True)
        low, high = m['low'], m['high']
        bmin = np.array([low[i:i+block].min() for i in range(0, len(low), block)])
        bmax = np.array([high[i:i+block].max() for i in range(0, len(high), block)])
        for sequence in SEQUENCES:
            orders, funnel = candidates(m, grids, zones, sequence, cost, bmin, bmax, block)
            print(asset, sequence, 'orders', len(orders), 'funnel', funnel, flush=True)
            for ratio in RATIOS:
                rows, occupancy = replay(m, orders, cost, ratio, bmin, bmax, block)
                stats = summarize(rows, first_year)
                name = f'trades_{asset}_{sequence}_R{ratio}.csv'
                pd.DataFrame(rows, columns=['zone_id', 'side', 'ratio', 'validation_time_utc',
                    'entry_time_utc', 'exit_time_utc', 'entry_bid', 'entry_exec', 'stop_exec',
                    'target_exec', 'exit_bid', 'exit_exec', 'initial_risk', 'cost', 'gross',
                    'net', 'exit_type', 'entry_ny_year', 'curve_fraction']).to_csv(OUT/name, index=False)
                result = {'asset': asset, 'sequence': sequence, 'rr': ratio,
                          'cost': cost, 'ledger': name, **stats,
                          'funnel': funnel, 'occupancy': occupancy}
                all_results.append(result)
                print(asset, sequence, f'R{ratio}', stats['closed_trades'],
                      fmt(stats['net_expectancy']), fmt(stats['mbb_p'], 4),
                      'PASS' if stats['pass'] else 'FAIL', flush=True)
    matrix = pd.DataFrame([{k: v for k, v in r.items() if k in (
        'asset', 'sequence', 'rr', 'closed_trades', 'win_rate', 'profit_factor',
        'gross_expectancy', 'net_expectancy', 'net_total', 'net_sharpe_per_trade',
        'mbb_p', 'pass')} for r in all_results])
    matrix.to_csv(OUT/'matrix.csv', index=False)
    meta = {'study': 'alfonso_snd', 'requested_label': 'Hypothesis #94',
            'label_collision': 'us100_weekday_hold_h94 already uses #94',
            'spec_sha256': sha(spec), 'gate_sha256': sha(gate),
            'script_sha256': sha(Path(__file__)), 'inputs': input_meta,
            'zones': {r['asset']+'_'+r['sequence']: r['funnel'] for r in all_results if r['rr'] == 1},
            'all_cells': all_results}
    (OUT/'run_meta.json').write_text(json.dumps(meta, indent=2, allow_nan=True), encoding='utf-8')
    lines = ['ALFONSO SUPPLY AND DEMAND PHASE 1 - HYPOTHESIS #94 REQUESTED LABEL',
             'Study identity alfonso_snd; #94 already labels us100_weekday_hold_h94 elsewhere.',
             'Full retrospective BID M1 proxy tapes; fixed all-inclusive spread floors; no swap/slippage.',
             f'MBB: {BOOT} moving-block resamples, block {BLOCK}, seed {SEED}.',
             'Net/gross expectancy and total are native price units; multiply EURUSD by 10000 for pips.',
             'Curve/trend use only completed HTF bars. M1 first-touch and adverse-first stops.',
             'PASS here means eligible for separate execution/forward validation, not validated.', '',
             'ASSET  SEQUENCE   R:R TRADES WIN%   PF     GROSS/TRADE NET/TRADE NET TOTAL SHARPE   MBB P  GATE']
    for r in all_results:
        lines.append(f"{r['asset']:<7}{r['sequence']:<11}{r['rr']:<4}"
                     f"{r['closed_trades']:<7}{fmt(100*r['win_rate'], 1):<7}"
                     f"{fmt(r['profit_factor'], 3):<7}{fmt(r['gross_expectancy'], 6):<12}"
                     f"{fmt(r['net_expectancy'], 6):<11}{fmt(r['net_total'], 5):<10}"
                     f"{fmt(r['net_sharpe_per_trade'], 4):<9}{fmt(r['mbb_p'], 4):<7}"
                     f"{'PASS' if r['pass'] else 'FAIL'}")
    lines += ['', 'ANNUAL ENTRY COHORTS AND CELL DIAGNOSTICS']
    for r in all_results:
        lines += [f"{r['asset']} {r['sequence']} R{r['rr']} | G1-G5 {r['gates']} | "
                  f"annual {r['annual']} | open_end {r['open_end']} | funnel {r['funnel']} "
                  f"| occupancy {r['occupancy']}"]
    lines += ['', 'LIMITATIONS',
              'This rule encoding uses confirmed 2-left/2-right pivots and event-based respect/break trend, not discretionary zone grading.',
              'Synthetic filled grid bars carry indicators but never signal. M1 OHLC cannot resolve within-minute ordering; entry-bar target is disallowed.',
              'Constant BID/ASK floor is a proxy, not variable broker spread or actual limit-fill priority. Zero financing is optimistic for held trades.',
              'Prior project research inspected overlapping history, so any positive cell is exploratory and needs independent forward/MT5 execution work.']
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Wrote', report, flush=True)


if __name__ == '__main__':
    main()
