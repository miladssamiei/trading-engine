"""Frozen daily RSI2 index screen and FTMO 2-Step scenario; see SPEC_RSI2_DAILY.txt."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/results/connors_rsi2_daily'
SOURCES = {
    'US500': (ROOT/'data/results/vwap_2021_2026/us500tick_m1.csv', .55, 2021),
    'US100': (ROOT/'data/results/virgin_assets/ustech_m1.csv', 1.66, 2020),
}
GER_NATIVE = ROOT/'data/results/ger40_native_source/ger40cash_native_tick_m1_2022_2026.csv'
THRESHOLDS = (10, 5)
VARIANTS = (('A', 0), ('B', 3), ('B', 5), ('B', 7),
            ('C', 3), ('C', 5), ('C', 7))
BOOT, BLOCK, SEED = 10_000, 5, 20261010
ACCOUNT, DAILY_LOSS, MAX_LOSS, FEE, SPLIT = 10000., 500., 1000., 175., .80
N_PATHS, MAX_DAYS, MAX_TRADES = 10000, 3650, 2000


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for part in iter(lambda: fh.read(1 << 20), b''):
            h.update(part)
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


def daily(m: dict) -> dict:
    t = m['time']
    ny_wall = pd.to_datetime(t, unit='s', utc=True).tz_convert(
        'America/New_York').tz_localize(None)
    # Subtract wall-clock hours from naive local time, avoiding a UTC DST shift.
    session_day = (ny_wall-pd.Timedelta(hours=17)).normalize()+pd.Timedelta(days=1)
    frame = pd.DataFrame({'day': session_day, 'o': m['open'], 'h': m['high'],
                          'l': m['low'], 'c': m['close'],
                          'first_ix': np.arange(len(t)), 'last_ix': np.arange(len(t))})
    agg = frame.groupby('day', sort=True).agg(
        o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'),
        n=('o', 'size'), first_ix=('first_ix', 'first'), last_ix=('last_ix', 'last'))
    agg = agg[(agg.index.dayofweek < 5) & (agg.n >= 300)]
    if len(agg) < 200:
        raise AssertionError('Too few complete observed daily sessions')
    ends_wall = pd.DatetimeIndex(agg.index+pd.Timedelta(hours=17))
    ends = (ends_wall.tz_localize('America/New_York')
            .tz_convert('UTC').asi8//1_000_000_000).astype(np.int64)
    if np.any(t[agg.last_ix.to_numpy(int)] >= ends):
        raise AssertionError('D1 includes a future M1 row')
    return {'date': agg.index.to_numpy('datetime64[D]'), 'end': ends,
            **{k: agg[k].to_numpy(float) for k in ('o', 'h', 'l', 'c')},
            'n': agg.n.to_numpy(int), 'first_ix': agg.first_ix.to_numpy(int),
            'last_ix': agg.last_ix.to_numpy(int)}


def wilder_rsi2(c: np.ndarray) -> np.ndarray:
    out = np.full(len(c), np.nan)
    if len(c) < 3:
        return out
    d = np.diff(c)
    gain = np.maximum(d, 0.)
    loss = np.maximum(-d, 0.)
    ag, al = float(gain[:2].mean()), float(loss[:2].mean())
    for i in range(2, len(c)):
        if i > 2:
            ag = (ag+gain[i-1])/2
            al = (al+loss[i-1])/2
        out[i] = (100. if ag > 0 else 50.) if al == 0 else (
            0. if ag == 0 else 100.-100./(1.+ag/al))
    return out


def wilder_atr14(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> np.ndarray:
    prev = np.r_[c[0], c[:-1]]
    tr = np.maximum(h-l, np.maximum(np.abs(h-prev), np.abs(l-prev)))
    out = np.full(len(c), np.nan)
    if len(c) >= 14:
        out[13] = tr[:14].mean()
        for i in range(14, len(c)):
            out[i] = (13*out[i-1]+tr[i])/14
    return out


def indicators(q: dict) -> dict:
    c = pd.Series(q['c'])
    return {'rsi': wilder_rsi2(q['c']),
            'sma200': c.rolling(200, min_periods=200).mean().to_numpy(float),
            'sma5': c.rolling(5, min_periods=5).mean().to_numpy(float),
            'atr': wilder_atr14(q['h'], q['l'], q['c'])}


def backtest(m: dict, q: dict, ind: dict, asset: str, cost: float,
             threshold: int, variant: str, cap: int):
    t, o, h, l, c = (m[k] for k in ('time', 'open', 'high', 'low', 'close'))
    condition = ((q['c'] > ind['sma200']) & (ind['rsi'] < threshold)
                 & np.isfinite(ind['atr']) & (ind['atr'] > 0))
    candidates = np.flatnonzero(condition)
    rows = []
    funnel = {'signals': len(candidates), 'no_next_quote': 0,
              'shadowed': 0, 'no_scheduled_exit': 0, 'entry_gap_over_2h': 0}
    occupied_until = -1
    for sig in candidates:
        en = int(np.searchsorted(t, q['end'][sig]))
        if en >= len(t) or (sig+1 < len(q['end']) and t[en] >= q['end'][sig+1]):
            funnel['no_next_quote'] += 1
            continue
        if en <= occupied_until:
            funnel['shadowed'] += 1
            continue
        gap_mins = (int(t[en])-int(q['end'][sig]))/60
        funnel['entry_gap_over_2h'] += int(gap_mins > 120)
        entry_bid = float(o[en])
        entry_ask = entry_bid+cost
        risk = float(2.5*ind['atr'][sig])
        stop = entry_ask-risk
        planned_day = -1
        for j in range(sig+1, len(q['c'])):
            sma_exit = np.isfinite(ind['sma5'][j]) and q['c'][j] > ind['sma5'][j]
            capped = variant != 'A' and j-sig >= cap
            if sma_exit or capped:
                planned_day = j
                break
        plan_ix = int(np.searchsorted(t, q['end'][planned_day])) if planned_day >= 0 else len(t)
        if plan_ix >= len(t):
            funnel['no_scheduled_exit'] += 1
            plan_ix = len(t)
        ex = plan_ix
        reason = 'SMA5' if planned_day >= 0 and q['c'][planned_day] > ind['sma5'][planned_day] else 'TIME_CAP'
        exit_bid = float(o[ex]) if ex < len(t) else float(c[-1])
        if variant == 'C':
            stop_hits = np.flatnonzero(l[en:min(plan_ix+1, len(t))] <= stop)
            if len(stop_hits):
                ex = en+int(stop_hits[0])
                exit_bid = min(stop, float(o[ex]))
                reason = 'EMERGENCY_STOP'
        if ex >= len(t):
            ex = len(t)-1
            reason = 'OPEN_END'
        occupied_until = ex
        # A scheduled exit executes at its opening quote, before that M1's range.
        range_end = ex+1 if reason == 'EMERGENCY_STOP' else ex
        if range_end > en:
            path_low = float(np.min(l[en:range_end]))
            path_high = float(np.max(h[en:range_end]))
        else:
            path_low = path_high = entry_bid
        path_low = min(path_low, exit_bid)
        path_high = max(path_high, exit_bid)
        gross = exit_bid-entry_bid
        net = exit_bid-entry_ask
        if abs(net-(gross-cost)) > 1e-8:
            raise AssertionError('Cost posted more or less than once')
        entry_ny = pd.Timestamp(int(t[en]), unit='s', tz='UTC').tz_convert('America/New_York')
        exit_ny = pd.Timestamp(int(t[ex]), unit='s', tz='UTC').tz_convert('America/New_York')
        rows.append({'asset': asset, 'threshold': threshold,
                     'variant': variant, 'cap': cap,
                     'signal_d1_ix': int(sig), 'planned_exit_d1_ix': int(planned_day),
                     'entry_ix': en, 'exit_ix': ex,
                     'signal_end_utc': int(q['end'][sig]),
                     'entry_time_utc': int(t[en]), 'exit_time_utc': int(t[ex]),
                     'entry_bid': entry_bid, 'entry_ask': entry_ask,
                     'exit_bid': exit_bid, 'stop_bid': stop if variant == 'C' else np.nan,
                     'atr14': float(ind['atr'][sig]), 'reference_risk': risk,
                     'rsi2': float(ind['rsi'][sig]), 'sma200': float(ind['sma200'][sig]),
                     'sma5_signal': float(ind['sma5'][sig]),
                     'cost': cost, 'gross': gross, 'net': net, 'net_r': net/risk,
                     'mae_r': (path_low-entry_ask)/risk,
                     'mfe_r': (path_high-entry_ask)/risk,
                     'exit_type': reason, 'holding_trading_days': (
                         planned_day-sig if planned_day >= 0 and reason != 'EMERGENCY_STOP'
                         else max(1, int(np.searchsorted(q['end'], t[ex], side='right'))-sig)),
                     'holding_calendar_days': (int(t[ex])-int(t[en]))/86400,
                     'entry_gap_minutes': gap_mins,
                     'ny_overnight': bool(entry_ny.date() != exit_ny.date()),
                     'entry_ny_year': int(entry_ny.year)})
    return rows, funnel


def mbb_p(x: np.ndarray) -> float:
    n = len(x)
    if n < BLOCK:
        return np.nan
    rng = np.random.default_rng(SEED)
    block_sums = np.convolve(x, np.ones(BLOCK), 'valid')
    count, extra = divmod(n, BLOCK)
    tail = np.convolve(x, np.ones(extra), 'valid') if extra else None
    bad = 0
    for offset in range(0, BOOT, 250):
        k = min(250, BOOT-offset)
        ix = rng.integers(0, n-BLOCK+1, (k, count+int(extra > 0)))
        sums = block_sums[ix[:, :count]].sum(axis=1)
        if extra:
            sums += tail[ix[:, count]]
        bad += int(np.count_nonzero(sums <= 0))
    return bad/BOOT


def metrics(rows: list[dict], first_year: int) -> dict:
    closed = [r for r in rows if r['exit_type'] != 'OPEN_END']
    x = np.array([r['net'] for r in closed], float)
    r = np.array([z['net_r'] for z in closed], float)
    gains = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    annual = {str(y): {'trades': 0, 'net': 0.} for y in range(first_year, 2027)}
    for z in closed:
        y = str(z['entry_ny_year'])
        if y not in annual:
            annual[y] = {'trades': 0, 'net': 0.}
        annual[y]['trades'] += 1
        annual[y]['net'] += z['net']
    mean = float(x.mean()) if len(x) else np.nan
    mean_r = float(r.mean()) if len(r) else np.nan
    sd = float(x.std(ddof=1)) if len(x) >= 2 else np.nan
    sharpe = mean/sd if np.isfinite(sd) and sd > 0 else np.nan
    p = mbb_p(x)
    gates = {'G1': len(x) >= 150, 'G2': bool(mean > 0 and mean_r > 0),
             'G3': bool(sharpe > 0), 'G4': bool(p <= .05),
             'G5': all(annual[str(y)]['net'] >= -1e-10 for y in range(first_year, 2026))}
    def distribution(field, adverse=False):
        a = np.array([z[field] for z in closed], float)
        if adverse:
            a = -np.minimum(a, 0.)
        return {'mean': float(np.mean(a)) if len(a) else np.nan,
                'median': float(np.median(a)) if len(a) else np.nan,
                'p90': float(np.percentile(a, 90)) if len(a) else np.nan}
    return {'closed_trades': len(x), 'open_end': len(rows)-len(x),
            'win_rate': float((x > 0).mean()) if len(x) else np.nan,
            'profit_factor': gains/losses if losses else (np.inf if gains else np.nan),
            'gross_expectancy': float(np.mean([z['gross'] for z in closed])) if len(x) else np.nan,
            'net_expectancy': mean, 'mean_net_r': mean_r, 'net_total': float(x.sum()),
            'average_holding_trading_days': float(np.mean([z['holding_trading_days'] for z in closed])) if len(x) else np.nan,
            'mae_r_magnitude': distribution('mae_r', True),
            'mfe_r': distribution('mfe_r'),
            'net_sharpe_per_trade': sharpe, 'mbb_p': p,
            'ny_overnight_trades': sum(z['ny_overnight'] for z in closed),
            'exit_counts': {name: sum(z['exit_type'] == name for z in closed)
                            for name in ('SMA5', 'TIME_CAP', 'EMERGENCY_STOP')},
            'annual': annual, 'gates': gates, 'pass': all(gates.values())}


def primary_templates(primary: dict[str, list[dict]], tapes: dict[str, dict]):
    """Risk-normalized Prague-day paths for common-period primary trades."""
    first = int(tapes['US500']['time'][0])
    last = int(tapes['US500']['time'][-1])
    all_trades = sorted((z for rows in primary.values() for z in rows
                         if z['exit_type'] != 'OPEN_END' and first <= z['entry_time_utc']
                         and z['exit_time_utc'] <= last),
                        key=lambda z: (z['entry_time_utc'], z['asset']))
    if len(all_trades) < 40 or not {z['asset'] for z in all_trades} == {'US500', 'US100'}:
        raise AssertionError('Insufficient two-asset common-period primary pool')
    cet_days = {}
    for asset, m in tapes.items():
        wall = pd.to_datetime(m['time'], unit='s', utc=True).tz_convert(
            'Europe/Prague').tz_localize(None)
        cet_days[asset] = wall.to_numpy(dtype='datetime64[D]').astype(np.int64)
    records = []
    for z in all_trades:
        asset = z['asset']
        m = tapes[asset]
        days = cet_days[asset]
        en, ex = int(z['entry_ix']), int(z['exit_ix'])
        day0, day1 = int(days[en]), int(days[ex])
        duration = day1-day0+1
        if duration < 1:
            raise AssertionError('Nonpositive sampled duration')
        low_path = np.empty(duration, float)
        close_path = np.empty(duration, float)
        carried = float(z['entry_bid'])
        for k, day in enumerate(range(day0, day1+1)):
            left = max(en, int(np.searchsorted(days, day, side='left')))
            right = min(ex+1, int(np.searchsorted(days, day+1, side='left')))
            if z['exit_type'] != 'EMERGENCY_STOP' and day == day1:
                right = min(right, ex)  # scheduled exit is at this M1 open
            if left < right:
                lowest = float(np.min(m['low'][left:right]))
                carried = float(m['close'][right-1])
            else:
                lowest = carried
            if day == day1:
                lowest = min(lowest, float(z['exit_bid']))
                carried = float(z['exit_bid'])
            low_path[k] = (lowest-z['entry_ask'])/z['reference_risk']
            close_path[k] = (carried-z['entry_ask'])/z['reference_risk']
            low_path[k] = min(low_path[k], close_path[k])
        if abs(close_path[-1]-z['net_r']) > 1e-8:
            raise AssertionError('Sampled path does not end at empirical net R')
        records.append((z, day0, low_path, close_path))
    durations = np.array([len(v[2]) for v in records], np.int32)
    max_len = int(durations.max())
    lows = np.zeros((len(records), max_len), float)
    closes = np.zeros_like(lows)
    for i, (_, _, low, close) in enumerate(records):
        lows[i, :len(low)] = low
        closes[i, :len(close)] = close
    entry_days = np.array([v[1] for v in records], np.int64)
    gaps = np.diff(entry_days).astype(np.int32)
    if np.any(gaps < 0):
        raise AssertionError('Chronological entry gaps reversed')
    assets = np.array([0 if v[0]['asset'] == 'US500' else 1 for v in records], np.int8)
    r = np.array([v[0]['net_r'] for v in records], float)
    return {'r': r, 'asset': assets, 'duration': durations,
            'low': lows, 'close': closes, 'gaps': gaps}, {
                'trades': len(records), 'by_asset': {
                    'US500': int(np.count_nonzero(assets == 0)),
                    'US100': int(np.count_nonzero(assets == 1))},
                'mean_r': float(r.mean()), 'median_r': float(np.median(r)),
                'min_r': float(r.min()), 'max_r': float(r.max()),
                'max_duration_cet_days': max_len,
                'median_duration_cet_days': float(np.median(durations)),
                'median_entry_gap_days': float(np.median(gaps)),
                'same_day_gap_fraction': float(np.mean(gaps == 0))}


@njit(cache=True)
def run_phase(r, asset, duration, low, close, gaps, cash_risk, target, funded):
    """Return state, static/daily flags, days, closed trades, final profit."""
    balance = ACCOUNT
    next_entry_day = 0
    active = np.full(2, -1, np.int32)
    age = np.zeros(2, np.int32)
    trading_days = 0
    first_entry = -1
    closed = 0
    for day in range(MAX_DAYS):
        opened_today = 0
        for _ in range(2):
            if next_entry_day > day or closed >= MAX_TRADES:
                break
            if funded and closed + int(active[0] >= 0) + int(active[1] >= 0) >= 30:
                break
            if not funded and balance >= ACCOUNT+target:
                break
            pick = np.random.randint(len(r))
            a = int(asset[pick])
            if active[a] < 0:
                active[a] = pick
                age[a] = 0
                opened_today += 1
                if first_entry < 0:
                    first_entry = day
            gap = int(gaps[np.random.randint(len(gaps))])
            next_entry_day = day+gap
        if next_entry_day <= day:
            next_entry_day = day+1
        if opened_today:
            trading_days += 1
        day_start_balance = balance
        worst = balance
        for a in range(2):
            pick = active[a]
            if pick >= 0:
                worst += cash_risk*low[pick, age[a]]
        static = worst <= ACCOUNT-MAX_LOSS
        daily = worst <= day_start_balance-DAILY_LOSS
        if static or daily:
            return -1, int(static), int(daily), day+1, closed, balance-ACCOUNT
        for a in range(2):
            pick = active[a]
            if pick >= 0:
                if age[a]+1 >= duration[pick]:
                    balance += cash_risk*r[pick]
                    active[a] = -1
                    closed += 1
                else:
                    age[a] += 1
        flat = active[0] < 0 and active[1] < 0
        if funded:
            if closed >= 30 and flat and day-first_entry >= 14:
                if balance > ACCOUNT:
                    return 1, 0, 0, day+1, closed, balance-ACCOUNT
                return -2, 0, 0, day+1, closed, balance-ACCOUNT
        else:
            if balance >= ACCOUNT+target and flat and trading_days >= 4:
                return 1, 0, 0, day+1, closed, balance-ACCOUNT
        if closed >= MAX_TRADES:
            return 0, 0, 0, day+1, closed, balance-ACCOUNT
    return 0, 0, 0, MAX_DAYS, closed, balance-ACCOUNT


@njit(cache=True)
def simulate_profile(r, asset, duration, low, close, gaps, cash_risk, seed):
    np.random.seed(seed)
    outcomes = np.zeros((N_PATHS, 16), float)
    for i in range(N_PATHS):
        p1, s1, d1, days1, n1, pnl1 = run_phase(
            r, asset, duration, low, close, gaps, cash_risk, 1000., False)
        outcomes[i, 0] = p1
        outcomes[i, 1] = s1
        outcomes[i, 2] = d1
        outcomes[i, 3] = days1
        outcomes[i, 4] = n1
        outcomes[i, 5] = pnl1
        if p1 != 1:
            continue
        p2, s2, d2, days2, n2, pnl2 = run_phase(
            r, asset, duration, low, close, gaps, cash_risk, 500., False)
        outcomes[i, 6] = p2
        outcomes[i, 7] = s2
        outcomes[i, 8] = d2
        outcomes[i, 9] = days2
        outcomes[i, 10] = n2
        outcomes[i, 11] = pnl2
        if p2 != 1:
            continue
        pf, sf, df, daysf, nf, pnlf = run_phase(
            r, asset, duration, low, close, gaps, cash_risk, 0., True)
        outcomes[i, 12] = pf
        outcomes[i, 13] = daysf
        outcomes[i, 14] = nf
        outcomes[i, 15] = pnlf
    return outcomes


def mc_stats(outcomes: np.ndarray, risk_fraction: float) -> tuple[dict, pd.DataFrame]:
    p1 = outcomes[:, 0] == 1
    p2 = outcomes[:, 6] == 1
    paid = outcomes[:, 12] == 1
    receipt = np.where(paid, SPLIT*outcomes[:, 15]+FEE, 0.)
    net_ev = float(receipt.mean()-FEE)
    funded_days = outcomes[p2, 3]+outcomes[p2, 9]
    funded_trades = outcomes[p2, 4]+outcomes[p2, 10]
    def quant(x):
        return {'median': float(np.median(x)) if len(x) else None,
                'p90': float(np.percentile(x, 90)) if len(x) else None}
    def failures(mask, flag_static, flag_daily):
        static = outcomes[:, flag_static] > 0
        daily = outcomes[:, flag_daily] > 0
        return {'static_only': int((mask & static & ~daily).sum()),
                'daily_only': int((mask & daily & ~static).sum()),
                'both': int((mask & static & daily).sum()),
                'static_inclusive': int((mask & static).sum()),
                'daily_inclusive': int((mask & daily).sum())}
    summary = {
        'risk_fraction': risk_fraction, 'cash_risk_per_reference_r': risk_fraction*ACCOUNT,
        'paths': N_PATHS, 'phase1_pass_count': int(p1.sum()),
        'phase1_pass_probability': float(p1.mean()),
        'phase2_pass_count': int(p2.sum()),
        'phase2_conditional_pass_probability': float(p2.sum()/p1.sum()) if p1.any() else 0.,
        'phase2_unconditional_pass_probability': float(p2.mean()),
        'funded_status_probability': float(p2.mean()),
        'first_payout_count': int(paid.sum()),
        'first_payout_probability': float(paid.mean()),
        'phase1_failures': failures(outcomes[:, 0] == -1, 1, 2),
        'phase2_failures': failures(outcomes[:, 6] == -1, 7, 8),
        'phase1_unresolved': int((outcomes[:, 0] == 0).sum()),
        'phase2_unresolved': int((p1 & (outcomes[:, 6] == 0)).sum()),
        'funded_fail_loss_count': int(((outcomes[:, 12] == -1) & p2).sum()),
        'funded_no_profit_count': int(((outcomes[:, 12] == -2) & p2).sum()),
        'funded_unresolved_count': int(((outcomes[:, 12] == 0) & p2).sum()),
        'days_to_funded_conditional': quant(funded_days),
        'trades_to_funded_conditional': quant(funded_trades),
        'mean_gross_receipt': float(receipt.mean()), 'fee': FEE,
        'net_ev_per_challenge': net_ev,
        'ev_mc_standard_error': float((receipt-FEE).std(ddof=1)/np.sqrt(N_PATHS)),
    }
    if not (paid.sum() <= p2.sum() <= p1.sum()
            and np.all(receipt[~paid] == 0)
            and abs(summary['net_ev_per_challenge']-(summary['mean_gross_receipt']-FEE)) < 1e-8):
        raise AssertionError('Monte Carlo accounting identity failed')
    names = ['phase1_state', 'phase1_static', 'phase1_daily', 'phase1_days',
             'phase1_trades', 'phase1_pnl', 'phase2_state', 'phase2_static',
             'phase2_daily', 'phase2_days', 'phase2_trades', 'phase2_pnl',
             'funded_state', 'funded_days', 'funded_trades', 'funded_pnl']
    table = pd.DataFrame(outcomes, columns=names)
    table.insert(0, 'risk_fraction', risk_fraction)
    table.insert(1, 'path', np.arange(N_PATHS))
    table['gross_receipt'] = receipt
    table['net_after_fee'] = receipt-FEE
    return summary, table


def portfolio_concurrency(rows: list[dict]) -> dict:
    events = []
    for z in rows:
        if z['exit_type'] == 'OPEN_END':
            continue
        events.append((z['entry_time_utc'], 1, z['asset']))
        events.append((z['exit_time_utc'], -1, z['asset']))
    events.sort(key=lambda e: (e[0], e[1]))
    open_assets = {'US500': 0, 'US100': 0}
    maximum = 0
    concurrent_entries = 0
    for _, direction, asset in events:
        open_assets[asset] += direction
        if open_assets[asset] not in (0, 1):
            raise AssertionError('Same-asset overlap in primary ledger')
        if direction == 1 and sum(open_assets.values()) == 2:
            concurrent_entries += 1
        maximum = max(maximum, sum(open_assets.values()))
    if any(open_assets.values()):
        raise AssertionError('Unclosed portfolio sweep')
    return {'max_simultaneous_positions': maximum,
            'entries_while_other_asset_open': concurrent_entries}


def fmt(value, places=4):
    return 'NA' if value is None or not np.isfinite(value) else f'{value:.{places}f}'


def main():
    spec, gate, report = (OUT/n for n in ('SPEC_RSI2_DAILY.txt',
                                         'GATE_RSI2_DAILY.txt',
                                         'REPORT_RSI2_DAILY.txt'))
    if not spec.exists() or not gate.exists():
        raise FileNotFoundError('Freeze SPEC and GATE before running this script')
    if report.exists():
        raise FileExistsError('Historical REPORT is immutable; use a new study')
    extracted = '\n'.join(line for line in spec.read_text(encoding='utf-8').splitlines()
                          if line.startswith('G') and len(line) > 2 and line[1].isdigit())+'\n'
    if extracted != gate.read_text(encoding='utf-8'):
        raise AssertionError('GATE differs from the frozen SPEC extraction')
    tapes = {}
    primary = {}
    results = []
    sources = {}
    for asset, (path, cost, first_year) in SOURCES.items():
        m = load(path)
        q = daily(m)
        ind = indicators(q)
        tapes[asset] = m
        sources[asset] = {'file': str(path), 'sha256': digest(path),
                          'm1_rows': len(m['time']), 'd1_sessions': len(q['c']),
                          'first_utc': int(m['time'][0]), 'last_utc': int(m['time'][-1]),
                          'cost': cost}
        print(asset, 'M1', len(m['time']), 'D1', len(q['c']), flush=True)
        for threshold in THRESHOLDS:
            for variant, cap in VARIANTS:
                cell = f'{asset}_RSI{threshold}_{variant}{cap if variant != "A" else ""}'
                rows, funnel = backtest(m, q, ind, asset, cost, threshold, variant, cap)
                if threshold == 10 and variant == 'C' and cap == 5:
                    primary[asset] = rows
                ledger = OUT/f'trades_{cell}.csv'
                pd.DataFrame(rows).to_csv(ledger, index=False)
                stats = metrics(rows, first_year)
                results.append({'asset': asset, 'threshold': threshold,
                                'variant': variant, 'cap': cap, 'cell': cell,
                                'ledger': str(ledger), 'funnel': funnel, **stats})
                print(cell, 'closed', stats['closed_trades'], 'net',
                      fmt(stats['net_expectancy'], 5), 'R', fmt(stats['mean_net_r'], 5),
                      'p', fmt(stats['mbb_p']), 'gate',
                      'PASS' if stats['pass'] else 'FAIL', flush=True)
    if len(results) != 28 or set(primary) != {'US500', 'US100'}:
        raise AssertionError('Incomplete fixed 28-cell experiment')
    primary_rows = sorted(primary['US500']+primary['US100'],
                          key=lambda z: (z['entry_time_utc'], z['asset']))
    concurrency = portfolio_concurrency(primary_rows)
    pd.DataFrame(primary_rows).to_csv(OUT/'primary_portfolio.csv', index=False)
    templates, template_meta = primary_templates(primary, tapes)
    mc_results = []
    all_paths = []
    for n, fraction in enumerate((.005, .01, .015, .02)):
        outcomes = simulate_profile(templates['r'], templates['asset'],
                                    templates['duration'], templates['low'],
                                    templates['close'], templates['gaps'],
                                    ACCOUNT*fraction, SEED+1000*n)
        summary, paths = mc_stats(outcomes, fraction)
        mc_results.append(summary)
        all_paths.append(paths)
        print('MC', fraction, 'P1', fmt(summary['phase1_pass_probability']),
              'P2 conditional', fmt(summary['phase2_conditional_pass_probability']),
              'payout', fmt(summary['first_payout_probability']),
              'EV', fmt(summary['net_ev_per_challenge'], 2), flush=True)
    pd.concat(all_paths, ignore_index=True).to_csv(OUT/'mc_paths.csv', index=False)
    pd.DataFrame([{k: v for k, v in r.items() if k in (
        'asset', 'threshold', 'variant', 'cap', 'cell', 'closed_trades',
        'win_rate', 'profit_factor', 'gross_expectancy', 'net_expectancy',
        'mean_net_r', 'net_total', 'average_holding_trading_days',
        'net_sharpe_per_trade', 'mbb_p', 'pass')} for r in results]).to_csv(
            OUT/'matrix.csv', index=False)
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, dict)}
                  for r in mc_results]).to_csv(OUT/'mc_summary.csv', index=False)
    primary_pass = all(r['pass'] for r in results if r['threshold'] == 10
                       and r['variant'] == 'C' and r['cap'] == 5)
    meta = {'study': 'connors_rsi2_daily', 'spec_sha256': digest(spec),
            'gate_sha256': digest(gate), 'script_sha256': digest(Path(__file__)),
            'ger40': {'status': 'DATA_UNAVAILABLE_FOR_REQUESTED_DUKASCOPY_SOURCE',
                      'native_broker_m1_exists': GER_NATIVE.exists(),
                      'native_broker_m1': str(GER_NATIVE)},
            'sources': sources, 'primary_pass': primary_pass,
            'primary_concurrency': concurrency, 'mc_template_pool': template_meta,
            'cells': results, 'monte_carlo': mc_results}
    (OUT/'run_meta.json').write_text(json.dumps(meta, indent=2, allow_nan=True),
                                     encoding='utf-8')
    lines = [
        'DAILY CONNORS RSI(2) INDICES AND FTMO 2-STEP SCENARIO - REPORT',
        'GER40: DATA UNAVAILABLE FOR REQUESTED DUKASCOPY SOURCE. A verified broker-native GER40.cash M1 tape exists, but is not this source and has no predeclared GER40 cost/USD conversion.',
        'D1 sessions end at 17:00 America/New_York; actual orders use first available subsequent M1 open.',
        'Cost floors: US500 0.55 points, US100 1.66 points round trip; swap 0 scenario.',
        f'MBB: {BOOT} moving blocks, block {BLOCK}, seed {SEED}.',
        f"PRIMARY RSI<10 C5 JOINT HISTORICAL GATE: {'PASS' if primary_pass else 'FAIL'}",
        '', 'BACKTEST MATRIX: net points and R are per closed trade; Sharpe is unannualized.',
        'ASSET | RSI< | EXIT | TRADES | WIN% | PF | GROSS/TRADE | NET/TRADE | NET R | HOLD D1 | SHARPE | MBB P | GATE']
    for r in results:
        name = r['variant']+(str(r['cap']) if r['variant'] != 'A' else '')
        lines.append(' | '.join((r['asset'], str(r['threshold']), name,
            str(r['closed_trades']), fmt(100*r['win_rate'], 1),
            fmt(r['profit_factor'], 3), fmt(r['gross_expectancy'], 4),
            fmt(r['net_expectancy'], 4), fmt(r['mean_net_r'], 4),
            fmt(r['average_holding_trading_days'], 2),
            fmt(r['net_sharpe_per_trade'], 4), fmt(r['mbb_p'], 4),
            'PASS' if r['pass'] else 'FAIL')))
    lines += ['', 'PRIMARY PORTFOLIO AND EMPIRICAL R POOL',
              f'Primary historical concurrency {concurrency}; Monte Carlo common-period pool {template_meta}.',
              'The empirical Monte Carlo permits US500/US100 overlap, one sampled position per asset, at most two new positions per synthetic day.',
              '', 'MONTE CARLO: 10000 paths per fixed initial-$10000 cash risk profile.',
              'RISK | PHASE1 PASS | PHASE2 PASS GIVEN P1 | FUNDED FROM PURCHASE | FIRST PAYOUT | NET EV / $175 FEE | FUNDED DAYS MED/P90 | FUNDED TRADES MED/P90']
    for x in mc_results:
        d, n = x['days_to_funded_conditional'], x['trades_to_funded_conditional']
        lines.append(' | '.join((f"{100*x['risk_fraction']:.1f}%",
            fmt(x['phase1_pass_probability']), fmt(x['phase2_conditional_pass_probability']),
            fmt(x['funded_status_probability']), fmt(x['first_payout_probability']),
            fmt(x['net_ev_per_challenge'], 2),
            f"{fmt(d['median'], 1)}/{fmt(d['p90'], 1)}",
            f"{fmt(n['median'], 1)}/{fmt(n['p90'], 1)}")))
    lines += ['', 'MC FAILURE BREAKDOWN (counts among 10000 purchased paths; BOTH counted separately and inclusively).']
    for x in mc_results:
        lines.append(f"{100*x['risk_fraction']:.1f}% | Phase1 {x['phase1_failures']} "
                     f"unresolved {x['phase1_unresolved']} | Phase2 {x['phase2_failures']} "
                     f"unresolved {x['phase2_unresolved']} | funded loss {x['funded_fail_loss_count']} "
                     f"no-profit {x['funded_no_profit_count']} unresolved {x['funded_unresolved_count']} "
                     f"| receipt ${x['mean_gross_receipt']:.2f} | EV SE ${x['ev_mc_standard_error']:.2f}")
    lines += ['', 'FULL CELL DIAGNOSTICS AND ANNUAL ENTRY COHORTS']
    for r in results:
        lines.append(f"{r['cell']} | total net {r['net_total']:.3f} | "
                     f"exit {r['exit_counts']} | overnight {r['ny_overnight_trades']} | "
                     f"MAE magnitude R {r['mae_r_magnitude']} | MFE R {r['mfe_r']} | "
                     f"G1-G5 {r['gates']} | annual {r['annual']} | funnel {r['funnel']} "
                     f"| ledger {r['ledger']}")
    lines += ['', 'METHOD AND LIMITS',
              'MC samples common-period primary trade R and associated daily adverse paths with empirical spacing. The worst daily marks of overlapping trades are summed as if simultaneous, a conservative proxy, not tick-exact portfolio equity.',
              'Daily and static loss guards use this path proxy; original trade exits are not reoptimized or interrupted. Phases reset balance and sample independently; no strategy decay, changing spread, swap, leverage or account-correlation effect is modeled.',
              'The 30-funded-trade, >=14-calendar-day first-payout window and $175 fee are scenario inputs; real reward eligibility and account terms require separate review.',
              'The source tapes have been previously researched. Even a positive historical or simulated result would require independent forward and broker-execution validation.',
              'Official 2-Step objectives, fee refund and reward references checked 2026-09-28: https://ftmo.com/en/trading-objectives/ ; https://ftmo.com/en/faq/is-the-entry-fee-refunded/ ; https://ftmo.com/faq/how-do-i-withdraw-my-profits/ .']
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Wrote', report, flush=True)


if __name__ == '__main__':
    main()
