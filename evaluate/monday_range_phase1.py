# -*- coding: utf-8 -*-
"""
Hypothesis #33 - Monday Range Strategy - Phase 1 signal-quality screen.

SPEC + GATE frozen at data/results/monday_range_screen/ BEFORE this file existed.
Run with: C:\\Users\\Milad\\miniconda3\\envs\\reinforce_rl\\python.exe evaluate/monday_range_phase1.py
"""
import json
import math
import os
import platform
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import pytz

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
STUDY = os.path.join(ROOT, 'data', 'results', 'monday_range_screen')
os.makedirs(STUDY, exist_ok=True)

M1_CSV = os.path.join(ROOT, 'data', 'results', 'vwap_2021_2026', 'us500tick_m1.csv')

NY = pytz.timezone('America/New_York')
WALK_LAST_MIN = 15 * 60 + 55        # Friday 15:55 NY cutoff
SEED = 20260921
BOOT_ITERS = 10000
BOOT_BLOCK = 5                      # ~5 weeks of trades per block (weekly cadence)
G1_MIN_SIGNALS = 75
G3_MAX_P = 0.20
T0 = time.time()


def p(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ [1] load + M15
p('[1/5] loading M1 tape and aggregating M15 ...')
m1 = pd.read_csv(M1_CSV)
m1 = m1.sort_values('time').reset_index(drop=True)
t = m1['time'].to_numpy(np.int64)
o = m1['open'].to_numpy(np.float64)
h = m1['high'].to_numpy(np.float64)
lo = m1['low'].to_numpy(np.float64)
c = m1['close'].to_numpy(np.float64)
n = len(m1)

ny = pd.Series(pd.to_datetime(t, unit='s', utc=True)).dt.tz_convert(NY)

bcode = pd.factorize(t // 900)[0]
g = m1.groupby(bcode, sort=True)
m15 = pd.DataFrame({
    't': g['time'].first().to_numpy(np.int64),
    'o': g['open'].first().to_numpy(np.float64),
    'h': g['high'].max().to_numpy(np.float64),
    'l': g['low'].min().to_numpy(np.float64),
    'c': g['close'].last().to_numpy(np.float64),
    'nbars': g['close'].size().to_numpy(np.int64),
})
m15['t_close'] = m15['t'] + 900
m15ny = pd.Series(pd.to_datetime(m15['t'].to_numpy(), unit='s', utc=True)).dt.tz_convert(NY)
m15['ny_date'] = m15ny.dt.date
m15['weekday'] = m15ny.dt.weekday
iso = m15ny.dt.isocalendar()
m15['iso_yw'] = iso['year'].astype(int) * 100 + iso['week'].astype(int)
sparse = int((m15['nbars'] < 15).sum())
p('  M1 bars=%d | M15 buckets=%d (sparse <15 M1 bars: %d, %.2f%%)'
  % (n, len(m15), sparse, 100.0 * sparse / len(m15)))

# ------------------------------------------------------------------ [2] Monday ranges + first-claim signals
p('[2/5] Monday ranges and first-claim sweep/reclaim signals ...')
weeks = m15.groupby('iso_yw')
mon = m15[m15['weekday'] == 0]
mon_rng = mon.groupby('iso_yw').agg(MH=('h', 'max'), ML=('l', 'min'), monday=('ny_date', 'first'))

funnel = dict(iso_weeks=m15['iso_yw'].nunique(), weeks_with_monday=len(mon_rng))
rows = []
for yw, wk in m15[m15['weekday'].isin((1, 2, 3))].groupby('iso_yw'):
    if yw not in mon_rng.index:
        continue
    mh, ml = float(mon_rng.loc[yw, 'MH']), float(mon_rng.loc[yw, 'ML'])
    wk = wk.sort_values('t')
    long_hit = (wk['l'] < ml) & (wk['c'] > ml)
    short_hit = (wk['h'] > mh) & (wk['c'] < mh)
    cand = wk[long_hit | short_hit]
    if len(cand) == 0:
        continue
    b = cand.iloc[0]
    d = 1 if (b['l'] < ml and b['c'] > ml) else -1
    rows.append(dict(yw=int(yw), monday=mon_rng.loc[yw, 'monday'], mh=mh, ml=ml,
                     bar_t=int(b['t']), bar_tc=int(b['t_close']), d=d,
                     sweep_extreme=float(b['l'] if d > 0 else b['h'])))
funnel['weeks_with_signal'] = len(rows)
funnel['long_signals'] = sum(1 for r in rows if r['d'] > 0)
funnel['short_signals'] = sum(1 for r in rows if r['d'] < 0)
p('  ISO weeks=%d | with Monday=%d | with first-claim signal=%d (long=%d short=%d)'
  % (funnel['iso_weeks'], funnel['weeks_with_monday'], funnel['weeks_with_signal'],
     funnel['long_signals'], funnel['short_signals']))

# ------------------------------------------------------------------ [3] M1 entry + walk
p('[3/5] M1 entries and TP/SL/time walk ...')
n_no_entry = n_degen = 0
out = []
for r in rows:
    friday = pd.Timestamp(r['monday']) + timedelta(days=4)
    cutoff = int(NY.localize(friday.replace(hour=15, minute=55)).timestamp())
    e = int(np.searchsorted(t, r['bar_tc'], side='left'))
    if e >= n or t[e] > cutoff:
        n_no_entry += 1
        continue
    entry = float(o[e])
    sl = r['sweep_extreme']
    if r['d'] > 0:
        tp = r['mh']
        risk = entry - sl
    else:
        tp = r['ml']
        risk = sl - entry
    if not (risk > 0):
        n_degen += 1
        continue
    last = int(np.searchsorted(t, cutoff, side='right')) - 1
    if last < e:
        n_no_entry += 1
        continue
    hh = h[e:last + 1]
    ll = lo[e:last + 1]
    cc = c[e:last + 1]
    tt = t[e:last + 1]
    if r['d'] > 0:
        sl_idx = np.flatnonzero(ll <= sl)
        tp_idx = np.flatnonzero(hh >= tp)
    else:
        sl_idx = np.flatnonzero(hh >= sl)
        tp_idx = np.flatnonzero(ll <= tp)
    i_sl = int(sl_idx[0]) if len(sl_idx) else None
    i_tp = int(tp_idx[0]) if len(tp_idx) else None
    if i_sl is not None and (i_tp is None or i_sl <= i_tp):
        reason, xp, xi = 'SL', float(sl), i_sl
    elif i_tp is not None:
        reason, xp, xi = 'TP', float(tp), i_tp
    else:
        reason, xp, xi = 'TIME', float(cc[-1]), len(cc) - 1
    gross = (xp - entry) if r['d'] > 0 else (entry - xp)
    if r['d'] > 0:
        fav = float(hh[:xi + 1].max()) - entry
        adv = entry - float(ll[:xi + 1].min())
    else:
        fav = entry - float(ll[:xi + 1].min())
        adv = float(hh[:xi + 1].max()) - entry
    out.append(dict(
        yw=r['yw'], dir=('LONG' if r['d'] > 0 else 'SHORT'),
        signal_m15=pd.to_datetime(r['bar_t'], unit='s', utc=True).tz_convert(NY).strftime('%Y-%m-%d %H:%M'),
        entry_time=ny.iloc[e].strftime('%Y-%m-%d %H:%M'),
        exit_time=ny.iloc[e + xi].strftime('%Y-%m-%d %H:%M'),
        monday_high=r['mh'], monday_low=r['ml'], entry=entry, stop=sl, target=tp,
        risk=risk, exit_reason=reason, exit_price=xp, gross=gross,
        mfe=max(0.0, fav), mae=max(0.0, adv),
        mfe_r=max(0.0, fav) / risk, mae_r=max(0.0, adv) / risk,
        bars_held=xi + 1,
    ))
funnel['drop_no_entry'] = n_no_entry
funnel['drop_degenerate_risk'] = n_degen
funnel['final_signals'] = len(out)
p('  executed=%d (drops: no-entry=%d degenerate-risk=%d)' % (len(out), n_no_entry, n_degen))

sig = pd.DataFrame(out)
sig['year'] = pd.to_datetime(sig['signal_m15']).dt.year
sig_csv = os.path.join(STUDY, 'monday_range_signals.csv')
sig.to_csv(sig_csv, index=False)

# ------------------------------------------------------------------ [4] stats + bootstrap + gates
p('[4/5] stats, bootstrap, gates ...')
gross = sig['gross'].to_numpy(float)
mean_g = float(gross.mean())
win = float((gross > 0).mean())
outcome_mix = sig['exit_reason'].value_counts().to_dict()

rng = np.random.default_rng(SEED)
nn = len(gross)
nb = int(math.ceil(nn / BOOT_BLOCK))
starts = rng.integers(0, nn, size=(BOOT_ITERS, nb))
idx = (starts[..., None] + np.arange(BOOT_BLOCK)) % nn
means = gross[idx].mean(axis=(1, 2))
pneg = float((means <= 0).mean())

g1 = nn >= G1_MIN_SIGNALS
g2 = bool(mean_g > 0)
g3 = bool(pneg <= G3_MAX_P)
overall = bool(g1 and g2 and g3)
verdict = 'PASS' if overall else 'FAIL'

by_year = sig.groupby('year')['gross'].agg(['count', 'mean']).to_dict('index')

# ------------------------------------------------------------------ [5] report
p('[5/5] writing REPORT ...')
L = []
A = L.append
A('REPORT: MONDAY RANGE STRATEGY - PHASE 1 (HYPOTHESIS #33)')
A('Generated: %s  |  script: evaluate/monday_range_phase1.py' % time.strftime('%Y-%m-%d %H:%M:%S'))
A('SPEC/GATE: data/results/monday_range_screen/{SPEC,GATE}_MONDAY_RANGE.txt (frozen before code)')
A('Mechanism: Monday-range liquidity sweep/reclaim, first claim per week; discretionary')
A('oscillator divergence deliberately omitted (pre-declared, conservative).')
A('')
A('VERDICT: %s   (G-MON1=%s, G-MON2=%s, G-MON3=%s)' % (verdict, g1, g2, g3))
A('')
A('1. DATA AND INTEGRITY')
A('  Tape: %s (%d M1 bars, %s .. %s NY)' % (os.path.relpath(M1_CSV, ROOT), n, ny.iloc[0], ny.iloc[-1]))
A('  M15 buckets: %d (sparse <15 M1 bars: %d, %.2f%%) built from M1 per tasking'
  % (len(m15), sparse, 100.0 * sparse / len(m15)))
A('')
A('2. WEEKLY FUNNEL')
A('  ISO weeks on tape: %d | weeks with Monday data: %d | skipped (no Monday): %d'
  % (funnel['iso_weeks'], funnel['weeks_with_monday'], funnel['iso_weeks'] - funnel['weeks_with_monday']))
A('  weeks with first-claim signal: %d (LONG %d / SHORT %d)'
  % (funnel['weeks_with_signal'], funnel['long_signals'], funnel['short_signals']))
A('  drops: no-M1-entry-before-Friday-cutoff=%d degenerate-risk=%d' % (funnel['drop_no_entry'], funnel['drop_degenerate_risk']))
A('  FINAL valid signals: %d' % funnel['final_signals'])
A('')
A('3. OUTCOMES AND EXPECTANCY (points, pre-cost; reference spread floor 0.51 pt)')
A('  n=%d | gross mean=%.3f | median=%.3f | total=%.2f | win share (gross>0)=%.1f%%'
  % (nn, mean_g, float(np.median(gross)), float(gross.sum()), 100 * win))
A('  exit mix: %s' % ', '.join('%s=%d' % kv for kv in sorted(outcome_mix.items())))
A('  MFE/R mean=%.3f | MAE/R mean=%.3f (descriptive)'
  % (float(sig['mfe_r'].mean()), float(sig['mae_r'].mean())))
A('  LONG: n=%d gross mean=%.3f | SHORT: n=%d gross mean=%.3f'
  % (int((sig['dir'] == 'LONG').sum()), float(sig.loc[sig['dir'] == 'LONG', 'gross'].mean()),
     int((sig['dir'] == 'SHORT').sum()), float(sig.loc[sig['dir'] == 'SHORT', 'gross'].mean())))
A('  by year: %s' % ', '.join('%d: n=%d mean=%.2f' % (y, v['count'], v['mean'])
                              for y, v in sorted(by_year.items())))
A('')
A('4. GATES (frozen verbatim in GATE file)')
A('  G-MON1 valid signals >= %d: %d -> %s' % (G1_MIN_SIGNALS, nn, g1))
A('  G-MON2 gross expectancy > 0: %.4f -> %s' % (mean_g, g2))
A('  G-MON3 block bootstrap P(mean gross <= 0) <= %.2f: %.4f (block=%d, %d iters) -> %s'
  % (G3_MAX_P, pneg, BOOT_BLOCK, BOOT_ITERS, g3))
A('  OVERALL PHASE-1: %s' % verdict)
A('')
A('5. NOTES AND CAVEATS')
A('  - SL-first when TP and SL are both touched inside one M1 bar (conservative, pre-declared).')
A('  - Gross is pre-cost; the edge must clear the 0.51 pt spread floor before economics.')
A('  - Weekly cadence (~1 trade/week) makes year splits thin; they are descriptive only.')
A('  - Null result is final for this artifact.')

report_txt = '\n'.join(L) + '\n'
with open(os.path.join(STUDY, 'REPORT_MONDAY_RANGE_PHASE1.txt'), 'w', encoding='utf-8') as f:
    f.write(report_txt)

meta = dict(
    study='monday_range_screen', hypothesis='#33', phase=1, verdict=verdict,
    generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    elapsed_s=round(time.time() - T0, 1),
    python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
    seed=SEED, boot_iters=BOOT_ITERS, boot_block=BOOT_BLOCK,
    funnel={k: int(v) for k, v in funnel.items()},
    n_signals=nn, gross_mean=mean_g, win=win, outcome_mix={k: int(v) for k, v in outcome_mix.items()},
    pneg=pneg, gates=dict(g_mon1=bool(g1), g_mon2=bool(g2), g_mon3=bool(g3)),
    by_year={int(k): dict(count=int(v['count']), mean=float(v['mean'])) for k, v in by_year.items()},
)
with open(os.path.join(STUDY, 'run_meta.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=1)

p('')
p('================ PHASE 1 SUMMARY (Hypothesis #33, monday_range_screen) ================')
p(report_txt)
p('Artifacts: %s' % STUDY)
