# -*- coding: utf-8 -*-
"""
Hypothesis #31 - VRP-Conditioned ORB - Phase 1 signal-quality / filter screen.

SPEC + GATE frozen at data/results/orb_vrp_filter/ BEFORE this file existed.
Run with: C:\\Users\\Milad\\miniconda3\\envs\\reinforce_rl\\python.exe evaluate/orb_vrp_filter_phase1.py
"""
import json
import math
import os
import platform
import time

import numpy as np
import pandas as pd
import pytz

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
STUDY = os.path.join(ROOT, 'data', 'results', 'orb_vrp_filter')
os.makedirs(STUDY, exist_ok=True)

TRADES_CSV = os.path.join(ROOT, 'data', 'results', 'orb_2021_present', 'py_trades.csv')
FEAT_CSV = os.path.join(ROOT, 'data', 'results', 'daily_feature_engineering', 'daily_features_stationary.csv')

SPREAD_RT = 0.51      # verified real-tick median round-trip spread, index pts
ROLL_N = 252          # trailing percentile window (feature-file observations)
ANNUAL = 252
G1_MIN_TRADES = 100
NY = pytz.timezone('America/New_York')
T0 = time.time()


def p(msg):
    print(msg, flush=True)


def metrics(df):
    """Trade Count / Win Rate / Gross / Net / Avg Net / Net Sharpe on a trade list."""
    n = len(df)
    if n == 0:
        return dict(n=0, win=float('nan'), gross=0.0, net=0.0, avg_net=float('nan'),
                    sharpe=float('nan'), avg_gross=float('nan'))
    net = df['net'].to_numpy(dtype=float)
    gross = df['gross'].to_numpy(dtype=float)
    m = float(np.mean(net))
    s = float(np.std(net, ddof=1)) if n > 1 else float('nan')
    sharpe = m / s * math.sqrt(ANNUAL) if (n > 1 and s > 0) else float('nan')
    return dict(n=n, win=float(np.mean(net > 0)), gross=float(gross.sum()), net=float(net.sum()),
                avg_net=m, avg_gross=float(np.mean(gross)), sharpe=sharpe)


def fmt(md, base_avg=None):
    if md['n'] == 0:
        return 'n=0'
    s = ('n=%d win=%.1f%% gross=%.2f net=%.2f avg_net=%.3f net_sharpe=%.3f'
         % (md['n'], 100 * md['win'], md['gross'], md['net'], md['avg_net'], md['sharpe']))
    if base_avg is not None and base_avg != 0:
        s += ' edge_conc=%+.1f%%' % (100.0 * (md['avg_net'] - base_avg) / abs(base_avg))
    return s


# ------------------------------------------------------------------ [1] load + join
p('[1/5] loading ORB trades and VRP features ...')
tr = pd.read_csv(TRADES_CSV, sep=';')
tr['open_dt'] = pd.to_datetime(tr['open_time'], format='%Y.%m.%d %H:%M')
tr['ny_date'] = pd.Series(pd.to_datetime(tr['open_dt'].to_numpy(), utc=True)).dt.tz_convert(NY).dt.date
tr['gross'] = tr['profit'].astype(float)
tr['net'] = tr['gross'] - SPREAD_RT

fe = pd.read_csv(FEAT_CSV)
fe['date'] = pd.to_datetime(fe['date'])
fe['d'] = fe['date'].dt.date

p('  ORB trades: %d (%s .. %s) | feature rows: %d (%s .. %s)'
  % (len(tr), tr['ny_date'].min(), tr['ny_date'].max(), len(fe), fe['d'].min(), fe['d'].max()))

j = tr.merge(fe[['d', 'vrp']], left_on='ny_date', right_on='d', how='inner').reset_index(drop=True)
p('  inner join on NY trading date: %d trades kept (%s .. %s)'
  % (len(j), j['ny_date'].min(), j['ny_date'].max()))

# ------------------------------------------------------------------ [2] causal trailing percentiles
p('[2/5] computing trailing %d-observation percentiles (inclusive, no future data) ...' % ROLL_N)
vrp_s = pd.Series(fe['vrp'].to_numpy(dtype=float))
roll = vrp_s.rolling(ROLL_N, min_periods=ROLL_N)
fe['p50'] = roll.quantile(0.50).to_numpy()
fe['p75'] = roll.quantile(0.75).to_numpy()
fe['p90'] = roll.quantile(0.90).to_numpy()
fe['pct_ready'] = ~(np.isnan(fe['p50'].to_numpy()))

j = j.merge(fe[['d', 'p50', 'p75', 'p90', 'pct_ready']], left_on='ny_date', right_on='d',
            how='left', suffixes=('', '_r'))
n_ready = int(j['pct_ready'].fillna(False).sum())
first_ready = j.loc[j['pct_ready'].fillna(False), 'ny_date'].min() if n_ready else None
p('  trades with trailing percentiles available: %d/%d (first arm-eligible date: %s)'
  % (n_ready, len(j), first_ready))

# ------------------------------------------------------------------ [3] arms
p('[3/5] filtering arms ...')
ready = j['pct_ready'].fillna(False)
vrp = j['vrp'].to_numpy(dtype=float)
arms = {
    'A (High VRP > P50)': ready & (vrp > j['p50'].to_numpy()),
    'B (Very High VRP > P75)': ready & (vrp > j['p75'].to_numpy()),
    'C (Extreme VRP > P90)': ready & (vrp > j['p90'].to_numpy()),
    'D (Low VRP < P50, control)': ready & (vrp < j['p50'].to_numpy()),
}

base = metrics(j)
base_wu = metrics(j[ready])

res = {'BASELINE (joined timeframe)': base,
       'BASELINE (warmup-matched, descriptive)': base_wu}
for name, mask in arms.items():
    res[name] = metrics(j[mask])

# ------------------------------------------------------------------ [4] gates
p('[4/5] evaluating frozen gates ...')
pos_arms = ['A (High VRP > P50)', 'B (Very High VRP > P75)', 'C (Extreme VRP > P90)']
qualifying = [a for a in pos_arms if res[a]['n'] >= G1_MIN_TRADES]
g1 = len(qualifying) > 0
if g1:
    best = sorted(qualifying, key=lambda a: (res[a]['sharpe'], res[a]['n']), reverse=True)[0]
    g2 = bool(res[best]['sharpe'] > base['sharpe'])
    g3 = bool(res[best]['win'] > base['win'])
else:
    best = None
    g2 = False
    g3 = False
overall = bool(g1 and g2 and g3)
verdict = 'PASS' if overall else 'FAIL'

# ------------------------------------------------------------------ [5] artifacts
p('[5/5] writing CSV and REPORT ...')
if best is not None:
    out = j[arms[best]].copy()
    out['filter_arm'] = best[0]
else:
    out = j.copy()
    out['filter_arm'] = 'NONE'
out['vrp'] = out['vrp'].astype(float)
out['p50'] = out['p50'].astype(float)
out['p75'] = out['p75'].astype(float)
out['p90'] = out['p90'].astype(float)
cols = ['ticket', 'open_time', 'close_time', 'type', 'open_price', 'stop_loss', 'take_profit',
        'close_price', 'exit_reason', 'ny_date', 'gross', 'net', 'vrp', 'p50', 'p75', 'p90',
        'filter_arm']
out[cols].to_csv(os.path.join(STUDY, 'orb_vrp_filtered_trades.csv'), index=False)

L = []
A = L.append
A('REPORT: VRP-CONDITIONED ORB - PHASE 1 (HYPOTHESIS #31)')
A('Generated: %s  |  script: evaluate/orb_vrp_filter_phase1.py' % time.strftime('%Y-%m-%d %H:%M:%S'))
A('SPEC/GATE: data/results/orb_vrp_filter/{SPEC,GATE}_ORB_VRP.txt (frozen before code)')
A('')
A('VERDICT: %s   (G-VRP1=%s, G-VRP2=%s, G-VRP3=%s, best_arm=%s)'
  % (verdict, g1, g2, g3, best if best else 'NONE'))
A('')
A('1. DATA AND JOIN')
A('  Trades: %s (source rows=%d, %s .. %s)' % (os.path.relpath(TRADES_CSV, ROOT), len(tr),
   tr['ny_date'].min(), tr['ny_date'].max()))
A('  Features: %s (rows=%d, %s .. %s, column vrp; s->s+1 shift already applied in source)'
  % (os.path.relpath(FEAT_CSV, ROOT), len(fe), fe['d'].min(), fe['d'].max()))
A('  Inner join on NY trading date: %d trades, %s .. %s (BASELINE timeframe)'
  % (len(j), j['ny_date'].min(), j['ny_date'].max()))
A('  Trailing %d-obs percentiles inclusive of d (causal; vrp(d) known by day-d open);' % ROLL_N)
A('  first arm-eligible trade date: %s (%d trades arm-eligible, %d in warmup dead-zone)'
  % (first_ready, n_ready, len(j) - n_ready))
A('')
A('2. METRICS (net = gross - %.2f pt spread; no swap, ORB is intraday; per-trade Sharpe x sqrt(252))' % SPREAD_RT)
A('  BASELINE (joined timeframe):     %s' % fmt(base))
A('  BASELINE (warmup-matched, descr):%s' % fmt(base_wu))
for name in arms:
    A('  Arm %s: %s' % (name, fmt(res[name], base['avg_net'])))
A('')
A('3. EDGE CONCENTRATION (avg net PnL/trade vs joined baseline)')
for name in pos_arms:
    r = res[name]
    if r['n'] > 0 and base['avg_net'] != 0:
        A('  Arm %s: %+.1f%%  (avg_net %.3f vs baseline %.3f)'
          % (name.split(' ')[0], 100.0 * (r['avg_net'] - base['avg_net']) / abs(base['avg_net']),
             r['avg_net'], base['avg_net']))
    else:
        A('  Arm %s: n/a' % name.split(' ')[0])
A('')
A('4. GATES (frozen verbatim in GATE file)')
A('  G-VRP1 at least one of A/B/C retains >= %d trades: %s -> %s'
  % (G1_MIN_TRADES, '; '.join('%s n=%d' % (a.split(' ')[0], res[a]['n']) for a in pos_arms), g1))
if best is not None:
    A('  G-VRP2 best arm (%s) net Sharpe %.4f > baseline %.4f: %s'
      % (best.split(' ')[0], res[best]['sharpe'], base['sharpe'], g2))
    A('  G-VRP3 best arm (%s) win rate %.4f > baseline %.4f: %s'
      % (best.split(' ')[0], res[best]['win'], base['win'], g3))
else:
    A('  G-VRP2/G-VRP3: not evaluable - no arm qualified under G-VRP1 -> False/False')
A('  OVERALL PHASE-1: %s' % verdict)
A('')
A('5. NOTES AND CAVEATS')
A('  - Arms cover only post-warmup dates (%s onward); the joined baseline includes the warmup year,' % first_ready)
A('    so arm-vs-baseline comparisons embed a regime-shift; the warmup-matched baseline row isolates this.')
A('  - Arm D is the inverted control (bottom 50%); it is never eligible as the best passing arm.')
A('  - Multiple arms are nested (B is a subset of A); no multiple-comparison correction is applied -')
A('    this is a pre-declared screen, not a selection tournament.')
A('  - Null result is final for this artifact.')

report_txt = '\n'.join(L) + '\n'
with open(os.path.join(STUDY, 'REPORT_ORB_VRP.txt'), 'w', encoding='utf-8') as f:
    f.write(report_txt)

meta = dict(
    study='orb_vrp_filter', hypothesis='#31', phase=1, verdict=verdict,
    generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    elapsed_s=round(time.time() - T0, 1),
    python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
    spread_rt=SPREAD_RT, roll_n=ROLL_N,
    trades_source=len(tr), joined=len(j), arm_eligible=n_ready,
    first_arm_date=str(first_ready),
    gates=dict(g_vrp1=bool(g1), g_vrp2=bool(g2), g_vrp3=bool(g3)),
    best_arm=(best if best else 'NONE'),
    baseline=dict(n=base['n'], win=base['win'], net=base['net'], sharpe=base['sharpe']),
    arms={k.split(' ')[0]: dict(n=v['n'], win=v['win'], net=v['net'], sharpe=v['sharpe'])
          for k, v in res.items() if k.startswith(('A ', 'B ', 'C ', 'D '))},
)
with open(os.path.join(STUDY, 'run_meta.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=1)

p('')
p('================ PHASE 1 SUMMARY (Hypothesis #31, orb_vrp_filter) ================')
p(report_txt)
p('Artifacts: %s' % STUDY)
