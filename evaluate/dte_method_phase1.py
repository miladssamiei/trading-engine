# -*- coding: utf-8 -*-
"""
Hypothesis #32 - DTE Method (Delivery + Execution) - Phase 1 MFE/MAE excursion screen.

SPEC + GATE frozen at data/results/dte_method_screen/ BEFORE this file existed.
Run with: C:\\Users\\Milad\\miniconda3\\envs\\reinforce_rl\\python.exe evaluate/dte_method_phase1.py
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
STUDY = os.path.join(ROOT, 'data', 'results', 'dte_method_screen')
os.makedirs(STUDY, exist_ok=True)

M1_CSV = os.path.join(ROOT, 'data', 'results', 'vwap_2021_2026', 'us500tick_m1.csv')

NY = pytz.timezone('America/New_York')
SESS_OPEN_MIN = 9 * 60 + 30      # 09:30 NY, setup bars strictly inside [570, 930)
SESS_LAST_SETUP_MIN = 15 * 60 + 30
WALK_LAST_MIN = 15 * 60 + 55     # walk to the 15:55 bar inclusive
INV_WINDOW = 6                   # printed M1 bars after formation
THRESH_R = [0.5, 1.0, 1.5]
SEED = 20260921
BOOT_ITERS = 10000
G1_MIN_SIGNALS = 150
G3_MAX_P = 0.20
T0 = time.time()


def p(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ [1] load tape
p('[1/6] loading M1 tape ...')
m1 = pd.read_csv(M1_CSV)
m1 = m1.sort_values('time').reset_index(drop=True)
t = m1['time'].to_numpy(np.int64)
o = m1['open'].to_numpy(np.float64)
h = m1['high'].to_numpy(np.float64)
lo = m1['low'].to_numpy(np.float64)
c = m1['close'].to_numpy(np.float64)
n = len(m1)

ny = pd.Series(pd.to_datetime(t, unit='s', utc=True)).dt.tz_convert(NY)
minute = (ny.dt.hour * 60 + ny.dt.minute).to_numpy(np.int32)
ny_date = ny.dt.date
dcode = pd.factorize(ny_date)[0]
setup_ok = (minute >= SESS_OPEN_MIN) & (minute < SESS_LAST_SETUP_MIN)
p('  M1 bars=%d (%s .. %s) | cash-session [09:30,15:30) bars=%d'
  % (n, ny.iloc[0], ny.iloc[-1], int(setup_ok.sum())))

# ------------------------------------------------------------------ [2] M15 aggregation (from M1 tape, per tasking)
p('[2/6] aggregating M15 ...')
bcode = pd.factorize(t // 900)[0]
g = m1.groupby(bcode, sort=True)
m15_t = g['time'].first().to_numpy(np.int64)      # bucket open time
m15_o = g['open'].first().to_numpy(np.float64)
m15_h = g['high'].max().to_numpy(np.float64)
m15_l = g['low'].min().to_numpy(np.float64)
m15_c = g['close'].last().to_numpy(np.float64)
m15_n = g['close'].size().to_numpy(np.int64)
m15_end = m15_t + 900
n15 = len(m15_t)
sparse = int((m15_n < 15).sum())
p('  M15 buckets=%d (sparse <15 M1 bars: %d, %.2f%%)' % (n15, sparse, 100.0 * sparse / n15))

# ------------------------------------------------------------------ [3] M1 FVG formations + inversion (vectorized passes)
p('[3/6] M1 FVG formations and 6-bar inversion window ...')
bull_form = np.zeros(n, bool)
bear_form = np.zeros(n, bool)
bull_form[2:] = lo[2:] > h[:-2]
bear_form[2:] = h[2:] < lo[:-2]

funnel = {}
funnel['m1_bars'] = n
funnel['m1_session_bars'] = int(setup_ok.sum())
funnel['m15_buckets'] = n15
funnel['m15_sparse_buckets'] = sparse

cand_bull = np.where(bull_form & setup_ok)[0]
cand_bear = np.where(bear_form & setup_ok)[0]
funnel['m1_fvg_bull_insession'] = len(cand_bull)
funnel['m1_fvg_bear_insession'] = len(cand_bear)

# gap bounds per candidate: bull gap (h[i-2], lo[i]); bear gap (h[i], lo[i-2])
sigs = []  # (form_i, inv_j, dir) dir=+1 long, -1 short
for cd, isbull in ((cand_bull, True), (cand_bear, False)):
    if len(cd) == 0:
        continue
    if isbull:
        gl = h[cd - 2]
        gh = lo[cd]
    else:
        gl = h[cd]
        gh = lo[cd - 2]
    inv = np.full(len(cd), -1, np.int64)
    for k in range(1, INV_WINDOW + 1):
        jj = cd + k
        ok = (jj < n) & (inv < 0)
        if not ok.any():
            break
        jjk = jj[ok]
        if isbull:
            trig = c[jjk] < gl[ok]      # close below bull gap bottom -> SHORT
        else:
            trig = c[jjk] > gh[ok]      # close above bear gap top -> LONG
        idx_ok = np.where(ok)[0][trig]
        inv[idx_ok] = jjk[trig]
    done = inv >= 0
    d = -1 if isbull else 1
    for i0, j0 in zip(cd[done], inv[done]):
        sigs.append((int(i0), int(j0), d))
funnel['ifvg_short_raw'] = sum(1 for _, _, d in sigs if d < 0)
funnel['ifvg_long_raw'] = sum(1 for _, _, d in sigs if d > 0)
p('  in-session M1 FVGs: bull=%d bear=%d -> inverted within %d bars: short=%d long=%d'
  % (len(cand_bull), len(cand_bear), INV_WINDOW,
     funnel['ifvg_short_raw'], funnel['ifvg_long_raw']))

# ------------------------------------------------------------------ [4] session validity, risk, degenerate drop
p('[4/6] session validity, stop reference, risk ...')
rows = []
n_entry_missing = n_inv_session = n_date_mix = n_degen = 0
for i0, j0, d in sigs:
    if j0 + 1 >= n:
        n_entry_missing += 1
        continue
    if not setup_ok[j0]:
        n_inv_session += 1
        continue
    e = j0 + 1
    if dcode[e] != dcode[i0] or dcode[j0] != dcode[i0]:
        n_date_mix += 1
        continue
    if minute[e] > WALK_LAST_MIN:
        n_entry_missing += 1
        continue
    entry = o[e]
    if d < 0:
        stop = float(np.max(h[i0 - 1:j0 + 1]))
        risk = stop - entry
        m1_gl, m1_gh = float(h[i0 - 2]), float(lo[i0])
    else:
        stop = float(np.min(lo[i0 - 1:j0 + 1]))
        risk = entry - stop
        m1_gl, m1_gh = float(h[i0]), float(lo[i0 - 2])
    if not (risk > 0):
        n_degen += 1
        continue
    rows.append(dict(i=i0, j=j0, e=e, d=d, entry=entry, stop=stop, risk=risk,
                     m1_gl=m1_gl, m1_gh=m1_gh))
funnel['drop_inv_out_of_session'] = n_inv_session
funnel['drop_entry_missing_or_late'] = n_entry_missing
funnel['drop_date_mixed'] = n_date_mix
funnel['drop_degenerate_risk'] = n_degen
funnel['pre_context_signals'] = len(rows)
p('  session/risk valid: %d (drops: inv-session=%d entry=%d date=%d degen-risk=%d)'
  % (len(rows), n_inv_session, n_entry_missing, n_date_mix, n_degen))

# ------------------------------------------------------------------ [5] M15 delivery sweep + context join
p('[5/6] M15 delivery sweep + context join ...')
rows.sort(key=lambda r: t[r['e']])
bull_act, bear_act = [], []
b_ptr = 0
n_join_short = n_join_long = 0
for r in rows:
    tt = t[r['e']]
    pc = c[r['j']]
    while b_ptr < n15 and m15_end[b_ptr] <= tt:
        b = b_ptr
        if b >= 2:
            if m15_l[b] > m15_h[b - 2]:
                bull_act.append((float(m15_h[b - 2]), float(m15_l[b]), int(m15_t[b])))
            if m15_h[b] < m15_l[b - 2]:
                bear_act.append((float(m15_h[b]), float(m15_l[b - 2]), int(m15_t[b])))
        cc = m15_c[b]
        if cc is not None:
            bull_act[:] = [f for f in bull_act if not (cc < f[0])]
            bear_act[:] = [f for f in bear_act if not (cc > f[1])]
        b_ptr += 1
    if r['d'] < 0:
        hit = next((f for f in reversed(bear_act) if f[0] <= pc <= f[1]), None)
        if hit:
            r['m15_gl'], r['m15_gh'], r['m15_ft'] = hit
            n_join_short += 1
            r['joined'] = True
    else:
        hit = next((f for f in reversed(bull_act) if f[0] <= pc <= f[1]), None)
        if hit:
            r['m15_gl'], r['m15_gh'], r['m15_ft'] = hit
            n_join_long += 1
            r['joined'] = True
rows = [r for r in rows if r.get('joined')]
funnel['context_joined_short'] = n_join_short
funnel['context_joined_long'] = n_join_long
funnel['final_signals'] = len(rows)
p('  context-joined: short=%d long=%d -> final signals=%d' % (n_join_short, n_join_long, len(rows)))

# ------------------------------------------------------------------ [6] excursion walk + stats + gates
p('[6/6] excursion walk, stats, bootstrap, artifacts ...')
for r in rows:
    e = r['e']
    k = e
    while k + 1 < n and dcode[k + 1] == dcode[e] and minute[k + 1] <= WALK_LAST_MIN:
        k += 1
    hh = h[e:k + 1]
    ll = lo[e:k + 1]
    if r['d'] < 0:
        fav = r['entry'] - float(ll.min())
        adv = float(hh.max()) - r['entry']
    else:
        fav = float(hh.max()) - r['entry']
        adv = r['entry'] - float(ll.min())
    r['mfe'] = max(0.0, fav)
    r['mae'] = max(0.0, adv)
    r['walk_bars'] = k + 1 - e

sig_df = pd.DataFrame([dict(
    ny_date=str(ny_date.iloc[r['e']]),
    form_time=ny.iloc[r['i']].strftime('%Y-%m-%d %H:%M'),
    inv_time=ny.iloc[r['j']].strftime('%Y-%m-%d %H:%M'),
    entry_time=ny.iloc[r['e']].strftime('%Y-%m-%d %H:%M'),
    dir=('SHORT' if r['d'] < 0 else 'LONG'),
    m1_gap_low=r['m1_gl'], m1_gap_high=r['m1_gh'],
    m15_gap_low=r['m15_gl'], m15_gap_high=r['m15_gh'],
    m15_form_time=pd.to_datetime(r['m15_ft'], unit='s', utc=True).tz_convert(NY).strftime('%Y-%m-%d %H:%M'),
    entry=r['entry'], stop=r['stop'], risk=r['risk'],
    mfe=r['mfe'], mae=r['mae'], mfe_r=r['mfe'] / r['risk'], mae_r=r['mae'] / r['risk'],
    walk_bars=r['walk_bars'],
) for r in rows])
sig_csv = os.path.join(STUDY, 'dte_phase1_signals.csv')
sig_df.to_csv(sig_csv, index=False)


def excursion_stats(sub):
    if len(sub) == 0:
        return dict(n=0)
    mfe = sub['mfe'].to_numpy(float)
    mae = sub['mae'].to_numpy(float)
    rk = sub['risk'].to_numpy(float)
    out = dict(n=len(sub), mfe=float(mfe.mean()), mae=float(mae.mean()),
               diff=float((mfe - mae).mean()),
               mfe_r=float((mfe / rk).mean()), mae_r=float((mae / rk).mean()),
               fav_share=float((mfe > mae).mean()))
    for k in THRESH_R:
        out['ex_mfe_%s' % k] = float(np.maximum(mfe - k * rk, 0).mean())
        out['ex_mae_%s' % k] = float(np.maximum(mae - k * rk, 0).mean())
    return out


agg = excursion_stats(sig_df)
st_long = excursion_stats(sig_df[sig_df['dir'] == 'LONG'])
st_short = excursion_stats(sig_df[sig_df['dir'] == 'SHORT'])
sig_df['year'] = pd.to_datetime(sig_df['ny_date']).dt.year
by_year = {int(y): len(s) for y, s in sig_df.groupby('year')}

# day-block bootstrap on pooled mean(MFE-MAE)
rng = np.random.default_rng(SEED)
days = sig_df.groupby('ny_date').apply(lambda s: (s['mfe'] - s['mae']).sum(), include_groups=False)
cnts = sig_df.groupby('ny_date').size()
ds = days.to_numpy(float)
dc = cnts.to_numpy(float)
idx = rng.integers(0, len(ds), size=(BOOT_ITERS, len(ds)))
means = ds[idx].sum(axis=1) / dc[idx].sum(axis=1)
pneg = float((means <= 0).mean())

g1 = len(sig_df) >= G1_MIN_SIGNALS
g2_checks = {('%.1fR' % k): (agg['ex_mfe_%s' % k] > agg['ex_mae_%s' % k]) for k in THRESH_R}
g2 = bool(agg['mfe'] > agg['mae'] and all(g2_checks.values()))
g3 = bool(pneg <= G3_MAX_P)
overall = bool(g1 and g2 and g3)
verdict = 'PASS' if overall else 'FAIL'

L = []
A = L.append
A('REPORT: DTE METHOD (DELIVERY + EXECUTION) - PHASE 1 (HYPOTHESIS #32)')
A('Generated: %s  |  script: evaluate/dte_method_phase1.py' % time.strftime('%Y-%m-%d %H:%M:%S'))
A('SPEC/GATE: data/results/dte_method_screen/{SPEC,GATE}_DTE_METHOD.txt (frozen before code)')
A('Mechanism: M15 FVG delivery context + M1 inverted-FVG (V-spike) execution; discretionary')
A('liquidity-target trendlines deliberately omitted (core directional edge only).')
A('')
A('VERDICT: %s   (G-DTE1=%s, G-DTE2=%s, G-DTE3=%s)' % (verdict, g1, g2, g3))
A('')
A('1. DATA AND INTEGRITY')
A('  Tape: %s (%d M1 bars, %s .. %s NY)' % (os.path.relpath(M1_CSV, ROOT), n,
  ny.iloc[0], ny.iloc[-1]))
A('  Cash-session [09:30,15:30) M1 bars: %d | M15 buckets: %d (sparse <15 bars: %d, %.2f%%)'
  % (funnel['m1_session_bars'], n15, sparse, 100.0 * sparse / n15))
A('  M15 built from the M1 tape per tasking (project M5 convention overridden; integrity above).')
A('')
A('2. FUNNEL')
A('  in-session M1 FVG formations: bull=%d bear=%d' % (funnel['m1_fvg_bull_insession'], funnel['m1_fvg_bear_insession']))
A('  inverted within %d bars: short=%d long=%d' % (INV_WINDOW, funnel['ifvg_short_raw'], funnel['ifvg_long_raw']))
A('  drops: displacement-out-of-session=%d entry-missing/late=%d date-mixed=%d degenerate-risk=%d'
  % (funnel['drop_inv_out_of_session'], funnel['drop_entry_missing_or_late'],
     funnel['drop_date_mixed'], funnel['drop_degenerate_risk']))
A('  pre-context session-valid: %d' % funnel['pre_context_signals'])
A('  context-joined (active same-direction M15 FVG): short=%d long=%d' % (n_join_short, n_join_long))
A('  FINAL excursion-qualified signals: %d | by year: %s'
  % (len(sig_df), ', '.join('%d:%d' % kv for kv in sorted(by_year.items()))))
A('')
A('3. EXCURSION EXPECTANCY (points, pre-cost; reference spread floor 0.51 pt round trip)')
for nm, st in (('AGGREGATE', agg), ('LONG', st_long), ('SHORT', st_short)):
    if st['n'] == 0:
        A('  %s: n=0' % nm)
        continue
    A('  %s: n=%d MFE=%.3f MAE=%.3f diff=%.3f (MFE/R=%.3f MAE/R=%.3f, MFE>MAE share=%.1f%%)'
      % (nm, st['n'], st['mfe'], st['mae'], st['diff'], st['mfe_r'], st['mae_r'], 100 * st['fav_share']))
    for k in THRESH_R:
        A('    %.1fR: excess MFE=%.3f vs excess MAE=%.3f' % (k, st['ex_mfe_%s' % k], st['ex_mae_%s' % k]))
A('')
A('4. GATES (frozen verbatim in GATE file)')
A('  G-DTE1 total session-valid signals >= %d: %d -> %s' % (G1_MIN_SIGNALS, len(sig_df), g1))
A('  G-DTE2 mean MFE > mean MAE: %.3f vs %.3f -> %s; threshold excesses -> %s'
  % (agg['mfe'], agg['mae'], agg['mfe'] > agg['mae'], g2_checks))
A('  G-DTE3 day-block bootstrap P(mean(MFE-MAE)<=0) <= %.2f: %.4f (n_day_blocks=%d) -> %s'
  % (G3_MAX_P, pneg, len(ds), g3))
A('  OVERALL PHASE-1: %s' % verdict)
A('')
A('5. NOTES AND CAVEATS')
A('  - Excursions are pre-cost; before the 0.51 pt spread any residual diff must clear it to matter.')
A('  - No discretionary target trendlines modeled (omission pre-declared, conservative for the method).')
A('  - M15 context uses completed buckets only; M1 sequencing is printed-bar based.')
A('  - Null result is final for this artifact; a PASS would only justify a separate Phase-2 SPEC.')

report_txt = '\n'.join(L) + '\n'
with open(os.path.join(STUDY, 'REPORT_DTE_PHASE1.txt'), 'w', encoding='utf-8') as f:
    f.write(report_txt)

meta = dict(
    study='dte_method_screen', hypothesis='#32', phase=1, verdict=verdict,
    generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    elapsed_s=round(time.time() - T0, 1),
    python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
    seed=SEED, boot_iters=BOOT_ITERS, inv_window=INV_WINDOW, thresholds_r=THRESH_R,
    funnel={k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in funnel.items()},
    agg={k: v for k, v in agg.items()},
    pneg=pneg, gates=dict(g_dte1=bool(g1), g_dte2=bool(g2), g_dte3=bool(g3)),
    by_year=by_year,
)
with open(os.path.join(STUDY, 'run_meta.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=1)

p('')
p('================ PHASE 1 SUMMARY (Hypothesis #32, dte_method_screen) ================')
p(report_txt)
p('Artifacts: %s' % STUDY)
