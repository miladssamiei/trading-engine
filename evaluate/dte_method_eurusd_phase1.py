# -*- coding: utf-8 -*-
"""
Hypothesis #34 - DTE Method (Delivery + Execution) - EURUSD Phase 1
pre-cost MFE/MAE excursion screen ONLY: no trade rule beyond first-touch
excursions, no EA, no tester, no Phase 2 implied by any outcome.

Frozen contract this script executes (where any code choice is ambiguous,
the SPEC text wins):
  SPEC: data/results/dte_method_eurusd/SPEC_DTE_EURUSD.txt
  GATE: data/results/dte_method_eurusd/GATE_DTE_EURUSD.txt
(both frozen BEFORE this file existed)

Run with:
  C:\\Users\\Milad\\miniconda3\\envs\\reinforce_rl\\python.exe evaluate/dte_method_eurusd_phase1.py

Template lineage: evaluate/dte_method_phase1.py (Hypothesis #32, US500-tick).
Intentional deviations from the template per SPEC #34 tasking: full-tape M1 FVG
scan with the session filter applied to the DISPLACEMENT bar (not formation);
stop slice i-2..j inclusive; M15 join anchored at the displacement bar close
time; per-bucket remove-then-append update order; earliest-formed gap match.
"""
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import pytz

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
STUDY = os.path.join(ROOT, 'data', 'results', 'dte_method_eurusd')
os.makedirs(STUDY, exist_ok=True)

M1_CSV = os.path.join(ROOT, 'data', 'results', 'eurusd_onboarding', 'eurusdtick_m1.csv')

NY = pytz.timezone('America/New_York')
SESS_LO = 480          # 08:00 ET; displacement bar open wall strictly inside [480, 960)
SESS_HI = 960          # 16:00 ET (exclusive)
WALK_LAST_MIN = 1020   # walk includes M1 bars with open wall < 17:00 ET only (17:00 bar = next FX day)
INV_WINDOW = 6         # printed M1 bars after formation
THRESH_R = [0.5, 1.0, 1.5]
SEED = 20260921
BOOT_ITERS = 10000
G1_MIN = 150
G3_MAX_P = 0.20
PIP = 1e-4             # 1 pip = 0.00010 (5-decimal EURUSD)
COST_FLOOR_PIPS = 0.3  # verified London/NY overlap median spread (eurusd_onboarding 2026-09-22)
T0 = time.time()


def p(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ [1/7] load tape
p('[1/7] loading M1 tape ...')
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
setup_ok = (minute >= SESS_LO) & (minute < SESS_HI)

funnel = {}
funnel['m1_bars'] = n
funnel['m1_session_bars'] = int(setup_ok.sum())
p('  M1 bars=%d (%s .. %s) | displacement-window [08:00,16:00) ET bars=%d'
  % (n, ny.iloc[0], ny.iloc[-1], funnel['m1_session_bars']))

# ------------------------------------------------------------------ [2/7] M15 aggregation (from M1 tape, per SPEC tasking)
p('[2/7] aggregating M15 ...')
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
funnel['m15_buckets'] = n15
funnel['m15_sparse_buckets'] = sparse
funnel['m15_mean_m1_per_bucket'] = float(m15_n.mean())
p('  M15 buckets=%d (sparse <15 M1 bars: %d, %.2f%%; mean M1 bars/bucket: %.2f)'
  % (n15, sparse, 100.0 * sparse / n15, funnel['m15_mean_m1_per_bucket']))

# ------------------------------------------------------------------ [3/7] full-tape M1 FVG formations + inversion window
p('[3/7] M1 FVG formations (full tape) and %d-bar inversion window ...' % INV_WINDOW)
bull_form = np.zeros(n, bool)
bear_form = np.zeros(n, bool)
bull_form[2:] = lo[2:] > h[:-2]
bear_form[2:] = h[2:] < lo[:-2]
funnel['m1_fvg_bull_fulltape'] = int(bull_form.sum())
funnel['m1_fvg_bear_fulltape'] = int(bear_form.sum())

# no session pre-filter on formations (template filtered them; SPEC #34 filters the displacement bar)
cand_bull = np.where(bull_form)[0]
cand_bear = np.where(bear_form)[0]

sigs = []  # (form_i, inv_j, dir) dir=+1 long, -1 short
raw_short = raw_long = insess_short = insess_long = 0
for cd, isbull in ((cand_bull, True), (cand_bear, False)):
    if len(cd) == 0:
        continue
    if isbull:
        gl = h[cd - 2]     # bull gap bottom
        gh = lo[cd]        # bull gap top
    else:
        gl = h[cd]         # bear gap bottom
        gh = lo[cd - 2]    # bear gap top
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
    safe_inv = np.where(done, inv, 0)
    keep = done & setup_ok[safe_inv]    # displacement bar open wall in [480, 960)
    d = -1 if isbull else 1
    for i0, j0 in zip(cd[keep], safe_inv[keep]):
        sigs.append((int(i0), int(j0), d))
    if isbull:
        raw_short = int(done.sum())
        insess_short = int(keep.sum())
    else:
        raw_long = int(done.sum())
        insess_long = int(keep.sum())
funnel['ifvg_short_raw'] = raw_short
funnel['ifvg_long_raw'] = raw_long
funnel['ifvg_short_insession'] = insess_short
funnel['ifvg_long_insession'] = insess_long
p('  full-tape M1 FVGs: bull=%d bear=%d -> inverted within %d printed bars (raw): short=%d long=%d'
  % (funnel['m1_fvg_bull_fulltape'], funnel['m1_fvg_bear_fulltape'], INV_WINDOW,
     raw_short, raw_long))
p('  displacement bar in-session [08:00,16:00) ET: short=%d long=%d -> session-valid setups=%d'
  % (insess_short, insess_long, len(sigs)))

# ------------------------------------------------------------------ [4/7] entry/date validity + initial risk
p('[4/7] entry/date validity, stop reference, risk ...')
rows = []
n_entry_missing = n_date_mix = n_degen = 0
for i0, j0, d in sigs:
    if j0 + 1 >= n:
        n_entry_missing += 1
        continue
    e = j0 + 1                        # immediately next printed bar
    if dcode[e] != dcode[j0]:
        n_date_mix += 1
        continue
    if minute[e] >= WALK_LAST_MIN:
        n_entry_missing += 1
        continue
    entry = float(o[e])
    if d < 0:
        # DEVIATION A: formation-through-displacement i-2..j inclusive (template started at i-1)
        stop = float(np.max(h[i0 - 2:j0 + 1]))
        risk = abs(entry - stop)
        m1_gl, m1_gh = float(h[i0 - 2]), float(lo[i0])
    else:
        stop = float(np.min(lo[i0 - 2:j0 + 1]))
        risk = abs(entry - stop)
        m1_gl, m1_gh = float(h[i0]), float(lo[i0 - 2])
    if not (risk > 0):
        n_degen += 1
        continue
    rows.append(dict(i=i0, j=j0, e=e, d=d, entry=entry, stop=stop, risk=risk,
                     m1_gl=m1_gl, m1_gh=m1_gh))
funnel['drop_entry_missing_or_late'] = n_entry_missing
funnel['drop_date_mixed'] = n_date_mix
funnel['drop_degenerate_risk'] = n_degen
funnel['pre_context_signals'] = len(rows)
p('  entry/date/risk valid: %d (drops: entry-missing/late=%d date-mixed=%d degen-risk=%d)'
  % (len(rows), n_entry_missing, n_date_mix, n_degen))

# ------------------------------------------------------------------ [5/7] M15 delivery sweep + context join
p('[5/7] M15 delivery sweep + context join ...')


class ActiveGaps(object):
    """Active same-direction M15 FVG store: preallocated parallel numpy arrays,
    per-bucket vectorized mitigation mask + compaction, amortized-doubling
    append. Storage order == formation (append) order; compaction preserves it."""

    def __init__(self, cap=1024):
        self._cap = cap
        self.size = 0
        self.glo = np.empty(cap, np.float64)   # gap bottom
        self.ghi = np.empty(cap, np.float64)   # gap top
        self.gft = np.empty(cap, np.int64)     # formation bucket open time

    def mitigate(self, cc, is_bull):
        s = self.size
        if s == 0:
            return
        if is_bull:
            keep = cc >= self.glo[:s]          # invalidated when a bucket closes below gap bottom
        else:
            keep = cc <= self.ghi[:s]          # invalidated when a bucket closes above gap top
        nk = int(keep.sum())
        if nk < s:
            self.glo[:nk] = self.glo[:s][keep]
            self.ghi[:nk] = self.ghi[:s][keep]
            self.gft[:nk] = self.gft[:s][keep]
            self.size = nk

    def append(self, lo_v, hi_v, ft_v):
        if self.size >= self._cap:
            new_cap = self._cap * 2
            for name in ('glo', 'ghi', 'gft'):
                arr = getattr(self, name)
                grown = np.empty(new_cap, arr.dtype)
                grown[:self.size] = arr[:self.size]
                setattr(self, name, grown)
            self._cap = new_cap
        self.glo[self.size] = lo_v
        self.ghi[self.size] = hi_v
        self.gft[self.size] = ft_v
        self.size += 1

    def match_first(self, pc):
        s = self.size
        if s == 0:
            return None
        m = (self.glo[:s] <= pc) & (pc <= self.ghi[:s])
        if not m.any():
            return None
        k = int(np.argmax(m))                  # DEVIATION D: earliest-formed match (append order)
        return float(self.glo[k]), float(self.ghi[k]), int(self.gft[k])


rows.sort(key=lambda r: int(t[r['j']] + 60))   # DEVIATION B: consume in displacement-close order
bull_act = ActiveGaps()
bear_act = ActiveGaps()
b_ptr = 0
n_join_short = n_join_long = 0
for r in rows:
    tt = int(t[r['j']]) + 60                   # displacement bar close time
    pc = float(c[r['j']])                      # displacement bar close
    while b_ptr < n15 and m15_end[b_ptr] <= tt:
        b = b_ptr
        cc = float(m15_c[b])
        bull_act.mitigate(cc, True)            # (a) remove gaps mitigated by THIS bucket's close ...
        bear_act.mitigate(cc, False)
        if b >= 2:                             # (b) ... THEN append gaps newly formed at b
            if m15_l[b] > m15_h[b - 2]:
                bull_act.append(float(m15_h[b - 2]), float(m15_l[b]), int(m15_t[b]))
            if m15_h[b] < m15_l[b - 2]:
                bear_act.append(float(m15_h[b]), float(m15_l[b - 2]), int(m15_t[b]))
        b_ptr += 1
    if r['d'] < 0:
        hit = bear_act.match_first(pc)
    else:
        hit = bull_act.match_first(pc)
    if hit is not None:
        r['m15_gl'], r['m15_gh'], r['m15_ft'] = hit
        r['joined'] = True
        if r['d'] < 0:
            n_join_short += 1
        else:
            n_join_long += 1
rows = [r for r in rows if r.get('joined')]
funnel['context_joined_short'] = n_join_short
funnel['context_joined_long'] = n_join_long
funnel['final_signals'] = len(rows)
p('  context-joined (active same-direction M15 FVG at displacement close): short=%d long=%d -> final signals=%d'
  % (n_join_short, n_join_long, len(rows)))

# ------------------------------------------------------------------ [6/7] excursion walk (to 17:00 ET of the entry bar's NY date)
p('[6/7] excursion walk to 17:00 ET ...')
walk_ok = minute < WALK_LAST_MIN
ndates = int(dcode[-1]) + 1
Ld = np.full(ndates, -1, np.int64)
wk = np.where(walk_ok)[0]
Ld[dcode[wk]] = wk    # wk ascending -> last walk-ok index per NY date (contiguous prefix per date)
for r in rows:
    e = r['e']
    k = int(Ld[dcode[e]])                     # window = [e, Ld[date(e)]] inclusive; entry bar included
    hh = h[e:k + 1]
    ll = lo[e:k + 1]
    if r['d'] < 0:
        fav = r['entry'] - float(ll.min())
        adv = float(hh.max()) - r['entry']
    else:
        fav = float(hh.max()) - r['entry']
        adv = r['entry'] - float(ll.min())
    r['mfe_pips'] = max(0.0, fav) / PIP
    r['mae_pips'] = max(0.0, adv) / PIP
    r['risk_pips'] = r['risk'] / PIP
    r['walk_bars'] = k + 1 - e
if rows:
    p('  walked %d signals (mean walk bars=%.1f)'
      % (len(rows), sum(r['walk_bars'] for r in rows) / float(len(rows))))
else:
    p('  no signals to walk')

# ------------------------------------------------------------------ [7/7] stats, bootstrap, gates, artifacts
p('[7/7] excursion stats, day-block bootstrap, gates, artifacts ...')
sig_df = pd.DataFrame([dict(
    ny_date=str(ny_date.iloc[r['e']]),
    form_time=ny.iloc[r['i']].strftime('%Y-%m-%d %H:%M'),
    inv_time=ny.iloc[r['j']].strftime('%Y-%m-%d %H:%M'),
    entry_time=ny.iloc[r['e']].strftime('%Y-%m-%d %H:%M'),
    dir=('SHORT' if r['d'] < 0 else 'LONG'),
    m1_gap_low=r['m1_gl'], m1_gap_high=r['m1_gh'],
    m15_gap_low=r['m15_gl'], m15_gap_high=r['m15_gh'],
    m15_form_time=pd.to_datetime(r['m15_ft'], unit='s', utc=True).tz_convert(NY).strftime('%Y-%m-%d %H:%M'),
    entry=r['entry'], stop=r['stop'], risk_pips=r['risk_pips'],
    mfe_pips=r['mfe_pips'], mae_pips=r['mae_pips'],
    mfe_r=r['mfe_pips'] / r['risk_pips'], mae_r=r['mae_pips'] / r['risk_pips'],
    walk_bars=r['walk_bars'],
) for r in rows])
sig_csv = os.path.join(STUDY, 'dte_eurusd_signals.csv')
sig_df.to_csv(sig_csv, index=False)


def excursion_stats(sub):
    if len(sub) == 0:
        return dict(n=0)
    mfe = sub['mfe_pips'].to_numpy(float)
    mae = sub['mae_pips'].to_numpy(float)
    rk = sub['risk_pips'].to_numpy(float)
    out = dict(
        n=len(sub),
        mfe=float(mfe.mean()), mae=float(mae.mean()), diff=float((mfe - mae).mean()),
        mfe_med=float(np.median(mfe)), mae_med=float(np.median(mae)), diff_med=float(np.median(mfe - mae)),
        mfe_r=float((mfe / rk).mean()), mae_r=float((mae / rk).mean()),
        mfe_r_med=float(np.median(mfe / rk)), mae_r_med=float(np.median(mae / rk)),
        fav_share=float((mfe > mae).mean()),
    )
    for k in THRESH_R:
        out['ex_mfe_%s' % k] = float(np.maximum(mfe - k * rk, 0).mean())
        out['ex_mae_%s' % k] = float(np.maximum(mae - k * rk, 0).mean())
    return out


agg = excursion_stats(sig_df)
if len(sig_df) > 0:
    st_long = excursion_stats(sig_df[sig_df['dir'] == 'LONG'])
    st_short = excursion_stats(sig_df[sig_df['dir'] == 'SHORT'])
    sig_df['year'] = pd.to_datetime(sig_df['ny_date']).dt.year
    by_year = {int(y): int(len(s)) for y, s in sig_df.groupby('year')}
else:
    st_long = dict(n=0)
    st_short = dict(n=0)
    by_year = {}

# day-block bootstrap: resample NY dates with replacement, pooled mean(MFE-MAE) in pips
if len(sig_df) > 0:
    rng = np.random.default_rng(SEED)
    days = sig_df.groupby('ny_date').apply(lambda s: (s['mfe_pips'] - s['mae_pips']).sum(), include_groups=False)
    cnts = sig_df.groupby('ny_date').size()
    ds = days.to_numpy(float)
    dc = cnts.to_numpy(float)
    idx = rng.integers(0, len(ds), size=(BOOT_ITERS, len(ds)))
    means = ds[idx].sum(axis=1) / dc[idx].sum(axis=1)
    pneg = float((means <= 0).mean())
    n_day_blocks = len(ds)
else:
    pneg = 1.0
    n_day_blocks = 0

g1 = bool(len(sig_df) >= G1_MIN)
if agg.get('n', 0) > 0:
    g2_checks = {('%.1fR' % k): bool(agg['ex_mfe_%s' % k] > agg['ex_mae_%s' % k]) for k in THRESH_R}
    g2 = bool(all(g2_checks.values()))
    mean_mfe_gt_mae = bool(agg['mfe'] > agg['mae'])   # descriptive only, NOT part of the gate
else:
    g2_checks = {('%.1fR' % k): False for k in THRESH_R}
    g2 = False
    mean_mfe_gt_mae = False
g3 = bool(pneg <= G3_MAX_P)
overall = bool(g1 and g2 and g3)
verdict = 'PASS' if overall else 'FAIL'

L = []
A = L.append
A('REPORT: DTE METHOD (DELIVERY + EXECUTION) - EURUSD PHASE 1 (HYPOTHESIS #34)')
A('Generated: %s  |  script: evaluate/dte_method_eurusd_phase1.py' % time.strftime('%Y-%m-%d %H:%M:%S'))
A('SPEC/GATE: data/results/dte_method_eurusd/{SPEC,GATE}_DTE_EURUSD.txt (frozen before code)')
A('Mechanism: M15 FVG delivery context + M1 inverted-FVG (V-spike) execution on EURUSD;')
A('discretionary liquidity-target trendlines / oscillator confluence deliberately omitted.')
A('Phase-1 pre-cost excursion screen ONLY (no trade rule, no EA, no tester, no Phase 2 implied).')
A('')
A('VERDICT: %s   (G-DTE1=%s, G-DTE2=%s, G-DTE3=%s)' % (verdict, g1, g2, g3))
A('')
A('1. DATA AND INTEGRITY')
A('  Tape: %s (%d M1 bars, %s .. %s NY; Bid, 5-decimal)'
  % (os.path.relpath(M1_CSV, ROOT), n, ny.iloc[0], ny.iloc[-1]))
A('  Displacement window [08:00,16:00) ET M1 bars: %d | walk cutoff: open wall < 17:00 ET (wall < %d)'
  % (funnel['m1_session_bars'], WALK_LAST_MIN))
A('  M15 buckets: %d (sparse <15 M1 bars: %d, %.2f%%; mean M1 bars/bucket: %.2f)'
  % (n15, sparse, 100.0 * sparse / n15, funnel['m15_mean_m1_per_bucket']))
A('  M15 built from the M1 tape per SPEC tasking (project M5 convention overridden; integrity above).')
A('  All excursion statistics in pips (1 pip = 0.00010).')
A('')
A('2. FUNNEL')
A('  full-tape M1 FVG formations: bull=%d bear=%d'
  % (funnel['m1_fvg_bull_fulltape'], funnel['m1_fvg_bear_fulltape']))
A('  inverted within %d printed bars (raw, before session filter): short=%d long=%d'
  % (INV_WINDOW, funnel['ifvg_short_raw'], funnel['ifvg_long_raw']))
A('  displacement bar open wall in [08:00,16:00) ET: short=%d long=%d -> session-valid setups=%d'
  % (funnel['ifvg_short_insession'], funnel['ifvg_long_insession'],
     funnel['ifvg_short_insession'] + funnel['ifvg_long_insession']))
A('  drops: entry-missing-or-late=%d date-mixed=%d degenerate-risk=%d'
  % (funnel['drop_entry_missing_or_late'], funnel['drop_date_mixed'], funnel['drop_degenerate_risk']))
A('  pre-context session-valid: %d' % funnel['pre_context_signals'])
A('  context-joined (active same-direction M15 FVG at displacement close): short=%d long=%d'
  % (n_join_short, n_join_long))
A('  FINAL excursion-qualified signals: %d | by year: %s'
  % (len(sig_df), ', '.join('%d:%d' % kv for kv in sorted(by_year.items()))))
A('')
A('3. EXCURSION EXPECTANCY (pips, pre-cost; R-normalized)')
for nm, st in (('AGGREGATE', agg), ('LONG', st_long), ('SHORT', st_short)):
    if st['n'] == 0:
        A('  %s: n=0' % nm)
        continue
    A('  %s: n=%d MFE=%.2f MAE=%.2f diff=%.2f pips | medians MFE=%.2f MAE=%.2f diff=%.2f | MFE>MAE share=%.1f%%'
      % (nm, st['n'], st['mfe'], st['mae'], st['diff'], st['mfe_med'], st['mae_med'], st['diff_med'],
         100 * st['fav_share']))
    A('    R-multiples: MFE/R=%.3f MAE/R=%.3f (medians %.3f / %.3f)'
      % (st['mfe_r'], st['mae_r'], st['mfe_r_med'], st['mae_r_med']))
    for k in THRESH_R:
        A('    %.1fR: excess MFE=%.3f pips vs excess MAE=%.3f pips'
          % (k, st['ex_mfe_%s' % k], st['ex_mae_%s' % k]))
A('')
A('4. COST-FLOOR CONTEXT (descriptive only; Phase 1 stays gross)')
A('  Verified EURUSD London/NY overlap median spread: %.1f pips round trip (eurusd_onboarding 2026-09-22).'
  % COST_FLOOR_PIPS)
if agg.get('n', 0) > 0:
    A('  Aggregate vs floor: mean MFE=%.2f, mean MAE=%.2f, mean (MFE-MAE)=%.2f pips;'
      % (agg['mfe'], agg['mae'], agg['diff']))
    A('  medians: MFE=%.2f, MAE=%.2f, (MFE-MAE)=%.2f pips. The mean/median diff must clear the'
      % (agg['mfe_med'], agg['mae_med'], agg['diff_med']))
    A('  %.1f-pip round-trip floor for any residual edge to survive costs; no net gate exists at Phase 1.'
      % COST_FLOOR_PIPS)
else:
    A('  No signals: no cost-floor comparison possible.')
A('')
A('5. GATES (frozen verbatim in GATE file)')
A('  G-DTE1 total session-valid signals >= %d: %d -> %s' % (G1_MIN, len(sig_df), g1))
A('  G-DTE2 excess-excursion dominance at ALL of 0.5R/1.0R/1.5R (strict, pips):')
for k in THRESH_R:
    A('    %.1fR: mean(max(MFE-%.1fR,0))=%.3f vs mean(max(MAE-%.1fR,0))=%.3f -> %s'
      % (k, k, agg.get('ex_mfe_%s' % k, float('nan')), k,
         agg.get('ex_mae_%s' % k, float('nan')), g2_checks['%.1fR' % k]))
A('    -> G-DTE2=%s (descriptive only, NOT part of the gate: plain mean MFE %.2f vs MAE %.2f -> %s)'
  % (g2, agg.get('mfe', float('nan')), agg.get('mae', float('nan')), mean_mfe_gt_mae))
A('  G-DTE3 day-block bootstrap P(pooled mean(MFE-MAE)<=0) <= %.2f: %.4f (n_day_blocks=%d) -> %s'
  % (G3_MAX_P, pneg, n_day_blocks, g3))
A('  OVERALL PHASE-1: %s' % verdict)
A('')
A('6. NOTES AND CAVEATS')
A('  - Excursions are pre-cost (gross); the %.1f-pip spread floor is descriptive context only.'
  % COST_FLOOR_PIPS)
A('  - No discretionary liquidity-target trendlines or oscillator confluence modeled')
A('    (omission pre-declared, conservative for the method).')
A('  - M15 context is causal: state as of the last bucket closing at or before the displacement')
A('    bar close; M1/M15 sequencing is printed-bar based; gaps across weekend closures allowed.')
A('  - Null result is final for this artifact; a PASS would only justify a separate Phase-2 SPEC.')
A('  - EURUSD is a new domain per tasking; this is NOT a rescue attempt for the #32 dte_method')
A('    or fvg_continuation nulls on US500-tick (frozen verdicts stand).')

report_txt = '\n'.join(L) + '\n'
with open(os.path.join(STUDY, 'REPORT_DTE_EURUSD_PHASE1.txt'), 'w', encoding='utf-8') as f:
    f.write(report_txt)

meta = dict(
    study='dte_method_eurusd', hypothesis='#34', phase=1, verdict=verdict,
    generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    elapsed_s=round(time.time() - T0, 1),
    python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
    seed=SEED, boot_iters=BOOT_ITERS, inv_window=INV_WINDOW, thresholds_r=THRESH_R,
    sess_lo=SESS_LO, sess_hi=SESS_HI, walk_last_min=WALK_LAST_MIN,
    pip=PIP, cost_floor_pips=COST_FLOOR_PIPS, g1_min=G1_MIN, g3_max_p=G3_MAX_P,
    funnel={k: (int(v) if isinstance(v, (int, np.integer))
                else float(v) if isinstance(v, (float, np.floating)) else v)
            for k, v in funnel.items()},
    agg={k: v for k, v in agg.items()},
    pneg=pneg, gates=dict(g_dte1=bool(g1), g_dte2=bool(g2), g_dte3=bool(g3)),
    by_year=by_year,
)
with open(os.path.join(STUDY, 'run_meta.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=1)

p('')
p('================ PHASE 1 SUMMARY (Hypothesis #34, dte_method_eurusd) ================')
p(report_txt)
p('Artifacts: %s' % STUDY)
