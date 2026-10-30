# -*- coding: utf-8 -*-
"""
Hypothesis #35 - VWAP Mean Reversion - EURUSD Phase 1 signal-quality screen
pre-cost MFE/MAE excursion screen ONLY: no trade rule beyond first-touch
excursions, no EA, no tester, no Phase 2 implied by any outcome.

Frozen contract this script executes (where any code choice is ambiguous,
the SPEC text wins):
  SPEC: data/results/vwap_mr_eurusd/SPEC_VWAP_MR_EURUSD.txt
  GATE: data/results/vwap_mr_eurusd/GATE_VWAP_MR_EURUSD.txt
(both frozen BEFORE this file existed)

Run with:
  C:\\Users\\Milad\\miniconda3\\envs\\reinforce_rl\\python.exe evaluate/vwap_mr_eurusd_phase1.py

Template lineage: evaluate/dte_method_eurusd_phase1.py (Hypothesis #34; same
EURUSD M1 tape, pips convention, excursion / day-block-bootstrap / report
skeleton). US500-tick VWAP-MR nulls (vwap_rr_verdict) are frozen lineage
context, NOT re-litigated here: EURUSD is a new instrument with a new
08:00 ET session anchor, FX-scaled percentage bands and a new cost floor.
"""
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import pytz

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
STUDY = os.path.join(ROOT, 'data', 'results', 'vwap_mr_eurusd')
os.makedirs(STUDY, exist_ok=True)

M1_CSV = os.path.join(ROOT, 'data', 'results', 'eurusd_onboarding', 'eurusdtick_m1.csv')

NY = pytz.timezone('America/New_York')   # DST-aware, computed from epoch UTC
SESS_LO = 480          # 08:00 ET session open wall (inclusive)
SESS_HI = 960          # 16:00 ET session end wall (exclusive)
SIG_LO = 510           # signal window: open wall strictly > 08:30 ET
SIG_HI = 930           # signal window: open wall strictly < 15:30 ET
SLOPE_S = 1800         # 30-minute VWAP slope reference span (seconds)
BAND = 0.0025          # +/-0.25% deviation bands around session VWAP
MAXMOVE = 0.0040       # session range strictly > 0.40% of session open invalidates
PIP = 1e-4             # 1 pip = 0.00010 (5-decimal EURUSD)
COST_FLOOR_PIPS = 0.3  # verified London/NY overlap median spread (eurusd_onboarding 2026-09-22)
SEED = 20260921
BOOT_ITERS = 10000
G1_MIN = 150
G3_MAX_P = 0.20
T0 = time.time()


def p(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ [1/6] load tape
p('[1/6] loading M1 tape ...')
m1 = pd.read_csv(M1_CSV)
m1 = m1.sort_values('time').reset_index(drop=True)
t = m1['time'].to_numpy(np.int64)
o = m1['open'].to_numpy(np.float64)
h = m1['high'].to_numpy(np.float64)
lo = m1['low'].to_numpy(np.float64)
c = m1['close'].to_numpy(np.float64)
v = m1['tick_volume'].to_numpy(np.float64)
n = len(m1)

ny = pd.Series(pd.to_datetime(t, unit='s', utc=True)).dt.tz_convert(NY)
minute = (ny.dt.hour * 60 + ny.dt.minute).to_numpy(np.int32)
ny_date = ny.dt.date
dcode = pd.factorize(ny_date)[0]
setup_ok = (minute >= SESS_LO) & (minute < SESS_HI)

funnel = {}
funnel['m1_bars'] = n
funnel['m1_session_bars'] = int(setup_ok.sum())
p('  M1 bars=%d (%s .. %s) | session [08:00,16:00) ET bars=%d'
  % (n, ny.iloc[0], ny.iloc[-1], funnel['m1_session_bars']))

# ------------------------------------------------------------------ [2/6] per-NY-date session state machine + first-touch walks
p('[2/6] per-NY-date session VWAP state machine (signals + first-touch walks) ...')
ndates = int(dcode[-1]) + 1
day_lo = np.searchsorted(dcode, np.arange(ndates), side='left')
day_hi = np.searchsorted(dcode, np.arange(ndates), side='right')

funnel['days_total'] = ndates
funnel['days_with_session'] = 0
funnel['days_maxmove_ever'] = 0
funnel['signal_candidates_short'] = 0
funnel['signal_candidates_long'] = 0
funnel['fired_short'] = 0
funnel['fired_long'] = 0
funnel['drop_no_entry'] = 0

rows = []
for d in range(ndates):
    seg = slice(int(day_lo[d]), int(day_hi[d]))
    if seg.stop - seg.start <= 0:
        continue
    sess = np.where((minute[seg] >= SESS_LO) & (minute[seg] < SESS_HI))[0] + seg.start
    if len(sess) < 2:
        continue
    funnel['days_with_session'] += 1

    t_s = t[sess]
    o_s = o[sess]
    h_s = h[sess]
    l_s = lo[sess]
    c_s = c[sess]
    v_s = v[sess]
    w_s = minute[sess]

    tp = (h_s + l_s + c_s) / 3.0
    cvv = np.cumsum(v_s)                        # tick_volume >= 1 -> always > 0; guard anyway
    vwap = np.cumsum(tp * v_s) / np.maximum(cvv, 1e-12)
    band_up = vwap * (1.0 + BAND)
    band_lo = vwap * (1.0 - BAND)

    rmax = np.maximum.accumulate(h_s)
    rmin = np.minimum.accumulate(l_s)
    open_px = float(o_s[0])                     # open of the session's first printed bar
    range_ok = (rmax - rmin) <= MAXMOVE * open_px   # inclusive, sequential trending-day filter
    if not range_ok.all():
        funnel['days_maxmove_ever'] += 1

    # slope reference: LAST session bar with open time <= t_open - 1800 s
    ref = np.searchsorted(t_s, t_s - SLOPE_S, side='right') - 1
    slope_ok = ref >= 0
    slope = np.where(slope_ok, vwap - vwap[np.where(slope_ok, ref, 0)], 0.0)

    win = (w_s > SIG_LO) & (w_s < SIG_HI) & slope_ok & range_ok
    short_m = win & (c_s > band_up) & (slope <= 0)
    long_m = win & (c_s < band_lo) & (slope >= 0)
    funnel['signal_candidates_short'] += int(short_m.sum())
    funnel['signal_candidates_long'] += int(long_m.sum())

    for mask, dname, dsign in ((short_m, 'SHORT', -1), (long_m, 'LONG', +1)):
        if not mask.any():
            continue
        if dsign < 0:
            funnel['fired_short'] += 1
        else:
            funnel['fired_long'] += 1
        jp = int(np.argmax(mask))               # first-touch cap: first True per direction per day
        jg = int(sess[jp])                      # global signal bar index
        if jg + 1 >= n or dcode[jg + 1] != d or minute[jg + 1] >= SESS_HI:
            funnel['drop_no_entry'] += 1
            continue
        ep = jp + 1
        assert int(sess[ep]) == jg + 1          # entry bar is the next session bar by construction
        eg = int(sess[ep])
        entry = float(o[eg])

        # walk over the rest of the session slice; VWAP keeps evolving (never a frozen snapshot)
        if dsign < 0:
            touch = l_s[ep:] <= vwap[ep:]
        else:
            touch = h_s[ep:] >= vwap[ep:]
        hit = bool(touch.any())
        kep = ep + (int(np.argmax(touch)) if hit else len(touch) - 1)
        exit_reason = 'TARGET' if hit else 'TIME'
        hh = h_s[ep:kep + 1]
        ll = l_s[ep:kep + 1]
        if dsign < 0:
            mfe = max(0.0, entry - float(ll.min())) / PIP
            mae = max(0.0, float(hh.max()) - entry) / PIP
        else:
            mfe = max(0.0, float(hh.max()) - entry) / PIP
            mae = max(0.0, entry - float(ll.min())) / PIP
        rows.append(dict(
            ny_date=str(ny_date.iloc[jg]),
            dir=dname,
            signal_time=ny.iloc[jg].strftime('%Y-%m-%d %H:%M'),
            entry_time=ny.iloc[eg].strftime('%Y-%m-%d %H:%M'),
            entry=entry,
            vwap_at_signal=float(vwap[jp]),
            slope_pips=float(slope[jp]) / PIP,
            range_pct_at_signal=float(rmax[jp] - rmin[jp]) / open_px,
            exit_reason=exit_reason,
            walk_bars=kep + 1 - ep,
            mfe_pips=mfe,
            mae_pips=mae,
            touch_bar_time=ny.iloc[int(sess[kep])].strftime('%Y-%m-%d %H:%M'),
        ))

funnel['final_signals'] = len(rows)
p('  days: total=%d | with session (>=2 session bars)=%d | maxmove-ever-triggered=%d'
  % (funnel['days_total'], funnel['days_with_session'], funnel['days_maxmove_ever']))
p('  signal candidates pre-cap: short=%d long=%d | fired (first per direction per day): short=%d long=%d'
  % (funnel['signal_candidates_short'], funnel['signal_candidates_long'],
     funnel['fired_short'], funnel['fired_long']))
p('  no-entry drops (next bar different NY date or wall >= 16:00)=%d | FINAL signals=%d'
  % (funnel['drop_no_entry'], funnel['final_signals']))

# ------------------------------------------------------------------ [3/6] signal table + CSV
p('[3/6] building signal table + vwap_mr_eurusd_signals.csv ...')
CSV_COLS = ['ny_date', 'dir', 'signal_time', 'entry_time', 'entry', 'vwap_at_signal',
            'slope_pips', 'range_pct_at_signal', 'exit_reason', 'walk_bars',
            'mfe_pips', 'mae_pips', 'diff_pips']
if rows:
    sig_df = pd.DataFrame(rows)
    sig_df['diff_pips'] = sig_df['mfe_pips'] - sig_df['mae_pips']
else:
    sig_df = pd.DataFrame(columns=CSV_COLS)
sig_csv = os.path.join(STUDY, 'vwap_mr_eurusd_signals.csv')
sig_df[CSV_COLS].to_csv(sig_csv, index=False)
p('  wrote %d signal rows -> %s' % (len(sig_df), os.path.relpath(sig_csv, ROOT)))

# ------------------------------------------------------------------ [4/6] excursion stats (pips, pre-cost)
p('[4/6] excursion stats (pips, pre-cost) ...')


def excursion_stats(sub):
    if len(sub) == 0:
        return dict(n=0)
    mfe = sub['mfe_pips'].to_numpy(float)
    mae = sub['mae_pips'].to_numpy(float)
    return dict(
        n=len(sub),
        mfe=float(mfe.mean()), mae=float(mae.mean()), diff=float((mfe - mae).mean()),
        mfe_med=float(np.median(mfe)), mae_med=float(np.median(mae)), diff_med=float(np.median(mfe - mae)),
        fav_share=float((mfe > mae).mean()),
    )


agg = excursion_stats(sig_df)
if len(sig_df) > 0:
    st_long = excursion_stats(sig_df[sig_df['dir'] == 'LONG'])
    st_short = excursion_stats(sig_df[sig_df['dir'] == 'SHORT'])
    sig_df['year'] = pd.to_datetime(sig_df['ny_date']).dt.year
    by_year = {int(y): int(len(s)) for y, s in sig_df.groupby('year')}
    by_dir = {k: int(v) for k, v in sig_df['dir'].value_counts().items()}
    vc = sig_df['exit_reason'].value_counts()
    exit_mix = {k: int(vc.get(k, 0)) for k in ('TARGET', 'TIME')}
    exit_mix_dir = {}
    for dname in ('LONG', 'SHORT'):
        vcd = sig_df[sig_df['dir'] == dname]['exit_reason'].value_counts()
        exit_mix_dir[dname] = {k: int(vcd.get(k, 0)) for k in ('TARGET', 'TIME')}
else:
    st_long = dict(n=0)
    st_short = dict(n=0)
    by_year = {}
    by_dir = {'LONG': 0, 'SHORT': 0}
    exit_mix = {'TARGET': 0, 'TIME': 0}
    exit_mix_dir = {dn: {'TARGET': 0, 'TIME': 0} for dn in ('LONG', 'SHORT')}

# ------------------------------------------------------------------ [5/6] day-block bootstrap
p('[5/6] day-block bootstrap (%d resamples) ...' % BOOT_ITERS)
if len(sig_df) > 0:
    rng = np.random.default_rng(SEED)
    diff = sig_df['mfe_pips'] - sig_df['mae_pips']
    sums = diff.groupby(sig_df['ny_date']).sum().to_numpy(float)
    cnts = sig_df.groupby('ny_date').size().to_numpy(float)
    n_day_blocks = len(sums)
    idx = rng.integers(0, n_day_blocks, size=(BOOT_ITERS, n_day_blocks))
    means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    pneg = float((means <= 0).mean())
else:
    pneg = 1.0
    n_day_blocks = 0
p('  P(pooled mean(MFE-MAE) <= 0) = %.4f over %d day blocks' % (pneg, n_day_blocks))

# ------------------------------------------------------------------ [6/6] gates, report, run_meta
p('[6/6] gates, report, run_meta ...')
g1 = bool(len(sig_df) >= G1_MIN)
g2 = bool(agg.get('n', 0) > 0 and agg['mfe'] > agg['mae'])
g3 = bool(pneg <= G3_MAX_P)
overall = bool(g1 and g2 and g3)
verdict = 'PASS' if overall else 'FAIL'

L = []
A = L.append
A('REPORT: VWAP MEAN REVERSION - EURUSD PHASE 1 SIGNAL-QUALITY SCREEN (HYPOTHESIS #35)')
A('Generated: %s  |  script: evaluate/vwap_mr_eurusd_phase1.py' % time.strftime('%Y-%m-%d %H:%M:%S'))
A('SPEC/GATE: data/results/vwap_mr_eurusd/{SPEC,GATE}_VWAP_MR_EURUSD.txt (frozen before code)')
A('Mechanism: EURUSD session VWAP mean reversion - 08:00 ET session anchor, TP-volume VWAP,')
A('+/-0.25% deviation bands, 30-minute VWAP slope filter, 0.40% max-move trending-day filter,')
A('first-touch-per-direction excursion walk to the evolving VWAP or 16:00 ET (pre-cost).')
A('Phase-1 signal-quality screen ONLY (no trade rule, no EA, no tester, no Phase 2 implied).')
A('')
A('VERDICT: %s   (G-VW1=%s, G-VW2=%s, G-VW3=%s)' % (verdict, g1, g2, g3))
A('')
A('1. DATA AND INTEGRITY')
A('  Tape: %s (%d M1 bars, %s .. %s NY; Bid, 5-decimal)'
  % (os.path.relpath(M1_CSV, ROOT), n, ny.iloc[0], ny.iloc[-1]))
A('  Session = NY-date M1 bars with open wall in [08:00,16:00) ET: %d bars; VWAP/range reset daily.'
  % funnel['m1_session_bars'])
A('  All state causal: VWAP, bands, rolling range and slope at bar t use session bars up to t only.')
A('  All excursion statistics in pips (1 pip = 0.00010).')
A('')
A('2. FUNNEL')
A('  NY dates: total=%d | with session (>=2 session bars)=%d | maxmove-ever-triggered=%d'
  % (funnel['days_total'], funnel['days_with_session'], funnel['days_maxmove_ever']))
A('  signal candidates pre-cap (510<wall<930, slope defined, range ok, band+direction): short=%d long=%d'
  % (funnel['signal_candidates_short'], funnel['signal_candidates_long']))
A('  fired after first-touch-per-direction-per-day cap: short=%d long=%d'
  % (funnel['fired_short'], funnel['fired_long']))
A('  no-entry drops (next printed bar different NY date or open wall >= 16:00): %d'
  % funnel['drop_no_entry'])
A('  FINAL excursion-qualified signals: %d | by dir: %s | by year: %s'
  % (len(sig_df), by_dir, ', '.join('%d:%d' % kv for kv in sorted(by_year.items()))))
A('')
A('3. EXCURSION STATS (pips, pre-cost)')
for nm, st in (('AGGREGATE', agg), ('LONG', st_long), ('SHORT', st_short)):
    if st['n'] == 0:
        A('  %s: n=0' % nm)
        continue
    A('  %s: n=%d MFE=%.2f MAE=%.2f diff=%.2f pips | medians MFE=%.2f MAE=%.2f diff=%.2f | MFE>MAE share=%.1f%%'
      % (nm, st['n'], st['mfe'], st['mae'], st['diff'], st['mfe_med'], st['mae_med'], st['diff_med'],
         100 * st['fav_share']))
A('  Exit mix: TARGET=%d TIME=%d | LONG: TARGET=%d TIME=%d | SHORT: TARGET=%d TIME=%d'
  % (exit_mix['TARGET'], exit_mix['TIME'],
     exit_mix_dir['LONG']['TARGET'], exit_mix_dir['LONG']['TIME'],
     exit_mix_dir['SHORT']['TARGET'], exit_mix_dir['SHORT']['TIME']))
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
A('  G-VW1 total session-valid signals >= %d: %d -> %s' % (G1_MIN, len(sig_df), g1))
A('  G-VW2 aggregate mean MFE > mean MAE (strict, pips): %.4f vs %.4f -> %s'
  % (agg.get('mfe', float('nan')), agg.get('mae', float('nan')), g2))
A('  G-VW3 day-block bootstrap P(pooled mean(MFE-MAE)<=0) <= %.2f: %.4f (n_day_blocks=%d) -> %s'
  % (G3_MAX_P, pneg, n_day_blocks, g3))
A('  OVERALL PHASE-1: %s' % verdict)
A('')
A('6. NOTES AND CAVEATS')
A('  - Excursions are pre-cost (gross); the %.1f-pip spread floor is descriptive context only.'
  % COST_FLOOR_PIPS)
A('  - US500-tick VWAP-MR lineage is a frozen null (vwap_rr_verdict: 273 trades, expectancy')
A('    -$0.075, Sharpe -0.048); EURUSD is a new instrument/anchor per SPEC tasking - this is')
A('    NOT a re-litigation of the US500 verdicts.')
A('  - No parameter search: band width, slope span, filters, caps, seed frozen in the SPEC;')
A('    nothing tuned after seeing results.')
A('  - Null result is final for this artifact; no re-run without a new pre-declared SPEC.')
A('  - Phase-1 excursions are signal-quality measurements, not tradable results (no')
A('    spread/swap/slippage modeling beyond the descriptive floor; no walk-forward claim).')

report_txt = '\n'.join(L) + '\n'
with open(os.path.join(STUDY, 'REPORT_VWAP_MR_EURUSD_PHASE1.txt'), 'w', encoding='utf-8') as f:
    f.write(report_txt)

meta = dict(
    study='vwap_mr_eurusd', hypothesis='#35', phase=1, verdict=verdict,
    generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    elapsed_s=round(time.time() - T0, 1),
    python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
    seed=SEED, boot_iters=BOOT_ITERS,
    sess_lo=SESS_LO, sess_hi=SESS_HI, sig_lo=SIG_LO, sig_hi=SIG_HI,
    slope_s=SLOPE_S, band=BAND, maxmove=MAXMOVE,
    pip=PIP, cost_floor_pips=COST_FLOOR_PIPS, g1_min=G1_MIN, g3_max_p=G3_MAX_P,
    funnel={k: (int(v) if isinstance(v, (int, np.integer))
                else float(v) if isinstance(v, (float, np.floating)) else v)
            for k, v in funnel.items()},
    agg={k: v for k, v in agg.items()},
    stats_long={k: v for k, v in st_long.items()},
    stats_short={k: v for k, v in st_short.items()},
    pneg=pneg, n_day_blocks=n_day_blocks,
    gates=dict(g_vw1=bool(g1), g_vw2=bool(g2), g_vw3=bool(g3)),
    exit_mix=exit_mix, exit_mix_dir=exit_mix_dir, by_dir=by_dir, by_year=by_year,
)
with open(os.path.join(STUDY, 'run_meta.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=1)

p('')
p('============ PHASE 1 SUMMARY (Hypothesis #35, vwap_mr_eurusd) ============')
p(report_txt)
p('Artifacts: %s' % STUDY)
