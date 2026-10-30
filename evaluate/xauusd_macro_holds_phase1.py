# -*- coding: utf-8 -*-
"""
Hypothesis #30 - XAUUSD Macro-Conditioned Holds - Phase 1 signal quality / strategy screen.

SPEC + GATE were frozen at data/results/xauusd_macro_holds/ BEFORE this file existed.
Run with: C:\\Users\\Milad\\miniconda3\\envs\\reinforce_rl\\python.exe evaluate/xauusd_macro_holds_phase1.py
"""
import io
import json
import math
import os
import platform
import time
import urllib.request

import numpy as np
import pandas as pd
import pytz

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
STUDY = os.path.join(ROOT, 'data', 'results', 'xauusd_macro_holds')
M5_CSV = os.path.join(ROOT, 'data', 'results', 'xauusd_onboarding', 'xauusdtick_m5.csv')

FRED_IDS = ['DFII10', 'DTWEXBGS']
FRED_COSD = '2015-01-01'
LOOKBACK = 21
HORIZONS = [5, 21]
SPREAD_RT = 0.41          # $/oz round trip, onboarding full-tape real median
SWAP_ANNUAL = 0.035       # institutional contango carry proxy, per annum on entry price
TDAYS = 252
STALENESS_CAP_DAYS = 10   # calendar days; older joined macro obs -> feature unavailable
SEED = 20260921
BOOT_ITERS = 10000
G1_MIN_SIGNALS = 150
G2_THRESHOLD = 0.82
G3_THRESHOLD = 0.20

NY = pytz.timezone('America/New_York')
T0 = time.time()


def p(msg):
    print(msg, flush=True)


def stats(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return dict(n=0, mean=float('nan'), med=float('nan'), std=float('nan'),
                    t=float('nan'), win=float('nan'), total=0.0)
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1)) if n > 1 else float('nan')
    t = m / (s / math.sqrt(n)) if (n > 1 and s > 0) else float('nan')
    return dict(n=n, mean=m, med=float(np.median(x)), std=s, t=t,
                win=float(np.mean(x > 0)), total=float(np.sum(x)))


def boot_pneg(x, block):
    """Circular moving-block bootstrap P(mean <= 0). Pre-declared in GATE."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return float('nan')
    rng = np.random.default_rng(SEED)
    b = int(min(max(1, block), n))
    nb = int(math.ceil(n / b))
    starts = rng.integers(0, n, size=(BOOT_ITERS, nb))
    offs = np.arange(b)
    idx = (starts[..., None] + offs) % n
    means = x[idx].mean(axis=(1, 2))
    return float(np.mean(means <= 0))


H10_BROAD_GUID = '122e3bcb627e8e53f1bf72a1a09cfb81'  # H10/H10/JRXWTFB_N.B Nominal Broad Dollar Index (daily)


def _http_get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (research phase1 script)'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode('utf-8-sig')


def _fetch_primary_source(sid):
    """Network-forced fallback to the PRIMARY source of the same series (fred.stlouisfed.org
    is reset by this network and no FRED API key is configured; decided BEFORE any results
    existed - delivery path only, series identity unchanged):
      DFII10    <- Treasury daily real yield curve '10 Yr' (the series FRED republishes as DFII10)
      DTWEXBGS  <- Fed H.10 JRXWTFB_N.B nominal broad dollar index (FRED republishes it as DTWEXBGS;
                   base normalization may differ by a positive constant - sign(Delta_21) invariant)
    Returns (date,value) dataframe + provenance dict."""
    import csv as _csv
    if sid == 'DFII10':
        rows, urls = [], []
        for year in range(2015, 2026 + 1):
            u = ('https://home.treasury.gov/resource-center/data-chart-center/interest-rates/'
                 'daily-treasury-rates.csv/%d/all?type=daily_treasury_real_yield_curve'
                 '&field_tdr_date_value=%d&page&_format=csv' % (year, year))
            urls.append(u)
            txt = _http_get(u)
            rd = _csv.DictReader(io.StringIO(txt))
            for r in rd:
                v = r.get('10 YR', r.get('10 Yr', ''))
                rows.append((r.get('Date', ''), v.strip()))
        df = pd.DataFrame(rows, columns=['date', 'value'])
        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
        src = 'US Treasury daily real yield curve (10 Yr) - primary source of DFII10'
        url_out = urls[0] + ' .. ' + urls[-1]
    elif sid == 'DTWEXBGS':
        to_dt = time.strftime('%m/%d/%Y')
        u = ('https://www.federalreserve.gov/datadownload/Output.aspx?rel=H10&series=%s'
             '&lastobs=&from=01/01/2015&to=%s&filetype=csv&label=include&layout=seriescolumn'
             % (H10_BROAD_GUID, to_dt))
        txt = _http_get(u)
        lines = txt.splitlines()
        hdr = next(i for i, ln in enumerate(lines) if ln.startswith('"Time Period"'))
        cols = next(_csv.reader(io.StringIO(lines[hdr])))
        data_col = cols[1]
        rows = []
        for ln in lines[hdr + 1:]:
            if not ln.strip():
                continue
            parts = next(_csv.reader(io.StringIO(ln)))
            rows.append((parts[0], parts[1].strip()))
        df = pd.DataFrame(rows, columns=['date', 'value'])
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
        src = 'Fed H.10 DataDownload JRXWTFB_N.B (nominal broad dollar index) - primary source of DTWEXBGS'
        url_out = u
    else:
        raise ValueError(sid)
    df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    df['value'] = df['value'].str.replace('ND', '', regex=False).str.replace('N/A', '', regex=False)
    return df, dict(source=src, url=url_out)


def fetch_fred(sid):
    """FRED public CSV endpoint (fredgraph.csv, keyless); verbatim cache for provenance.
    Falls back to the primary source when the FRED web host is unreachable."""
    cache = os.path.join(STUDY, 'fred_raw_%s.csv' % sid)
    url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=%s&cosd=%s' % (sid, FRED_COSD)
    provenance = dict(source='FRED fredgraph.csv (keyless public endpoint)', url=url, via='fredgraph')
    if os.path.exists(cache):
        with open(cache, 'r', encoding='utf-8-sig') as f:
            raw = f.read()
    else:
        try:
            raw = _http_get(url)
        except Exception as e:
            p('  [%s] fredgraph unreachable (%s); falling back to PRIMARY source' % (sid, type(e).__name__))
            dfp, prov = _fetch_primary_source(sid)
            provenance = dict(source=prov['source'], url=prov['url'], via='primary_source_fallback')
            with open(cache, 'w', encoding='utf-8') as f:
                f.write('date,value\n')
                f.write(dfp.assign(value=dfp['value'].mask(dfp['value'] == '')).to_csv(index=False, header=False))
            raw = None
    if raw is not None:
        df = pd.read_csv(io.StringIO(raw), dtype=str)
        df = df.iloc[:, :2]
        df.columns = ['date', 'value']
        df['date'] = pd.to_datetime(df['date'])
    else:
        df = pd.read_csv(cache, dtype=str)
        df.columns = ['date', 'value']
        df['date'] = pd.to_datetime(df['date'])
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'].replace('.', np.nan), errors='coerce')
    meta = dict(rows_total=int(len(df)), rows_missing=int(df['value'].isna().sum()),
                first=str(df['date'].min().date()), last=str(df['date'].max().date()),
                url=url, source=provenance['source'], via=provenance['via'])
    df = df.dropna(subset=['value']).reset_index(drop=True)
    df['delta21'] = df['value'] - df['value'].shift(LOOKBACK)
    return df, meta


# ------------------------------------------------------------------ [1] D1 build
p('[1/7] loading M5 tape and building D1 closes at 16:00 ET ...')
m5 = pd.read_csv(M5_CSV)
m5 = m5.sort_values('time').reset_index(drop=True)
ny_open = pd.Series(pd.to_datetime(m5['time'].to_numpy(), unit='s', utc=True)).dt.tz_convert(NY)
m5['day'] = (ny_open + pd.Timedelta(hours=8)).dt.date

grp = m5.groupby('day', sort=True)
d1 = grp.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'),
             close=('close', 'last'), n_bars=('close', 'size')).reset_index()
d1['date'] = pd.to_datetime(d1['day'])
last_open_epoch = grp['time'].max().to_numpy()
last_open_ny = pd.Series(pd.to_datetime(last_open_epoch, unit='s', utc=True)).dt.tz_convert(NY)
last_bar_min = (last_open_ny.dt.hour * 60 + last_open_ny.dt.minute).to_numpy()
n_1555 = int(np.sum(last_bar_min == 15 * 60 + 55))
n_early = int(np.sum(last_bar_min < 15 * 60 + 55))

close = d1['close'].to_numpy(dtype=float)
n_days = len(d1)
close_s = pd.Series(close)
ret21 = (close_s / close_s.shift(LOOKBACK) - 1).to_numpy()

# M5 arrays + last-bar position per day for the excursion walk
day_codes = pd.factorize(m5['day'], sort=True)[0]
is_last = np.append(day_codes[1:] != day_codes[:-1], True)
last_pos = np.where(is_last)[0]
hi = m5['high'].to_numpy(dtype=float)
lo = m5['low'].to_numpy(dtype=float)

p('  M5 bars=%d  D1 days=%d  %s .. %s  (1555 closes=%d, early closes=%d)'
  % (len(m5), n_days, d1['date'].iloc[0].date(), d1['date'].iloc[-1].date(), n_1555, n_early))

# ------------------------------------------------------------------ [2] FRED
p('[2/7] fetching FRED series %s ...' % FRED_IDS)
fred = {}
fred_meta = {}
for sid in FRED_IDS:
    fr, meta = fetch_fred(sid)
    fred[sid] = fr
    fred_meta[sid] = meta
    p('  %s: obs=%d (missing rows dropped=%d) %s .. %s' %
      (sid, len(fr), meta['rows_missing'], meta['first'], meta['last']))

# ------------------------------------------------------------------ [3] strict s->s+1 join
p('[3/7] joining macro with strict s->s+1 publication shift (exact matches disallowed) ...')
gold = d1[['date', 'close']].copy()
join_diag = {}
for sid in FRED_IDS:
    fr = fred[sid][['date', 'value', 'delta21']].rename(
        columns={'date': sid + '_obs_date', 'value': sid, 'delta21': sid + '_d21'})
    gold = pd.merge_asof(gold, fr, left_on='date', right_on=sid + '_obs_date',
                         direction='backward', allow_exact_matches=False)
    stale = (gold['date'] - gold[sid + '_obs_date']).dt.days
    join_diag[sid] = dict(
        stale_med=float(stale.median()), stale_max=float(stale.max()),
        n_beyond_cap=int((stale > STALENESS_CAP_DAYS).sum()),
        joined_last=str(gold[sid + '_obs_date'].dropna().max().date()))
    # staleness cap: older than CAP days -> feature unavailable (no frozen-forever deltas)
    gold.loc[stale > STALENESS_CAP_DAYS, sid + '_d21'] = np.nan
    p('  %s staleness: med=%.0f d max=%.0f d  days-beyond-cap=%d  last-joined-obs=%s'
      % (sid, join_diag[sid]['stale_med'], join_diag[sid]['stale_max'],
         join_diag[sid]['n_beyond_cap'], join_diag[sid]['joined_last']))

dd = gold['DFII10_d21'].to_numpy(dtype=float)
dw = gold['DTWEXBGS_d21'].to_numpy(dtype=float)
rr = ret21
ready = ~(np.isnan(dd) | np.isnan(dw) | np.isnan(rr))
tailwind = ready & (dd < 0) & (dw < 0)
headwind = ready & (dd > 0) & (dw > 0)
long_sig = tailwind & (rr < 0)
short_sig = headwind & (rr > 0)
sig_dir = np.where(long_sig, 1, np.where(short_sig, -1, 0))

# ------------------------------------------------------------------ [4] signals
p('[4/7] forming signals and fixed-horizon holds ...')
fwd_close = {h: close_s.shift(-h).to_numpy() for h in HORIZONS}
idx_sig = np.where(sig_dir != 0)[0]
recs = []
for i in idx_sig:
    d = int(sig_dir[i])
    e = float(close[i])
    carry_day = e * SWAP_ANNUAL / TDAYS
    rec = dict(ny_date=gold['date'].iloc[i].strftime('%Y-%m-%d'), dir=d, entry_close=e,
               dfii10=gold['DFII10'].iloc[i], dtwexbgs=gold['DTWEXBGS'].iloc[i],
               dfii10_d21=float(dd[i]), dtwexbgs_d21=float(dw[i]), ret21=float(rr[i]))
    for h in HORIZONS:
        j = i + h
        if j < n_days:
            x = float(close[j])
            gross = d * (x - e)
            swap = -d * carry_day * h
            net = gross - SPREAD_RT + swap
            eb, xb = int(last_pos[i]), int(last_pos[j])
            if d > 0:
                mfe = float(hi[eb + 1:xb + 1].max() - e)
                mae = float(lo[eb + 1:xb + 1].min() - e)
            else:
                mfe = float(e - lo[eb + 1:xb + 1].min())
                mae = float(e - hi[eb + 1:xb + 1].max())
            rec.update({'exit_h%d' % h: gold['date'].iloc[j].strftime('%Y-%m-%d'),
                        'gross_h%d' % h: gross, 'swap_h%d' % h: swap, 'net_h%d' % h: net,
                        'mfe_h%d' % h: mfe, 'mae_h%d' % h: mae})
        else:
            for k in ('exit', 'gross', 'swap', 'net', 'mfe', 'mae'):
                rec['%s_h%d' % (k, h)] = np.nan
    recs.append(rec)

sig = pd.DataFrame(recs)
sig['year'] = pd.to_datetime(sig['ny_date']).dt.year
sig_csv = os.path.join(STUDY, 'xauusd_macro_signals.csv')
sig.to_csv(sig_csv, index=False)
p('  signals formed=%d (long=%d short=%d) -> %s'
  % (len(sig), int((sig['dir'] > 0).sum()), int((sig['dir'] < 0).sum()), os.path.basename(sig_csv)))

# ------------------------------------------------------------------ [5] economics + bootstrap
p('[5/7] economics, per-year split, block bootstrap ...')
res = {}
for h in HORIZONS:
    g = sig['gross_h%d' % h]
    n_h = sig['net_h%d' % h]
    res[h] = dict(
        gross=stats(g), net=stats(n_h),
        swap=stats(sig['swap_h%d' % h]),
        pneg=boot_pneg(n_h.to_numpy(), h))
    for nm, sub in (('long', sig[sig['dir'] > 0]), ('short', sig[sig['dir'] < 0])):
        res[h][nm] = dict(gross=stats(sub['gross_h%d' % h]), net=stats(sub['net_h%d' % h]))
    yr = {}
    for y, sub in sig.groupby('year'):
        yr[int(y)] = dict(n=len(sub), net_mean=float(np.nanmean(sub['net_h%d' % h])),
                          gross_mean=float(np.nanmean(sub['gross_h%d' % h])))
    res[h]['by_year'] = yr
    era_mask = sig['year'].between(2022, 2024)
    res[h]['era_2022_2024'] = dict(gross=stats(sig.loc[era_mask, 'gross_h%d' % h]),
                                   net=stats(sig.loc[era_mask, 'net_h%d' % h]))
    res[h]['era_rest'] = dict(gross=stats(sig.loc[~era_mask, 'gross_h%d' % h]),
                              net=stats(sig.loc[~era_mask, 'net_h%d' % h]))

# ------------------------------------------------------------------ [6] gates
p('[6/7] evaluating frozen gates ...')
n_exec_h5 = int(sig['net_h5'].notna().sum())
n_exec_h21 = int(sig['net_h21'].notna().sum())
gates = {'G-X1': n_exec_h5 >= G1_MIN_SIGNALS}
for h in HORIZONS:
    gates['G-X2_h%d' % h] = bool(res[h]['gross']['n'] > 0 and res[h]['gross']['mean'] >= G2_THRESHOLD)
    gates['G-X3_h%d' % h] = bool(res[h]['net']['n'] > 0 and res[h]['pneg'] <= G3_THRESHOLD)
qualifying = [h for h in HORIZONS if gates['G-X2_h%d' % h] and gates['G-X3_h%d' % h]]
overall = bool(gates['G-X1'] and qualifying)
verdict = 'PASS' if overall else 'FAIL'

# ------------------------------------------------------------------ [7] report
p('[7/7] writing REPORT ...')
L = []
A = L.append
A('REPORT: XAUUSD MACRO-CONDITIONED HOLDS - PHASE 1 (HYPOTHESIS #30)')
A('Generated: %s  |  script: evaluate/xauusd_macro_holds_phase1.py' % time.strftime('%Y-%m-%d %H:%M:%S'))
A('SPEC/GATE: data/results/xauusd_macro_holds/{SPEC,GATE}_XAUUSD_MACRO_HOLDS.txt (frozen before code)')
A('Mechanism: Erb-Harvey 2013 dual-driver conditioning (real yield + broad dollar) with anti-momentum entry.')
A('')
A('VERDICT: %s   (G-X1=%s%s)' % (verdict, gates['G-X1'],
  ''.join(', G-X2(h%d)=%s/G-X3(h%d)=%s' % (h, gates['G-X2_h%d' % h], h, gates['G-X3_h%d' % h]) for h in HORIZONS)))
A('')
A('1. DATA AND INTEGRITY (non-gating)')
A('  Price: %s (%d M5 bars -> %d D1 days %s .. %s; 1555-close days=%d, early-close days=%d)'
  % (os.path.relpath(M5_CSV, ROOT), len(m5), n_days, d1['date'].iloc[0].date(),
     d1['date'].iloc[-1].date(), n_1555, n_early))
for sid in FRED_IDS:
    m = fred_meta[sid]
    jd = join_diag[sid]
    A('  FRED %s: %d observations (%s .. %s, %d missing rows dropped)' % (sid, len(fred[sid]), m['first'], m['last'], m['rows_missing']))
    A('    delivered via: %s | %s' % (m['via'], m['source']))
    A('    s->s+1 join: staleness median %.0f cal-days, max %.0f; days beyond %d-day cap=%d; last joined obs=%s'
      % (jd['stale_med'], jd['stale_max'], STALENESS_CAP_DAYS, jd['n_beyond_cap'], jd['joined_last']))
A('')
A('2. FUNNEL')
A('  D1 days: %d' % n_days)
A('  Feature-ready days (both Delta_21 live post s->s+1 shift, Return_21 live): %d' % int(ready.sum()))
A('  Macro tailwind days (both Delta_21 < 0): %d | headwind days (both > 0): %d | conflicted: %d'
  % (int(tailwind.sum()), int(headwind.sum()), int((ready & ~tailwind & ~headwind).sum())))
A('  Anti-momentum pass: tailwind & Ret21<0 -> LONG signals: %d | headwind & Ret21>0 -> SHORT signals: %d'
  % (int(long_sig.sum()), int(short_sig.sum())))
A('  Executed (complete window): h=5: %d | h=21: %d | truncated (no h=21 window): %d'
  % (n_exec_h5, n_exec_h21, len(sig) - n_exec_h21))
A('')
A('3. ECONOMICS ($/oz per 1-oz notional; spread $0.41 round trip; swap = +/-3.50%% p.a. of entry / 252 per trading day held)')
for h in HORIZONS:
    r = res[h]
    A('  -- horizon h=%d (%s) --' % (h, 'one week' if h == 5 else 'one month'))
    A('    gross: n=%d mean=%.3f med=%.3f std=%.3f t=%.2f win=%.1f%% total=%.1f'
      % (r['gross']['n'], r['gross']['mean'], r['gross']['med'], r['gross']['std'],
         r['gross']['t'], 100 * r['gross']['win'], r['gross']['total']))
    A('    swap:  mean=%.3f total=%.1f (longs pay, shorts receive)'
      % (r['swap']['mean'], r['swap']['total']))
    A('    net:   n=%d mean=%.3f med=%.3f std=%.3f t=%.2f win=%.1f%% total=%.1f'
      % (r['net']['n'], r['net']['mean'], r['net']['med'], r['net']['std'],
         r['net']['t'], 100 * r['net']['win'], r['net']['total']))
    A('    M5 excursion walk: mean MFE=%.2f  mean MAE=%.2f ($/oz, descriptive)'
      % (float(np.nanmean(sig['mfe_h%d' % h])), float(np.nanmean(sig['mae_h%d' % h]))))
    A('    long: n=%d gross_mean=%.3f net_mean=%.3f | short: n=%d gross_mean=%.3f net_mean=%.3f'
      % (r['long']['gross']['n'], r['long']['gross']['mean'], r['long']['net']['mean'],
         r['short']['gross']['n'], r['short']['gross']['mean'], r['short']['net']['mean']))
A('')
A('4. YEAR-BY-YEAR AND THE 2022-2024 CENTRAL-BANK-FLOW ERA (descriptive, non-gating)')
for h in HORIZONS:
    r = res[h]
    A('  -- h=%d --' % h)
    for y in sorted(r['by_year']):
        yd = r['by_year'][y]
        tag = '  <-- CB-flow era' if y in (2022, 2023, 2024) else ''
        A('    %d: n=%d gross_mean=%.3f net_mean=%.3f%s'
          % (y, yd['n'], yd['gross_mean'], yd['net_mean'], tag))
    A('    era 2022-2024 combined: n=%d gross_mean=%.3f net_mean=%.3f (t=%.2f)'
      % (r['era_2022_2024']['gross']['n'], r['era_2022_2024']['gross']['mean'],
         r['era_2022_2024']['net']['mean'], r['era_2022_2024']['net']['t']))
    A('    all other years:        n=%d gross_mean=%.3f net_mean=%.3f (t=%.2f)'
      % (r['era_rest']['gross']['n'], r['era_rest']['gross']['mean'],
         r['era_rest']['net']['mean'], r['era_rest']['net']['t']))
A('')
A('5. GATES (frozen verbatim in GATE file)')
A('  G-X1 total executed signals >= %d: %d -> %s' % (G1_MIN_SIGNALS, n_exec_h5, gates['G-X1']))
for h in HORIZONS:
    A('  G-X2 gross expectancy h=%d >= %.2f: %.4f -> %s'
      % (h, G2_THRESHOLD, res[h]['gross']['mean'], gates['G-X2_h%d' % h]))
    A('  G-X3 bootstrap P(mean net<=0) h=%d <= %.2f: %.4f -> %s'
      % (h, G3_THRESHOLD, res[h]['pneg'], gates['G-X3_h%d' % h]))
A('  OVERALL PHASE-1: %s%s' % (verdict, (' (qualifying horizon(s): %s)' % qualifying) if qualifying else ''))
A('')
A('6. NOTES AND CAVEATS')
A('  - Publication shift s->s+1 applied to BOTH FRED series (strict backward as-of join, exact match disallowed).')
A('  - DTWEXBGS is itself published with a lag; s->s+1 is a lower bound on its true availability (known optimism on the dollar leg).')
A('  - Swap proxy is the institutional contango convention (long debit / short credit) at a flat 3.50%% p.a.; no Wednesday-x3.')
A('  - Signals near the tape end without a full window are excluded from that horizon (funnel line 2).')
A('  - MAE can be positive if the tape never traded adversely after the entry bar (entry-bar extremes excluded).')
A('  - Null result is final for this artifact; Phase 2 only on PASS.')

report_txt = '\n'.join(L) + '\n'
with open(os.path.join(STUDY, 'REPORT_XAUUSD_MACRO_HOLDS.txt'), 'w', encoding='utf-8') as f:
    f.write(report_txt)

meta = dict(
    study='xauusd_macro_holds', hypothesis='#30', phase=1, verdict=verdict,
    generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    elapsed_s=round(time.time() - T0, 1),
    python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
    seed=SEED, boot_iters=BOOT_ITERS, spread_rt=SPREAD_RT, swap_annual=SWAP_ANNUAL,
    horizons=HORIZONS, lookback=LOOKBACK, staleness_cap_days=STALENESS_CAP_DAYS,
    d1_days=n_days, signals=len(sig), executed_h5=n_exec_h5, executed_h21=n_exec_h21,
    fred=fred_meta, gates={k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in gates.items()},
    pneg_h5=res[5]['pneg'], pneg_h21=res[21]['pneg'],
    gross_mean_h5=res[5]['gross']['mean'], gross_mean_h21=res[21]['gross']['mean'],
    net_mean_h5=res[5]['net']['mean'], net_mean_h21=res[21]['net']['mean'],
)
with open(os.path.join(STUDY, 'run_meta.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=1)

p('')
p('================ PHASE 1 SUMMARY (Hypothesis #30, xauusd_macro_holds) ================')
p(report_txt)
p('Artifacts: %s' % STUDY)
