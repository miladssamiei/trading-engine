"""Frozen five-card XGBoost meta-label test on gold A/B raw signal candidates."""
from __future__ import annotations

import heapq
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, __version__ as xgboost_version

from futures_prop_validation_phase1 import engine_a, engine_b, grids, sha
from gold_ensemble_real_cost import mbb_p, replay as baseline_replay
from unseen_asset_saleh_donchian_phase1 import wilder_adx, wilder_atr

ROOT = Path(__file__).resolve().parents[1]
EARLY = Path(r'C:\Users\Milad\download\xauusd-m1-bid-2010-01-01-2019-12-31.csv')
LATE = ROOT/'data/results/xauusd_onboarding/xauusdtick_m1.csv'
OUT = ROOT/'data/results/gold_xgboost_metalabel'
COST = .585
NS_H = 3_600_000_000_000
NS_DAY = 24*NS_H
O17 = 17*NS_H
FEATURES = ['atr_ratio', 'adx_1h', 'candle_size', 'macro_pullback', 'recent_win_rate']


def load_extended():
    if not EARLY.exists() or not LATE.exists():
        raise FileNotFoundError('Both owner 2010-2019 and verified 2020-2026 tapes are required')
    early = pd.read_csv(EARLY, usecols=['timestamp', 'open', 'high', 'low', 'close'])
    stamp = early.timestamp.to_numpy(np.int64)
    if len(stamp) < 4_000_000 or not np.all(stamp[1:] > stamp[:-1]) or not np.all(stamp % 60000 == 0):
        raise AssertionError('Early M1 timestamps are sparse, duplicate or off-grid')
    t0 = stamp//1000
    o0, h0, l0, c0 = (early[k].to_numpy(float) for k in ('open', 'high', 'low', 'close'))
    carry = ((o0 == h0) & (h0 == l0) & (l0 == c0) &
             np.r_[False, c0[1:] == c0[:-1]])
    keep = ~carry
    m0 = {'time': t0[keep], 'open': o0[keep], 'high': h0[keep],
          'low': l0[keep], 'close': c0[keep]}
    early_count = len(early)
    del early, stamp, t0, o0, h0, l0, c0, carry, keep
    late = pd.read_csv(LATE, usecols=['time', 'open', 'high', 'low', 'close', 'tick_volume'])
    if not (late.tick_volume.to_numpy() > 0).all():
        raise AssertionError('Verified later M1 tape has a nonpositive tick-volume row')
    m1 = {k: late[k].to_numpy(np.int64 if k == 'time' else float)
          for k in ('time', 'open', 'high', 'low', 'close')}
    late_count = len(late)
    del late
    m = {k: np.concatenate((m0[k], m1[k]))
         for k in ('time', 'open', 'high', 'low', 'close')}
    t, o, h, l, c = (m[k] for k in ('time', 'open', 'high', 'low', 'close'))
    if not (t[0] < int(pd.Timestamp('2010-01-02',tz='UTC').timestamp()) and
            t[-1] >= int(pd.Timestamp('2026-09-01',tz='UTC').timestamp()) and
            m0['time'][-1] < m1['time'][0] and np.all(t[1:] > t[:-1]) and
            np.all(t % 60 == 0) and np.isfinite(np.column_stack((o,h,l,c))).all() and
            np.all(o > 0) and np.all(l > 0) and
            np.all(h >= np.maximum(o,c)) and np.all(l <= np.minimum(o,c))):
        raise AssertionError('Stitched gold M1 integrity failed')
    years = pd.to_datetime(t, unit='s', utc=True).year.to_numpy()
    annual = {str(y): int((years == y).sum()) for y in range(2010, 2027)}
    if min(annual.values()) < 100_000:
        raise AssertionError(f'Missing gold M1 calendar year: {annual}')
    return m, {'early_path': str(EARLY), 'early_sha256': sha(EARLY),
               'early_raw_rows': early_count, 'early_carried_flat_removed': early_count-len(m0['time']),
               'early_kept_rows': len(m0['time']), 'late_path': str(LATE),
               'late_sha256': sha(LATE), 'late_rows': late_count,
               'stitched_rows': len(t), 'annual_rows': annual,
               'first_utc': pd.Timestamp(int(t[0]),unit='s',tz='UTC').isoformat(),
               'last_utc': pd.Timestamp(int(t[-1]),unit='s',tz='UTC').isoformat()}


def simulate_candidate(m, e, i, left, right):
    """Independent raw-signal counterfactual, matching the strict-cost M1 replay."""
    t, o, h, l, c = (m[k] for k in ('time', 'open', 'high', 'low', 'close'))
    if left[i] >= right[i]:
        return None
    side = int(e['side'][i])
    entry_ix = int(left[i])
    entry_bid = float(o[entry_ix])
    atr = float(e['atr'][i-1])
    if not np.isfinite(atr) or atr <= 0:
        raise AssertionError('Bad entry ATR')
    if e['name'] == 'A_KC':
        distance = max(abs(entry_bid-float(e['ema'][i-1])), 1.5*atr)
    else:
        distance = 4.0*atr
    stop = entry_bid-side*distance
    initial_stop = float(stop)
    entry_fill = entry_bid+COST if side == 1 else entry_bid
    exit_ix = exit_bar = None
    exit_bid = exit_fill = None
    reason = None
    for j in range(i, len(e['side'])):
        a, z = int(left[j]), int(right[j])
        if j == i:
            a = entry_ix
        if a < z:
            if side == 1:
                hits = np.flatnonzero(l[a:z] <= stop)
            else:
                ask_stop = stop+COST if e['name'] == 'A_KC' else stop
                hits = np.flatnonzero(h[a:z]+COST >= ask_stop)
            if len(hits):
                exit_ix = a+int(hits[0])
                exit_bar = j
                if side == 1:
                    exit_bid = min(float(stop), float(o[exit_ix]))
                    exit_fill = exit_bid
                else:
                    exit_fill = max(float(ask_stop), float(o[exit_ix])+COST)
                    exit_bid = exit_fill-COST
                reason = 'SL_TRAIL' if side*(stop-initial_stop) > 1e-9 else 'SL_INIT'
                break
        if e['name'] == 'A_KC':
            if np.isfinite(e['ema'][j]):
                stop = max(stop, float(e['ema'][j])) if side == 1 else min(stop, float(e['ema'][j]))
        elif a < z and e['real'][j] and np.isfinite(e['atr'][j]):
            candidate = float(e['close'][j]-side*4.0*e['atr'][j])
            stop = max(stop, candidate) if side == 1 else min(stop, candidate)
    if exit_ix is None:
        exit_ix = len(t)-1
        exit_bar = len(e['side'])-1
        exit_bid = float(c[-1])
        exit_fill = exit_bid if side == 1 else exit_bid+COST
        reason = 'END'
    gross = side*(exit_bid-entry_bid)
    net = side*(exit_fill-entry_fill)
    if abs(net-(gross-COST)) > 1e-8:
        raise AssertionError('Candidate cost identity failed')
    entry_ny = pd.Timestamp(int(t[entry_ix]),unit='s',tz='UTC').tz_convert('America/New_York')
    exit_ny = pd.Timestamp(int(t[exit_ix]),unit='s',tz='UTC').tz_convert('America/New_York')
    return {'engine': e['name'], 'signal_bar': int(i), 'signal_open': int(e['start'][i]),
            'side': side, 'entry_time': int(t[entry_ix]), 'exit_time': int(t[exit_ix]),
            'exit_complete_time': int(t[exit_ix])+60, 'exit_bar': int(exit_bar),
            'entry_bid': entry_bid, 'exit_bid': exit_bid,
            'entry_fill': entry_fill, 'exit_fill': exit_fill,
            'gross': gross, 'net': net, 'y': int(net > 0),
            'exit_type': reason, 'entry_ny_year': int(entry_ny.year),
            'ny_overnight': entry_ny.date() != exit_ny.date()}


def feature_context(g):
    r1 = g['r1']
    h1, l1, c1 = (r1[k].to_numpy(float) for k in ('h','l','c'))
    atr1 = wilder_atr(h1,l1,c1,14)
    adx1 = wilder_adx(pd.Series(h1),pd.Series(l1),pd.Series(c1),14)
    b4 = g['b4']
    h4,l4,c4 = (b4[k].to_numpy(float) for k in ('h','l','c'))
    atr4 = wilder_atr(h4,l4,c4,14)
    d = g['d']
    dkeys = d.index.to_numpy(np.int64)
    dema = pd.Series(d.c.to_numpy(float)).ewm(span=50,adjust=False).mean().to_numpy()
    dema[:49] = np.nan
    return {'atr1': atr1, 'adx1': adx1, 'atr4': atr4,
            'dkeys': dkeys, 'dema': dema, 'h1starts': g['grid1']*3600}


def feature_card(g, ctx, e, i):
    start = int(e['start'][i])
    hi = int(np.searchsorted(ctx['h1starts'], start, side='left'))
    hp = hi-1
    if hi >= len(ctx['h1starts']) or hp < 0 or ctx['h1starts'][hi] != start:
        raise AssertionError('Signal not aligned to completed H1 grid')
    if ctx['h1starts'][hp]+3600 > start:
        raise AssertionError('H1 feature looks past signal open')
    containing = int(g['key4_1'][hi])
    k4 = int(np.searchsorted(g['bkeys'], containing, side='right')-2)
    if k4 < 0:
        return None
    if not int(g['bkeys'][k4]) < containing:
        raise AssertionError('Incomplete 4H feature anchor')
    ny = pd.Timestamp(start,unit='s',tz='UTC').tz_convert('America/New_York')
    wall_ns = int(ny.tz_localize(None).value)
    last_dkey = wall_ns-(wall_ns-O17)%NS_DAY-NS_DAY
    di = int(np.searchsorted(ctx['dkeys'],last_dkey))
    if di >= len(ctx['dkeys']) or ctx['dkeys'][di] != last_dkey:
        return None
    if not int(ctx['dkeys'][di]) < wall_ns:
        raise AssertionError('Incomplete D1 feature anchor')
    atr1 = float(ctx['atr1'][hp])
    atr4 = float(ctx['atr4'][k4])
    if e['name'] == 'A_KC':
        trigger_open = float(g['a4'].o.iloc[i-1])
        trigger_close = float(g['a4'].c.iloc[i-1])
    else:
        trigger_open = float(g['r1'].o.iloc[i-1])
        trigger_close = float(g['r1'].c.iloc[i-1])
    ema50 = float(ctx['dema'][di])
    values = {'atr_ratio': atr1/atr4 if atr4 > 0 else np.nan,
              'adx_1h': float(ctx['adx1'][hp]),
              'candle_size': abs(trigger_close-trigger_open)/atr1 if atr1 > 0 else np.nan,
              'macro_pullback': (trigger_close-ema50)/ema50 if ema50 > 0 else np.nan}
    return values if np.isfinite(list(values.values())).all() else None


def build_candidates(m, g, ctx, e):
    t = m['time']
    left = np.searchsorted(t,e['start'])
    right = np.searchsorted(t,e['end'])
    candidates = []
    heap = []
    recent = deque(maxlen=10)
    no_real = invalid = 0
    for i in np.flatnonzero(e['side']):
        signal_start = int(e['start'][i])
        while heap and heap[0][0] < signal_start:
            _, _, label = heapq.heappop(heap)
            recent.append(label)
        row = simulate_candidate(m,e,int(i),left,right)
        if row is None:
            no_real += 1
            continue
        card = feature_card(g,ctx,e,int(i))
        if card is None or len(recent) < 10:
            invalid += 1
            row.update({f: np.nan for f in FEATURES})
            row['feature_valid'] = False
        else:
            card['recent_win_rate'] = float(np.mean(recent))
            row.update(card)
            row['feature_valid'] = True
        row['probability'] = np.nan
        row['predicted_take'] = np.nan
        candidates.append(row)
        heapq.heappush(heap,(row['exit_complete_time'],int(i),row['y']))
    print(f"{e['name']}: {len(candidates)} raw candidates, {no_real} no-real entries, {invalid} unusable cards",flush=True)
    return candidates, {'raw_signal_bars': int(np.count_nonzero(e['side'])),
                        'real_entry_candidates':len(candidates),
                        'no_real_entries':no_real,'unusable_feature_cards':invalid}


def walk_forward(candidates, name):
    fold_rows = []
    for year in range(2016,2027):
        cutoff = int(pd.Timestamp(f'{year}-01-01',tz='America/New_York').timestamp())
        train = [r for r in candidates if r['feature_valid'] and
                 r['entry_ny_year'] < year and r['exit_complete_time'] < cutoff]
        test = [r for r in candidates if r['feature_valid'] and r['entry_ny_year'] == year]
        if len(train) < 50 or len({r['y'] for r in train}) != 2:
            raise RuntimeError(f'{name} {year}: too few resolved two-class training candidates')
        if not test:
            raise RuntimeError(f'{name} {year}: no test-year candidates')
        X = np.array([[r[k] for k in FEATURES] for r in train],float)
        y = np.array([r['y'] for r in train],np.int8)
        V = np.array([[r[k] for k in FEATURES] for r in test],float)
        model = XGBClassifier(max_depth=4,learning_rate=.05,n_estimators=100,
                              objective='binary:logistic',tree_method='hist',
                              random_state=42,n_jobs=1,eval_metric='logloss')
        model.fit(X,y)
        prob = model.predict_proba(V)[:,1]
        if not np.isfinite(prob).all():
            raise AssertionError('Nonfinite XGB probability')
        for r,p in zip(test,prob):
            r['probability'] = float(p)
            r['predicted_take'] = int(p >= .5)
        fold_rows.append({'engine':name,'test_year':year,'train_cutoff_utc':
                          pd.Timestamp(cutoff,unit='s',tz='UTC').isoformat(),
                          'train_candidates':len(train),'train_positive_share':float(y.mean()),
                          'test_candidates':len(test),'test_predicted_take':int((prob>=.5).sum())})
        print(f'{name}: trained through {year-1}, predicted {year} ({len(train)} train, {len(test)} test)',flush=True)
    return fold_rows


def execute_menu(candidates, filtered):
    base_pre = [r for r in candidates if r['entry_ny_year'] < 2016]
    occupied = -1
    if filtered:
        for r in base_pre:
            if r['signal_bar'] > occupied:
                occupied = r['exit_bar']
    selected = []
    action_skip_loss = action_skip_win = unavailable = shadowed = 0
    for r in candidates:
        if r['entry_ny_year'] < 2016:
            if not filtered and r['signal_bar'] > occupied:
                selected.append(r)
                occupied = r['exit_bar']
            continue
        if r['signal_bar'] <= occupied:
            shadowed += 1
            continue
        if filtered:
            if not np.isfinite(r['probability']):
                unavailable += 1
                continue
            if not r['predicted_take']:
                if r['y']:
                    action_skip_win += 1
                else:
                    action_skip_loss += 1
                continue
        selected.append(r)
        occupied = r['exit_bar']
    oos = [r for r in selected if r['entry_ny_year'] >= 2016]
    for a,b in zip(oos,oos[1:]):
        if b['signal_bar'] <= a['exit_bar'] or b['entry_time'] <= a['exit_time']:
            raise AssertionError('Overlapping selected trades')
    return oos, {'actionable_skipped_losing':action_skip_loss,
                 'actionable_skipped_winning':action_skip_win,
                 'unavailable_card_skips':unavailable,'occupied_signal_shadows':shadowed}


def summary(rows):
    x = np.array([r['net'] for r in rows],float)
    if len(x) < 2:
        raise AssertionError('Too few OOS trades')
    annual = {}
    for r in rows:
        y = str(r['entry_ny_year'])
        v = annual.setdefault(y,{'trades':0,'net_units':0.0})
        v['trades'] += 1
        v['net_units'] += r['net']
    return {'trades':len(rows),'win_rate':float((x>0).mean()),
            'net_expectancy':float(x.mean()),'net_total':float(x.sum()),
            'net_sharpe_per_trade':float(x.mean()/x.std(ddof=1)),
            'mbb_p':mbb_p(x),'ny_overnight':sum(r['ny_overnight'] for r in rows),
            'yearly':annual}


def main():
    spec,gate,report = (OUT/n for n in ('SPEC_XGB.txt','GATE_XGB.txt','REPORT_XGB.txt'))
    if not spec.exists() or not gate.exists():
        raise FileNotFoundError('Frozen SPEC/GATE must predate run')
    if report.exists():
        raise FileExistsError('Historical report is immutable; use a named correction')
    m,source = load_extended()
    print(f"Stitched {source['stitched_rows']:,} observed M1 bars; 2010-2019 carry removed {source['early_carried_flat_removed']:,}",flush=True)
    g = grids(m)
    ctx = feature_context(g)
    engines = (engine_a(g),engine_b(g))
    results,folds = [],[]
    for e in engines:
        name = e['name']
        candidates,funnel = build_candidates(m,g,ctx,e)
        fold = walk_forward(candidates,name)
        folds.extend(fold)
        base,base_funnel = execute_menu(candidates,False)
        filtered,filtered_funnel = execute_menu(candidates,True)
        # Independent parent replay is the baseline execution audit.
        parent,_ = baseline_replay(m,e,COST)
        parent_oos = [r for r in parent if
                      pd.Timestamp(r['entry_time'],unit='s',tz='UTC').tz_convert('America/New_York').year >= 2016]
        if len(parent_oos) != len(base):
            raise AssertionError(f'{name}: baseline trade count differs from frozen M1 replay')
        for a,b in zip(parent_oos,base):
            if (a['entry_bar'] != b['signal_bar'] or a['exit_bar'] != b['exit_bar'] or
                    abs(a['net']-b['net']) > 1e-8):
                raise AssertionError(f'{name}: candidate baseline differs from frozen M1 replay')
        prediction_skipped_loss = sum(r['feature_valid'] and r['entry_ny_year']>=2016 and
                                      r['predicted_take']==0 and r['y']==0 for r in candidates)
        prediction_skipped_win = sum(r['feature_valid'] and r['entry_ny_year']>=2016 and
                                     r['predicted_take']==0 and r['y']==1 for r in candidates)
        bs,fs = summary(base),summary(filtered)
        year_checks = {}
        for y in (2021,2024):
            old = bs['yearly'].get(str(y),{}).get('net_units',0.0)
            new = fs['yearly'].get(str(y),{}).get('net_units',0.0)
            year_checks[str(y)] = {'base_net':old,'filtered_net':new,'change':new-old,
                                   'base_drawdown':old<0,'turned_positive':old<0 and new>0}
        gate_pass = (fs['trades']>=50 and fs['net_expectancy']>0 and
                     fs['net_total']>bs['net_total'] and fs['mbb_p']<=.05 and
                     all(v['change']>=0 for v in year_checks.values()))
        for r in candidates:
            if abs(r['net']-(r['gross']-COST))>1e-8:
                raise AssertionError('Candidate cost identity')
        if filtered_funnel['actionable_skipped_losing']+filtered_funnel['actionable_skipped_winning'] > prediction_skipped_loss+prediction_skipped_win:
            raise AssertionError('Actionable skip counts exceed all-prediction skips')
        pd.DataFrame(candidates).to_csv(OUT/f'candidates_{name}.csv',index=False)
        pd.DataFrame(base).to_csv(OUT/f'baseline_trades_{name}.csv',index=False)
        pd.DataFrame(filtered).to_csv(OUT/f'filtered_trades_{name}.csv',index=False)
        results.append({'engine':name,'candidates':funnel,'base':bs,'filtered':fs,
                        'baseline_funnel':base_funnel,'filtered_funnel':filtered_funnel,
                        'all_prediction_skipped_losing':int(prediction_skipped_loss),
                        'all_prediction_skipped_winning':int(prediction_skipped_win),
                        'year_checks':year_checks,'overlay_gate_pass':bool(gate_pass)})
        print(f"{name}: OOS base {bs['trades']} trades / {bs['net_total']:.2f} units; "
              f"filtered {fs['trades']} / {fs['net_total']:.2f}; gate {gate_pass}",flush=True)
    pd.DataFrame(folds).to_csv(OUT/'folds.csv',index=False)
    meta = {'study':'gold_xgboost_metalabel','date':'2026-09-28','cost':COST,
            'xgboost_version':xgboost_version,'features':FEATURES,
            'source':source,'script_sha256':sha(Path(__file__)),
            'spec_sha256':sha(spec),'gate_sha256':sha(gate),
            'overall_gate_pass':all(r['overlay_gate_pass'] for r in results),
            'folds':folds,'results':results}
    (OUT/'run_meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    lines = ['GOLD A/B XGBOOST META-LABEL OVERLAY - REPORT',
             f"OVERALL GATE {'PASS' if meta['overall_gate_pass'] else 'FAIL'}",
             '2010-2015 expanding-training start; 2016-2026 yearly OOS predictions; 2026 partial.',
             f"Stitched BID M1 {source['stitched_rows']:,} rows, {source['first_utc']} through {source['last_utc']}; 2010-2019 carry removed {source['early_carried_flat_removed']:,}.",
             f"Sources SHA256: early {source['early_sha256']}; verified later {source['late_sha256']}.",
             f'Cost 0.585 gold units embedded once in fills; XGBoost {xgboost_version}; five fixed causal features; threshold 0.50.',
             'All metrics below use common OOS NY entry years 2016-2026. Candidate labels include counterfactual overlap signals; baseline and filtered trade books enforce one open position.', '']
    for r in results:
        b,f = r['base'],r['filtered']
        sk = r['filtered_funnel']
        lines += [f"{r['engine']} | GATE {'PASS' if r['overlay_gate_pass'] else 'FAIL'} | raw candidates {r['candidates']}",
                  f"BASE: trades {b['trades']}, win {b['win_rate']:.2%}, net expectancy {b['net_expectancy']:.6f} units/trade, total {b['net_total']:.4f} units, Sharpe {b['net_sharpe_per_trade']:.5f}, MBB p {b['mbb_p']:.4f}, NY overnight {b['ny_overnight']}.",
                  f"XGB FILTERED: trades {f['trades']}, win {f['win_rate']:.2%}, net expectancy {f['net_expectancy']:.6f} units/trade, total {f['net_total']:.4f} units, Sharpe {f['net_sharpe_per_trade']:.5f}, MBB p {f['mbb_p']:.4f}, NY overnight {f['ny_overnight']}.",
                  f"Actionable skipped losing {sk['actionable_skipped_losing']}; actionable skipped winning {sk['actionable_skipped_winning']}; skipped for unavailable card {sk['unavailable_card_skips']}; occupied shadows {sk['occupied_signal_shadows']}.",
                  f"All predictions skipped losing {r['all_prediction_skipped_losing']}; winning {r['all_prediction_skipped_winning']} (includes signals shadowed by occupancy).",
                  f"Year 2021: {r['year_checks']['2021']}; year 2024: {r['year_checks']['2024']}",
                  f"Baseline yearly entry-cohort PnL: {b['yearly']}",
                  f"Filtered yearly entry-cohort PnL: {f['yearly']}", '']
    lines += ['Training rows require both a signal before the test year and a completed exit before its January 1 New York cutoff. The recent-win feature uses only closed candidate shadow-book outcomes at each signal open.',
              'This is a BID proxy with fixed all-inclusive cost and no swaps, historical ASK quotes, slippage, FTMO account guard or MT5 execution. The 2010-2019 CSV is owner-supplied and lacks volume; carried flat filtering may drop genuine unchanged-price minutes. Prior strategy research exposure prevents a claim of untouched validation.']
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Wrote {report}',flush=True)


if __name__ == '__main__':
    main()
