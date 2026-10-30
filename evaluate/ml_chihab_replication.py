"""Hourly RF+Probit walk-forward adaptation; frozen SPEC_ML.txt."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from categorical_omnibus import digest, load_tape, continuous_h1, mbb_p
from unseen_asset_saleh_donchian_phase1 import wilder_adx, wilder_atr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/results/ml_replication'
ASSETS = (
    ('GC', 'data/results/xauusd_onboarding/xauusdtick_m1.csv', .41, .13),
    ('NQ', 'data/results/virgin_assets/ustech_m1.csv', 1.00, .39),
    ('ES', 'data/results/vwap_2021_2026/us500tick_m1.csv', .51, .30),
)
FEATURES = ('sma20_rel', 'sma50_rel', 'rsi14', 'macd_rel',
            'macd_signal_rel', 'bb_width_rel', 'adx14')


def feature_table(grid):
    keys, q, real = grid
    h, l, c = (q[k].to_numpy(float) for k in ('h', 'l', 'c'))
    close = pd.Series(c)
    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    std20 = close.rolling(20, min_periods=20).std(ddof=0)
    change = close.diff()
    up = change.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    down = (-change.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rsi = 100-100/(1+up/down.replace(0, np.nan))
    rsi = rsi.mask((down == 0) & (up > 0), 100)
    rsi = rsi.mask((up == 0) & (down > 0), 0)
    rsi = rsi.mask((up == 0) & (down == 0), 50)
    rsi.iloc[:14] = np.nan
    macd = close.ewm(span=12, adjust=False).mean()-close.ewm(span=26, adjust=False).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    adx = wilder_adx(pd.Series(h), pd.Series(l), close, 14)
    adx[:14] = np.nan
    atr = wilder_atr(h, l, c, 14)
    atr[:14] = np.nan
    frame = pd.DataFrame({
        'sma20_rel': (close-sma20)/close,
        'sma50_rel': (close-sma50)/close,
        'rsi14': rsi,
        'macd_rel': macd/close,
        'macd_signal_rel': macd_signal/close,
        'macd_hist_rel': (macd-macd_signal)/close,
        'bb_width_rel': 4*std20/sma20,
        'adx14': adx,
    })
    values = frame.loc[:, FEATURES].to_numpy(float)
    valid = np.isfinite(values).all(axis=1)
    valid[:50] = False
    assert np.isfinite(atr[valid]).all()
    return values, valid, atr


def walk_forward(asset, grid, features, valid):
    keys, q, real = grid
    starts = keys*3600
    c = q.c.to_numpy(float)
    y = np.zeros(len(keys), np.int8)
    y[:-1] = (c[1:] > c[:-1]).astype(np.int8)
    next_real = np.r_[real[1:], False]
    first = pd.Timestamp(int(starts[0]), unit='s', tz='UTC').to_period('M').to_timestamp().tz_localize('UTC')
    last = pd.Timestamp(int(starts[-1]), unit='s', tz='UTC').to_period('M').to_timestamp().tz_localize('UTC')
    months = pd.date_range(first+pd.DateOffset(months=6), last, freq='MS', tz='UTC')
    predictions, folds = [], []
    for fold_no, test_start in enumerate(months, 1):
        train_start = test_start-pd.DateOffset(months=6)
        test_end = test_start+pd.DateOffset(months=1)
        ts, te, tr = int(test_start.timestamp()), int(test_end.timestamp()), int(train_start.timestamp())
        # Label close for row t is at start(t)+2 hours, strictly before test month.
        train = np.flatnonzero((starts >= tr) & (starts+7200 < ts) & real & next_real & valid)
        test = np.flatnonzero((starts >= ts) & (starts < te) & real & valid)
        if not len(test):
            continue
        if len(train) < 200 or np.unique(y[train]).size != 2:
            raise RuntimeError(f'{asset} {test_start:%Y-%m}: insufficient two-class training rows')
        scaler = StandardScaler().fit(features[train])
        xt, xv = scaler.transform(features[train]), scaler.transform(features[test])
        rf = RandomForestClassifier(n_estimators=64, max_depth=5,
                                    min_samples_leaf=25, max_features='sqrt',
                                    random_state=42, n_jobs=1).fit(xt, y[train])
        rf_up = rf.predict_proba(xv)[:, int(np.flatnonzero(rf.classes_ == 1)[0])]
        model = sm.GLM(y[train], sm.add_constant(xt, has_constant='add'),
                       family=sm.families.Binomial(link=sm.families.links.Probit()))
        fit = model.fit(maxiter=100, tol=1e-8, disp=0)
        if not fit.converged or not np.isfinite(fit.params).all():
            raise RuntimeError(f'{asset} {test_start:%Y-%m}: Probit did not converge')
        probit_up = np.asarray(fit.predict(sm.add_constant(xv, has_constant='add')), float)
        if not np.isfinite(probit_up).all():
            raise AssertionError('Non-finite Probit test prediction')
        majority = int(y[train].mean() >= .5)
        scored = next_real[test]
        folds.append({'asset': asset, 'fold': test_start.strftime('%Y-%m'),
                      'train_start_utc': train_start.isoformat(),
                      'train_label_end_before_utc': test_start.isoformat(),
                      'test_end_utc': test_end.isoformat(),
                      'train_rows': len(train), 'test_predictions': len(test),
                      'scored_rows': int(scored.sum()), 'train_up_share': float(y[train].mean())})
        for n, k in enumerate(test):
            predictions.append({'asset': asset, 'fold': test_start.strftime('%Y-%m'),
                                'signal_index': int(k), 'signal_hour_open': int(starts[k]),
                                'signal_close_time': int(starts[k]+3600),
                                'rf_prob_up': float(rf_up[n]), 'probit_prob_up': float(probit_up[n]),
                                'rf_pred': int(rf_up[n] >= .5), 'probit_pred': int(probit_up[n] >= .5),
                                'scored': bool(scored[n]),
                                'true_up': int(y[k]) if scored[n] else np.nan,
                                'train_majority': majority})
        if fold_no % 12 == 0:
            print(f'{asset}: fitted {fold_no}/{len(months)} monthly folds', flush=True)
    pred = pd.DataFrame.from_records(predictions)
    if pred.empty:
        raise RuntimeError(f'{asset}: no OOS predictions')
    return pred, pd.DataFrame.from_records(folds)


def accuracy_summary(pred):
    scored = pred.loc[pred.scored].copy()
    agree = scored.loc[scored.rf_pred == scored.probit_pred]
    scored['year'] = pd.to_datetime(scored.signal_hour_open, unit='s', utc=True).dt.year
    yearly = {}
    for year, g in scored.groupby('year'):
        a = g.loc[g.rf_pred == g.probit_pred]
        yearly[str(year)] = {'scored': len(g),
                             'rf_accuracy': float((g.rf_pred == g.true_up).mean()),
                             'probit_accuracy': float((g.probit_pred == g.true_up).mean()),
                             'agree_count': len(a),
                             'agree_accuracy': float((a.rf_pred == a.true_up).mean()) if len(a) else None}
    return {'predictions': len(pred), 'scored_hours': len(scored),
            'up_base_rate': float(scored.true_up.mean()),
            'rf_accuracy': float((scored.rf_pred == scored.true_up).mean()),
            'probit_accuracy': float((scored.probit_pred == scored.true_up).mean()),
            'train_majority_baseline_accuracy': float((scored.train_majority == scored.true_up).mean()),
            'agreement_scored': len(agree), 'agreement_coverage': len(agree)/len(scored),
            'agreement_accuracy': float((agree.rf_pred == agree.true_up).mean()) if len(agree) else None,
            'agreement_correct': int((agree.rf_pred == agree.true_up).sum()),
            'agreement_incorrect': int((agree.rf_pred != agree.true_up).sum()),
            'yearly': yearly}


def guard_scan(m, start_ix, end_ix, position, cost):
    o, h, l = (m[k] for k in ('open', 'high', 'low'))
    side, entry = position['side'], position['entry_bid']
    peak, distance = position['peak_open_equity'], position['guard_distance']
    for ix in range(start_ix, end_ix):
        if side == 1:
            adverse = float(l[ix]-entry-cost)
            favorable = float(h[ix]-entry-cost)
        else:
            adverse = float(entry-h[ix]-cost)
            favorable = float(entry-l[ix]-cost)
        if adverse <= peak-distance:
            threshold = peak-distance
            if side == 1:
                bid_level = entry+cost+threshold
                exit_bid = min(bid_level, float(o[ix]))
            else:
                ask_level = entry-threshold
                exit_bid = max(ask_level, float(o[ix])+cost)-cost
            position['peak_open_equity'] = peak
            return ix, exit_bid
        peak = max(peak, favorable)
    position['peak_open_equity'] = peak
    return None


def close_position(m, position, exit_ix, exit_bid, reason, cost):
    net = position['side']*(exit_bid-position['entry_bid'])-cost
    entry_utc = pd.Timestamp(position['entry_time'], unit='s', tz='UTC')
    exit_utc = pd.Timestamp(int(m['time'][exit_ix]), unit='s', tz='UTC')
    entry_ny = entry_utc.tz_convert('America/New_York')
    exit_ny = exit_utc.tz_convert('America/New_York')
    return {'asset': position['asset'], 'environment': position['environment'],
            'side': position['side'], 'entry_time': position['entry_time'],
            'exit_time': int(m['time'][exit_ix]), 'entry_bid': position['entry_bid'],
            'exit_bid': exit_bid, 'entry_atr14': position['entry_atr14'],
            'guard_distance': position['guard_distance'],
            'peak_open_equity': position['peak_open_equity'], 'cost': cost,
            'exit_type': reason, 'net': net if reason != 'OPEN_END' else np.nan,
            'open_end_mark_net': net if reason == 'OPEN_END' else np.nan,
            'entry_ny_year': int(entry_ny.year),
            'ny_overnight': bool(entry_ny.date() != exit_ny.date()),
            'hold_hours': float((m['time'][exit_ix]-position['entry_time'])/3600),
            'entry_fold': position['entry_fold'],
            'entry_signal_gap_hours': position['entry_signal_gap_hours']}


def execute(m, pred, atr, asset, environment, cost):
    times = m['time']
    actions = pred.copy()
    actions['action_ix'] = np.searchsorted(times, actions.signal_close_time.to_numpy(np.int64))
    no_open = int((actions.action_ix >= len(times)).sum())
    actions = actions.loc[actions.action_ix < len(times)].sort_values(
        ['action_ix', 'signal_close_time'])
    rows, position, scan_from = [], None, 0
    funnel = {'actions': len(actions), 'no_next_m1_open': no_open,
              'flat_disagreement': 0, 'entries': 0, 'model_warning_exits': 0,
              'prop_liquidations': 0, 'same_open_reversal_suppressed': 0}
    prop = environment == 'FUTURES'
    for action in actions.itertuples(index=False):
        ix = int(action.action_ix)
        if prop and position is not None:
            liquidation = guard_scan(m, scan_from, ix, position, cost)
            if liquidation is not None:
                gx, bid = liquidation
                rows.append(close_position(m, position, gx, bid,
                                           'PROP_LIQUIDATION', cost))
                funnel['prop_liquidations'] += 1
                position = None
            else:
                scan_from = ix
        rf_class, probit_class = int(action.rf_pred), int(action.probit_pred)
        if position is not None:
            held_class = 1 if position['side'] == 1 else 0
            if rf_class != held_class or probit_class != held_class:
                rows.append(close_position(m, position, ix, float(m['open'][ix]),
                                           'MODEL_WARNING', cost))
                position = None
                funnel['model_warning_exits'] += 1
                if rf_class == probit_class:
                    funnel['same_open_reversal_suppressed'] += 1
                continue
        if position is None:
            if rf_class != probit_class:
                funnel['flat_disagreement'] += 1
                continue
            signal_ix = int(action.signal_index)
            entry_atr = float(atr[signal_ix])
            if not np.isfinite(entry_atr) or entry_atr <= 0:
                raise AssertionError('Invalid entry H1 ATR')
            position = {'asset': asset, 'environment': environment,
                        'side': 1 if rf_class == 1 else -1,
                        'entry_time': int(times[ix]), 'entry_bid': float(m['open'][ix]),
                        'entry_atr14': entry_atr,
                        'guard_distance': 6*entry_atr if prop else np.nan,
                        'peak_open_equity': 0.0, 'entry_fold': action.fold,
                        'entry_signal_gap_hours': float((times[ix]-action.signal_close_time)/3600)}
            scan_from = ix
            funnel['entries'] += 1
    if position is not None:
        if prop:
            liquidation = guard_scan(m, scan_from, len(times), position, cost)
            if liquidation is not None:
                gx, bid = liquidation
                rows.append(close_position(m, position, gx, bid,
                                           'PROP_LIQUIDATION', cost))
                funnel['prop_liquidations'] += 1
                position = None
        if position is not None:
            rows.append(close_position(m, position, len(times)-1,
                                       float(m['close'][-1]), 'OPEN_END', cost))
    return rows, funnel


def trade_summary(rows, accuracy, asset):
    closed = [r for r in rows if r['exit_type'] != 'OPEN_END']
    x = np.array([r['net'] for r in closed], float)
    n = len(x)
    mean = float(x.mean()) if n else float('nan')
    sd = float(x.std(ddof=1)) if n > 1 else float('nan')
    sharpe = mean/sd if sd > 0 else float('nan')
    win = float((x > 0).mean()) if n else float('nan')
    yearly = {}
    for r in closed:
        y = str(r['entry_ny_year'])
        v = yearly.setdefault(y, {'trades': 0, 'net_total': 0.0})
        v['trades'] += 1
        v['net_total'] += r['net']
    years = range(2022, 2026) if asset == 'ES' else range(2021, 2026)
    year_sign_pass = all(yearly.get(str(y), {}).get('net_total', 0) > 0 for y in years)
    p = mbb_p(x)
    return {'entries': len(rows), 'closed_trades': n, 'open_end': len(rows)-n,
            'open_end_mark_net': sum(r['open_end_mark_net'] for r in rows
                                     if r['exit_type'] == 'OPEN_END'),
            'net_expectancy': mean, 'net_total': float(x.sum()),
            'net_sharpe': sharpe, 'mbb_p': p, 'win_rate': win,
            'accuracy_minus_trade_win_rate_pp':
                100*(accuracy['agreement_accuracy']-win)
                if accuracy['agreement_accuracy'] is not None and n else None,
            'year_sign_pass': year_sign_pass,
            'gate_pass': bool(n >= 150 and mean > 0 and sharpe > 0 and
                              p <= .05 and year_sign_pass),
            'exit_counts': {name: sum(r['exit_type'] == name for r in rows)
                            for name in ('MODEL_WARNING', 'PROP_LIQUIDATION', 'OPEN_END')},
            'ny_overnight_entries': sum(r['ny_overnight'] for r in rows),
            'max_signal_to_open_gap_hours': max((r['entry_signal_gap_hours'] for r in rows), default=None),
            'yearly': yearly}


def main():
    if not (OUT/'SPEC_ML_V2.txt').exists() or not (OUT/'GATE_ML_V2.txt').exists():
        raise RuntimeError('SPEC and GATE must predate the evaluator')
    report = OUT/'REPORT_ML_V2.txt'
    if report.exists():
        raise FileExistsError('Historical report is immutable; use a named correction')
    results, sources, accuracy_by_asset, fold_frames = [], [], {}, []
    for asset, rel, cfd_cost, futures_cost in ASSETS:
        path = ROOT/rel
        m = load_tape(path)
        grid = continuous_h1(m)
        features, valid, atr = feature_table(grid)
        pred, folds = walk_forward(asset, grid, features, valid)
        pred.to_csv(OUT/f'predictions_{asset}.csv', index=False)
        fold_frames.append(folds)
        accuracy = accuracy_summary(pred)
        accuracy_by_asset[asset] = accuracy
        sources.append({'asset': asset, 'path': rel, 'sha256': digest(path),
                        'm1_rows': len(m['time']), 'h1_rows': len(grid[0]),
                        'h1_real': int(grid[2].sum()),
                        'first_utc': pd.Timestamp(int(m['time'][0]),unit='s',tz='UTC').isoformat(),
                        'last_utc': pd.Timestamp(int(m['time'][-1]),unit='s',tz='UTC').isoformat()})
        print(f"{asset}: {len(folds)} fitted folds, {accuracy['scored_hours']} scored hours, "
              f"RF={accuracy['rf_accuracy']:.4f}, Probit={accuracy['probit_accuracy']:.4f}, "
              f"agree={accuracy['agreement_accuracy']:.4f}", flush=True)
        for environment, cost in (('CFD', cfd_cost), ('FUTURES', futures_cost)):
            rows, funnel = execute(m, pred, atr, asset, environment, cost)
            stats = trade_summary(rows, accuracy, asset)
            pd.DataFrame(rows).to_csv(OUT/f'trades_{environment}_{asset}.csv', index=False)
            results.append({'asset': asset, 'environment': environment,
                            'cost': cost, 'funnel': funnel, 'stats': stats})
            print(f"{asset} {environment}: n={stats['closed_trades']} "
                  f"mean={stats['net_expectancy']:.5f} total={stats['net_total']:.2f} "
                  f"win={stats['win_rate']:.3f} prop={stats['exit_counts']['PROP_LIQUIDATION']} "
                  f"gate={stats['gate_pass']}", flush=True)
        del m, grid, features, pred
    pd.concat(fold_frames, ignore_index=True).to_csv(OUT/'folds.csv', index=False)
    meta = {'study': 'ml_replication_v2', 'date': '2026-09-27',
            'paper': 'https://onlinelibrary.wiley.com/doi/10.1155/2019/8342461',
            'script_sha256': digest(Path(__file__)), 'features': list(FEATURES),
            'model': {'rf_trees': 64, 'rf_depth': 5, 'rf_min_leaf': 25,
                      'rf_seed': 42, 'probit': 'Binomial GLM probit link',
                      'train_months': 6, 'test_months': 1},
            'bootstrap': {'replicates': 10000, 'block': 5, 'seed': 20260927},
            'sources': sources, 'accuracy': accuracy_by_asset,
            'overall_gate_pass': all(r['stats']['gate_pass'] for r in results),
            'results': results}
    (OUT/'run_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    lines = ['CHIHAB-STYLE RF + PROBIT HOURLY CROSS-ASSET ADAPTATION - REPORT',
             'OVERALL GATE: '+('PASS' if meta['overall_gate_pass'] else 'FAIL'),
             'The original Chihab et al. (2019) study concerns intraweek FX; this is the user-defined hourly GC/NQ/ES adaptation, not an exact paper replication.',
             meta['paper'],
             'V2 removes a redundant MACD histogram from the failed V1 rank-deficient design. Frozen 7 features, 64-tree depth-5 RF and Probit GLM; rolling six prior UTC months train, next UTC month test; no test-label leakage into fitting.',
             'Bid-only CFD proxy tapes; futures costs/guard are hypothetical transfers. No structural stop or TP was specified. MBB uses 10000 five-trade blocks, seed 20260927.', '']
    for src in sources:
        asset = src['asset']
        a = accuracy_by_asset[asset]
        lines += [f"TAPE {asset}: {src['path']} | M1 {src['m1_rows']} | H1 real {src['h1_real']}/{src['h1_rows']} | {src['first_utc']} through {src['last_utc']}",
                  f"SHA256 {src['sha256']}",
                  f"OOS classification: predictions {a['predictions']}, scored real-next-H1 {a['scored_hours']}, UP base rate {a['up_base_rate']:.4f}, train-majority baseline {a['train_majority_baseline_accuracy']:.4f}",
                  f"RF accuracy {a['rf_accuracy']:.4f}; Probit accuracy {a['probit_accuracy']:.4f}; both-agree accuracy {a['agreement_accuracy']:.4f}, coverage {a['agreement_coverage']:.4f}, correct/incorrect {a['agreement_correct']}/{a['agreement_incorrect']}",
                  'OOS accuracy by year: '+json.dumps(a['yearly'], sort_keys=True), '']
    for r in results:
        s = r['stats']
        lines += [f"{r['asset']} / {r['environment']} | cost {r['cost']:.2f} | GATE {'PASS' if s['gate_pass'] else 'FAIL'}",
                  f"Closed trades {s['closed_trades']}; net expectancy {s['net_expectancy']:.6f} price units/trade; total net PnL {s['net_total']:.3f}; Net Sharpe {s['net_sharpe']:.6f}; MBB p {s['mbb_p']:.4f}; win rate {s['win_rate']:.2%}",
                  f"Accuracy minus trade win rate {s['accuracy_minus_trade_win_rate_pp']:.2f} percentage points (different denominators: scored H1 labels versus closed trades)",
                  f"Exits {s['exit_counts']}; overnight entries {s['ny_overnight_entries']}; maximum signal-to-next-open gap {s['max_signal_to_open_gap_hours']} hours",
                  f"Complete-year sign gate {'PASS' if s['year_sign_pass'] else 'FAIL'}; annual NY entry cohorts "+json.dumps(s['yearly'], sort_keys=True),
                  f"OPEN_END last-close mark excluded from closed stats {s['open_end_mark_net']:.3f}; funnel {r['funnel']}", '']
    lines += ['Interpretation: Directional classification accuracy is not a trading edge after entry/exit timing and bid/ask costs. The accuracy-minus-win-rate comparison has different units and denominators, so it is a diagnostic gap only.',
              'Realized trades may span nights/weekends; swap and financing are unmeasured. Futures executions use CFD-derived bars rather than native CME quotes, and the prop guard is per position rather than a firm account-wide equity rule.',
              'The walk-forward model predictions are out of sample for training, but feature architecture and these tapes have prior research exposure. No live execution or prop challenge pass is established.']
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
