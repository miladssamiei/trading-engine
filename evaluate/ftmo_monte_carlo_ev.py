"""Owner-specified simplified FTMO 10k two-step and first-payout Monte Carlo."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from futures_prop_validation_phase1 import sha

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/results/gold_ensemble_real_cost'
OUT = ROOT/'data/results/ftmo_monte_carlo'
N_PATHS = 10_000
MAX_PHASE_TRADES = 20_000
SEED = 20260928
FEE = 175.0
SPLIT = .80
MAX_LOSS = -1_000.0
PROFILES = (.005, .01, .02)


def load_r_pool():
    pool = []
    source_meta = []
    for engine in ('A_KC', 'B_DONCHIAN'):
        path = SOURCE/f'trades_{engine}_strict.csv'
        d = pd.read_csv(path)
        required = {'engine', 'side', 'entry_bid', 'entry_fill', 'exit_fill',
                    'initial_stop', 'cost', 'gross', 'net', 'entry_ny_year'}
        if not required.issubset(d.columns) or len(d) < 150:
            raise AssertionError(f'{engine}: incomplete strict-cost trade ledger')
        if not d.engine.eq(engine).all() or not d.side.isin([-1, 1]).all():
            raise AssertionError(f'{engine}: wrong identity or trade direction')
        side = d.side.to_numpy(np.int8)
        cost = d.cost.to_numpy(float)
        entry_bid = d.entry_bid.to_numpy(float)
        entry_fill = d.entry_fill.to_numpy(float)
        exit_fill = d.exit_fill.to_numpy(float)
        initial = d.initial_stop.to_numpy(float)
        gross = d.gross.to_numpy(float)
        net = d.net.to_numpy(float)
        if (not np.isfinite(np.column_stack((cost,entry_bid,entry_fill,exit_fill,
                                            initial,gross,net))).all() or
                not np.allclose(cost,.585,rtol=0,atol=1e-10) or
                np.max(np.abs(net-side*(exit_fill-entry_fill))) > 1e-8 or
                np.max(np.abs(net-(gross-cost))) > 1e-8 or
                np.max(np.abs(entry_fill-entry_bid-(side==1)*cost)) > 1e-8):
            raise AssertionError(f'{engine}: source cost or fill identity failed')
        ask_stop = initial+(side==-1)*cost if engine == 'A_KC' else initial
        risk = np.where(side==1,entry_fill-initial,ask_stop-entry_fill)
        if not np.isfinite(risk).all() or np.any(risk <= 0):
            raise AssertionError(f'{engine}: invalid initial fill-to-stop risk')
        r = net/risk
        if not np.isfinite(r).all():
            raise AssertionError(f'{engine}: nonfinite trade R')
        pool.append(r)
        source_meta.append({'engine':engine,'path':str(path),'sha256':sha(path),
                            'trades':len(r),'mean_initial_risk_units':float(risk.mean()),
                            'median_initial_risk_units':float(np.median(risk)),
                            'mean_r':float(r.mean()),'median_r':float(np.median(r)),
                            'min_r':float(r.min()),'max_r':float(r.max()),
                            'r_below_minus_one':int((r < -1-1e-10).sum()),
                            'net_total_gold_units':float(net.sum()),
                            'yearly_count':{str(int(y)):int(n) for y,n in
                                            d.groupby('entry_ny_year').size().items()}})
    pooled = np.concatenate(pool)
    pooled_meta = {'trades':len(pooled),'mean_r':float(pooled.mean()),
                   'median_r':float(np.median(pooled)),
                   'min_r':float(pooled.min()),'max_r':float(pooled.max()),
                   'std_r':float(pooled.std(ddof=1)),
                   'r_below_minus_one':int((pooled < -1-1e-10).sum()),
                   'a_share':len(pool[0])/len(pooled),
                   'b_share':len(pool[1])/len(pooled)}
    return pooled,source_meta,pooled_meta


def run_boundary(rng, empirical_r, count, cash_risk, target):
    equity = np.zeros(count,float)
    state = np.zeros(count,np.int8)  # 0 unresolved, +1 pass, -1 fail
    trades = np.zeros(count,np.int32)
    for _ in range(MAX_PHASE_TRADES):
        active = np.flatnonzero(state == 0)
        if not len(active):
            break
        pick = rng.integers(0,len(empirical_r),size=len(active))
        equity[active] += cash_risk*empirical_r[pick]
        trades[active] += 1
        passed = equity[active] >= target
        failed = equity[active] <= MAX_LOSS
        state[active[passed]] = 1
        state[active[failed]] = -1
    return equity,state,trades


def run_funded(rng, empirical_r, count, cash_risk):
    equity = np.zeros(count,float)
    alive = np.ones(count,bool)
    trades = np.zeros(count,np.int8)
    for _ in range(30):
        active = np.flatnonzero(alive)
        if not len(active):
            break
        pick = rng.integers(0,len(empirical_r),size=len(active))
        equity[active] += cash_risk*empirical_r[pick]
        trades[active] += 1
        alive[active[equity[active] <= MAX_LOSS]] = False
    positive = alive & (equity > 0) & (trades == 30)
    receipt = np.zeros(count,float)
    receipt[positive] = SPLIT*equity[positive]+FEE
    return equity,alive,positive,trades,receipt


def wilson(successes,n):
    if n == 0:
        return [float('nan'),float('nan')]
    p=successes/n
    z=1.96
    den=1+z*z/n
    center=(p+z*z/(2*n))/den
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [center-radius,center+radius]


def minimum_accounts_95(p):
    if p <= 0:
        return None
    if p >= 1:
        return 1
    return math.ceil(math.log(.05)/math.log1p(-p))


def scenario(rng, empirical_r, risk_fraction):
    cash_risk=10_000*risk_fraction
    p1_equity,p1_state,p1_trades=run_boundary(rng,empirical_r,N_PATHS,cash_risk,1_000)
    p1_pass=np.flatnonzero(p1_state==1)
    p2_equity=np.zeros(N_PATHS,float)
    p2_state=np.zeros(N_PATHS,np.int8)
    p2_trades=np.zeros(N_PATHS,np.int32)
    if len(p1_pass):
        eq,st,tr=run_boundary(rng,empirical_r,len(p1_pass),cash_risk,500)
        p2_equity[p1_pass],p2_state[p1_pass],p2_trades[p1_pass]=eq,st,tr
    p2_pass=np.flatnonzero(p2_state==1)
    funded_equity=np.zeros(N_PATHS,float)
    funded_alive=np.zeros(N_PATHS,bool)
    first_payout=np.zeros(N_PATHS,bool)
    funded_trades=np.zeros(N_PATHS,np.int8)
    receipt=np.zeros(N_PATHS,float)
    if len(p2_pass):
        eq,alive,positive,tr,rec=run_funded(rng,empirical_r,len(p2_pass),cash_risk)
        funded_equity[p2_pass],funded_alive[p2_pass]=eq,alive
        first_payout[p2_pass],funded_trades[p2_pass]=positive,tr
        receipt[p2_pass]=rec
    if (len(p2_pass)>len(p1_pass) or int(first_payout.sum())>len(p2_pass) or
            np.any(receipt[~first_payout] != 0) or
            np.any(receipt[first_payout] < FEE) or
            np.any(receipt < 0)):
        raise AssertionError('Monte Carlo phase/payout accounting failed')
    net=receipt-FEE
    ev=float(net.mean())
    se=float(net.std(ddof=1)/math.sqrt(N_PATHS))
    p=float(first_payout.mean())
    unresolved1=int((p1_state==0).sum())
    unresolved2=int(((p2_state==0)&(p1_state==1)).sum())
    funded_blow=int((~funded_alive[p2_pass]).sum())
    summary={'risk_fraction':risk_fraction,'risk_cash_per_trade':cash_risk,
             'paths':N_PATHS,'phase1_pass_count':len(p1_pass),
             'phase1_pass_rate':len(p1_pass)/N_PATHS,
             'phase1_unresolved':unresolved1,
             'phase2_pass_count':len(p2_pass),
             'phase2_pass_rate_given_phase1':len(p2_pass)/len(p1_pass) if len(p1_pass) else 0.0,
             'phase2_pass_rate_from_purchase':len(p2_pass)/N_PATHS,
             'phase2_unresolved':unresolved2,
             'funded_blow_count':funded_blow,
             'funded_blow_rate_given_funded':funded_blow/len(p2_pass) if len(p2_pass) else 0.0,
             'first_payout_count':int(first_payout.sum()),
             'first_payout_probability':p,
             'first_payout_probability_wilson95':wilson(int(first_payout.sum()),N_PATHS),
             'first_payout_rate_given_funded':int(first_payout.sum())/len(p2_pass) if len(p2_pass) else 0.0,
             'mean_gross_receipt':float(receipt.mean()),'fee':FEE,
             'ev_per_challenge':ev,'ev_monte_carlo_se':se,
             'ev_monte_carlo_ci95':[ev-1.96*se,ev+1.96*se],
             'positive_ev_signal':bool(ev-1.96*se>0),
             'model_accounts_for_at_least_95pct':minimum_accounts_95(p),
             'median_phase1_trades':float(np.median(p1_trades)),
             'p90_phase1_trades':float(np.percentile(p1_trades,90)),
             'median_phase2_trades_given_phase1':float(np.median(p2_trades[p1_pass])) if len(p1_pass) else None,
             'p90_phase2_trades_given_phase1':float(np.percentile(p2_trades[p1_pass],90)) if len(p1_pass) else None,
             'median_funded_trades_given_funded':float(np.median(funded_trades[p2_pass])) if len(p2_pass) else None}
    if abs(summary['ev_per_challenge']-(summary['mean_gross_receipt']-FEE))>1e-9:
        raise AssertionError('EV identity failed')
    rows=pd.DataFrame({'risk_fraction':risk_fraction,'path':np.arange(N_PATHS),
                       'phase1_state':p1_state,'phase1_trades':p1_trades,
                       'phase1_end_pnl':p1_equity,
                       'phase2_state':p2_state,'phase2_trades':p2_trades,
                       'phase2_end_pnl':p2_equity,
                       'funded_trades':funded_trades,'funded_end_pnl':funded_equity,
                       'funded_alive':funded_alive,'first_payout':first_payout,
                       'gross_receipt':receipt,'net_after_fee':net})
    return summary,rows


def main():
    spec=OUT/'SPEC_MC.txt'
    gate=OUT/'GATE_MC.txt'
    report=OUT/'REPORT_FTMO_MC.txt'
    if not spec.exists() or not gate.exists():
        raise FileNotFoundError('Frozen SPEC/GATE must predate simulation')
    if report.exists():
        raise FileExistsError('Historical Monte Carlo report is immutable')
    empirical,source_meta,pooled_meta=load_r_pool()
    results=[]
    paths=[]
    for n,risk in enumerate(PROFILES):
        rng=np.random.default_rng(SEED+1000*n)
        s,rows=scenario(rng,empirical,risk)
        results.append(s)
        paths.append(rows)
        print(f"risk {risk:.1%}: phase1 {s['phase1_pass_rate']:.3%}, "
              f"phase2|phase1 {s['phase2_pass_rate_given_phase1']:.3%}, "
              f"payout {s['first_payout_probability']:.3%}, EV ${s['ev_per_challenge']:.2f}",flush=True)
    pd.DataFrame(results).to_csv(OUT/'summary.csv',index=False)
    pd.concat(paths,ignore_index=True).to_csv(OUT/'per_path_outcomes.csv',index=False)
    validity=all(s['paths']==N_PATHS and s['phase1_unresolved']==0 and
                 s['phase2_unresolved']==0 for s in results)
    meta={'study':'ftmo_monte_carlo_ev','date':'2026-09-28','seed':SEED,
          'paths_per_profile':N_PATHS,'max_phase_trades':MAX_PHASE_TRADES,
          'spec_sha256':sha(spec),'gate_sha256':sha(gate),
          'script_sha256':sha(Path(__file__)),'sources':source_meta,
          'pooled_r':pooled_meta,'validity_gate_pass':validity,
          'scenarios':results}
    (OUT/'run_meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    lines=['FTMO 10K TWO-STEP GOLD TRADE-R MONTE CARLO - REPORT',
           f"SIMULATION VALIDITY GATE {'PASS' if validity else 'FAIL'}",
           '10,000 IID-with-replacement purchased paths per fixed-dollar risk profile; 2020-2026 strict 0.585-cost A/B executed trades.',
           'Owner scenario: $10,000 initial, $175 paid fee, 80% profit split, +$1,000/+500 phase targets, -$1,000 static closed-loss cutoff, 30 funded trades.',
           'R uses exact fill-to-initial-stop risk where available; no average-risk approximation.',
           f'Pooled trade R: {pooled_meta}.',
           f'Engine source details: {source_meta}.','']
    for s in results:
        lines += [f"RISK {s['risk_fraction']:.1%} (${s['risk_cash_per_trade']:.0f} initial cash risk/trade) | 10,000 paths",
                  f"  Phase 1 pass {s['phase1_pass_rate']:.2%} ({s['phase1_pass_count']}); Phase 2 pass conditional on Phase 1 {s['phase2_pass_rate_given_phase1']:.2%}, unconditional {s['phase2_pass_rate_from_purchase']:.2%} ({s['phase2_pass_count']}).",
                  f"  First positive payout probability from purchase {s['first_payout_probability']:.2%} ({s['first_payout_count']}); Wilson 95% Monte Carlo interval {s['first_payout_probability_wilson95']}; funded blow count {s['funded_blow_count']}.",
                  f"  Expected gross receipt ${s['mean_gross_receipt']:,.2f}; paid fee ${FEE:.2f}; NET EV per challenge ${s['ev_per_challenge']:,.2f}; Monte Carlo SE ${s['ev_monte_carlo_se']:.2f}; 95% MC interval {s['ev_monte_carlo_ci95']}; positive-EV signal {'YES' if s['positive_ev_signal'] else 'NO'}.",
                  f"  Model-implied independent accounts for >=95% chance of >=1 payout: {s['model_accounts_for_at_least_95pct']} (not a guarantee or purchase recommendation).",
                  f"  Phase 1 trades median/p90 {s['median_phase1_trades']:.0f}/{s['p90_phase1_trades']:.0f}; Phase 2 conditional trades median/p90 {s['median_phase2_trades_given_phase1']}/{s['p90_phase2_trades_given_phase1']}; unresolved P1/P2 {s['phase1_unresolved']}/{s['phase2_unresolved']}.",'']
    lines += ['CRITICAL RULE GAP: The owner-defined simulator checks only closed-trade -10% loss. Actual FTMO Challenge: 2-Step also enforces a 5% daily EQUITY-loss limit and minimum trading days; intratrade floating losses, swap and timing are absent here. The simulated pass/payout rates and EV must not be treated as real FTMO probabilities.',
              'Official objectives: https://ftmo.com/en/trading-objectives/ ; FTMO 2-Step fee refund terms: https://ftmo.com/en/faq/is-the-entry-fee-refunded/ .',
              'The $175 fee is supplied by the owner, not verified current pricing. Accounts trading the same strategy are correlated, so the binomial independent-account count cannot guarantee a payout. Historical R sampling from 754 prior-researched trades ignores regime concentration, strategy decay, leverage, overlapping A/B positions and actual account-specific execution.']
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Wrote {report}',flush=True)


if __name__=='__main__':
    main()
