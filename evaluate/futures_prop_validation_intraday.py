"""Intraday frozen-rule transfers with M1 prop equity guard; see study SPEC."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import cross_asset_rsi_supertrend_pullback_phase1 as h52
import xauusd_sp2l_m1_phase1 as h56

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/futures_prop_validation_intraday"
SPEC_NAME = "SPEC_FUTURES_INTRADAY.txt"
GATE_NAME = "GATE_FUTURES_INTRADAY.txt"
REPORT_NAME = "REPORT_FUTURES_INTRADAY.txt"
ASSETS = [
    ("GC", "data/results/xauusd_onboarding/xauusdtick_m1.csv", .13),
    ("NQ", "data/results/virgin_assets/ustech_m1.csv", .39),
    ("ES", "data/results/vwap_2021_2026/us500tick_m1.csv", .30),
]
NS_H = 3600 * 10**9
NS_D = 24 * NS_H
SEED, BOOT, BLOCK = 20260926, 10000, 5


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    d = pd.read_csv(path, usecols=["time", "open", "high", "low", "close"])
    assert d.time.is_monotonic_increasing and d.time.is_unique
    assert d[["open", "high", "low", "close"]].notna().all().all()
    return d


def bracket(m, side, entry_ix, entry_bid, stop_bid, target_bid, risk, cost,
            entry_extreme=False):
    """M1 bracket, then per-position peak-equity guard; bid references."""
    t, o, h, l, c = (m[k] for k in ("time", "open", "high", "low", "close"))
    peak = 0.0
    n = len(t)
    for v in range(entry_ix, n):
        first = v == entry_ix
        if first and not entry_extreme and abs(entry_bid-o[v]) <= 1e-9:
            # H53 entry-gap rule: an already-crossed bracket exits at the
            # simultaneous entry open, so gross PnL is zero.
            if (side == 1 and o[v] <= stop_bid) or (side == -1 and o[v] >= stop_bid):
                return v, float(o[v]), "STOP", peak
            if (side == 1 and o[v] >= target_bid) or (side == -1 and o[v] <= target_bid):
                return v, float(o[v]), "TARGET", peak
        if side == 1:
            stop = l[v] <= stop_bid and not (first and entry_extreme)
            target = h[v] >= target_bid
        else:
            # At an extreme short entry the stop is unreachable on that
            # minute unless the ask-side floor exceeds structural risk.
            stop = h[v] >= stop_bid and not (
                first and entry_extreme and stop_bid > entry_bid)
            target = l[v] <= target_bid
        if stop:
            exit_bid = float(stop_bid if first else
                             min(stop_bid, o[v]) if side == 1 else max(stop_bid, o[v]))
            return v, exit_bid, "STOP", peak
        if target:
            exit_bid = float(target_bid if first else
                             max(target_bid, o[v]) if side == 1 else min(target_bid, o[v]))
            return v, exit_bid, "TARGET", peak
        favorable = (h[v]-entry_bid-cost) if side == 1 else (entry_bid-l[v]-cost)
        peak = max(peak, favorable)
        open_net = side*(c[v]-entry_bid)-cost
        if open_net <= peak-2.5*risk:
            return v, float(c[v]), "PROP_LIQUIDATION", peak
    return n-1, float(c[-1]), "END", peak


def row(asset, engine, m, side, entry_ix, entry_bid, stop_bid, target_bid,
        risk, cost, exit_result, exec_seconds, setup_ix=None):
    x, exit_bid, reason, peak = exit_result
    entry_t, exit_t = int(m["time"][entry_ix]), int(m["time"][x])
    return dict(asset=asset, engine=engine, side=side,
                setup_time=int(m["time"][setup_ix]) if setup_ix is not None else entry_t,
                entry_time=entry_t, exit_time=exit_t, entry_bid=float(entry_bid),
                exit_bid=float(exit_bid), stop_bid=float(stop_bid),
                target_bid=float(target_bid), initial_risk=float(risk),
                guard_distance=float(2.5*risk), peak_open_equity=float(peak),
                exit_type=reason, gross=float(side*(exit_bid-entry_bid)),
                cost=cost, net=float(side*(exit_bid-entry_bid)-cost),
                same_bar=bool(entry_t//exec_seconds == exit_t//exec_seconds),
                same_minute=bool(entry_ix == x),
                ny_year=int(pd.Timestamp(entry_t, unit="s", tz="UTC").tz_convert("America/New_York").year))


def run_c(df, asset, cost):
    t, o, h, l, c = (df[k].to_numpy(np.int64 if k == "time" else float)
                     for k in ("time", "open", "high", "low", "close"))
    ny = pd.to_datetime(t, unit="s", utc=True).tz_convert("America/New_York")
    wall = np.asarray(ny.hour*60+ny.minute, np.int32)
    day = np.asarray(ny.year*10000+ny.month*100+ny.day, np.int64)
    ends = np.r_[np.flatnonzero(day[1:] != day[:-1])+1, len(t)]
    day_end = np.repeat(ends, np.diff(np.r_[0, ends]))
    ema = df.close.ewm(span=60, adjust=False).mean().to_numpy(float)
    adx = h56.calc_adx(df)
    setups, _, _, _ = h56.detect_setups(dict(o=o, h=h, l=l, c=c, N=len(t)))
    m = dict(time=t, open=o, high=h, low=l, close=c)
    trades, cur, cooldown = [], -1, -1
    counts = dict(setups=len(setups), busy=0, cooldown=0, killed=0,
                  expired=0, entered=0, trend_dead=0)
    for d, side, struct in setups:
        if d <= cur:
            counts["busy"] += 1
            continue
        if t[d] <= cooldown:
            counts["cooldown"] += 1
            continue
        opp = 0
        dead = False
        outcome = "expired"
        u = d+1
        while u < day_end[d]:
            if wall[u] >= 300:
                break
            if (h[u] > h[u-1]) if side == 1 else (l[u] < l[u-1]):
                opp = 0
            else:
                opp += 1
            if opp > 1:
                dead = True
                break
            trigger = (l[u] < l[u-1]) if side == 1 else (h[u] > h[u-1])
            if not trigger:
                u += 1
                continue
            entry_bid = float(l[u] if side == 1 else h[u])
            risk = side*(entry_bid-struct)
            if risk <= 0:
                u += 1
                continue
            if risk > 10.0:
                outcome = "killed"
                break
            if (side == 1 and c[u] <= ema[u]) or (side == -1 and c[u] >= ema[u]):
                u += 1
                continue
            if not np.isfinite(adx[u]) or adx[u] < 20.0 or wall[u] < 60:
                u += 1
                continue
            outcome = "entered"
            break
        if dead:
            counts["trend_dead"] += 1
        if outcome == "killed":
            counts["killed"] += 1
            cur = u
            continue
        if outcome != "entered":
            counts["expired"] += 1
            cur = int(day_end[d]-1)
            continue
        counts["entered"] += 1
        stop_bid = struct if side == 1 else struct-cost
        target_bid = entry_bid+cost+risk if side == 1 else entry_bid-risk-cost
        x = bracket(m, side, u, entry_bid, stop_bid, target_bid,
                    risk, cost, entry_extreme=True)
        cur = x[0]
        if x[2] == "PROP_LIQUIDATION":
            cooldown = int(t[cur])+86400
        trades.append(row(asset, "C_H56_M1", m, side, u, entry_bid,
                          stop_bid, target_bid, risk, cost, x, 60, d))
    return trades, counts


def continuous(m, seconds):
    t = m["time"]
    slots = t//seconds
    g = pd.DataFrame({"key": slots, "o": m["open"], "h": m["high"],
                      "l": m["low"], "c": m["close"]}).groupby("key", sort=True).agg(
                          o=("o", "first"), h=("h", "max"),
                          l=("l", "min"), c=("c", "last"))
    grid = np.arange(slots[0], slots[-1]+1, dtype=np.int64)
    r = g.reindex(grid).ffill()
    assert np.isfinite(r.to_numpy()).all()
    return grid, r, np.isin(grid, g.index.to_numpy())


def adx_with_finite_seed(h, l, c, period=14):
    """H52 Wilder ADX, repairing only a poisoned initial DX seed."""
    original = h52.wilder_adx(h, l, c, period)
    if np.isfinite(original).any():
        return original
    n = len(h)
    tr = np.empty(n, float)
    plus = np.zeros(n, float)
    minus = np.zeros(n, float)
    tr[0] = h[0]-l[0]
    for i in range(1,n):
        up = h[i]-h[i-1]
        down = l[i-1]-l[i]
        plus[i] = up if up>down and up>0 else 0.0
        minus[i] = down if down>up and down>0 else 0.0
        tr[i] = max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    st = np.full(n,np.nan);sp=np.full(n,np.nan);sm=np.full(n,np.nan)
    dx = np.full(n,np.nan);out=np.full(n,np.nan)
    if n < 2*period:
        return out
    st[period] = tr[1:period+1].sum()
    sp[period] = plus[1:period+1].sum()
    sm[period] = minus[1:period+1].sum()
    for i in range(period,n):
        if i>period:
            st[i]=st[i-1]-st[i-1]/period+tr[i]
            sp[i]=sp[i-1]-sp[i-1]/period+plus[i]
            sm[i]=sm[i-1]-sm[i-1]/period+minus[i]
        if st[i]>0 and sp[i]+sm[i]>0:
            dx[i]=100.0*abs(sp[i]-sm[i])/(sp[i]+sm[i])
    finite=np.isfinite(dx).astype(np.int8)
    run=pd.Series(finite).rolling(period,min_periods=period).sum().to_numpy()
    candidate=np.flatnonzero((run>=period)&(np.arange(n)>=2*period-1))
    if len(candidate)==0:
        return out
    seed=int(candidate[0])
    out[seed]=dx[seed-period+1:seed+1].mean()
    for i in range(seed+1,n):
        if np.isfinite(dx[i]):
            out[i]=(out[i-1]*(period-1)+dx[i])/period
        else:
            out[i]=out[i-1]
    return out


def signal_d52(m):
    grid, b, real = continuous(m, 900)
    o,h,l,c = (b[k].to_numpy(float) for k in ("o","h","l","c"))
    ny = pd.to_datetime(grid*900, unit="s", utc=True).tz_convert("America/New_York")
    wall = ny.tz_localize(None).asi8
    key4 = wall-wall%(4*NS_H)
    g4 = pd.DataFrame({"key":key4,"h":h,"l":l,"c":c}).groupby("key",sort=True).agg(
        h=("h","max"),l=("l","min"),c=("c","last"))
    keys4 = g4.index.to_numpy(np.int64)
    h4,l4,c4 = (g4[k].to_numpy(float) for k in ("h","l","c"))
    rsi4 = h52.wilder_rsi(c4,14)
    st4 = h52.supertrend_dir(h4,l4,c4,10,3.0)
    adx4 = adx_with_finite_seed(h4,l4,c4,14)
    prev4 = key4-4*NS_H
    ix = np.searchsorted(keys4,prev4)
    valid = ix<len(keys4)
    ix = np.clip(ix,0,len(keys4)-1)
    valid &= keys4[ix] == prev4
    sma = pd.Series(c).rolling(20).mean().to_numpy()
    std = pd.Series(c).rolling(20).std(ddof=0).to_numpy()
    upper,lower = sma+2*std,sma-2*std
    adx15 = adx_with_finite_seed(h,l,c,14)
    atr15 = h52.wilder_atr(h,l,c,14)
    side = np.zeros(len(grid),np.int8)
    j = np.arange(1,len(grid))
    arm = valid[j] & np.isfinite(upper[j-1]) & np.isfinite(lower[j-1]) & \
        np.isfinite(adx15[j-1]) & np.isfinite(atr15[j-1]) & \
        np.isfinite(rsi4[ix[j]]) & np.isfinite(st4[ix[j]]) & np.isfinite(adx4[ix[j]])
    side[j] = np.where(arm & (rsi4[ix[j]]>70) & (st4[ix[j]]==1) &
                       (adx4[ix[j]]>40) & (adx15[j-1]>20),1,
                       np.where(arm & (rsi4[ix[j]]<30) & (st4[ix[j]]==-1) &
                                (adx4[ix[j]]>40) & (adx15[j-1]>20),-1,0))
    return dict(grid=grid,side=side,real=real,upper=upper,lower=lower,atr=atr15)


def signal_d53(m):
    grid,b,real = continuous(m,3600)
    o,h,l,c = (b[k].to_numpy(float) for k in ("o","h","l","c"))
    ny = pd.to_datetime(grid*3600,unit="s",utc=True).tz_convert("America/New_York")
    wall = ny.tz_localize(None).asi8
    ses = wall-(wall-17*NS_H)%NS_D
    gd = pd.DataFrame({"key":ses,"h":h,"l":l}).groupby("key",sort=True).agg(
        h=("h","max"),l=("l","min"))
    dkeys = gd.index.to_numpy(np.int64)
    dh,dl = gd.h.to_numpy(float),gd.l.to_numpy(float)
    n=len(grid)
    bn=np.full(n,np.nan);sn=np.full(n,np.nan)
    bn[3:]=np.where(l[2:n-1]>h[:n-3],h[:n-3],np.nan)
    sn[3:]=np.where(h[2:n-1]<l[:n-3],l[:n-3],np.nan)
    bull=pd.Series(bn).ffill().to_numpy(float)
    bear=pd.Series(sn).ffill().to_numpy(float)
    pdh=np.full(n,np.nan);pdl=np.full(n,np.nan)
    shi=np.full(n,np.nan);slo=np.full(n,np.nan)
    cur=None;hi=-np.inf;lo=np.inf;swh=swl=False;ph=pl=np.nan
    for i in range(n):
        if ses[i]!=cur:
            cur=ses[i]
            p=np.searchsorted(dkeys,cur-NS_D)
            ph,pl=(dh[p],dl[p]) if p<len(dkeys) and dkeys[p]==cur-NS_D else (np.nan,np.nan)
            hi,lo=-np.inf,np.inf;swh=swl=False
        if np.isfinite(ph):
            if h[i]>ph:swh=True;hi=max(hi,h[i])
            if l[i]<pl:swl=True;lo=min(lo,l[i])
        pdh[i],pdl[i]=ph,pl
        if swh:shi[i]=hi
        if swl:slo[i]=lo
    side=np.zeros(n,np.int8)
    same=ses[1:]==ses[:-1]
    side[1:]=np.where(same & np.isfinite(shi[:-1]) & (c[:-1]<bull[1:]),-1,
                      np.where(same & np.isfinite(slo[:-1]) & (c[:-1]>bear[1:]),1,0))
    return dict(grid=grid,side=side,real=real,pdh=pdh,pdl=pdl,shi=shi,slo=slo)


def run_d(m,asset,cost,engine,signal,seconds):
    t,o,h,l = (m[k] for k in ("time","open","high","low"))
    grid,side = signal["grid"],signal["side"]
    left=np.searchsorted(t,grid*seconds)
    right=np.searchsorted(t,(grid+1)*seconds)
    trades=[];occupied=-1;cooldown=-1
    counts=dict(candidates=int(np.count_nonzero(side)),busy=0,cooldown=0,
                unfilled=0,no_real=0,invalid_risk=0)
    for i in np.flatnonzero(side):
        if i<=occupied:
            counts["busy"]+=1;continue
        if grid[i]*seconds<=cooldown:
            counts["cooldown"]+=1;continue
        if left[i]>=right[i]:
            counts["no_real"]+=1;continue
        dr=int(side[i]);a,z=int(left[i]),int(right[i])
        if engine=="D52_H52_M15":
            level=float(signal["lower"][i-1] if dr==1 else signal["upper"][i-1])
            touch=np.flatnonzero(l[a:z]<=level if dr==1 else h[a:z]>=level)
            if len(touch)==0:
                counts["unfilled"]+=1;continue
            u=a+int(touch[0])
            entry=float(min(o[u],level) if dr==1 else max(o[u],level))
            risk=6.0*float(signal["atr"][i-1])
            sl=entry-dr*risk
            tp=float(signal["upper"][i-1] if dr==1 else signal["lower"][i-1])
        else:
            u=a;entry=float(o[u])
            sl=float(signal["slo"][i-1] if dr==1 else signal["shi"][i-1])
            tp=float(signal["pdh"][i-1] if dr==1 else signal["pdl"][i-1])
            risk=dr*(entry-sl)
        if not np.isfinite(risk) or risk<=0 or not np.isfinite(tp):
            counts["invalid_risk"]+=1;continue
        x=bracket(m,dr,u,entry,sl,tp,risk,cost)
        occupied=int(t[x[0]]//seconds-grid[0])
        if x[2]=="PROP_LIQUIDATION":
            cooldown=int(t[x[0]])+86400
        trades.append(row(asset,engine,m,dr,u,entry,sl,tp,risk,cost,x,seconds,a))
    return trades,counts


def mbb(x):
    n=len(x)
    if n<BLOCK:return float("nan")
    rng=np.random.default_rng(SEED)
    bad=0
    for _ in range(BOOT):
        starts=rng.integers(0,n-BLOCK+1,size=math.ceil(n/BLOCK))
        sample=np.concatenate([x[s:s+BLOCK] for s in starts])[:n]
        bad+=sample.mean()<=0
    return bad/BOOT


def stats(trades):
    x=np.array([r["net"] for r in trades],float);n=len(x)
    mean=float(x.mean()) if n else float("nan")
    sd=float(x.std(ddof=1)) if n>1 else float("nan")
    sharpe=mean/sd if sd>0 else float("nan")
    p=mbb(x)
    return dict(n=n,mean=mean,total=float(x.sum()),sharpe=sharpe,mbb_p=p,
                passed=bool(n>=30 and sharpe>0 and p<=.20),
                same_bar_share=float(np.mean([r["same_bar"] for r in trades])) if n else float("nan"),
                same_minute_share=float(np.mean([r["same_minute"] for r in trades])) if n else float("nan"),
                exits={k:sum(r["exit_type"]==k for r in trades)
                       for k in ("STOP","TARGET","PROP_LIQUIDATION","END")},
                yearly={str(y):dict(n=sum(r["ny_year"]==y for r in trades),
                                    net=sum(r["net"] for r in trades if r["ny_year"]==y))
                        for y in sorted({r["ny_year"] for r in trades})})


def main():
    assert (OUT/SPEC_NAME).exists()
    assert (OUT/GATE_NAME).exists()
    if (OUT/REPORT_NAME).exists():
        raise FileExistsError("Frozen report exists; use a newly named remeasurement study")
    all_trades=[];results=[]
    for asset,rel,cost in ASSETS:
        path=ROOT/rel
        df=load(path)
        m={k:df[k].to_numpy(np.int64 if k=="time" else float)
           for k in ("time","open","high","low","close")}
        runs=[("C_H56_M1",lambda:run_c(df,asset,cost)),
              ("D52_H52_M15",lambda:run_d(m,asset,cost,"D52_H52_M15",signal_d52(m),900)),
              ("D53_H53_H1",lambda:run_d(m,asset,cost,"D53_H53_H1",signal_d53(m),3600))]
        for name,fn in runs:
            trades,funnel=fn()
            s=stats(trades)
            all_trades.extend(trades)
            results.append(dict(asset=asset,engine=name,cost=cost,tape=rel,
                                tape_sha256=digest(path),m1_rows=len(m["time"]),
                                first_utc=pd.Timestamp(m["time"][0],unit="s",tz="UTC").isoformat(),
                                last_utc=pd.Timestamp(m["time"][-1],unit="s",tz="UTC").isoformat(),
                                funnel=funnel,stats=s))
            print(f"{asset} {name}: {s['n']} trades mean={s['mean']:.5f} "
                  f"Sharpe={s['sharpe']:.4f} MBB={s['mbb_p']:.4f} PASS={s['passed']}",flush=True)
        del df,m
    pd.DataFrame(all_trades).to_csv(OUT/"trades.csv",index=False)
    meta=dict(study="futures_prop_validation_intraday",date="2026-09-26",
              seed=SEED,bootstrap=BOOT,block=BLOCK,script_sha256=digest(Path(__file__)),
              overall_pass=all(r["stats"]["passed"] for r in results),results=results)
    (OUT/"run_meta.json").write_text(json.dumps(meta,indent=2,allow_nan=True),encoding="utf-8")
    lines=["FUTURES PROP VALIDATION INTRADAY - REPORT",
           "VERDICT: "+("PASS" if meta["overall_pass"] else "FAIL"),
           "H52 is a 15m Bollinger pullback; H53 is a 1h sweep/FVG. No frozen 15m SMC order-block rule exists. Both unselected D interpretations are reported.",
           "Bid-only CFD proxy bars, owner-supplied futures cost floors in fills; no CME executions or account-dollar prop threshold.",
           "Per-trade Net Sharpe is unannualized. Same-bar share uses each engine's execution timeframe.",""]
    for r in results:
        s=r["stats"];e=s["exits"]
        lines += [f"{r['asset']} / {r['engine']} | {r['first_utc']} .. {r['last_utc']} | {r['m1_rows']} M1 rows",
                  f"  Executed {s['n']}; net expectancy {s['mean']:.6f} price units/trade; total {s['total']:.4f}; Net Sharpe {s['sharpe']:.5f}; MBB p {s['mbb_p']:.4f}; gate {'PASS' if s['passed'] else 'FAIL'}",
                  f"  Exits: normal stops {e['STOP']}, targets {e['TARGET']}, PROP_LIQUIDATION {e['PROP_LIQUIDATION']}, END {e['END']}",
                  f"  Same execution-bar exit share {s['same_bar_share']:.2%}; exact same-M1-minute share {s['same_minute_share']:.2%}",
                  f"  Funnel {r['funnel']}",
                  "  Annual entry cohorts: "+", ".join(f"{y}: n={v['n']} net={v['net']:.3f}" for y,v in s["yearly"].items()),""]
    lines += ["Overnight swap in the new model = 0. Comparable baseline CFD swap charges for all three mapped sources are not defined; total swap saved is N/A.",
              "Apex's real threshold is account-wide and a breach ends an evaluation; this per-trade 2.5R guard with cooldown is the requested diagnostic overlay, not firm-rule compliance.",
              "Frozen source parameters were transferred without optimization. Retrospective full-tape gates are exploratory, not walk-forward confirmation."]
    (OUT/REPORT_NAME).write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
