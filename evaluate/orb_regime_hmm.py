#!/usr/bin/env python3
"""Walk-forward HMM regime labels for the ORB strategy filter (research only).

ANTI-LOOKAHEAD DESIGN (pre-declared, fixed -- do not tune after seeing results):
  Constraints (study plan):
    1. No smoothing at decision time: the regime label for session day T is
       computed with a hand-rolled log-space FORWARD filter (alpha recursion)
       only, evaluated at the position of T's PRECEDING observation day. No
       Viterbi, no forward-backward, anywhere in the label path.
    2. No full-history in-sample fit: parameters are fitted on a rolling
       TRAILING window of observation days, frozen for one application month
       (walk-forward), then refitted and rolled forward.
    3. Prior-session-only observations: features for day D use ONLY D's RTH
       bars (09:30-15:55 NY); day T's label depends only on observations with
       NY date < T, i.e. strictly before T's 09:30 NY decision cutoff.

  Variants: w252_k2 (PRIMARY, ~12mo window, 2 states), w126_k2 / w504_k2
  (window sensitivity), w252_k3 (state-count sensitivity). Uniform applied
  range: session days 2022-01-01 .. 2026-04-30.

  Features per observation day D (causal, RTH-only):
    f1 = ln(RV + 1e-12), RV = sum of squared log close-to-close returns of
         D's RTH bar closes (minutes 570..955)
    f2 = opening-range width % = (max high - min low of the 6 OR bars
         09:30-09:55) / first OR bar open * 100
    f3 = |session return| % = |last RTH close - first OR open| / first OR
         open * 100

  This script computes REGIME LABELS ONLY. It must not read py_trades.csv or
  compute any PnL/performance number. The only other project file it reads is
  py_days.csv (ny_date + or_bars columns, session-day cross-check only).
"""

import argparse
import hashlib
import json
import math
import os
import sys
from bisect import bisect_left
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore

NY = ZoneInfo("America/New_York")
UTC = timezone.utc
APPLIED_FIRST, APPLIED_LAST = 20220101, 20260430
VARIANTS = [("w252_k2", 252, 2), ("w126_k2", 126, 2), ("w504_k2", 504, 2), ("w252_k3", 252, 3)]
PRIMARY = "w252_k2"
RTH_LO, RTH_HI = 570, 955   # NY minute-of-day, bar opens (inclusive)
OR_LO, OR_HI = 570, 595     # the six 09:30-09:55 OR bars
SCORE_TOL = 1e-4


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-csv", default=os.path.join("data", "results", "orb_2021_present", "us500tick_m5.csv"))
    ap.add_argument("--py-days-csv", default=None, help="default: py_days.csv next to --data-csv")
    ap.add_argument("--out-dir", default=os.path.join("data", "results", "orb_regime"))
    ap.add_argument("--leak-check", action="store_true", help="write LEAKCHECK.md (also writes labels+fit_blocks)")
    ap.add_argument("--verify-determinism", action="store_true", help="rebuild labels in-process and compare hash")
    return ap.parse_args()


def ny_of(ts):
    dt = datetime.fromtimestamp(ts, tz=UTC).astimezone(NY)
    return dt, dt.year * 10000 + dt.month * 100 + dt.day, dt.hour * 60 + dt.minute


def fmt_ts(ts):
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M")


def cutoff_ts(ny_date):
    y, m, d = ny_date // 10000, (ny_date // 100) % 100, ny_date % 100
    return int(datetime(y, m, d, 9, 30, tzinfo=NY).timestamp())


def gap_str(sec):
    return "%dh%02dm" % (sec // 3600, (sec % 3600) // 60)


def load_days(data_csv):
    df = pd.read_csv(data_csv)
    if list(df.columns) != ["time", "open", "high", "low", "close"]:
        raise SystemExit("unexpected M5 schema: %s" % list(df.columns))
    days = {}
    for ts, o, h, l, c in zip(df["time"].to_numpy(), df["open"].to_numpy(),
                              df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()):
        _, nyd, minute = ny_of(int(ts))
        days.setdefault(nyd, []).append((int(ts), int(minute), float(o), float(h), float(l), float(c)))
    print("bars=%d first=%s last=%s ny_dates=%d" % (len(df), fmt_ts(int(df["time"].iloc[0])),
                                                    fmt_ts(int(df["time"].iloc[-1])), len(days)))
    return days


def build_obs(days):
    """Session day = NY date with >=1 RTH bar (minute 570..955)."""
    session_days, obs, nonqual = [], [], []
    for nyd in sorted(days):
        bars = [b for b in days[nyd] if RTH_LO <= b[1] <= RTH_HI]
        if not bars:
            continue
        session_days.append(nyd)
        or_bars = [b for b in bars if OR_LO <= b[1] <= OR_HI]
        closes = [b[5] for b in bars]
        if len(or_bars) != 6 or len(closes) < 2:
            nonqual.append(nyd)
            continue
        lr = np.diff(np.log(np.array(closes)))
        rv = float(np.sum(lr * lr))
        first_open = or_bars[0][2]
        f1 = math.log(rv + 1e-12)
        f2 = (max(b[3] for b in or_bars) - min(b[4] for b in or_bars)) / first_open * 100.0
        f3 = abs(closes[-1] - first_open) / first_open * 100.0
        obs.append({"ny": nyd, "f": (f1, f2, f3), "last_ts": bars[-1][0], "n_rth": len(bars)})
    return session_days, obs, nonqual


def cross_check_session_days(py_days_csv, session_days):
    df = pd.read_csv(py_days_csv, sep=";", usecols=["ny_date", "or_bars"])
    ref = {int(r.ny_date) for r in df.itertuples() if int(r.or_bars) > 0}
    mine = set(session_days)
    if ref != mine:
        print("SESSION CROSS-CHECK FAIL: only-mine=%s only-py=%s" % (sorted(mine - ref), sorted(ref - mine)))
        raise SystemExit("session-day cross-check failed")
    print("session cross-check: %d days == py_days(or_bars>0) PASS" % len(mine))


def relabel(model, perm):
    cv = model.covars_
    if cv.ndim == 3:  # hmmlearn 0.3.3 stores diag covariances as (K,D,D)
        cv = np.diagonal(cv, axis1=1, axis2=2)
    cv = np.maximum(cv, 1e-12)  # numerical floor for the log-space filter
    return {
        "startprob": model.startprob_[perm].copy(),
        "transmat": model.transmat_[np.ix_(perm, perm)].copy(),
        "means": model.means_[perm].copy(),
        "covars": cv[perm].copy(),  # diag variances (K,D)
    }


def forward_filter(X, P):
    """Log-space scaled forward recursion. Returns (total loglik, filtered
    posteriors array (T,K) where row t uses observations 0..t ONLY)."""
    means, covars, K = P["means"], P["covars"], P["means"].shape[0]
    with np.errstate(divide="ignore"):  # log(0) = -inf on EM-zeroed entries is intended
        dx = X[:, None, :] - means[None, :, :]
        logb = -0.5 * np.sum(np.log(2.0 * np.pi * covars)[None, :, :] + dx * dx / covars[None, :, :], axis=2)
        logA = np.log(P["transmat"])
    T = X.shape[0]
    loga = np.empty((T, K))
    a = np.log(P["startprob"]) + logb[0]
    s = logsumexp(a)
    ll, loga[0] = s, a - s
    for t in range(1, T):
        a = logsumexp(loga[t - 1][:, None] + logA, axis=0) + logb[t]
        s = logsumexp(a)
        ll += s
        loga[t] = a - s
    return float(ll), np.exp(loga)


def month_list(first, last):
    y, m = first // 100, first % 100
    out = []
    while y * 100 + m <= last:
        out.append(y * 100 + m)
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def build_labels(obs, session_days):
    obs_nys = [o["ny"] for o in obs]
    obs_pos = {ny: i for i, ny in enumerate(obs_nys)}
    out_rows, blocks, max_delta, nonconv = [], {}, 0.0, 0
    names2 = ["LOWVOL", "HIGHVOL"]
    names3 = ["LOWVOL", "MIDVOL", "HIGHVOL"]
    for vname, n_fit, K in VARIANTS:
        for M in month_list(202201, 202604):
            m_sessions = [d for d in session_days if d // 100 == M and APPLIED_FIRST <= d <= APPLIED_LAST]
            if not m_sessions:
                continue
            first_sess = min(d for d in session_days if d // 100 == M)
            hi = bisect_left(obs_nys, first_sess)
            if hi < n_fit:
                continue  # warmup: not enough trailing observations yet
            fit = obs[hi - n_fit:hi]
            F = np.array([o["f"] for o in fit], dtype=float)
            mu, sd = F.mean(axis=0), F.std(axis=0)
            last_sess_m = max(m_sessions)
            seq = [o for o in obs if fit[0]["ny"] <= o["ny"] <= last_sess_m]
            X = (np.array([o["f"] for o in seq], dtype=float) - mu) / sd
            model = GaussianHMM(n_components=K, covariance_type="diag", n_iter=500,
                                tol=1e-6, random_state=42).fit(X[:n_fit])
            if not model.monitor_.converged:
                nonconv += 1
            perm = np.argsort(model.means_[:, 0])  # ascending f1 mean: LOWVOL..HIGHVOL
            P = relabel(model, perm)
            ll, post = forward_filter(X, P)
            score = float(model.score(X))
            delta = abs(ll - score)
            max_delta = max(max_delta, delta)
            if delta > SCORE_TOL:
                raise SystemExit("forward/score mismatch %s/%s: %.3e" % (vname, M, delta))
            blocks.setdefault(vname, {})["%d-%02d" % (M // 100, M % 100)] = {
                "window_first_ny": fit[0]["ny"], "window_last_ny": fit[-1]["ny"], "n_fit": n_fit,
                "startprob": np.round(P["startprob"], 6).tolist(),
                "transmat": np.round(P["transmat"], 6).tolist(),
                "means": np.round(P["means"], 6).tolist(),
                "covars": np.round(P["covars"], 6).tolist(),
                "scaler_mean": np.round(mu, 6).tolist(), "scaler_std": np.round(sd, 6).tolist(),
                "loglik": round(ll, 6), "score": round(score, 6), "score_delta": float("%.3e" % delta),
                "converged": bool(model.monitor_.converged), "engine": "hmmlearn", "perm": perm.tolist(),
            }
            seq_nys = [o["ny"] for o in seq]
            names = names2 if K == 2 else names3
            fit_block = "%s/%d-%02d" % (vname, M // 100, M % 100)
            prev_row = None
            for T in m_sessions:
                cut = cutoff_ts(T)
                if T in obs_pos:
                    i = bisect_left(seq_nys, T) - 1  # posterior at preceding obs (ny < T)
                    pv = post[i]
                    lab = names[int(np.argmax(pv))]
                    ph = float(pv[K - 1])
                    pm = float(pv[1]) if K == 3 else None
                    pl = float(pv[0]) if K == 3 else None
                    row = {"variant": vname, "ny_date": T, "label": lab, "p_highvol": ph,
                           "p_midvol": pm, "p_lowvol": pl, "last_obs_ny_date": seq[i]["ny"],
                           "last_obs_bar_ts_utc": seq[i]["last_ts"], "decision_cutoff_utc": cut,
                           "fit_block": fit_block, "warmup_len": n_fit, "flag": "ok"}
                else:
                    row = dict(prev_row) if prev_row else None
                    if row is None:
                        continue
                    row.update({"ny_date": T, "decision_cutoff_utc": cut, "fit_block": fit_block,
                                "warmup_len": n_fit, "flag": "nonobs_carry"})
                prev_row = row
                out_rows.append(row)
    return out_rows, blocks, max_delta, nonconv


def write_labels_csv(rows, path):
    cols = ["variant", "ny_date", "label", "p_highvol", "p_midvol", "p_lowvol",
            "last_obs_ny_date", "last_obs_bar_ts_utc", "decision_cutoff_utc",
            "fit_block", "warmup_len", "flag"]
    lines = [",".join(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append("%.2f" % v if v is not None else "")
            else:
                vals.append(str(v) if v is not None else "")
        lines.append(",".join(vals))
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(text)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def f1_from_closes(closes):
    lr = np.diff(np.log(np.array(closes)))
    return math.log(float(np.sum(lr * lr)) + 1e-12)


def leak_check(rows, blocks, obs, session_days, days, path):
    prim = [r for r in rows if r["variant"] == PRIMARY]
    obs_by_ny = {o["ny"]: o for o in obs}
    sess_sorted = sorted(session_days)
    lines = ["# HMM regime labels -- leak check", ""]
    # (a) sample-day table
    picks = [min(r["ny_date"] for r in prim), max(r["ny_date"] for r in prim)]
    for mfirst in (202206, 202306, 202406, 202506, 202604):
        cand = [r["ny_date"] for r in prim if r["ny_date"] // 100 == mfirst]
        if cand:
            picks.append(min(cand))
    picks += [20220314, 20221107]
    rng = np.random.RandomState(7)
    picks += [int(prim[i]["ny_date"]) for i in rng.choice(len(prim), 4, replace=False)]
    seen, samples = set(), []
    for ny in picks:
        r = next((x for x in prim if x["ny_date"] == ny), None)
        if r and ny not in seen:
            seen.add(ny)
            samples.append(r)
    samples.sort(key=lambda r: r["ny_date"])
    lines += ["## (a) Sample days: exact bar range used for the regime state", "",
              "| ny_date T | decision cutoff (UTC, epoch) | last obs day | last obs bar (UTC, epoch) | gap |",
              "|---|---|---|---|---|"]
    for r in samples:
        gap = r["decision_cutoff_utc"] - r["last_obs_bar_ts_utc"]
        lines.append("| %d | %s (%d) | %d | %s (%d) | %s |" % (
            r["ny_date"], fmt_ts(r["decision_cutoff_utc"]), r["decision_cutoff_utc"],
            r["last_obs_ny_date"], fmt_ts(r["last_obs_bar_ts_utc"]), r["last_obs_bar_ts_utc"], gap_str(gap)))
    # (b) global assertion
    viol = [r for r in rows if r["last_obs_bar_ts_utc"] >= r["decision_cutoff_utc"]]
    min_gap = min(r["decision_cutoff_utc"] - r["last_obs_bar_ts_utc"] for r in rows)
    lines += ["", "## (b) Global assertion: last_obs_bar_ts_utc < decision_cutoff_utc (ALL variants x ALL days)",
              "", "- rows checked: %d" % len(rows), "- violations: %d" % len(viol),
              "- tightest margin: %s" % gap_str(min_gap),
              "- verdict: " + ("PASS" if not viol else "FAIL"), ""]
    # (c) feature-bar containment
    lines += ["## (c) Feature-bar containment (audit bites)", "",
              "| obs day D | next session T | f1 stored | f1 recomputed from bars < cutoff(T) | equal? | f1 incl. bars >= cutoff | changed? |",
              "|---|---|---|---|---|---|---|"]
    rng = np.random.RandomState(11)
    c_ok = True
    for i in rng.choice(len(obs), 5, replace=False):
        o = obs[int(i)]
        j = bisect_left(sess_sorted, o["ny"]) + 1
        if j >= len(sess_sorted):
            continue
        T = sess_sorted[j]
        cut = cutoff_ts(T)
        bars_D = [b for b in days[o["ny"]] if RTH_LO <= b[1] <= RTH_HI and b[0] < cut]
        f1_clean = f1_from_closes([b[5] for b in bars_D])
        bars_incl = [b for b in days[o["ny"]] + days.get(T, []) if RTH_LO <= b[1] <= RTH_HI]
        f1_incl = f1_from_closes([b[5] for b in sorted(bars_incl)])
        eq = abs(f1_clean - o["f"][0]) < 1e-12
        chg = abs(f1_incl - o["f"][0]) > 1e-12
        c_ok = c_ok and eq and chg
        lines.append("| %d | %d | %.6f | %.6f | %s | %.6f | %s |" % (
            o["ny"], T, o["f"][0], f1_clean, "yes" if eq else "NO", f1_incl, "yes" if chg else "NO"))
    lines += ["", "- verdict: " + ("PASS" if c_ok else "FAIL"), ""]
    # (d) filtering invariance
    lines += ["## (d) Filtering invariance: future observations cannot change a label", "",
              "| ny_date T | label obs[..T-1] | p_highvol | label obs +5 future | p_highvol | |dp| | verdict |",
              "|---|---|---|---|---|---|---|"]
    rng = np.random.RandomState(13)
    ok_rows = [r for r in prim if r["flag"] == "ok"]
    d_ok = True
    obs_nys = [o["ny"] for o in obs]
    for i in rng.choice(len(ok_rows), 5, replace=False):
        r = ok_rows[int(i)]
        blk = blocks[r["variant"]][r["fit_block"].split("/")[1]]
        P = {"startprob": np.array(blk["startprob"]), "transmat": np.array(blk["transmat"]),
             "means": np.array(blk["means"]), "covars": np.array(blk["covars"])}
        mu, sd = np.array(blk["scaler_mean"]), np.array(blk["scaler_std"])
        w_first = blk["window_first_ny"]
        seq = [o for o in obs if o["ny"] >= w_first and o["ny"] < r["ny_date"]]
        i_pred = len(seq) - 1  # position of T's preceding observation day
        extra = [o for o in obs if o["ny"] >= r["ny_date"]][:5]
        Xa = (np.array([o["f"] for o in seq]) - mu) / sd
        Xb = (np.array([o["f"] for o in seq + extra]) - mu) / sd
        _, pa = forward_filter(Xa, P)
        _, pb = forward_filter(Xb, P)
        la = "HIGHVOL" if pa[i_pred][1] > pa[i_pred][0] else "LOWVOL"
        lb = "HIGHVOL" if pb[i_pred][1] > pb[i_pred][0] else "LOWVOL"
        dp = abs(float(pa[i_pred][1]) - float(pb[i_pred][1]))
        ok = la == lb and dp < 1e-9  # invariance: +5 future obs cannot move the filtered posterior
        d_ok = d_ok and ok
        lines.append("| %d | %s | %.9f | %s | %.9f | %.2e | %s |" % (
            r["ny_date"], la, pa[i_pred][1], lb, pb[i_pred][1], dp, "PASS" if ok else "FAIL"))
    lines += ["", "- verdict: " + ("PASS" if d_ok else "FAIL"), ""]
    lines += ["## (e) Structural statement", "",
              "- The ONLY inference used for any label is the hand-rolled log-space scaled",
              "  forward recursion in forward_filter() (alpha update, causal).",
              "- model.predict (Viterbi) and model.predict_proba (forward-backward smoothing)",
              "  are never called anywhere in this script; model.score is used solely as an",
              "  independent cross-check of the forward recursion's total log-likelihood.",
              "- Scaler stats, like all parameters, are frozen on the trailing fit window",
              "  before the application month begins.", ""]
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("leak check (b)=%s (c)=%s (d)=%s -> %s" % (
        "PASS" if not viol else "FAIL", "PASS" if c_ok else "FAIL", "PASS" if d_ok else "FAIL", path))


def main():
    args = parse_args()
    np.random.seed(42)
    os.makedirs(args.out_dir, exist_ok=True)
    py_days_csv = args.py_days_csv or os.path.join(os.path.dirname(os.path.abspath(args.data_csv)), "py_days.csv")
    days = load_days(args.data_csv)
    session_days, obs, nonqual = build_obs(days)
    print("session_days=%d obs_days=%d nonqualifying_session_days=%d (%s)" % (
        len(session_days), len(obs), len(nonqual), nonqual))
    cross_check_session_days(py_days_csv, session_days)
    rows, blocks, max_delta, nonconv = build_labels(obs, session_days)
    labels_csv = os.path.join(args.out_dir, "regime_labels.csv")
    h1 = write_labels_csv(rows, labels_csv)
    for vname, _, _ in VARIANTS:
        vrows = [r for r in rows if r["variant"] == vname]
        if not vrows:
            print("%s: NO labeled days (warmup never satisfied)" % vname)
            continue
        counts = {}
        for r in vrows:
            counts[r["label"]] = counts.get(r["label"], 0) + 1
        print("%s: first=%d last=%d days=%d labels=%s nonconv_blocks=%d" % (
            vname, min(r["ny_date"] for r in vrows), max(r["ny_date"] for r in vrows), len(vrows),
            counts, nonconv))
    print("max forward/score delta=%.3e sha256=%s" % (max_delta, h1[:16]))
    with open(os.path.join(args.out_dir, "fit_blocks.json"), "w", encoding="ascii") as f:
        json.dump({"engine": "hmmlearn", "variants": [v[0] for v in VARIANTS], "blocks": blocks}, f, indent=1)
    if args.verify_determinism:
        rows2, _, _, _ = build_labels(obs, session_days)
        h2 = write_labels_csv(rows2, os.path.join(args.out_dir, "regime_labels_verify.tmp"))
        os.remove(os.path.join(args.out_dir, "regime_labels_verify.tmp"))
        print("determinism: %s (%s)" % ("PASS" if h1 == h2 else "FAIL", h2[:16]))
    if args.leak_check:
        leak_check(rows, blocks, obs, session_days, days, os.path.join(args.out_dir, "LEAKCHECK.md"))


if __name__ == "__main__":
    main()
