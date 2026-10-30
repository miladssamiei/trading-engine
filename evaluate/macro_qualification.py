"""
macro_qualification.py - executes the data-provenance and feasibility
qualification of the XAUUSD scheduled macro-surprise mechanism strictly
per data/results/macro_qualification/SPEC_MACRO_QUALIFICATION.txt and
GATE_MACRO_QUALIFICATION.txt (gates G-Q1..G-Q4). Read-only on price
data (the Dukascopy XAUUSD tick CSVs and all verified onboarding
exports); writes only into the study directory.

Expected user-placed input (G-Q1 primary source):
  data/results/macro_qualification/raw/macro_events_raw.csv
    columns (case-insensitive, common aliases accepted):
      family     : nfp / nonfarm payrolls / employment ...,
                   retail / advance retail sales ...,
                   durable / durable goods orders ...
      time       : scheduled release time; date-only rows assume the
                   frozen 08:30:00 ET release time; tz-aware strings,
                   epoch s/ms, YYYYMMDD[,HHMMSS] and common formats work;
                   a 'timezone' column may say UTC or ET (default ET)
      consensus  : forecast value (K/M/% suffixes tolerated)
      actual     : first-published actual value
      optional   : reference_period
  data/results/macro_qualification/raw/secondary_source.txt
    one or more lines naming the secondary audit source (G-Q1 clause a)

Usage (defaults per SPEC; overrides exist for operational flexibility
and self-testing; the audit log records the paths actually used):
  C:/Users/Milad/miniconda3/envs/reinforce_rl/python.exe \
      evaluate/macro_qualification.py
  optional: --raw-csv PATH --tick-glob GLOB [--tick-glob GLOB ...]
            --out-dir DIR
"""

import argparse
import glob as globmod
import hashlib
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

CHUNK = 5_000_000
NY = ZoneInfo("America/New_York")
UTC = timezone.utc

STUDY_DIR = Path("data/results/macro_qualification")
TICK_GLOBS = [r"C:\Users\Milad\download\xauusd-tick-*.csv",
              r"C:\Users\Milad\download\patches\xauusd-tick-*.csv"]

PRIMARY_WINDOW = ("2021-01-01", "2026-04-30")
FALLBACK_WINDOW = ("2020-01-01", "2026-08-31")

FAMILIES = ["NFP", "RETAIL", "DURABLE"]
FAMILY_PATTERNS = {
    "NFP": ["nfp", "nonfarm", "non-farm", "non farm", "payroll",
            "employment situation", "employment change"],
    "RETAIL": ["retail"],
    "DURABLE": ["durable"],
}
NON_HEADLINE_MARKERS = ["core", "ex-auto", "ex auto", "excluding", "revised"]

FLOOR = 0.41
GATE_MIN_PRIMARY_MEAN = 0.82
PRIMARY_CELL = (30, 300)
BUFFERS_S = [30, 90]
HORIZONS_S = [300, 900]

BASELINE_LO_S = -300
BASELINE_HI_S = -60
T0_MAX_GAP_MS = 5_000
ENTRY_STALE_MS = 120_000
EXIT_TAIL_MS = 60_000
BURST_MIN_RATIO = 5.0

N_MIN_EVENTS = 150
N_MIN_DATES = 150
COVERAGE_MIN = 0.95
INVARIANT_SYSTEMATIC_FRAC = 0.10

ALFRED_HINTS = {
    "NFP": "PAYEMS (BLS Employment Situation, bls.gov schedule + "
           "archived press release)",
    "RETAIL": "RSAFS (Census Advance Retail Sales, census.gov schedule "
              "+ press release)",
    "DURABLE": "DGORDER (Census Advance Durable Goods, census.gov M3 "
               "schedule + press release)",
}

TIME_COL_CANDIDATES = ["scheduled_et", "scheduled_utc", "scheduled",
                       "release_time", "datetime", "date_time",
                       "event_time", "timestamp", "gmt_time", "time",
                       "date"]
CONSENSUS_CANDIDATES = ["consensus", "forecast", "forecast_value",
                        "expected", "consensus_value", "estimate"]
ACTUAL_CANDIDATES = ["actual", "actual_value", "reported", "realized"]
TZ_COL_CANDIDATES = ["timezone", "tz", "tzinfo"]

FALLBACK_FORMATS = [
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d",
    "%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M", "%Y.%m.%d",
    "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y/%m/%d",
    "%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y",
    "%Y%m%d %H:%M", "%Y%m%d", "%Y-%b-%d %H:%M", "%Y-%b-%d",
]


def sha256_of(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def months_in_window(win_start, win_end):
    y, m = win_start.year, win_start.month
    out = []
    while (y, m) <= (win_end.year, win_end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def parse_num(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    s = str(x).strip().replace(",", "").replace("$", "")
    if s == "" or s.lower() in ("nan", "none", "n/a", "na", "-"):
        return np.nan
    mult = 1.0
    low = s.lower()
    for suf, k in (("k", 1e3), ("m", 1e6), ("b", 1e9)):
        if low.endswith(suf):
            mult = k
            s = s[:-1].strip()
            break
    s = s.rstrip("%").strip()
    try:
        return float(s) * mult
    except ValueError:
        return np.nan


def pick_column(cols, candidates, contains=False):
    low = {c.lower().strip(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    if contains:
        for cand in candidates:
            for lc, orig in low.items():
                if cand in lc:
                    return orig
    return None


def epoch_to_et_wall(epoch_s_float):
    return (datetime.fromtimestamp(epoch_s_float, tz=UTC)
            .astimezone(NY).replace(tzinfo=None))


def parse_one_time(s, tz_hint):
    s = str(s).strip()
    if s == "":
        return None, False, False
    if s.isdigit():
        v = int(s)
        nd = len(s)
        if nd == 8 and 19000101 <= v <= 99991231:
            return datetime.strptime(s, "%Y%m%d"), False, False
        if nd == 14:
            return datetime.strptime(s, "%Y%m%d%H%M%S"), False, False
        if 9 <= nd <= 10:
            return epoch_to_et_wall(float(v)), False, True
        if 12 <= nd <= 13:
            return epoch_to_et_wall(v / 1000.0), False, True
        return None, False, False
    iso = s[:-1] + "+00:00" if s.endswith("Z") else s
    dt = None
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        for fmt in FALLBACK_FORMATS:
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
    if dt is None:
        return None, False, False
    if dt.tzinfo is not None:
        return dt.astimezone(NY).replace(tzinfo=None), False, True
    if tz_hint and "utc" in str(tz_hint).lower():
        return (dt.replace(tzinfo=UTC).astimezone(NY)
                .replace(tzinfo=None)), False, True
    return dt, False, False


def load_raw_events(raw_csv):
    head = ""
    if Path(raw_csv).exists():
        head = Path(raw_csv).read_text(encoding="utf-8-sig",
                                       errors="replace").split("\n", 1)[0]
    sep = ";" if head.count(";") > head.count(",") else ","
    df = pd.read_csv(raw_csv, sep=sep, dtype=str, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)

    fam_col = pick_column(cols, ["family", "event", "event_name", "name",
                                 "indicator", "title"])
    time_col = pick_column(cols, TIME_COL_CANDIDATES)
    cons_col = pick_column(cols, CONSENSUS_CANDIDATES, contains=True)
    act_col = pick_column(cols, ACTUAL_CANDIDATES, contains=True)
    tz_col = pick_column(cols, TZ_COL_CANDIDATES)
    ref_col = pick_column(cols, ["reference_period", "period",
                                 "ref_period"])
    missing = [n for n, c in (("family", fam_col), ("time", time_col),
                              ("consensus", cons_col),
                              ("actual", act_col)) if c is None]
    if missing:
        raise SystemExit("G-Q1 FAIL (input schema): required columns not "
                         f"found: {missing}; columns present: {cols}")

    rows = []
    n_unknown = 0
    n_non_headline = 0
    for i, r in df.iterrows():
        fam_raw = str(r[fam_col]).strip().lower()
        fam = None
        for f in FAMILIES:
            if any(p in fam_raw for p in FAMILY_PATTERNS[f]):
                fam = f
                break
        if fam is None:
            n_unknown += 1
            continue
        if any(mk in fam_raw for mk in NON_HEADLINE_MARKERS):
            n_non_headline += 1
            continue
        tz_hint = r[tz_col] if tz_col else None
        et_wall, _, explicit_tz = parse_one_time(r[time_col], tz_hint)
        if et_wall is None:
            rows.append(dict(row=i, family=fam, parse_ok=False,
                             exclusion_code_fields="unparseable_time",
                             raw_family=str(r[fam_col])[:60],
                             raw_time=str(r[time_col])[:60],
                             reference_period="",
                             consensus=np.nan, actual=np.nan))
            continue
        date_only = (et_wall.hour == 0 and et_wall.minute == 0
                     and et_wall.second == 0 and not explicit_tz)
        if date_only:
            et_wall = et_wall.replace(hour=8, minute=30)
        scheduled_et = et_wall.replace(tzinfo=NY)
        scheduled_utc = scheduled_et.astimezone(UTC)
        rows.append(dict(
            row=i, family=fam, parse_ok=True,
            scheduled_et=scheduled_et, scheduled_utc=scheduled_utc,
            T_ms=int(scheduled_utc.timestamp() * 1000),
            date_only=bool(date_only), explicit_tz=bool(explicit_tz),
            consensus=parse_num(r[cons_col]), actual=parse_num(r[act_col]),
            reference_period=(str(r[ref_col]).strip() if ref_col else ""),
            raw_family=str(r[fam_col])[:60], raw_time=str(r[time_col])[:60],
            raw_consensus=str(r[cons_col])[:40],
            raw_actual=str(r[act_col])[:40],
            exclusion_code_fields=""))
    return pd.DataFrame(rows), n_unknown, n_non_headline, cols


def apply_field_exclusions(ev):
    if ev.empty:
        return ev
    def code_of(r):
        if not r.get("parse_ok", False):
            return r.get("exclusion_code_fields", "unparseable_time")
        if pd.isna(r["consensus"]):
            return "missing_consensus"
        if pd.isna(r["actual"]):
            return "missing_actual"
        return ""
    ev["exclusion_code_fields"] = ev.apply(code_of, axis=1)
    dup = ev.duplicated(subset=["family", "T_ms"], keep="first")
    ev.loc[dup & (ev["exclusion_code_fields"] == ""),
           "exclusion_code_fields"] = "duplicate"
    return ev


def dst_invariant_ok(scheduled_et, scheduled_utc):
    ok_wall = (scheduled_et.hour == 8 and scheduled_et.minute == 30
               and scheduled_et.second == 0)
    ok_utc = ((scheduled_utc.hour, scheduled_utc.minute)
              in ((12, 30), (13, 30)))
    return bool(ok_wall and ok_utc)


def collect_ticks(tick_files, ev):
    tms = ev["T_ms"].to_numpy(np.int64)
    starts = tms + BASELINE_LO_S * 1000
    ends = tms + max(HORIZONS_S) * 1000
    order = np.argsort(tms)
    starts = starts[order]
    ends = ends[order]
    lo, hi = int(starts[0]), int(ends[-1])
    store = {i: {"ts": [], "ask": [], "bid": []} for i in range(len(tms))}
    dts = {"timestamp": "int64", "askPrice": "float64",
           "bidPrice": "float64"}
    total_rows = 0
    matched = 0
    for path in tick_files:
        reader = pd.read_csv(path, chunksize=CHUNK, usecols=list(dts),
                             dtype=dts)
        ci = 0
        for chunk in reader:
            ci += 1
            ts = chunk["timestamp"].to_numpy(np.int64)
            total_rows += len(ts)
            if ts[-1] < lo or ts[0] > hi:
                continue
            idx = np.searchsorted(starts, ts, side="right") - 1
            valid = idx >= 0
            idxc = np.where(valid, idx, 0)
            valid &= ts <= ends[idxc]
            hits = np.flatnonzero(valid)
            matched += len(hits)
            ask = chunk["askPrice"].to_numpy(float)
            bid = chunk["bidPrice"].to_numpy(float)
            for j in hits:
                k = int(idxc[j])
                store[k]["ts"].append(int(ts[j]))
                store[k]["ask"].append(float(ask[j]))
                store[k]["bid"].append(float(bid[j]))
            if ci % 20 == 0:
                print(f"  ticks: {Path(path).name} chunk {ci} "
                      f"({total_rows:,} scanned, {matched:,} matched)",
                      flush=True)
    out = {}
    for k in range(len(tms)):
        if store[k]["ts"]:
            arr = np.column_stack([store[k]["ts"], store[k]["ask"],
                                   store[k]["bid"]])
            arr = arr[np.argsort(arr[:, 0], kind="stable")]
            arr = arr[np.concatenate(([True], np.diff(arr[:, 0]) > 0))]
            out[int(order[k])] = arr
    return out, total_rows, matched


def join_event(arr, t_ms, family):
    if arr is None or len(arr) == 0:
        return dict(exclusion_code="no_ticks_window",
                    exclusion_reason="no ticks in [T-300s,T+900s]")
    ts = arr[:, 0]

    def first_ge(x):
        i = int(np.searchsorted(ts, x, side="left"))
        return i if i < len(ts) else None

    def last_le(x):
        i = int(np.searchsorted(ts, x, side="right")) - 1
        return i if i >= 0 else None

    rec = dict(n_window_ticks=int(len(ts)))
    i0 = first_ge(t_ms)
    if i0 is None or (ts[i0] - t_ms) > T0_MAX_GAP_MS:
        reason = ("no tick within 5000ms after T0" if i0 is None else
                  f"first tick {int(ts[i0] - t_ms)}ms after T0 "
                  f"> {T0_MAX_GAP_MS}ms")
        rec.update(exclusion_code="t0_gap_gt_5s", exclusion_reason=reason)
        return rec
    rec["t0_lag_ms"] = int(ts[i0] - t_ms)
    rec["mid_t0"] = float(arr[i0, 1] + arr[i0, 2]) / 2.0
    base_lo = t_ms + BASELINE_LO_S * 1000
    base_hi = t_ms + BASELINE_HI_S * 1000
    n_base = int(np.searchsorted(ts, base_hi, "right")
                 - np.searchsorted(ts, base_lo, "left"))
    n_burst = int(np.searchsorted(ts, t_ms + 5_000, "right")
                  - np.searchsorted(ts, t_ms, "left"))
    rate_base = (n_base / float(BASELINE_HI_S - BASELINE_LO_S)
                 if n_base > 0 else 0.0)
    ratio = (n_burst / 5.0 / rate_base) if rate_base > 0 else 1e9
    rec["burst_ratio"] = float(ratio)
    rec["n_baseline"], rec["n_burst"] = n_base, n_burst
    if family == "NFP" and ratio < BURST_MIN_RATIO:
        rec.update(exclusion_code="nfp_burst_fail",
                   exclusion_reason=f"burst ratio {ratio:.1f} < "
                                    f"{BURST_MIN_RATIO:.0f} (manual "
                                    "review required)")
        return rec
    for b in BUFFERS_S:
        i = first_ge(t_ms + b * 1000)
        lag = None if i is None else int(ts[i] - (t_ms + b * 1000))
        if i is None or lag > ENTRY_STALE_MS:
            reason = (f"no entry tick after T+{b}s" if i is None else
                      f"T+{b}s entry staleness {lag}ms > "
                      f"{ENTRY_STALE_MS}ms")
            rec.update(exclusion_code=f"entry_stale_{b}s",
                       exclusion_reason=reason)
            return rec
        rec[f"i_entry_{b}"] = i
        rec[f"mid_entry_{b}"] = float(arr[i, 1] + arr[i, 2]) / 2.0
        rec[f"ask_entry_{b}"] = float(arr[i, 1])
        rec[f"bid_entry_{b}"] = float(arr[i, 2])
        rec[f"entry_{b}_lag_ms"] = lag
    for h in HORIZONS_S:
        i = last_le(t_ms + h * 1000)
        staleness = None if i is None else int(
            (t_ms + h * 1000) - ts[i])
        if i is None or staleness > EXIT_TAIL_MS:
            reason = (f"no exit tick before T+{h}s" if i is None else
                      f"T+{h}s exit tick {staleness}ms stale > "
                      f"{EXIT_TAIL_MS}ms")
            rec.update(exclusion_code=f"exit_window_empty_{h}s",
                       exclusion_reason=reason)
            return rec
        rec[f"i_exit_{h}"] = i
        rec[f"mid_exit_{h}"] = float(arr[i, 1] + arr[i, 2]) / 2.0
        rec[f"ask_exit_{h}"] = float(arr[i, 1])
        rec[f"bid_exit_{h}"] = float(arr[i, 2])
    rec["exclusion_code"] = ""
    rec["exclusion_reason"] = ""
    return rec


def pick_stratified(sub, n):
    if len(sub) == 0:
        return []
    pos = np.unique(np.linspace(0, len(sub) - 1,
                                min(n, len(sub))).astype(int))
    return [sub.iloc[p] for p in pos]


def fmt_ts(ms):
    return pd.Timestamp(int(ms), unit="ms", tz="UTC").strftime(
        "%Y-%m-%d %H:%M:%S.%f")[:-3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-csv", default=None)
    ap.add_argument("--tick-glob", action="append", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else STUDY_DIR
    raw_dir = out_dir / "raw"
    raw_csv = (Path(args.raw_csv) if args.raw_csv
               else raw_dir / "macro_events_raw.csv")
    tick_globs = args.tick_glob if args.tick_glob else TICK_GLOBS
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    t_start = datetime.now(UTC)
    audit = [f"CLOCK_SYNC_AUDIT / PROVENANCE LOG",
             f"generated {t_start.isoformat()}",
             "spec: data/results/macro_qualification/"
             "SPEC_MACRO_QUALIFICATION.txt", ""]
    rep = ["REPORT_MACRO_QUALIFICATION.txt",
           "Generated " + t_start.isoformat()
           + "  (tool: evaluate/macro_qualification.py)",
           "Gates executed verbatim from GATE_MACRO_QUALIFICATION.txt "
           "(SPEC_MACRO_QUALIFICATION.txt, 2026-09-06).", ""]

    tick_files = sorted(p for g in tick_globs for p in globmod.glob(g))
    if not tick_files:
        rep.append("OPERATIONAL FAIL: no tick files matched: "
                   + str(tick_globs))
        (out_dir / "REPORT_MACRO_QUALIFICATION.txt").write_text(
            "\n".join(rep), encoding="utf-8")
        print(rep[-1])
        return 2

    audit += ["INPUTS",
              f"primary source csv: {raw_csv}"]
    if not raw_csv.exists():
        rep.append(f"G-Q1 FAIL: primary source file missing: {raw_csv}")
        rep.append("Place the primary source at "
                   "raw/macro_events_raw.csv (see SPEC section 3) and "
                   "re-run.")
        (out_dir / "REPORT_MACRO_QUALIFICATION.txt").write_text(
            "\n".join(rep), encoding="utf-8")
        print(rep[-2])
        return 2
    audit += ["primary source sha256: " + sha256_of(raw_csv),
              "primary source size bytes: "
              + str(raw_csv.stat().st_size)]
    secondary = raw_dir / "secondary_source.txt"
    secondary_ok = secondary.exists() and secondary.stat().st_size > 0
    audit.append(f"secondary audit source file: {secondary} "
                 + ("PRESENT" if secondary_ok else "MISSING"))
    if secondary_ok:
        for line in secondary.read_text().splitlines()[:5]:
            audit.append("  secondary: " + line.strip())
    audit.append("tick files (read-only; verified by "
                 "data/results/xauusd_onboarding/REPORT.md):")
    for p in tick_files:
        audit.append(f"  {p}  ({Path(p).stat().st_size:,} bytes)")
    audit.append("")

    try:
        ev, n_unknown, n_non_headline, src_cols = load_raw_events(raw_csv)
    except SystemExit as e:
        rep.append("G-Q1 FAIL: " + str(e))
        (out_dir / "REPORT_MACRO_QUALIFICATION.txt").write_text(
            "\n".join(rep), encoding="utf-8")
        print(rep[-1])
        return 2
    ev = apply_field_exclusions(ev)
    audit += [f"raw rows ingested: "
              f"{len(ev) + n_unknown + n_non_headline}",
              f"unmapped-family rows dropped: {n_unknown}",
              f"non-headline rows dropped (core/ex/revised): "
              f"{n_non_headline}",
              f"source columns: {src_cols}", ""]
    if ev.empty or ev["parse_ok"].sum() == 0:
        rep.append("G-Q1 FAIL: no parseable frozen-family events in "
                   "source")
        (out_dir / "REPORT_MACRO_QUALIFICATION.txt").write_text(
            "\n".join(rep), encoding="utf-8")
        print(rep[-1])
        return 2

    window = PRIMARY_WINDOW
    used_fallback = False
    while True:
        ws = pd.Timestamp(window[0], tz=UTC)
        we = pd.Timestamp(window[1], tz=UTC) + pd.Timedelta(days=1) \
            - pd.Timedelta(milliseconds=1)
        tag = ("window " + window[0] + ".." + window[1]
               + (" (PRE-DECLARED FALLBACK)" if used_fallback
                  else " (PRIMARY)"))
        w = ev[(ev["T_ms"] >= int(ws.timestamp() * 1000))
               & (ev["T_ms"] <= int(we.timestamp() * 1000))].copy()
        w["ny_date"] = w["scheduled_et"].map(lambda d: d.strftime("%Y%m%d"))
        w["ny_year"] = w["scheduled_et"].map(lambda d: int(d.year))
        n_expected = len(months_in_window(
            datetime.strptime(window[0], "%Y-%m-%d").date(),
            datetime.strptime(window[1], "%Y-%m-%d").date()))
        print(f"[{tag}] parsed events in window: {len(w)}", flush=True)

        cov_rows = []
        for fam in FAMILIES:
            sub = w[w["family"] == fam]
            complete = sub[sub["exclusion_code_fields"] == ""]
            months_covered = len({(d.year, d.month)
                                  for d in complete["scheduled_et"]})
            cov_rows.append((fam, months_covered, n_expected,
                             months_covered / n_expected))
        coverage_ok = all(c >= COVERAGE_MIN for _, _, _, c in cov_rows)
        gq1_a = secondary_ok
        gq1_b = coverage_ok
        gq1_pass = gq1_a and gq1_b

        inv = w.apply(lambda r: dst_invariant_ok(r["scheduled_et"],
                                                 r["scheduled_utc"]),
                      axis=1) if len(w) else pd.Series(dtype=bool)
        n_inv_viol = int((~inv).sum()) if len(w) else 0
        frac_viol = (n_inv_viol / len(w)) if len(w) else 0.0
        systematic = frac_viol > INVARIANT_SYSTEMATIC_FRAC

        audit += ["=" * 70, "WINDOW: " + tag,
                  f"events parsed in window: {len(w)}; DST-invariant "
                  f"violations: {n_inv_viol} ({frac_viol:.1%})",
                  "DST mapping table (all window events; scheduled ET -> "
                  "scheduled UTC -> invariant):"]
        for _, r in w.sort_values("T_ms").iterrows():
            ok = dst_invariant_ok(r["scheduled_et"], r["scheduled_utc"])
            flag = "OK" if ok else "VIOLATION"
            note = ("  [date-only row, 08:30 ET assumed]"
                    if r.get("date_only") else "")
            audit.append(f"  {r['family']:8s} "
                         + r["scheduled_et"].strftime(
                             "%Y-%m-%d %H:%M:%S")
                         + " ET -> "
                         + r["scheduled_utc"].strftime("%m-%d %H:%M:%S")
                         + f" UTC  {flag}{note}")
        if systematic and n_inv_viol:
            naive_utc_hits = sum(
                1 for _, r in w[~inv].iterrows()
                if (r["scheduled_et"].replace(tzinfo=UTC).hour,
                    r["scheduled_et"].replace(tzinfo=UTC).minute)
                in ((12, 30), (13, 30)))
            if naive_utc_hits / n_inv_viol > 0.5:
                audit.append("HINT: most violations are consistent with "
                             "UTC-naive source times; set a timezone "
                             "column to UTC and re-run (per SPEC "
                             "strictness: no automatic rescue applied).")
        audit.append("")

        w["exclusion_code_join"] = ""
        w.loc[~inv, "exclusion_code_join"] = "dst_invariant"
        sim_counts = (w.loc[w["exclusion_code_join"] == ""]
                      .groupby("T_ms")["family"].nunique())
        sim_times = sim_counts[sim_counts > 1].index
        w.loc[(w["T_ms"].isin(sim_times))
              & (w["exclusion_code_join"] == ""),
              "exclusion_code_join"] = "simultaneous"
        audit.append(f"simultaneous frozen-family release timestamps: "
                     f"{len(sim_times)} (all participant events excluded, "
                     "SPEC section 5)")

        candidates = w[(w["exclusion_code_fields"] == "")
                       & (w["exclusion_code_join"] == "")].copy()
        candidates = candidates.sort_values("T_ms").reset_index(drop=True)
        print(f"[{tag}] candidate events for tick join: "
              f"{len(candidates)}", flush=True)

        if systematic or candidates.empty:
            ticks, rows_scanned, rows_matched = {}, 0, 0
            join_recs = {}
            code_counts = {}
            if systematic:
                code_counts = {"dst_invariant": n_inv_viol}
                audit.append("G-Q2 SYSTEMATIC DST-INARIANT FAILURE: tick "
                             "join not attempted (SPEC stop rule).")
            else:
                code_counts = {"no_candidate_events": len(w)}
                audit.append("no candidate events after exclusions; tick "
                             "join not attempted.")
        else:
            ticks, rows_scanned, rows_matched = collect_ticks(tick_files,
                                                              candidates)
            audit.append(f"tick scan: {rows_scanned:,} rows scanned, "
                         f"{rows_matched:,} rows matched event windows "
                         f"({len(ticks)} events with ticks)")
            join_recs = {}
            for i, r in candidates.iterrows():
                join_recs[i] = join_event(ticks.get(i), int(r["T_ms"]),
                                          r["family"])
            code_counts = {}
            for v in join_recs.values():
                if v["exclusion_code"]:
                    code_counts[v["exclusion_code"]] = \
                        code_counts.get(v["exclusion_code"], 0) + 1
            audit.append("join exclusions by code: "
                         + (str(code_counts) if code_counts else "none"))
        audit.append("")

        usable_idx = [i for i, v in join_recs.items()
                      if v["exclusion_code"] == ""]
        u = candidates.loc[usable_idx].copy() if usable_idx \
            else candidates.iloc[0:0].copy()
        for col in ("n_window_ticks", "t0_lag_ms", "burst_ratio",
                    "n_baseline", "n_burst"):
            u[col] = [join_recs[i].get(col, np.nan) for i in usable_idx]
        for b in BUFFERS_S:
            u[f"mid_entry_{b}"] = [join_recs[i][f"mid_entry_{b}"]
                                   for i in usable_idx]
            u[f"ask_entry_{b}"] = [join_recs[i][f"ask_entry_{b}"]
                                   for i in usable_idx]
            u[f"bid_entry_{b}"] = [join_recs[i][f"bid_entry_{b}"]
                                   for i in usable_idx]
        for h in HORIZONS_S:
            u[f"mid_exit_{h}"] = [join_recs[i][f"mid_exit_{h}"]
                                  for i in usable_idx]
            u[f"ask_exit_{h}"] = [join_recs[i][f"ask_exit_{h}"]
                                  for i in usable_idx]
            u[f"bid_exit_{h}"] = [join_recs[i][f"bid_exit_{h}"]
                                  for i in usable_idx]
        if usable_idx:
            u["mid_t0"] = [join_recs[i]["mid_t0"] for i in usable_idx]

        n_events = len(u)
        n_dates = u["ny_date"].nunique() if n_events else 0
        print(f"[{tag}] usable events: {n_events}; unique NY dates: "
              f"{n_dates}", flush=True)

        gq2_pass = (not systematic) and n_events > 0

        if not used_fallback and not (systematic or candidates.empty) \
                and (n_events < N_MIN_EVENTS or n_dates < N_MIN_DATES):
            audit.append("G-Q3 fallback invoked: window extended to "
                         + FALLBACK_WINDOW[0] + ".." + FALLBACK_WINDOW[1]
                         + " (pre-declared, SPEC section 1)")
            window = FALLBACK_WINDOW
            used_fallback = True
            continue

        gq3_pass = n_events >= N_MIN_EVENTS and n_dates >= N_MIN_DATES
        break

    audit_picks = []
    spot_picks = []
    for fam in FAMILIES:
        sub = u[u["family"] == fam].sort_values("T_ms")
        audit_picks += pick_stratified(sub, 4)
        spot_picks += pick_stratified(sub, 2)
    audit += ["", "=" * 70,
              "G-Q1 (c) MANUAL AUDIT SAMPLE - up to 12 events (4 per "
              "family). Verify each against the agency schedule page,",
              "the agency press release, and ALFRED first vintage; "
              "record results in this file. Per SPEC: each discrepancy",
              "removes the event; >3 of 12 discrepant -> G-Q1 FAIL."]
    for r in audit_picks:
        audit.append(f"  {r['family']} | scheduled "
                     + r["scheduled_et"].strftime("%Y-%m-%d %H:%M:%S")
                     + f" ET | reference_period={r['reference_period']} | "
                     f"consensus={r['consensus']} | actual="
                     f"{r['actual']} | source row {int(r['row'])} | "
                     f"verify vs {ALFRED_HINTS[r['family']]}")
    audit += ["", "G-Q2 MANUAL SPOT-CHECK TRANSCRIPTS - up to 6 events "
              "(2 per family); last 5 ticks pre-T, first 10 post-T.",
              "Any timezone ambiguity, DST offset mismatch, or "
              "unexplained gap -> G-Q2 HARD FAIL (SPEC section 4)."]
    for r in spot_picks:
        audit.append(f"  {r['family']} "
                     + r["scheduled_et"].strftime("%Y-%m-%d %H:%M:%S")
                     + f" ET (T={fmt_ts(r['T_ms'])} UTC)")
        arr = ticks.get(int(r.name)) if not systematic else None
        if arr is not None:
            ts = arr[:, 0]
            lo = int(np.searchsorted(ts, r["T_ms"], "left"))
            for j in range(max(0, lo - 5), lo):
                audit.append(f"    pre  {fmt_ts(ts[j])} "
                             f"ask={arr[j, 1]:.3f} bid={arr[j, 2]:.3f}")
            for j in range(lo, min(len(ts), lo + 10)):
                audit.append(f"    post {fmt_ts(ts[j])} "
                             f"ask={arr[j, 1]:.3f} bid={arr[j, 2]:.3f}")
        audit.append("")

    u["event_id"] = u["family"] + "-" + u["ny_date"].astype(str)
    dropped = w[(w["exclusion_code_fields"] != "")
                | (w["exclusion_code_join"] != "")].copy()
    if len(dropped):
        dropped["event_id"] = (dropped["family"] + "-"
                               + dropped["ny_date"].astype(str))

    run_gq4 = gq1_pass and gq2_pass and gq3_pass
    gq4_pass = False
    gq4_i = gq4_ii = gq4_iii = False
    primary_mean = np.nan
    fam_means = {f: np.nan for f in FAMILIES}
    n_fam_pos = 0
    best_fam = None
    pooled_ex_best = np.nan
    yr_share = pd.Series(dtype=float)
    total_pos = 0.0
    year_viol = []
    n_measured = 0
    n_zero = 0
    m = u.iloc[0:0].copy()

    if run_gq4:
        u["surprise"] = u["actual"] - u["consensus"]
        n_zero = int((u["surprise"] == 0).sum())
        sigma = {}
        for fam in FAMILIES:
            s = u.loc[u["family"] == fam, "surprise"].to_numpy(float)
            sigma[fam] = float(np.std(s, ddof=1)) if len(s) > 1 else np.nan
        u["surprise_std"] = [
            (r["surprise"] / sigma[r["family"]]
             if np.isfinite(sigma.get(r["family"], np.nan))
             and sigma.get(r["family"], np.nan) > 0 else np.nan)
            for _, r in u.iterrows()]
        u["direction"] = -np.sign(u["surprise"]).astype(int)
        u.loc[u["surprise"] == 0, "direction"] = 0

        m = u[u["surprise"] != 0].copy()
        n_measured = len(m)
        for b in BUFFERS_S:
            for h in HORIZONS_S:
                m[f"gross_{b}_{h}"] = m["direction"] * (
                    m[f"mid_exit_{h}"] - m[f"mid_entry_{b}"])
                entry_px = np.where(m["direction"] > 0,
                                    m[f"ask_entry_{b}"],
                                    m[f"bid_entry_{b}"])
                exit_px = np.where(m["direction"] > 0,
                                   m[f"bid_exit_{h}"],
                                   m[f"ask_exit_{h}"])
                m[f"net_{b}_{h}"] = m["direction"] * (exit_px - entry_px)
        mfe_l, mae_l = [], []
        for i, r in m.iterrows():
            arr = ticks.get(int(i))
            if arr is None or len(arr) == 0:
                mfe_l.append(np.nan)
                mae_l.append(np.nan)
                continue
            ts = arr[:, 0]
            lo = int(np.searchsorted(
                ts, r["T_ms"] + PRIMARY_CELL[0] * 1000, "left"))
            hi = int(np.searchsorted(
                ts, r["T_ms"] + PRIMARY_CELL[1] * 1000, "right")) - 1
            mids = (arr[lo:hi + 1, 1] + arr[lo:hi + 1, 2]) / 2.0
            path = r["direction"] * (
                mids - r[f"mid_entry_{PRIMARY_CELL[0]}"])
            mfe_l.append(float(path.max()) if len(path) else np.nan)
            mae_l.append(float(path.min()) if len(path) else np.nan)
        m["mfe_primary"] = mfe_l
        m["mae_primary"] = mae_l
        m["t0_jump"] = m["direction"] * (
            m[f"mid_entry_{PRIMARY_CELL[0]}"] - m["mid_t0"])

        pcol = f"gross_{PRIMARY_CELL[0]}_{PRIMARY_CELL[1]}"
        primary_mean = float(m[pcol].mean()) if n_measured else np.nan
        fam_means = {f: (float(m.loc[m["family"] == f, pcol].mean())
                         if (m["family"] == f).any() else np.nan)
                     for f in FAMILIES}
        n_fam_pos = sum(1 for v in fam_means.values()
                        if np.isfinite(v) and v > 0)
        finite_fams = [f for f in FAMILIES if np.isfinite(fam_means[f])]
        best_fam = (max(finite_fams, key=lambda f: fam_means[f])
                    if finite_fams else None)
        m_ex = m[m["family"] != best_fam] if best_fam is not None else m
        pooled_ex_best = (float(m_ex[pcol].mean())
                          if len(m_ex) else np.nan)
        pos = m[pcol].clip(lower=0.0)
        yr_pos = pos.groupby(m["ny_year"]).sum()
        total_pos = float(yr_pos.sum())
        yr_share = (yr_pos / total_pos) if total_pos > 0 else yr_pos * 0.0
        year_viol = [int(y) for y, s_ in yr_share.items() if s_ > 0.5]

        gq4_i = bool(np.isfinite(primary_mean)
                     and primary_mean >= GATE_MIN_PRIMARY_MEAN)
        gq4_ii = bool(n_fam_pos >= 2 and np.isfinite(pooled_ex_best)
                      and pooled_ex_best > 0)
        gq4_iii = bool(total_pos > 0 and not year_viol)
        gq4_pass = bool(gq4_i and gq4_ii and gq4_iii)

    ds_cols = ["event_id", "family", "reference_period", "scheduled_et",
               "scheduled_utc", "T_ms", "consensus", "actual"]
    if len(u):
        ds = u[ds_cols].copy()
        if run_gq4:
            ds["surprise"] = u["surprise"]
            ds["surprise_std"] = u["surprise_std"]
            ds["direction"] = u["direction"]
            ds["status"] = ["usable_zero_surprise" if z else "usable"
                            for z in (u["surprise"] == 0)]
        else:
            ds["status"] = "usable"
    else:
        ds = pd.DataFrame(columns=ds_cols + ["status"])
    if len(dropped):
        dr = dropped[ds_cols[:8]].copy()
        dr["status"] = "excluded"
        dr["exclusion_code"] = [
            r["exclusion_code_fields"] or r["exclusion_code_join"]
            for _, r in dropped.iterrows()]
        ds = pd.concat([ds, dr], ignore_index=True) if len(ds) \
            else dr.copy()
    ds["window"] = ("fallback_2020-01_2026-08" if used_fallback
                    else "primary_2021-01_2026-04")
    ds.to_csv(out_dir / "macro_event_dataset.csv", index=False)

    if run_gq4 and n_measured:
        m_cols = (["event_id", "family", "ny_date", "ny_year",
                   "direction", "surprise", "surprise_std", "mid_t0",
                   f"mid_entry_{PRIMARY_CELL[0]}",
                   f"mid_exit_{PRIMARY_CELL[1]}"]
                  + [f"gross_{b}_{h}" for b in BUFFERS_S
                     for h in HORIZONS_S]
                  + [f"net_{b}_{h}" for b in BUFFERS_S
                     for h in HORIZONS_S]
                  + ["mfe_primary", "mae_primary", "t0_jump"])
        m[m_cols].to_csv(out_dir / "post_latency_returns.csv", index=False)
    (out_dir / "clock_sync_audit.txt").write_text("\n".join(audit),
                                                  encoding="utf-8")

    overall = bool(gq1_pass and gq2_pass and gq3_pass and gq4_pass)

    rep += ["=" * 70,
            "VERDICTS (thresholds verbatim from "
            "GATE_MACRO_QUALIFICATION.txt)",
            "window analyzed: " + tag, "",
            "G-Q1 EVENT PROVENANCE: "
            + ("PASS (provisional - manual audit pending)"
               if gq1_pass else "FAIL"),
            "  (a) primary source archived+hashed, secondary source "
            "named: "
            + ("PASS" if gq1_a
               else "FAIL (raw/secondary_source.txt missing or empty)")]
    for fam, cov, expd, frac in cov_rows:
        verdict_b = "PASS" if frac >= COVERAGE_MIN else "FAIL"
        rep.append(f"  (b) {fam}: {cov}/{expd} months field-complete = "
                   f"{frac:.1%} -> {verdict_b} (>= 95%; official "
                   "schedule count approximated by calendar months, one "
                   "scheduled release per family per month - reconcile "
                   "against BLS/Census schedule pages in the manual "
                   "audit)")
    drop_codes = {}
    if len(dropped):
        for _, r in dropped.iterrows():
            c = r["exclusion_code_fields"] or r["exclusion_code_join"]
            drop_codes[c] = drop_codes.get(c, 0) + 1
    rep.append(f"  excluded events by reason code: {drop_codes}")
    rep.append("  (c) agency/ALFRED manual audit of the sampled events: "
               "PENDING HUMAN VERIFICATION (sample + pointers written to "
               "clock_sync_audit.txt; each discrepancy removes the "
               "event; >3 of 12 discrepant -> G-Q1 FAIL)")
    rep.append("  (d) cleaned dataset written: macro_event_dataset.csv "
               "(PASS)")
    rep.append("")
    rep.append("G-Q2 CLOCK SYNCHRONIZATION: "
               + ("PASS (provisional - 6 manual spot-checks pending)"
                  if gq2_pass else "FAIL"))
    rep.append(f"  DST-invariant violations: {n_inv_viol} of {len(w)} "
               f"window events ({frac_viol:.1%}); systematic threshold "
               ">10% -> "
               + ("SYSTEMATIC FAIL (tick join not attempted)"
                  if systematic else "not triggered"))
    n_iso = int((w["exclusion_code_join"] == "dst_invariant").sum()) \
        if len(w) else 0
    n_sim = int((w["exclusion_code_join"] == "simultaneous").sum()) \
        if len(w) else 0
    rep.append(f"  isolated invariant violations dropped: {n_iso}; "
               f"simultaneous-release events excluded: {n_sim}")
    rep.append(f"  tick-join exclusions by code: "
               + (str(code_counts) if code_counts else "none"))
    rep.append(f"  events with valid T0/buffer/exit ticks: {n_events}")
    rep.append("  6 tick-window spot-check transcripts: PENDING HUMAN "
               "VERIFICATION (clock_sync_audit.txt)")
    rep.append("")
    rep.append("G-Q3 STATISTICAL POWER: " + ("PASS" if gq3_pass
                                             else "FAIL"))
    rep.append(f"  usable events: {n_events} (>= {N_MIN_EVENTS} "
               "required); unique NY dates: "
               f"{n_dates} (>= {N_MIN_DATES} required)")
    rep.append(f"  pre-declared fallback window used: "
               + ("YES" if used_fallback else "NO"))
    rep.append("")
    rep.append("G-Q4 POST-LATENCY MEASUREMENT: "
               + ("PASS" if gq4_pass else
                  ("FAIL" if run_gq4 else "NOT RUN (prior gate failed)")))
    if run_gq4:
        pcol = f"gross_{PRIMARY_CELL[0]}_{PRIMARY_CELL[1]}"
        rep.append(f"  measured events: {n_measured} (zero-surprise "
                   f"excluded from directional measurement: {n_zero})")
        rep.append(f"  (i) PRIMARY cell T+{PRIMARY_CELL[0]}s->"
                   f"T+{PRIMARY_CELL[1]}s mean gross directional move: "
                   f"${primary_mean:.4f}/oz "
                   f"(gate >= ${GATE_MIN_PRIMARY_MEAN:.2f} = 2.0 x "
                   f"${FLOOR:.2f} floor) -> "
                   + ("PASS" if gq4_i else "FAIL"))
        sec = ", ".join(
            f"T+{b}s->T+{h}s ${m[f'gross_{b}_{h}'].mean():.4f}/oz"
            for b in BUFFERS_S for h in HORIZONS_S
            if (b, h) != PRIMARY_CELL)
        rep.append("  secondary cells (reported, non-binding): " + sec)
        fam_str = ", ".join(f"{f} ${fam_means[f]:.4f}"
                            for f in FAMILIES)
        ex_str = (f"${pooled_ex_best:.4f} > 0"
                  if np.isfinite(pooled_ex_best) else "n/a")
        rep.append(f"  (ii) family robustness: means {fam_str}; positive "
                   f"in {n_fam_pos}/3 (>= 2 required); pooled ex-best "
                   f"({best_fam}) {ex_str} -> "
                   + ("PASS" if gq4_ii else "FAIL"))
        yr_str = ", ".join(f"{y} {s_:.0%}"
                           for y, s_ in yr_share.items())
        rep.append(f"  (iii) year robustness: positive-move shares by "
                   f"year: {yr_str} -> "
                   + ("PASS" if gq4_iii else
                      f"FAIL (year(s) {year_viol} exceed 50%)"))
        jump_str = ", ".join(
            f"{f} ${m.loc[m['family'] == f, 't0_jump'].mean():.4f}"
            if (m['family'] == f).any()
            and len(m.loc[m['family'] == f]) else f"{f} n/a"
            for f in FAMILIES)
        rep.append("  contemporaneous T0->T+30s jump per family "
                   "(informational, never credited): " + jump_str)
        rep.append(f"  net-of-spread primary cell (informational): "
                   f"${m['net_30_300'].mean():.4f}/oz; mean MFE "
                   f"${m['mfe_primary'].mean():.4f}; mean MAE "
                   f"${m['mae_primary'].mean():.4f}")
    rep.append("")

    rep.append("=" * 70)
    if overall:
        rep.append("QUALIFICATION VERDICT: PASS on all automated gate "
                   "clauses.")
        rep.append("Remaining before the four-gate PASS is final: "
                   "complete the sampled-event provenance audit and the "
                   "clock-sync spot-checks in clock_sync_audit.txt (any "
                   "timezone ambiguity -> G-Q2 HARD FAIL; >3 of 12 "
                   "provenance discrepancies -> G-Q1 FAIL). Only then "
                   "may a Phase-1 strategy SPEC be drafted (its own "
                   "pre-declared SPEC; nothing here authorizes one).")
    else:
        failed = [n for n, ok in (("G-Q1", gq1_pass),
                                  ("G-Q2", gq2_pass),
                                  ("G-Q3", gq3_pass),
                                  ("G-Q4", gq4_pass)) if not ok] \
            if run_gq4 else \
            [n for n, ok in (("G-Q1", gq1_pass), ("G-Q2", gq2_pass),
                             ("G-Q3", gq3_pass)) if not ok]
        rep.append("QUALIFICATION VERDICT: FAIL (failed gates: "
                   + ", ".join(failed)
                   + (")" if run_gq4
                      else "; G-Q4 not run per the SPEC stop rule)"))
        rep.append("Per SPEC section 7 this is a COMPLETE deliverable "
                   "and terminates the hypothesis: no family "
                   "substitution, no new buffers/horizons/thresholds, "
                   "no source switch, no 19th invented pattern. The "
                   "project accepts single-instrument CFD-tape "
                   "intraday exhaustion; remaining shortlist paths are "
                   "the daily-horizon candidates of "
                   "LITERATURE_EDGE_REVIEW.txt.")
    (out_dir / "REPORT_MACRO_QUALIFICATION.txt").write_text(
        "\n".join(rep), encoding="utf-8")
    print("\n".join(rep[-12:]))
    print("artifacts written to " + str(out_dir))
    print("elapsed "
          + str(int((datetime.now(UTC) - t_start).total_seconds())) + "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
