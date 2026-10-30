"""Hypothesis #21: causal HAR volatility scaling of the frozen Model=4 book.

This module deliberately contains no direction model and no MT5 calls.  It
uses the already executed one-lot ORB/SP2L trades, scales both legs by a
common volatility multiplier, and keeps the 2025--2026 overlay sealed until
all outer-validation gates pass.  The final study report is written by the
owner after reviewing the machine-readable outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "data" / "results" / "har_risk_scaling_model4_v1"
M5_PATH = ROOT / "data" / "results" / "orb_2021_present" / "us500tick_m5.csv"
COMMON_DAYS_PATH = ROOT / "data" / "results" / "orb_2021_present" / "mt5_days.csv"
ORB_TRADES = ROOT / "data" / "results" / "tick_model_revalidation_children" / "orb_model4_trades.csv"
SP2L_TRADES = ROOT / "data" / "results" / "tick_model_revalidation_children" / "sp2l_model4_trades.csv"
SPEC_PATH = STUDY / "SPEC_HAR_RISK_SCALING_MODEL4.txt"
GATE_PATH = STUDY / "GATE_HAR_RISK_SCALING_MODEL4.txt"

VAL_END = pd.Timestamp("2024-12-31", tz="America/New_York")
HOLDOUT_START = pd.Timestamp("2025-01-01", tz="America/New_York")
MIN_TRAIN = 252
SEED = 20260911
BLOCK = 20
BASE_ACCOUNT = 10000.0
REAL_SPREAD = 0.510
INCR_15 = REAL_SPREAD * 0.5
INCR_20 = REAL_SPREAD


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protocol_gate() -> tuple[bool, list[str]]:
    if not SPEC_PATH.is_file() or not GATE_PATH.is_file():
        return False, ["SPEC or GATE missing"]
    spec_lines = [line for line in SPEC_PATH.read_text(encoding="utf-8").splitlines()
                  if line.startswith(("V-", "H-"))]
    gate_lines = GATE_PATH.read_text(encoding="utf-8").splitlines()
    errors = [] if spec_lines == gate_lines else ["GATE is not the verbatim ordered SPEC extraction"]
    return not errors, errors


def _date_key(value) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def load_common_dates(path: Path = COMMON_DAYS_PATH) -> pd.DatetimeIndex:
    d = pd.read_csv(path, sep=";", dtype=str)
    for col in ("ny_date", "prev_close", "session_open"):
        if col not in d:
            raise ValueError(f"common-day file missing {col}")
    prev = pd.to_numeric(d["prev_close"], errors="coerce")
    op = pd.to_numeric(d["session_open"], errors="coerce")
    keep = (prev > 0) | (op > 0)
    dates = pd.to_datetime(d.loc[keep, "ny_date"], format="%Y%m%d")
    dates = pd.DatetimeIndex(dates).tz_localize("America/New_York").normalize()
    if len(dates) != len(set(dates)) or not dates.is_monotonic_increasing:
        raise ValueError("common dates are not unique and ordered")
    return dates


def load_m5(path: Path = M5_PATH) -> pd.DataFrame:
    d = pd.read_csv(path, usecols=["time", "open", "high", "low", "close"])
    if d.empty:
        raise ValueError("empty M5 input")
    ts = pd.to_datetime(pd.to_numeric(d["time"], errors="raise"), unit="s", utc=True)
    if ts.duplicated().any() or not ts.is_monotonic_increasing:
        raise ValueError("M5 timestamps are not strictly ordered")
    for col in ("open", "high", "low", "close"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    if not np.isfinite(d[["open", "high", "low", "close"]].to_numpy()).all():
        raise ValueError("non-finite M5 OHLC")
    if (d[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("non-positive M5 OHLC")
    d["ts"] = ts
    d["ny_date"] = ts.dt.tz_convert("America/New_York").dt.normalize()
    d["wall"] = ts.dt.tz_convert("America/New_York").dt.hour * 60 + ts.dt.tz_convert("America/New_York").dt.minute
    return d


def build_cash_rv(m5: pd.DataFrame, common_dates: Iterable[pd.Timestamp], strict: bool = True) -> pd.DataFrame:
    """Build RV from 09:30 open followed by every close through 15:55."""
    out = []
    for day in pd.DatetimeIndex(common_dates):
        rows = m5[(m5["ny_date"] == day) & (m5["wall"] >= 570) & (m5["wall"] < 960)]
        if rows.empty:
            if strict:
                raise ValueError(f"missing cash bars for {day.date()}")
            out.append({"date": day, "rv": np.nan, "n_bars": 0, "valid": False, "error": "missing cash bars"})
            continue
        open_rows = rows[rows["wall"] == 570]
        if len(open_rows) != 1 or len(rows) < 70:
            if strict:
                raise ValueError(f"invalid cash session {day.date()}: bars={len(rows)} open={len(open_rows)}")
            out.append({"date": day, "rv": np.nan, "n_bars": int(len(rows)), "valid": False,
                        "error": f"bars={len(rows)} open={len(open_rows)}"})
            continue
        if rows["ts"].duplicated().any() or not rows["ts"].is_monotonic_increasing:
            if strict:
                raise ValueError(f"unordered cash bars {day.date()}")
            out.append({"date": day, "rv": np.nan, "n_bars": int(len(rows)), "valid": False, "error": "unordered cash bars"})
            continue
        p = np.concatenate(([float(open_rows.iloc[0]["open"])], rows["close"].to_numpy(dtype=float)))
        if not np.isfinite(p).all() or (p <= 1e-12).any():
            if strict:
                raise ValueError(f"invalid cash path {day.date()}")
            out.append({"date": day, "rv": np.nan, "n_bars": int(len(rows)), "valid": False, "error": "invalid cash path"})
            continue
        rv = float(np.square(np.diff(np.log(np.maximum(p, 1e-12)))).sum())
        if not np.isfinite(rv) or rv <= 0:
            if strict:
                raise ValueError(f"non-positive RV {day.date()}")
            out.append({"date": day, "rv": np.nan, "n_bars": int(len(rows)), "valid": False, "error": "non-positive RV"})
            continue
        out.append({"date": day, "rv": rv, "n_bars": int(len(rows)), "valid": True, "error": ""})
    result = pd.DataFrame(out).set_index("date").sort_index()
    return result


def load_trades(path: Path, strict: bool = True) -> pd.DataFrame:
    d = pd.read_csv(path, sep=";")
    required = {"open_time", "close_time", "profit"}
    if not required.issubset(d.columns):
        raise ValueError(f"trade file missing {required - set(d.columns)}")
    d["open_ts"] = pd.to_datetime(d["open_time"], format="%Y.%m.%d %H:%M", utc=True)
    d["close_ts"] = pd.to_datetime(d["close_time"], format="%Y.%m.%d %H:%M", utc=True)
    d["open_date"] = d["open_ts"].dt.tz_convert("America/New_York").dt.normalize()
    d["close_date"] = d["close_ts"].dt.tz_convert("America/New_York").dt.normalize()
    d["profit"] = pd.to_numeric(d["profit"], errors="coerce")
    if d[["profit"]].isna().any().any() or not np.isfinite(d["profit"]).all():
        raise ValueError(f"non-finite trade profit in {path}")
    errors = []
    # Tester exports have minute precision, so equal minute stamps can be a
    # valid sub-minute trade. Only a strictly earlier close is contradictory.
    negative_duration = int((d["open_ts"] > d["close_ts"]).sum())
    overnight = int((d["open_date"] != d["close_date"]).sum())
    if negative_duration:
        errors.append("negative trade duration")
    if overnight:
        errors.append("overnight trade")
    if errors and strict:
        raise ValueError(f"{'; '.join(errors)} in {path}")
    d.attrs["validation_errors"] = errors
    d.attrs["validation_counts"] = {
        "negative_duration": negative_duration,
        "overnight": overnight,
        "equal_minute_timestamps": int((d["open_ts"] == d["close_ts"]).sum()),
    }
    return d


def _features_for_target(log_rv: np.ndarray, target: int) -> np.ndarray:
    return np.array([log_rv[target - 1], log_rv[target - 5:target].mean(), log_rv[target - 22:target].mean()], dtype=float)


def fit_har_forecasts(rv: pd.Series, min_train: int = MIN_TRAIN) -> pd.DataFrame:
    """Expanding, one-day-ahead HAR forecasts and frozen benchmarks."""
    vals = rv.to_numpy(dtype=float)
    log_rv = np.log(vals)
    rows = []
    # target i has features ending at i-1; all j < i are matured targets.
    for i in range(22, len(vals)):
        j = np.arange(22, i, dtype=int)
        if len(j) < min_train:
            continue
        X = np.vstack([_features_for_target(log_rv, int(k)) for k in j])
        y = log_rv[j]
        A = np.column_stack([np.ones(len(X)), X])
        beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        dof = max(1, len(y) - A.shape[1])
        resid_var = float(np.dot(resid, resid) / dof)
        xt = _features_for_target(log_rv, i)
        pred_log = float(np.r_[1.0, xt] @ beta)
        har = float(np.exp(pred_log + 0.5 * resid_var))
        b1 = float(vals[i - 1])
        b22 = float(vals[i - 22:i].mean())
        med = float(np.median(vals[:i]))
        rows.append({"date": rv.index[i], "rv": vals[i], "har_forecast": har, "b1_forecast": b1,
                     "b22_forecast": b22, "training_median_rv": med, "training_rows": len(j),
                     "residual_var": resid_var})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def multiplier(median_rv: float, forecast_rv: float) -> float:
    if not np.isfinite(median_rv) or not np.isfinite(forecast_rv) or median_rv <= 0 or forecast_rv <= 0:
        raise ValueError("invalid volatility scaling inputs")
    return float(np.clip(math.sqrt(median_rv / forecast_rv), 0.25, 1.0))


def add_multipliers(forecasts: pd.DataFrame) -> pd.DataFrame:
    d = forecasts.copy()
    d["har_multiplier"] = [multiplier(m, f) for m, f in zip(d.training_median_rv, d.har_forecast)]
    d["b1_multiplier"] = [multiplier(m, f) for m, f in zip(d.training_median_rv, d.b1_forecast)]
    d["b22_multiplier"] = [multiplier(m, f) for m, f in zip(d.training_median_rv, d.b22_forecast)]
    return d


def _trade_daily(d: pd.DataFrame, dates: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    pnl = d.groupby("open_date")["profit"].sum().reindex(dates, fill_value=0.0)
    counts = d.groupby("open_date").size().reindex(dates, fill_value=0).astype(float)
    return pnl.astype(float), counts


def build_portfolio(forecasts: pd.DataFrame, orb: pd.DataFrame, sp2l: pd.DataFrame, dates: pd.DatetimeIndex,
                    spread_increment: float = 0.0) -> pd.DataFrame:
    """Return fixed, B1, B22 and HAR daily PnL using the same dates."""
    idx = pd.DatetimeIndex(dates)
    op, oc = _trade_daily(orb, idx)
    sp, sc = _trade_daily(sp2l, idx)
    f = forecasts.reindex(idx).copy()
    for col in ("har_multiplier", "b1_multiplier", "b22_multiplier"):
        f[col] = f[col].fillna(1.0)
    base = op * 0.5 + sp * 0.5
    count = (oc + sc) * 0.5
    for name, mult_col in (("fixed", None), ("b1", "b1_multiplier"), ("b22", "b22_multiplier"), ("har", "har_multiplier")):
        mult = pd.Series(1.0, index=idx) if mult_col is None else f[mult_col]
        f[f"{name}_multiplier"] = mult
        f[f"{name}_pnl"] = base * mult - count * mult * spread_increment
    f["orb_trade_count"] = oc
    f["sp2l_trade_count"] = sc
    f["total_trade_count"] = oc + sc
    return f


def metrics(series: pd.Series) -> dict:
    x = pd.to_numeric(series, errors="raise").astype(float)
    total = float(x.sum())
    sd = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
    sharpe = float(x.mean() / sd * math.sqrt(252)) if np.isfinite(sd) and sd > 0 else None
    cum = x.cumsum()
    dd = float((cum.cummax() - cum).max()) if len(x) else 0.0
    year = x.groupby(x.index.year).sum() if len(x) else pd.Series(dtype=float)
    pos = year[year > 0]
    return {"total_pnl": round(total, 6), "sharpe": None if sharpe is None else round(sharpe, 6),
            "max_dd_usd": round(dd, 6), "max_dd_pct": round(dd / BASE_ACCOUNT * 100, 6),
            "worst_day": round(float(x.min()), 6) if len(x) else None,
            "trade_days": int((x != 0).sum()), "calendar_years": {str(k): round(float(v), 6) for k, v in year.items()},
            "positive_years": int((year > 0).sum()),
            "best_positive_year_share": round(float(pos.max() / pos.sum()), 6) if len(pos) and pos.sum() > 0 else None}


def qlike(y: pd.Series, f: pd.Series) -> float:
    yy, ff = np.asarray(y, dtype=float), np.asarray(f, dtype=float)
    if len(yy) == 0 or not np.isfinite(yy).all() or not np.isfinite(ff).all() or (yy <= 0).any() or (ff <= 0).any():
        raise ValueError("QLIKE requires finite positive values")
    return float(np.mean(yy / ff - np.log(yy / ff) - 1.0))


def forecast_stats(f: pd.DataFrame) -> dict:
    out = {}
    for name in ("har", "b1", "b22"):
        pred = f[f"{name}_forecast"]
        real = f["rv"]
        out[name] = {"mean_qlike": qlike(real, pred),
                     "log_mse": float(np.mean((np.log(real) - np.log(pred)) ** 2)),
                     "spearman": float(real.corr(pred, method="spearman")), "n": int(len(f))}
    q = f["har_forecast"].rank(method="first", pct=True)
    top, bot = f.loc[q > 0.8, "rv"].mean(), f.loc[q <= 0.2, "rv"].mean()
    out["har"]["top_bottom_realized_rv_ratio"] = float(top / bot) if bot > 0 else None
    return out


def bootstrap_sharpe_probability(har: pd.Series, benchmark: pd.Series, n: int = 10000,
                                 block: int = BLOCK, seed: int = SEED) -> float:
    a, b = np.asarray(har, dtype=float), np.asarray(benchmark, dtype=float)
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("paired bootstrap requires equal non-trivial series")
    rng = np.random.default_rng(seed)
    nrows = len(a)
    diffs = np.empty(n, dtype=float)
    starts = np.arange(nrows)
    for k in range(n):
        chosen = []
        while len(chosen) < nrows:
            s = int(rng.choice(starts))
            chosen.extend((s + np.arange(block)) % nrows)
        ix = np.asarray(chosen[:nrows], dtype=int)
        xa, xb = a[ix], b[ix]
        sa, sb = xa.std(ddof=1), xb.std(ddof=1)
        sha = xa.mean() / sa * math.sqrt(252) if sa > 0 else 0.0
        shb = xb.mean() / sb * math.sqrt(252) if sb > 0 else 0.0
        diffs[k] = sha - shb
    return float(np.mean(diffs <= 0.0))


def prefix_invariance(rv: pd.DataFrame, forecasts: pd.DataFrame, portfolio: pd.DataFrame,
                      orb: pd.DataFrame, sp2l: pd.DataFrame) -> bool:
    """Check that a validation prefix gives the same earlier causal outputs."""
    val_dates = forecasts.index[forecasts.index <= VAL_END]
    if len(val_dates) < 10:
        return True
    cut = val_dates[len(val_dates) // 2]
    prefix_rv = rv.loc[rv.index <= cut, "rv"]
    pf = add_multipliers(fit_har_forecasts(prefix_rv))
    compare = forecasts.loc[forecasts.index <= cut]
    if len(pf) != len(compare) or not np.allclose(pf[["har_forecast", "b1_forecast", "b22_forecast", "har_multiplier", "b1_multiplier", "b22_multiplier"]].to_numpy(), compare[["har_forecast", "b1_forecast", "b22_forecast", "har_multiplier", "b1_multiplier", "b22_multiplier"]].to_numpy(), rtol=1e-10, atol=1e-12):
        return False
    pfx = build_portfolio(pf, orb, sp2l, pf.index)
    full = portfolio.loc[portfolio.index <= cut]
    cols = ["fixed_pnl", "b1_pnl", "b22_pnl", "har_pnl"]
    return len(pfx) == len(full) and np.allclose(pfx[cols].to_numpy(), full[cols].to_numpy(), rtol=1e-10, atol=1e-10)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
        out.reset_index(drop=True, inplace=True)
    out.to_csv(path, index=False)


def run(output_dir: Path = STUDY, allow_network: bool = False) -> dict:
    """Execute Phase 1 and write machine-readable artifacts (never a report)."""
    protocol_ok, protocol_errors = protocol_gate()
    if not SPEC_PATH.exists() or not GATE_PATH.exists():
        raise RuntimeError("frozen HAR SPEC/GATE are required before implementation")
    if allow_network:
        raise ValueError("network access is forbidden for this offline study")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = [M5_PATH, COMMON_DAYS_PATH, ORB_TRADES, SP2L_TRADES]
    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in inputs}
    (output_dir / "input_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    common = load_common_dates()
    m5 = load_m5()
    universe_ok = bool(len(common) == 1369
                       and common[0] == pd.Timestamp("2021-01-05", tz="America/New_York")
                       and common[-1] == pd.Timestamp("2026-04-30", tz="America/New_York"))
    if not universe_ok:
        raise ValueError("common RV universe does not match frozen 1369-day universe")
    # The holdout overlay is sealed. A validation failure can be proven from
    # the pre-2025 prefix, so do not construct or write holdout RV outcomes.
    validation_common = common[common <= VAL_END]
    rv = build_cash_rv(m5, validation_common, strict=False)
    _write_csv(rv, output_dir / "daily_rv.csv")
    orb, sp2l = load_trades(ORB_TRADES, strict=False), load_trades(SP2L_TRADES, strict=False)
    d1, d2 = orb["profit"].sum(), sp2l["profit"].sum()
    trade_ok = not orb.attrs.get("validation_errors") and not sp2l.attrs.get("validation_errors")
    vd1 = bool(protocol_ok and universe_ok and trade_ok and len(orb) == 437
               and abs(d1 - 114.91) <= 0.01 and len(sp2l) == 1349
               and abs(d2 - 138.65) <= 0.01)
    rv_valid = bool(rv["valid"].all() and rv["rv"].notna().all())
    if not rv_valid or not vd1:
        gate = {"V-I1": protocol_ok, "V-D1": bool(vd1), "V-D2": False, "V-F1": False,
                "V-E1": False, "V-E2": False, "V-C1": False, "V-FINAL": False}
        failed = [name for name in ("V-I1", "V-D1", "V-D2") if not gate[name]]
        result = {"status": "FAIL", "gates": gate, "input_hashes": hashes,
                  "common_universe_rows": len(common),
                  "common_universe_first_eligible": str(common[0].date()),
                  "common_universe_last_eligible": str(common[-1].date()),
                  "validation_rv_rows": len(rv),
                  "invalid_validation_rv_days": rv.index[~rv["valid"]].strftime("%Y-%m-%d").tolist(),
                  "invalid_validation_rv_bar_counts": {
                      str(int(k)): int(v) for k, v in rv.loc[~rv["valid"], "n_bars"].value_counts().sort_index().items()},
                  "trade_validation_errors": {"orb": orb.attrs.get("validation_errors", []), "sp2l": sp2l.attrs.get("validation_errors", [])},
                  "trade_validation_counts": {"orb": orb.attrs.get("validation_counts", {}), "sp2l": sp2l.attrs.get("validation_counts", {})},
                  "protocol_errors": protocol_errors,
                  "validation_rows": 0, "holdout_overlay_scored": False,
                  "holdout_files_written": False,
                  "stop_reason": f"{', '.join(failed)} failed; no model or holdout overlay evaluation performed"}
        (output_dir / "validation_forecasts.csv").write_text("date\n", encoding="utf-8")
        (output_dir / "validation_daily_pnl.csv").write_text("date\n", encoding="utf-8")
        (output_dir / "validation_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    # Reaching this branch means the validation prefix is clean. Construct
    # the full RV series only now so a passing validation can conditionally
    # open the overlay holdout under the frozen rules.
    rv_full = build_cash_rv(m5, common, strict=True)
    f = add_multipliers(fit_har_forecasts(rv_full["rv"]))
    val = f.loc[(f.index <= VAL_END)].copy()
    if val.empty:
        raise ValueError("no validation forecasts")
    p = build_portfolio(val, orb, sp2l, val.index)
    _write_csv(val, output_dir / "validation_forecasts.csv")
    _write_csv(p, output_dir / "validation_daily_pnl.csv")
    fs = forecast_stats(val)
    primary = "b1" if fs["b1"]["mean_qlike"] <= fs["b22"]["mean_qlike"] else "b22"
    gate = {
        "V-I1": protocol_ok, "V-D1": bool(vd1), "V-D2": bool(prefix_invariance(rv_full, f, build_portfolio(f, orb, sp2l, f.index), orb, sp2l)),
        "V-F1": bool(fs["har"]["mean_qlike"] <= 0.98 * fs["b1"]["mean_qlike"] and fs["har"]["mean_qlike"] <= 0.98 * fs["b22"]["mean_qlike"] and fs["har"]["spearman"] >= 0.25 and (fs["har"].get("top_bottom_realized_rv_ratio") or 0) >= 1.5),
    }
    # Reindex each policy to identical outer-validation rows.
    pm = {name: metrics(p[f"{name}_pnl"]) for name in ("fixed", "b1", "b22", "har")}
    p15 = build_portfolio(val, orb, sp2l, val.index, INCR_15)
    p20 = build_portfolio(val, orb, sp2l, val.index, INCR_20)
    sm = {name: metrics(p15[f"{name}_pnl"]) for name in ("fixed", "b1", "b22", "har")}
    dm = {name: metrics(p20[f"{name}_pnl"]) for name in ("fixed", "b1", "b22", "har")}
    har_sh, fixed_sh, b22_sh = pm["har"]["sharpe"] or -np.inf, pm["fixed"]["sharpe"] or -np.inf, pm["b22"]["sharpe"] or -np.inf
    best_share = pm["har"]["best_positive_year_share"]
    prob = bootstrap_sharpe_probability(p["har_pnl"], p[f"{primary}_pnl"])
    gate["V-E1"] = bool(pm["har"]["total_pnl"] > 0 and har_sh >= fixed_sh + 0.05 and har_sh >= b22_sh + 0.05 and pm["har"]["max_dd_usd"] <= 0.85 * pm["fixed"]["max_dd_usd"] and pm["har"]["positive_years"] >= 2)
    gate["V-E2"] = bool(prob <= 0.10 and best_share is not None and best_share <= 0.60)
    gate["V-C1"] = bool(sm["har"]["total_pnl"] > 0 and (sm["har"]["sharpe"] or -np.inf) > 0 and sm["har"]["max_dd_usd"] <= sm["fixed"]["max_dd_usd"])
    gate["V-FINAL"] = all(gate.values())
    result = {"status": "PASS" if gate["V-FINAL"] else "FAIL", "gates": gate, "primary_benchmark": primary,
              "input_hashes": hashes, "rv_rows": len(rv_full), "validation_rows": len(val), "forecast_stats": fs,
              "validation_metrics": pm, "stress_1_5x_metrics": sm, "stress_2_0x_metrics": dm,
              "bootstrap_probability_har_le_primary": prob, "prefix_invariance": gate["V-D2"]}
    (output_dir / "validation_metrics.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    if gate["V-FINAL"]:
        hold = f.loc[f.index >= HOLDOUT_START].copy()
        hp = build_portfolio(hold, orb, sp2l, hold.index)
        _write_csv(hold, output_dir / "holdout_forecasts.csv")
        _write_csv(hp, output_dir / "holdout_daily_pnl.csv")
        (output_dir / "holdout_metrics.json").write_text(json.dumps({k: metrics(hp[f"{k}_pnl"]) for k in ("fixed", "b1", "b22", "har")}, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2, default=str))
    raise SystemExit(0 if r["status"] == "PASS" else 2)
