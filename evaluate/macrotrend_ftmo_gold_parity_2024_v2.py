"""Independent, frozen-input FTMO Gold EA signal and stop parity audit."""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/results/macrotrend_ftmo_gold_parity_2024_v2"
TERMINAL = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
LOG = Path(r"C:\Users\Milad\AppData\Roaming\MetaQuotes\Tester\81A933A9AFC5DE3C23B15CAB19C63850\Agent-127.0.0.1-3000\logs\20260929.log")
START = {"A": "18:29:27.", "B": "18:37:00."}
CHART = {"A": "H4", "B": "H1"}
EMPTY = 1.7976931348623157e308


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nth_sunday(year: int, month: int, n: int) -> datetime:
    first = datetime(year, month, 1)
    return first + timedelta(days=(6 - first.weekday()) % 7 + 7 * (n - 1))


def last_sunday(year: int, month: int) -> datetime:
    following = datetime(year + (month == 12), month % 12 + 1, 1)
    last = following - timedelta(days=1)
    return last - timedelta(days=(last.weekday() + 1) % 7)


def wall_buckets(server_seconds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h4, daily = [], []
    for second in server_seconds:
        server = datetime.fromtimestamp(int(second), timezone.utc).replace(tzinfo=None)
        summer = server - timedelta(hours=3)
        eu_start = last_sunday(summer.year, 3) + timedelta(hours=1)
        eu_end = last_sunday(summer.year, 10) + timedelta(hours=1)
        utc = summer if eu_start <= summer < eu_end else server - timedelta(hours=2)
        us_start = nth_sunday(utc.year, 3, 2) + timedelta(hours=7)
        us_end = nth_sunday(utc.year, 11, 1) + timedelta(hours=6)
        wall = utc - timedelta(hours=4 if us_start <= utc < us_end else 5)
        floor4 = wall.replace(hour=wall.hour // 4 * 4, minute=0, second=0)
        day = wall.replace(hour=17, minute=0, second=0)
        if wall < day:
            day -= timedelta(days=1)
        h4.append(int(floor4.replace(tzinfo=timezone.utc).timestamp()))
        daily.append(int(day.replace(tzinfo=timezone.utc).timestamp()))
    return np.asarray(h4, dtype=np.int64), np.asarray(daily, dtype=np.int64)


def grid_group(frame: pd.DataFrame, keys: np.ndarray) -> pd.DataFrame:
    work = frame.assign(bucket=keys)
    return work.groupby("bucket", sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), real=("real", "max")
    )


def ema(values: np.ndarray, period: int) -> np.ndarray:
    out = np.empty(len(values), dtype=float)
    out[0] = values[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def true_range(frame: pd.DataFrame) -> np.ndarray:
    high, low, close = (frame[x].to_numpy(dtype=float) for x in ("high", "low", "close"))
    tr = high - low
    tr[1:] = np.maximum.reduce([tr[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])])
    return tr


def atr(frame: pd.DataFrame, period: int, sma_seed: bool) -> np.ndarray:
    tr = true_range(frame)
    out = np.empty(len(tr), dtype=float)
    if sma_seed:
        out[:period-1] = np.nan
        out[period-1] = np.mean(tr[:period])
        begin = period
    else:
        out[0] = tr[0]
        begin = 1
    for i in range(begin, len(tr)):
        out[i] = (out[i-1] * (period-1) + tr[i]) / period
    return out


def adx(frame: pd.DataFrame, period: int) -> np.ndarray:
    high, low = (frame[x].to_numpy(dtype=float) for x in ("high", "low"))
    tr = true_range(frame)
    out = np.full(len(tr), np.nan)
    tr_sm = tr[0]
    plus = minus = 0.0
    value = 0.0
    seeded = False
    for i in range(len(tr)):
        pdm = mdm = 0.0
        if i:
            up, down = high[i]-high[i-1], low[i-1]-low[i]
            if up > down and up > 0:
                pdm = up
            if down > up and down > 0:
                mdm = down
            tr_sm = (tr_sm * (period-1) + tr[i]) / period
            plus = (plus * (period-1) + pdm) / period
            minus = (minus * (period-1) + mdm) / period
        if tr_sm <= 0 or plus+minus <= 0:
            continue
        dx = 100 * abs(plus-minus) / (plus+minus)
        value = ((value * (period-1) + dx) / period) if seeded else dx
        seeded = True
        out[i] = value
    return out


def supertrend(frame: pd.DataFrame, period: int, mult: float) -> np.ndarray:
    a = atr(frame, period, False)
    high, low, close = (frame[x].to_numpy(dtype=float) for x in ("high", "low", "close"))
    direction = np.empty(len(a), dtype=int)
    mid = (high + low) / 2
    upper, lower = mid[0] + mult*a[0], mid[0] - mult*a[0]
    direction[0] = -1
    for i in range(1, len(a)):
        basic_u, basic_l = mid[i]+mult*a[i], mid[i]-mult*a[i]
        prev_u, prev_l = upper, lower
        upper = basic_u if basic_u < prev_u or close[i-1] > prev_u else prev_u
        lower = basic_l if basic_l > prev_l or close[i-1] < prev_l else prev_l
        direction[i] = (1 if close[i] > upper else -1) if direction[i-1] == -1 else (-1 if close[i] < lower else 1)
    return direction


def tick_round(value: float) -> float:
    return math.floor(value * 100 + 0.5) / 100


def extract_log() -> dict[str, list[str]]:
    lines = LOG.read_text(encoding="utf-16", errors="replace").splitlines()
    segments = {}
    for engine in ("A", "B"):
        chart = CHART[engine]
        starts = [i for i, line in enumerate(lines) if START[engine] in line and
                  f"XAUUSD,{chart}: testing of Experts\\MacroTrendEnsemble_FTMO.ex5" in line]
        if len(starts) != 1:
            raise RuntimeError(f"Expected one {engine} tester start; found {len(starts)}")
        start = starts[0]
        ends = [i for i in range(start+1, len(lines)) if
                f"test Experts\\MacroTrendEnsemble_FTMO.ex5 on XAUUSD,{chart} thread finished" in lines[i]]
        if not ends:
            raise RuntimeError(f"Missing {engine} test finish")
        segment = lines[start:ends[0]+1]
        (OUT / f"mt5_{engine}_journal_snapshot.txt").write_text("\n".join(segment)+"\n", encoding="utf-8")
        segments[engine] = segment
    return segments


EVENT_TIME = re.compile(r"(20\d\d\.\d\d\.\d\d \d\d:\d\d:\d\d)\s+MTE:")
OPEN = re.compile(r"MTE: OPEN (BUY|SELL) engine=([AB]) lots=([\d.]+) entry=([\d.]+) SL=([\d.]+) spread=([\d.]+)")
SKIP = re.compile(r"MTE: ([\d.]+) lots below broker minimum ([\d.]+); skip")
RATCHET = re.compile(r"MTE: ratcheted ticket=(\d+) SL ([\d.]+) -> ([\d.]+)")


def events(segments: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for engine, segment in segments.items():
        for line in segment:
            dt = EVENT_TIME.search(line)
            if not dt:
                continue
            stamp = dt.group(1).replace(".", "-")
            o, s, r = OPEN.search(line), SKIP.search(line), RATCHET.search(line)
            if o:
                rows.append(dict(engine=engine, time=stamp, kind="OPEN", direction=1 if o[1]=="BUY" else -1,
                                 lots=float(o[3]), entry=float(o[4]), sl=float(o[5]), spread=float(o[6]), line=line))
            elif s:
                rows.append(dict(engine=engine, time=stamp, kind="MIN_LOT_SKIP", lots=float(s[1]), line=line))
            elif r:
                rows.append(dict(engine=engine, time=stamp, kind="RATCHET", ticket=int(r[1]),
                                 old_sl=float(r[2]), sl=float(r[3]), line=line))
            elif any(word in line for word in ("PANIC_FLATTEN", "TRAIL_CROSSED", "MTE: close", "MTE: REJECT")):
                rows.append(dict(engine=engine, time=stamp, kind="OTHER", line=line))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "mt5_events.csv", index=False)
    return result


def load_rates() -> pd.DataFrame:
    if not mt5.initialize(path=TERMINAL):
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        raw = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_H1,
                                   datetime(2024, 1, 1, tzinfo=timezone.utc),
                                   datetime(2026, 9, 25, tzinfo=timezone.utc))
    finally:
        mt5.shutdown()
    if raw is None:
        raise RuntimeError("FTMO XAUUSD H1 export failed")
    rates = pd.DataFrame(raw).sort_values("time").drop_duplicates("time")
    rates.to_csv(OUT / "ftmo_xauusd_h1_snapshot.csv", index=False)
    return rates


def signal_engine(rates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = np.arange(int(rates.time.iloc[0]), int(rates.time.iloc[-1])+3600, 3600, dtype=np.int64)
    grid = rates.set_index("time").reindex(times)
    grid["real"] = grid.open.notna()
    for field in ("open", "high", "low", "close"):
        grid[field] = grid[field].ffill()
    grid.index.name = "server_time"
    h4key, daykey = wall_buckets(times)
    four, daily = grid_group(grid, h4key), grid_group(grid, daykey)
    four_keys, day_keys = four.index.to_numpy(), daily.index.to_numpy()
    four_close, day_close = four.close.to_numpy(), daily.close.to_numpy()
    h1_close, h1_high, h1_low = (grid[x].to_numpy() for x in ("close", "high", "low"))
    h1_real = grid.real.to_numpy(dtype=bool)
    four_real = four.real.to_numpy(dtype=bool)
    ema4, atr4 = ema(four_close, 29), atr(four, 29, True)
    ema_day = ema(day_close, 50)
    atr1, adx1, st4 = atr(grid, 14, False), adx(grid, 14), supertrend(four, 10, 3)
    # Last index strictly before this entry hour's own bucket.
    idx4 = np.searchsorted(four_keys, h4key, side="left") - 1
    idxd = np.searchsorted(day_keys, daykey, side="left") - 1
    records = []
    contexts = []
    previous_real_h4 = None
    for i, second in enumerate(times):
        if not h1_real[i]:
            continue
        stamp = datetime.fromtimestamp(int(second), timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        j4, jd = int(idx4[i]), int(idxd[i])
        prev = i-1
        context = dict(time=stamp, hour_index=i, h4_index=j4, day_index=jd,
                       h1_atr14=atr1[prev] if prev>=0 else np.nan,
                       h1_adx14=adx1[prev] if prev>=0 else np.nan,
                       h1_close=h1_close[prev] if prev>=0 else np.nan,
                       h4_ema29=ema4[j4] if j4>=0 else np.nan,
                       h4_atr29=atr4[j4] if j4>=0 else np.nan,
                       h4_close=four_close[j4] if j4>=0 else np.nan,
                       h4_supertrend=st4[j4] if j4>=0 else 0,
                       d1_ema50=ema_day[jd] if jd>=0 else np.nan,
                       d1_close=day_close[jd] if jd>=0 else np.nan,
                       h4_bucket_changed=(previous_real_h4 is not None and h4key[i]!=previous_real_h4))
        contexts.append(context)
        if context["h4_bucket_changed"] and j4>=28 and jd>=49 and four_real[j4] and atr4[j4]>0:
            side = (1 if four_close[j4] > ema4[j4]+2.77*atr4[j4] and day_close[jd]>ema_day[jd]
                    else -1 if four_close[j4] < ema4[j4]-2.77*atr4[j4] and day_close[jd]<ema_day[jd] else 0)
            if side:
                records.append(dict(engine="A", time=stamp, direction=side, anchor=ema4[j4], atr=atr4[j4],
                                    trigger_close=four_close[j4], filter_close=day_close[jd], filter_value=ema_day[jd]))
        if prev>=22 and h1_real[prev] and j4>=15 and np.isfinite(adx1[prev]) and adx1[prev]>=20:
            upper = np.max(h1_high[prev-20:prev])
            lower = np.min(h1_low[prev-20:prev])
            side = (1 if h1_close[prev]>upper and st4[j4]==1
                    else -1 if h1_close[prev]<lower and st4[j4]==-1 else 0)
            if side:
                records.append(dict(engine="B", time=stamp, direction=side, anchor=0, atr=atr1[prev],
                                    trigger_close=h1_close[prev], filter_close=st4[j4], filter_value=adx1[prev],
                                    donchian_high=upper, donchian_low=lower))
        previous_real_h4 = h4key[i]
    signals = pd.DataFrame(records)
    context = pd.DataFrame(contexts)
    signals.to_csv(OUT / "python_signals_2024_2026.csv", index=False)
    context.to_csv(OUT / "python_hour_context.csv", index=False)
    return signals, context


def compare(signals: pd.DataFrame, context: pd.DataFrame, mt_events: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    sig = signals.set_index(["engine", "time"])
    ctx = context.set_index("time")
    rows = []
    for _, event in mt_events.iterrows():
        item = event.to_dict()
        engine, stamp, kind = item["engine"], item["time"], item["kind"]
        hour = stamp[:14] + "00:00"
        item["matched_h1_hour"] = hour
        item.update(signal_match=False, direction_match=np.nan, predicted_sl=np.nan, sl_error=np.nan, verdict="UNTESTED")
        if kind in ("OPEN", "MIN_LOT_SKIP"):
            if (engine, hour) in sig.index:
                ref = sig.loc[(engine, hour)]
                item["signal_match"] = True
                item["python_direction"] = int(ref.direction)
                if kind == "OPEN":
                    item["direction_match"] = int(item["direction"]) == int(ref.direction)
                    distance = max(abs(item["entry"]-ref.anchor), 1.5*ref.atr) if engine=="A" else 4*ref.atr
                    item["predicted_sl"] = tick_round(item["entry"]-item["direction"]*distance)
                    item["sl_error"] = abs(item["sl"]-item["predicted_sl"])
                    item["verdict"] = "PASS" if item["direction_match"] and item["sl_error"]<=0.0100001 else "FAIL"
                else:
                    item["verdict"] = "PASS"
            else:
                item["verdict"] = "FAIL"
        elif kind == "RATCHET" and hour in ctx.index:
            ref = ctx.loc[hour]
            if engine == "A" and ref.h4_bucket_changed:
                item["predicted_sl"] = tick_round(ref.h4_ema29)
            elif engine == "B":
                side = 1 if item["sl"] > item["old_sl"] else -1
                item["predicted_sl"] = tick_round(ref.h1_close-side*4*ref.h1_atr14)
            if np.isfinite(item["predicted_sl"]):
                item["sl_error"] = abs(item["sl"]-item["predicted_sl"])
                item["verdict"] = "PASS" if item["sl_error"]<=0.0100001 else "FAIL"
            else:
                item["verdict"] = "FAIL"
        elif kind == "RATCHET":
            item["verdict"] = "FAIL"
        rows.append(item)
    matched = pd.DataFrame(rows)
    matched.to_csv(OUT / "event_comparison.csv", index=False)
    summary = {}
    for engine, expected_open, expected_skip in (("A",23,265),("B",106,248)):
        ev = matched[matched.engine==engine]
        opens = ev[ev.kind=="OPEN"]
        skips = ev[ev.kind=="MIN_LOT_SKIP"]
        ratchets = ev[ev.kind=="RATCHET"]
        python = signals[(signals.engine==engine)&(signals.time>="2025-01-01")]
        summary[engine] = dict(mt5_open=len(opens), mt5_min_lot_skip=len(skips), mt5_ratchets=len(ratchets),
                               mt5_other=len(ev[ev.kind=="OTHER"]), python_2024_signals=len(signals[(signals.engine==engine)&(signals.time<"2025-01-01")]),
                               python_2025_26_raw_signals=len(python),
                               unmatched_open=int((opens.verdict=="FAIL").sum()),
                               unmatched_skips=int((skips.verdict=="FAIL").sum()),
                               ratchet_mismatches=int((ratchets.verdict=="FAIL").sum()),
                               preflight_counts_pass=len(opens)==expected_open and len(skips)==expected_skip,
                               signal_stop_gate_pass=bool((opens.verdict=="PASS").all() and (skips.verdict=="PASS").all() and (ratchets.verdict=="PASS").all()))
    return matched, summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("SPEC_GOLD_PARITY_V2.txt", "GATE_GOLD_PARITY_V2.txt"):
        if not (OUT/name).is_file():
            raise RuntimeError(f"Predeclared {name} missing")
    segments = extract_log()
    mt_events = events(segments)
    rates = load_rates()
    signals, context = signal_engine(rates)
    compared, summary = compare(signals, context, mt_events)
    first = datetime.fromtimestamp(int(rates.time.iloc[0]), timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    last = datetime.fromtimestamp(int(rates.time.iloc[-1]), timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    g1 = len(rates)==16181 and first=="2024-01-02 01:00:00" and last=="2026-09-24 23:00:00" and rates.time.is_monotonic_increasing and all(v["preflight_counts_pass"] for v in summary.values())
    failures = compared[compared.verdict=="FAIL"]
    status = "PASS" if g1 and all(v["signal_stop_gate_pass"] for v in summary.values()) else "FAIL"
    files = ["ftmo_xauusd_h1_snapshot.csv", "mt5_A_journal_snapshot.txt", "mt5_B_journal_snapshot.txt"]
    metadata = dict(gate=status, g1=g1, g4="UNTESTED", rows=len(rates), first=first, last=last,
                    source_terminal=TERMINAL, source_log=str(LOG), sha256={x:digest(OUT/x) for x in files}, summary=summary)
    (OUT/"audit_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    lines = ["FTMO GOLD EA VS INDEPENDENT PYTHON PARITY AUDIT", f"Gate verdict: {status}",
             f"G1 broker H1 and tester snapshots: {'PASS' if g1 else 'FAIL'}; {len(rates)} H1 bars {first} to {last}",
             "G4 exact tick entry and stop-exit parity: UNTESTED (broker M1/tick history incomplete)",
             "Scope: 2024 Python signal baseline; 2025-2026 v1.01 Gold A and B tester comparison.",
             "Journal events are matched to their enclosing server H1 candle.", ""]
    for engine in ("A","B"):
        s=summary[engine]
        lines += [f"Engine {engine}: 2024 Python signals {s['python_2024_signals']}; 2025-26 raw Python signals {s['python_2025_26_raw_signals']}; MT5 OPEN {s['mt5_open']}; min-lot skips {s['mt5_min_lot_skip']}; ratchets {s['mt5_ratchets']}.",
                  f"  Unmatched or wrong-stop OPEN {s['unmatched_open']}; unmatched skips {s['unmatched_skips']}; ratchet mismatches {s['ratchet_mismatches']}; G2/G3 {'PASS' if s['signal_stop_gate_pass'] else 'FAIL'}."]
    lines += ["", f"Total event divergences: {len(failures)}"]
    if len(failures):
        for _, row in failures.head(20).iterrows():
            lines.append(f"  {row.engine} {row.time} {row.kind}: logged SL={row.sl}, predicted SL={row.predicted_sl}; {row.line}")
    lines += ["", "Interpretation: Raw Python signals include occupied/broker-blocked cases; compare logged OPEN and volume skips to raw signals, not raw signal count to trade count.",
              "The tester journal reported discarded real ticks due M1 price mismatch. This audit cannot validate actual fill prices, intraminute stops, PnL, or live profitability.",
              "No historical SPEC/GATE/REPORT was edited. Input snapshot hashes are in audit_metadata.json."]
    report="\n".join(lines)+"\n"
    (OUT/"REPORT_GOLD_PARITY.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
