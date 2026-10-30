"""Frozen concurrency-cap and daily moving-block FTMO-like Phase-1 scenario."""

from __future__ import annotations

import hashlib
import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/results/crypto_gold_real_cost"
OUT = ROOT / "data/results/crypto_gold_concurrency"
REPORT = OUT / "REPORT_CONCURRENCY.txt"
HASHES = {
    "trades_BTCUSD_A_KC.csv": "f3b36e75d8e5ad2cee65284528fcce492f9d50b2221ddb83e91055b655b2a810",
    "trades_BTCUSD_B_DONCHIAN.csv": "b4e596d0940ef5764dc912a3d082512fffeb2e24c4b9c1d59eae6a2f1d370624",
    "trades_ETHUSD_A_KC.csv": "709e07e3cc18471bbded911e915efc57426fa064e189bae138d974dd9913a5cb",
    "trades_ETHUSD_B_DONCHIAN.csv": "54f209f6ef82e2f779eb94b973a88ae2e1ad7d0691d5588c0e62081561f782ae",
    "trades_XAUUSD_A_KC.csv": "5e8edc9f2bdc45858a89175f9f81e1dfc354db6651e5d137f621711a83699644",
    "trades_XAUUSD_B_DONCHIAN.csv": "530193b75b27467ac56cb9768c48b63679bc32896b9480fe9d9ae67dd0c70432",
}
CAPS = (1, 2, 3, 4, 6)
RISKS = (0.005, 0.010, 0.020)
PATHS = 10_000
BLOCK = 10
HORIZON = 365
SEED = 20261101


def read_master() -> pd.DataFrame:
    pieces = []
    for filename, expected_hash in HASHES.items():
        path = SOURCE / filename
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Frozen input hash changed: {filename}: {actual_hash}")
        part = pd.read_csv(path)
        required = {"asset", "engine", "entry_time", "exit_time", "exit_type", "net_r"}
        if not required.issubset(part.columns):
            raise ValueError(f"Trade ledger schema changed: {filename}")
        asset, engine = filename.removeprefix("trades_").removesuffix(".csv").split("_", 1)
        if not ((part.asset == asset) & (part.engine == engine)).all():
            raise ValueError(f"Asset/engine identity mismatch: {filename}")
        part["source_file"] = filename
        part["source_row"] = np.arange(1, len(part) + 1)
        pieces.append(part)
    master = pd.concat(pieces, ignore_index=True)
    if len(master) != 2_277:
        raise ValueError("Expected 2,277 source rows")
    for col in ("entry_time", "exit_time"):
        master[col] = pd.to_numeric(master[col], errors="raise").astype(np.int64)
    if not (master.exit_time > master.entry_time).all():
        raise ValueError("Entry/exit chronology failure")
    open_end = master.exit_type.eq("OPEN_END")
    finite = np.isfinite(master.net_r.to_numpy(float))
    if open_end.sum() != 2 or finite.sum() != 2_275 or not np.array_equal(open_end.to_numpy(), ~finite):
        raise ValueError("Expected 2 OPEN_END missing-R and 2,275 finite-R closed rows")
    if not np.isclose(master.loc[~open_end, "net_r"].sum(), 220.481515907202, rtol=0, atol=1e-8):
        raise ValueError("Source net-R total changed")
    master = master.sort_values(
        ["entry_time", "asset", "engine", "exit_time", "source_row"], kind="mergesort"
    ).reset_index(drop=True)
    master.insert(0, "master_trade_id", np.arange(1, len(master) + 1))
    if (np.diff(master.entry_time.to_numpy()) < 0).any():
        raise AssertionError("Master ledger is not entry-time sorted")
    return master


def cap_walk(master: pd.DataFrame, cap: int) -> tuple[np.ndarray, int]:
    accepted = np.zeros(len(master), dtype=bool)
    open_exits: list[int] = []
    max_live = 0
    for i, (entry, exit_) in enumerate(zip(master.entry_time, master.exit_time)):
        while open_exits and open_exits[0] <= entry:
            heapq.heappop(open_exits)
        if len(open_exits) < cap:
            accepted[i] = True
            heapq.heappush(open_exits, int(exit_))
            max_live = max(max_live, len(open_exits))
    return accepted, max_live


def independent_max_occupancy(trades: pd.DataFrame) -> int:
    events = [(int(t), 1) for t in trades.entry_time]
    events.extend((int(t), -1) for t in trades.exit_time)
    active = peak = 0
    for _, change in sorted(events, key=lambda event: (event[0], event[1])):
        active += change
        peak = max(peak, active)
        if active < 0:
            raise AssertionError("Exit event preceded its accepted entry")
    if active != 0:
        raise AssertionError("Accepted event ledger did not balance")
    return peak


def daily_series(accepted: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    closed = accepted.loc[accepted.exit_type.ne("OPEN_END")]
    exit_day = pd.to_datetime(closed.exit_time, unit="s", utc=True).dt.floor("D")
    grouped = closed.groupby(exit_day, sort=True).net_r.sum()
    net_r = grouped.reindex(calendar, fill_value=0.0).to_numpy(float)
    if len(net_r) != len(calendar) or not np.isclose(net_r.sum(), closed.net_r.sum(), atol=1e-8):
        raise AssertionError("Daily R failed to reconcile to accepted closed trades")
    return pd.DataFrame({"date_utc": calendar.strftime("%Y-%m-%d"), "daily_net_r": net_r})


def sample_indices(n_days: int) -> np.ndarray:
    if n_days < BLOCK:
        raise ValueError("Daily history shorter than a bootstrap block")
    rng = np.random.default_rng(SEED)
    n_blocks = (HORIZON + BLOCK - 1) // BLOCK
    starts = rng.integers(0, n_days - BLOCK + 1, size=(PATHS, n_blocks), dtype=np.int32)
    offsets = np.arange(BLOCK, dtype=np.int32)
    return (starts[:, :, None] + offsets).reshape(PATHS, -1)[:, :HORIZON]


def run_phase(sampled_daily_r: np.ndarray, risk: float) -> pd.DataFrame:
    status = np.zeros(PATHS, dtype=np.int8)  # 0 timeout, 1 pass, -1 daily, -2 static
    equity = np.zeros(PATHS, dtype=np.float64)
    stop_day = np.full(PATHS, HORIZON, dtype=np.int16)
    terminal_day_pnl = np.zeros(PATHS, dtype=np.float64)
    for day in range(HORIZON):
        active = np.flatnonzero(status == 0)
        if active.size == 0:
            break
        delta = sampled_daily_r[active, day] * risk
        equity[active] += delta
        terminal_day_pnl[active] = delta
        daily_loss = delta <= -0.05
        static_loss = (equity[active] <= -0.10) & ~daily_loss
        passed = (equity[active] >= 0.10) & ~daily_loss & ~static_loss
        status[active[daily_loss]] = -1
        status[active[static_loss]] = -2
        status[active[passed]] = 1
        stop_day[active[daily_loss | static_loss | passed]] = day + 1
    names = np.array(["TIMEOUT", "PASS", "DAILY_LOSS", "STATIC_LOSS"], dtype=object)
    labels = np.empty(PATHS, dtype=object)
    labels[status == 0] = names[0]
    labels[status == 1] = names[1]
    labels[status == -1] = names[2]
    labels[status == -2] = names[3]
    return pd.DataFrame(
        {
            "path_id": np.arange(1, PATHS + 1),
            "status": labels,
            "stop_day": stop_day,
            "terminal_equity_pct": equity,
            "terminal_daily_pnl_pct": terminal_day_pnl,
        }
    )


def audit_path_ledger(paths: pd.DataFrame) -> None:
    if len(paths) != len(CAPS) * len(RISKS) * PATHS:
        raise AssertionError("Wrong total number of paths")
    legal = {"PASS", "DAILY_LOSS", "STATIC_LOSS", "TIMEOUT"}
    for (cap, risk), group in paths.groupby(["cap", "risk_pct"], sort=True):
        if cap not in CAPS or risk not in {x * 100 for x in RISKS} or len(group) != PATHS:
            raise AssertionError("Missing or extra cap-risk cell")
        if not np.array_equal(group.path_id.to_numpy(), np.arange(1, PATHS + 1)):
            raise AssertionError("Path IDs missing or reordered")
        if not set(group.status).issubset(legal):
            raise AssertionError("Illegal path status")
        if not group.stop_day.between(1, HORIZON).all():
            raise AssertionError("Stop day outside 365-day horizon")
        if not (group.loc[group.status == "TIMEOUT", "stop_day"] == HORIZON).all():
            raise AssertionError("Early timeout")
        if not (group.loc[group.status == "PASS", "terminal_equity_pct"] >= 0.10 - 1e-10).all():
            raise AssertionError("Pass below target")
        if not (group.loc[group.status == "DAILY_LOSS", "terminal_daily_pnl_pct"] <= -0.05 + 1e-10).all():
            raise AssertionError("Daily-loss classification below threshold")
        if not (group.loc[group.status == "STATIC_LOSS", "terminal_equity_pct"] <= -0.10 + 1e-10).all():
            raise AssertionError("Static-loss classification below threshold")
        if not (group.loc[group.status == "STATIC_LOSS", "terminal_daily_pnl_pct"] > -0.05 - 1e-10).all():
            raise AssertionError("Daily-loss precedence failed")


def summarize(paths: pd.DataFrame, cap_counts: dict[int, tuple[int, int]], n_days: int) -> pd.DataFrame:
    rows = []
    for (cap, risk), group in paths.groupby(["cap", "risk_pct"], sort=True):
        passed = group.status.eq("PASS")
        rows.append(
            {
                "cap": int(cap),
                "risk_pct": float(risk),
                "accepted_all": cap_counts[int(cap)][0],
                "accepted_closed": cap_counts[int(cap)][1],
                "rejected_all": 2_277 - cap_counts[int(cap)][0],
                "daily_history_days": n_days,
                "paths": len(group),
                "pass_count": int(passed.sum()),
                "pass_rate": float(passed.mean()),
                "median_days_to_pass": float(group.loc[passed, "stop_day"].median()) if passed.any() else np.nan,
                "daily_loss_count": int(group.status.eq("DAILY_LOSS").sum()),
                "daily_loss_rate": float(group.status.eq("DAILY_LOSS").mean()),
                "static_loss_count": int(group.status.eq("STATIC_LOSS").sum()),
                "static_loss_rate": float(group.status.eq("STATIC_LOSS").mean()),
                "timeout_count": int(group.status.eq("TIMEOUT").sum()),
                "timeout_rate": float(group.status.eq("TIMEOUT").mean()),
            }
        )
    return pd.DataFrame(rows)


def report_text(summary: pd.DataFrame, master: pd.DataFrame, n_days: int) -> str:
    lines = [
        "CRYPTO + GOLD CONCURRENCY / 10-DAY-BLOCK PHASE-1 MONTE CARLO - 2026-09-29",
        "Frozen BTCUSD, ETHUSD and XAUUSD Engine A/B ledgers: 2,275 closed + 2 OPEN_END rows.",
        f"Master UTC entry range {pd.to_datetime(master.entry_time.min(), unit='s', utc=True)} to {pd.to_datetime(master.entry_time.max(), unit='s', utc=True)}.",
        f"Daily UTC calendar: {n_days:,} days from first entry date through last exit date; no-close days are zero.",
        "Open-end rows can occupy a cap slot but cannot contribute realized daily net R.",
        f"Bootstrap: {PATHS:,} paths/cell; noncircular {BLOCK}-day blocks; {HORIZON}-day maximum; seed {SEED}; common sampled calendar indices across cells.",
        "Phase 1: daily loss <= -5%, static loss <= -10%, pass >= +10%; daily-loss check has precedence.",
        "",
        "K | Risk | Accepted all | Closed R trades | Pass rate | Median pass days | Daily-loss fail | Static-loss fail | 365d timeout",
    ]
    for row in summary.itertuples(index=False):
        median = "NA" if pd.isna(row.median_days_to_pass) else f"{row.median_days_to_pass:.1f}"
        lines.append(
            f"{row.cap} | {row.risk_pct:.1f}% | {row.accepted_all:,} | {row.accepted_closed:,} | "
            f"{row.pass_rate:.2%} | {median} | {row.daily_loss_rate:.2%} | "
            f"{row.static_loss_rate:.2%} | {row.timeout_rate:.2%}"
        )
    lines += [
        "",
        "Failure rates and timeout rates use all 10,000 original paths per cell; median pass days uses passers only.",
        "GATE: PASS - frozen inputs, cap occupancy, daily aggregation and path accounting verified.",
        "LIMITATIONS: This is a capped walk over already executed trades. Exit-date realized R omits intraday unrealized losses, true FTMO daily equity checks and simultaneous cash exposure. Ten-day blocks preserve only local exit-date dependence. Phase-1 scenario rates are not live challenge probabilities.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    if REPORT.exists():
        raise FileExistsError(f"Historical report will not be overwritten: {REPORT}")
    if not (OUT / "SPEC_CONCURRENCY.txt").is_file() or not (OUT / "GATE_CONCURRENCY.txt").is_file():
        raise RuntimeError("Frozen SPEC and GATE must exist before execution")
    master = read_master()
    first_day = pd.to_datetime(master.entry_time.min(), unit="s", utc=True).floor("D")
    last_day = pd.to_datetime(master.exit_time.max(), unit="s", utc=True).floor("D")
    calendar = pd.date_range(first_day, last_day, freq="D")
    sampled_indices = sample_indices(len(calendar))
    cap_counts: dict[int, tuple[int, int]] = {}
    series: dict[int, pd.DataFrame] = {}
    accepted_ledgers: dict[int, pd.DataFrame] = {}
    for cap in CAPS:
        accept, max_live = cap_walk(master, cap)
        master[f"accepted_k{cap}"] = accept
        accepted = master.loc[accept].copy()
        independent_peak = independent_max_occupancy(accepted)
        if max_live != independent_peak or max_live > cap:
            raise AssertionError(f"Cap {cap} occupancy audit failed")
        if cap == 6 and not accept.all():
            raise AssertionError("K=6 must reproduce every source position")
        closed_count = int(accepted.exit_type.ne("OPEN_END").sum())
        cap_counts[cap] = (len(accepted), closed_count)
        accepted_ledgers[cap] = accepted
        series[cap] = daily_series(accepted, calendar)

    path_chunks = []
    for cap in CAPS:
        daily_r = series[cap].daily_net_r.to_numpy(float)[sampled_indices]
        for risk in RISKS:
            paths = run_phase(daily_r, risk)
            paths.insert(0, "risk_pct", risk * 100.0)
            paths.insert(0, "cap", cap)
            path_chunks.append(paths)
    all_paths = pd.concat(path_chunks, ignore_index=True)
    audit_path_ledger(all_paths)
    summary = summarize(all_paths, cap_counts, len(calendar))
    for row in summary.itertuples(index=False):
        if row.pass_count + row.daily_loss_count + row.static_loss_count + row.timeout_count != PATHS:
            raise AssertionError("Path outcome counts do not sum to 10,000")

    artifact_names = ["master_ledger.csv", "paths.csv", "summary.csv", "run_meta.json"]
    artifact_names += [f"accepted_k{k}.csv" for k in CAPS]
    artifact_names += [f"daily_k{k}.csv" for k in CAPS]
    if any((OUT / name).exists() for name in artifact_names):
        raise FileExistsError("Historical artifacts exist and will not be overwritten")
    master.to_csv(OUT / "master_ledger.csv", index=False, float_format="%.12f")
    for cap in CAPS:
        accepted_ledgers[cap].to_csv(OUT / f"accepted_k{cap}.csv", index=False, float_format="%.12f")
        series[cap].to_csv(OUT / f"daily_k{cap}.csv", index=False, float_format="%.12f")
    all_paths.to_csv(OUT / "paths.csv", index=False, float_format="%.12f")
    saved_paths = pd.read_csv(OUT / "paths.csv")
    audit_path_ledger(saved_paths)
    saved_summary = summarize(saved_paths, cap_counts, len(calendar))
    numeric = summary.select_dtypes(include="number").columns
    if not np.allclose(summary[numeric], saved_summary[numeric], atol=1e-8, rtol=0, equal_nan=True):
        raise AssertionError("Saved path ledger changed summary")
    saved_summary.to_csv(OUT / "summary.csv", index=False, float_format="%.12f")
    metadata = {
        "input_sha256": HASHES,
        "input_rows": len(master),
        "input_closed_rows": int(master.net_r.notna().sum()),
        "input_open_end_rows": int(master.net_r.isna().sum()),
        "calendar_first_utc": calendar[0].strftime("%Y-%m-%d"),
        "calendar_last_utc": calendar[-1].strftime("%Y-%m-%d"),
        "calendar_days": len(calendar),
        "caps": list(CAPS),
        "risk_fractions": list(RISKS),
        "paths_per_cell": PATHS,
        "block_days": BLOCK,
        "horizon_days": HORIZON,
        "seed": SEED,
        "cap_counts": {str(k): {"accepted_all": v[0], "accepted_closed": v[1]} for k, v in cap_counts.items()},
        "paths_sha256": hashlib.sha256((OUT / "paths.csv").read_bytes()).hexdigest(),
    }
    (OUT / "run_meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    output = report_text(saved_summary, master, len(calendar))
    with REPORT.open("x", encoding="utf-8") as handle:
        handle.write(output)
    print(output)


if __name__ == "__main__":
    main()
