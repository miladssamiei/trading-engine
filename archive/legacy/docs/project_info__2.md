I've read PROGRESS.md in full and saved my findings to `project_info__1.md`. Here's the summary:

---

# PROGRESS.md — Summary

## What this project is
**RR (Reinforce-Ready)** — an MT5 validation pipeline that bridges a Python-based ML trading stack (CAIRO/Reinforce) to MetaTrader 5. The goal: verify that model signals from Python backtests survive the trip into real MT5 execution.

## Where things stand (as of 2026-08-18)

**✅ Phase 1 — Bridge core extraction (done):** `bridge_core.py` contains the full feature pipeline (86 indicators), LiveModels (GBM/TQC/DT), BarValidator, BarBuffer, and `process_bar_core` signal generation. CAIRO training scripts modified to emit RR checkpoint structure.

**✅ Phase 2 — DLL client (done):** C-based Winsock2 TCP client compiled to `bridge_client.dll`, with `transport_dll.mqh` as the production MQL5 include. Critical permanent note (2.6b): MQL5 `string` → `const char*` marshaling corrupts JSON via CP_ACP — all transport must use raw byte arrays with explicit CP_UTF8 encoding.

**✅ Phase 3 — Orchestrator (done):** Complete job pipeline — terminal_manager, config_generator, job_queue (SQLite), bridge_manager, result_collector, orchestrator. Multiple permanent bug-fix notes: M15-vs-H1 timeframe bug, Sharpe bar-count sizing, direction inversion (exit vs entry DEAL_TYPE), per-symbol instrument params, DEAL_MAGIC filter, US500.c symbol mapping, EA auto-lot detection (batch .ini ignores `double` params).

**✅ Phase 4 — Validation (done, with a critical conclusion):** 3-symbol matrix completed, then narrowed to USA500-only 6-fold validation (folds 202601–202606).

## The KEY finding — GO/NO-GO
**The validation concluded NO-GO for VPS deployment.**

| Metric | Python backtest | MT5 execution |
|---|---|---|
| Mean Sharpe | **+17.1** (all folds 15–19) | **-1.24** (range -6.6..+4.4) |
| Signal correlation | — | Pearson +0.652 (p=0.161, NOT significant) |

5 of 6 folds: Python strongly positive (15–20) but MT5 negative. The correlation drops below 0.5 on leave-one-out analysis. **The execution chain works mechanically, but the checkpoint signal does NOT survive the Python→MT5 trip** — the same divergence pattern that sank live CAIRO.

## Deferred / Blocked items
- Terminal pod setup (setup_terminals.ps1 + golden image)
- `_train_*_live.py` script consumption of REINFORCE_CHECKPOINT_DIR
- 25-fold walk-forward matrix (only 6 folds currently produced)
- XAUUSD history data verification (prices in 4468–4567 range — looks like USA500 data, possible mislabel)

## Likely next work
Investigating the signal-fidelity gap — comparing bar-by-bar signals between the Python backtest and MT5 execution to find where the divergence originates (bar/feature mismatch, feed differences, or timestamp alignment).

---

I'm ready to proceed. What would you like to do next — e.g., dig deeper into the signal-fidelity gap, explore a specific module, or something else?