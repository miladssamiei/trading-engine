# RR (Reinforce-Ready) — MT5 Validation Pipeline

## Summary
RR is a quantitative trading validation pipeline that bridges a Python-based ML trading stack (CAIRO/Reinforce) to MetaTrader 5 for backtesting. It trains GBM/TQC/DT models on historical data, serves live predictions through a TCP socket bridge, and executes trades via an MQL5 EA in the MT5 Strategy Tester to verify checkpoint signals survive the trip from Python backtest to MT5 execution. The project's most recent phase (Aug 2026) concluded that the execution chain works mechanically, but the checkpoint signal does NOT survive into MT5 execution (Python Sharpes of 15–20 degrade to MT5 Sharpes of -6.6..+4.4) — the same divergence pattern that sank live CAIRO.

## Current State (from PROGRESS.md, last updated 2026-08-18)
- **Phase 1 (done):** Bridge core extraction — `bridge_core.py` contains the feature pipeline (86 indicators), LiveModels, BarValidator, BarBuffer, `process_bar_core` signal generation. Modified CAIRO training scripts to emit RR checkpoint structure.
- **Phase 2 (done):** Bridge DLL client — C-based Winsock2 TCP client (`bridge_client.c/h`) compiled to `bridge_client.dll`, MQL5 transport include (`transport_dll.mqh`) using CP_UTF8-safe `BridgeSendReceiveRaw`. Critical permanent note: MQL5 `string` → `const char*` marshaling goes through CP_ACP (CP1252) and corrupts JSON — all transport must use raw byte arrays with explicit UTF-8 encoding.
- **Phase 3 (done):** Orchestrator components — terminal_manager, config_generator, result_collector, bridge_manager, dll_preflight, job_queue (SQLite), orchestrator (master controller). Integrates terminal, bridge, EA, and result collection. Known permanent notes: timeframe bug (M15 vs H1), bar-count sizing for Sharpe, direction inversion (exit vs entry DEAL_TYPE), per-symbol instrument params, DEAL_MAGIC filter, US500.c symbol mapping, EA auto-lot detection (MT5 batch .ini ignores `double` params).
- **Phase 4 (done):** 3-symbol matrix (EURUSD, XAUUSD, USA500) completed with all bug fixes. Then narrowed to USA500-only signal-fidelity validation with 6 folds (202601–202606). Final go/no-go: **NO-GO for VPS deployment** — reconciliation between Python backtest Sharpe (~17) and MT5 Sharpe (-1.24 avg) shows non-significant, fragile correlation (Pearson +0.652, p=0.161; drops to +0.427 on leave-one-out).

## Directory Structure
```
RR/
├── config.py                  — Single source of truth: RR constants, RR_MT5_SYMBOL_MAP, RR_MT5_PERIOD
├── PROGRESS.md                — Master task/status tracker (36 tasks + validation phases)
├── GUIDE.md                   — Developer guide
├── bridge/                    — Python bridge serving model predictions
│   ├── bridge_core.py         — Core logic: gc_col, compute_features_df (86 indicators), LiveModels, BarValidator, BarBuffer, process_bar_core
│   ├── server_http.py         — HTTP wrapper (deprecated path; /bar /health /status /equity /history)
│   ├── server_socket.py       — TCP socket server (primary; used by orchestrator + EA)
│   ├── bridge_client.c/.h     — C DLL: Winsock2 TCP client, BridgePing / BridgeSendReceiveRaw
│   └── bridge_client.dll      — Compiled 16.5 KB DLL (MinGW-W64 x86_64-ucrt)
├── ea/                        — MQL5 experts
│   ├── MULTI_CAIRO_EA.mq5     — Production EA: bridge calls, trade execution, OnDeinit trades.csv writer
│   ├── transport_dll.mqh      — Canonical MQL5 include (CP_UTF8-safe, only Raw variant)
│   ├── test_bridge_client.mq5 — Sandbox DLL validation EA
│   └── ExportM1Data.mq5, ScanHistory.mq5 — Utility EAs
├── evaluate/                  — Orchestration (the "orchestrator" pattern)
│   ├── orchestrator.py        — Master controller: run_one_job, batch mode, Sharpe calc, CSV/log parsers
│   ├── terminal_manager.py    — MT5 terminal lifecycle (spawn, health, stop)
│   ├── config_generator.py    — UTF-8 .ini generation (tester_*.ini + ea_params_*.ini)
│   ├── result_collector.py    — XML report parser (deprioritized; batch mode emits no XML)
│   ├── bridge_manager.py      — Bridge subprocess lifecycle + health checks
│   ├── dll_preflight.py       — Pre-flight DLL sandbox validation via agent log parsing
│   ├── job_queue.py           — SQLite job queue (queued→running→completed/failed, 3 retries)
│   └── queue_usa500_folds.py  — Fold queueing script
├── data/checkpoints/USA500/fold_<YYYYMM>/ — Trained models per fold (splits.json, sizing_results.json, gbm_model.txt, tqc_meta.json, dt_meta.json, scalers/)
├── data/queue/jobs.db         — SQLite job database (also evaluate/jobs.db)
├── data/results/results.csv   — Validation results
├── deploy/compile_dll.ps1     — MinGW DLL compile script
├── tests/                     — smoke_roundtrip, dll_protocol, terminal_manager, single_job_match_port, bridge_signals, signals_near_end
└── temp_*.py                  — Throwaway diagnostic scripts (temp_count, temp_diag, temp_eurchk, temp_explore, temp_explore2, temp_queue, temp_verify)
```

## Key Architectural Decisions
1. **Socket-first transport:** `server_socket.py` (TCP JSON) is the primary bridge interface; the HTTP server exists but is the secondary path. The EA uses the DLL → TCP socket → Python bridge chain.
2. **CP_UTF8 raw-byte boundary (permanent note 2.6b):** MQL5 string marshaling through CP_ACP corrupts JSON. All DLL transport uses `uchar[]` arrays + `StringToCharArray`/`CharArrayToString` with explicit CP_UTF8. Any new .ini reading must use explicit code page handling.
3. **Single source of truth in config.py:** `REINFORCE_TIMEFRAME="15m"`, `RR_MT5_PERIOD="M15"`, `RR_MT5_SYMBOL_MAP={"USA500":"US500.c"}` — never hardcode timeframe/symbol in callers.
4. **MT5 batch mode limitations discovered:** (a) no XML/HTML reports in `/config:` batch mode → trade CSV from EA OnDeinit is primary source; (b) batch .ini ignores `double` EA params → auto-detect lot size from symbol properties in OnInit.
5. **Sharpe calculation:** canonical formula matches CAIRO `parallel/common.py:1104`: `sharpe = mean(pnl)/std(pnl)*sqrt(6048)` for M15, with per-bar returns sized to full test bar count (from agent log `bars` field), not trade span.
6. **Signal-fidelity is the core open problem:** Python backtest Sharpe ≈ 17 vs MT5 ≈ -1.24. The chain works mechanically but the signal degrades. VPS deployment is NOT justified on current evidence.

## Data Flow (per-fold validation job)
1. `orchestrator.py` pulls queued job from `job_queue` → marks running
2. Starts bridge (`server_socket.py` on port 9998+n) with `--symbols USA500` → loads models from `data/checkpoints/USA500/fold_<YYYYMM>/`
3. `config_generator` writes `tester_*.ini` (Symbol=US500.c, Period=M15, ShutdownTerminal=1) + `ea_params_*.ini`
4. `terminal_manager` spawns MT5 terminal64.exe /portable → Strategy Tester runs EA
5. EA (MULTI_CAIRO_EA.mq5) feeds M15 bars to bridge via `transport_dll.mqh` → `BridgeSendReceiveRaw` → `server_socket.py` → `process_bar_core` → action + cairo_weight back
6. EA executes orders; on Deinit writes `trades.csv` (semicolon-delimited, magic-filtered, per-symbol params)
7. Terminal exits (ShutdownTerminal=1) → orchestrator parses agent log (UTF-16LE, `bars` field) + trades.csv → computes Sharpe/PnL/win_rate → `complete_job` → stops bridge

## Status / Open Items
- **DONE:** Full pipeline (36 tasks): bridge extraction, DLL, orchestrator, 6-fold USA500 validation
- **GO/NO-GO:** NO-GO for VPS — signal does not survive Python→MT5
- **BLOCKED/DEFERRED:** terminal pod setup, _train_*_live.py script consumption, 25-fold walk-forward, XAUUSD history data verification (prices in USA500 range — possible mislabeled data)
- **Next likely work:** investigate signal-fidelity gap (bar/feature mismatch between Python and MT5 feeds), compare bar-by-bar signals between backtest and EA execution
