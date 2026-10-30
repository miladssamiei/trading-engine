# CAIRO Trading System — Complete Guide

> **Who this is for:** Anyone — whether you have never seen this codebase before or you built
> every line of it. This document explains what the system is, why each piece exists, how
> everything connects, and exactly what commands to run in what order.

---

## Table of Contents

1. [What Is CAIRO?](#1-what-is-cairo)
2. [The Big Picture — How It All Works](#2-the-big-picture)
3. [Key Concepts Explained Simply](#3-key-concepts)
4. [System Architecture — Every Component](#4-system-architecture)
5. [The 7 Models — What Each One Is](#5-the-7-models)
6. [The Decision Stack — What Happens on Every Bar](#6-the-decision-stack)
7. [Risk Management — The Four Safety Layers](#7-risk-management)
8. [The Training Pipeline (Local Machine)](#8-training-pipeline)
9. [The Optimisation Layer](#9-the-optimisation-layer)
10. [The Live Engine](#10-the-live-engine)
11. [Deploying to VPS](#11-deploying-to-vps)
12. [Monthly Maintenance — Rollover](#12-monthly-rollover)
13. [Project File Map](#13-project-file-map)
14. [Configuration Reference](#14-configuration-reference)
15. [Common Tasks — Quick Reference](#15-common-tasks)

---

## 1. What Is CAIRO?

**CAIRO** stands for **Collective AI Reinforced Intelligence for trading Outcomes**.

It is an automated trading system that:
- Watches three financial instruments: **EURUSD** (Euro vs Dollar), **USA500** (S&P 500 index), and **XAUUSD** (Gold)
- Runs **seven independent AI models** simultaneously on every price bar
- Has those seven models **vote** on whether to buy, hold, or sell
- Applies four layers of risk protection before placing any trade
- Runs 24 hours a day on a VPS server, completely hands-free

**The goal:** Compete in a prop firm challenge — a $10,000 funded account where the rules are:
- Maximum 5% daily loss (we stop at 4.5% for safety)
- Maximum 10% total drawdown (we stop at 9%)
- Target: reach 8% profit

**The philosophy:** No single AI model is reliably right all the time. By combining seven models that were trained differently and tend to disagree with each other, the *ensemble* is more consistent than any individual model — the same reason a jury of twelve reaches better verdicts than any one judge.

---

## 2. The Big Picture

Here is how the entire system flows from raw data to live trades:

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                     LOCAL MACHINE (you)                         │
  │                                                                 │
  │  Historical price data (PostgreSQL DB)                         │
  │          │                                                      │
  │          ▼                                                      │
  │  Walk-forward training  ──►  7 models × 3 symbols             │
  │  (parallel/ scripts)          = 21 trained model checkpoints   │
  │          │                                                      │
  │          ▼                                                      │
  │  Optimiser                                                      │
  │  - Which models to use?     (subset_results.csv)               │
  │  - What lot size?           (sizing_results.json)              │
  │  - How to weight each model? (model_weights in JSON)           │
  │          │                                                      │
  │          ▼                                                      │
  │  Live fold training    ──►  final checkpoints for VPS          │
  │  (multi_live/02+03)          (~50 MB per symbol)               │
  └───────────────────────────┬─────────────────────────────────────┘
                              │  scp (copy files to server)
  ┌───────────────────────────▼─────────────────────────────────────┐
  │                         VPS (server)                            │
  │                                                                 │
  │  multi_live/04_live_engine.py   (runs 24/7)                    │
  │  - Loads checkpoints for EURUSD, USA500, XAUUSD               │
  │  - Listens on port 9999                                        │
  │          │                           ▲                         │
  │          │  HTTP (localhost)         │                         │
  │          ▼                           │                         │
  │  MT5 Expert Advisor (MULTI_CAIRO_EA.mq5)                       │
  │  - Sends each new price bar to engine                          │
  │  - Receives: BUY / SELL / HOLD + lot size                     │
  │  - Executes the trade                                          │
  └─────────────────────────────────────────────────────────────────┘
```

**In plain English:**
1. You train AI models on years of price history on your local computer.
2. You package the trained models and copy them to a VPS.
3. The VPS runs the engine forever. MetaTrader 5 sends each new price bar to the engine and receives a trading decision back within milliseconds.
4. Every month, you retrain on the newest data and update the VPS checkpoints.

---

## 3. Key Concepts

### Walk-Forward Validation
**The problem:** If you train an AI on data from 2020–2024 and test it on that same data, it will look great — it already "memorised" the answers. This is called overfitting, and it doesn't work in real markets.

**The solution:** Train on the past, test on the future — repeatedly.

```
Fold 1:  Train Jan2022–Dec2022 ──► Test Jan2023
Fold 2:  Train Feb2022–Jan2023 ──► Test Feb2023
Fold 3:  Train Mar2022–Feb2023 ──► Test Mar2023
...
Fold 41: Train (most recent 12 months) ──► Test (last month)
```

Each model is tested on data it has *never seen*, one month at a time. This gives an honest picture of how the model would have performed if you had deployed it live. CAIRO has 25–54 such folds per model, representing years of real out-of-sample performance.

### Reinforcement Learning (RL)
**The analogy:** Like training a dog. The model tries an action (BUY / SELL / HOLD). If that action leads to profit, it gets a reward. If it leads to a loss, it gets a penalty. Over thousands of training episodes it learns which actions tend to produce rewards.

Unlike traditional machine learning (where you label correct answers), RL discovers its own strategy purely from trial and error against the price data.

### The Ensemble Vote
Seven models each output a probability distribution: e.g. `[15% SELL, 25% HOLD, 60% BUY]`. The system:
1. Takes each model's most confident vote (BUY in this example)
2. Tallies the votes, **weighted by each model's historical Sharpe ratio** from past fold performance
3. The winner is the action. The confidence (how unanimous the vote was) scales the position size.

If 4 high-weight models vote BUY and 1 low-weight model votes SELL, the result is BUY with high confidence → larger position. If there is a 2-2 split (tie), the system defaults to HOLD — never forces a trade.

### HMM Regime Detection
Markets cycle through states: trending (clear direction), ranging (sideways), high-volatility (chaotic). A Hidden Markov Model detects which state the market is currently in and adjusts ETA multipliers accordingly — trending conditions favour momentum models (PPO, IMPALA), ranging conditions favour entropy-regularised models (SAC, TQC).

### Sharpe Ratio
The standard measure of risk-adjusted return. A Sharpe of 1.0 is respectable for a human trader. CAIRO's backtested folds average Sharpe of 20–27 per symbol — very high, reflecting that the walk-forward training is capturing genuine edge.

---

## 4. System Architecture

### Four Subsystems

| Subsystem | Location | Purpose |
|---|---|---|
| **Backtest pipeline** | `parallel/` | Walk-forward train + evaluate all 7 models on historical data. Produces fold prob files, results CSVs. |
| **Optimiser** | `parallel/optimize_ensemble.py` | Finds the best model subset and lot size. Produces `sizing_results.json`. |
| **Live fold builder** | `multi_live/02+03` | Trains the selected models on the most recent 12 months, producing VPS-ready checkpoints. |
| **Live engine** | `multi_live/04_live_engine.py` | Runs on VPS 24/7. Receives bars from MT5, runs inference, applies risk layers, returns signals. |

### Data Directory Structure

```
Reinforce/
├── parallel/
│   ├── data/
│   │   ├── EURUSD/15M/
│   │   │   ├── PPO/
│   │   │   │   ├── results.csv           ← per-fold Sharpe, win rate, DD
│   │   │   │   └── probs/
│   │   │   │       └── fold_202604_probs.npy  ← model's probability outputs
│   │   │   ├── IMPALA/ SAC/ TQC/ GBM/ APPO/ DT/
│   │   │   ├── CAIRO/results/cairo_results.csv ← ensemble aggregate results
│   │   │   └── OPTUNA/CAIRO/
│   │   │       ├── subset_results.csv    ← all subset combinations ranked
│   │   │       └── sizing_results.json   ← FINAL: subset + lot + model weights
│   │   ├── USA500/15M/   (same structure)
│   │   └── XAUUSD/15M/   (same structure)
│
├── multi_live/
│   └── data/
│       ├── EURUSD/15m/live_fold/
│       │   ├── splits.json              ← feature list, fold dates
│       │   ├── scalers/scaler.pkl       ← feature normaliser
│       │   ├── sizing_results.json      ← model weights (copied from parallel/)
│       │   └── checkpoints/
│       │       ├── ppo_policy/          ← RLlib model
│       │       ├── impala_policy/       ← RLlib model
│       │       ├── tqc_model.zip        ← Stable-Baselines3 model
│       │       └── gbm_model.txt        ← LightGBM model
│       ├── USA500/15m/live_fold/   (same structure)
│       └── XAUUSD/15m/live_fold/   (same structure)
│
├── live/                    ← single-asset engine (port 9998) — for one symbol only
├── config.py                ← ALL global constants live here
└── deploy/                  ← deployment scripts
```

---

## 5. The 7 Models

Each model is a completely separate AI trained independently. They form the "jury."

| Model | Type | Library | Characteristic |
|---|---|---|---|
| **PPO** | Policy Gradient RL | RLlib | Stable, general-purpose. The "all-rounder". |
| **IMPALA** | Distributed RL | RLlib | Trained with more diverse experience. Good at trending markets. |
| **APPO** | Asynchronous PPO | RLlib | Similar to PPO but with asynchronous updates. |
| **SAC** | Soft Actor-Critic | RLlib | Entropy-regularised — stays cautious, prefers "HOLD" when uncertain. Valuable for its independence from the others. |
| **TQC** | Truncated Quantile Critics | Stable-Baselines3 | Distributional RL — models the full distribution of returns, not just the mean. |
| **GBM** | Gradient Boosting | LightGBM | Not RL — a classical ML model trained on labelled trade outcomes. Fast, interpretable. |
| **DT** | Decision Transformer | PyTorch | Frames trading as a sequence prediction problem. Reads the "story" of recent bars. |

**Currently trained:**
- EURUSD: PPO, IMPALA, APPO, SAC, TQC, GBM (DT not yet trained)
- USA500: all 7 including DT
- XAUUSD: PPO, IMPALA, APPO, SAC, TQC, GBM (DT not yet trained)

**Selected subsets** (from the optimiser — not all models are used in production):
- EURUSD: **IMPALA + PPO + SAC + TQC** (Sharpe 26.7)
- USA500: **GBM + IMPALA + PPO + SAC + TQC** (Sharpe 25.2)
- XAUUSD: **IMPALA + PPO + SAC** (Sharpe 24.3)

The optimiser excludes models that are redundant (e.g. APPO votes identically to PPO 78% of the time — including it adds no new information).

---

## 6. The Decision Stack

Every time a new price bar arrives from MetaTrader 5, the engine runs through this exact sequence:

```
Bar arrives (JSON with OHLC, spread, volume)
        │
        ▼
1. Feature computation
   300+ bars of history → 86 technical indicators (RSI, ATR, Bollinger, etc.)
   + RVR (Realized Vol Ratio: short-term vs long-term volatility)
        │
        ▼
2. HMM Regime Detection
   Classifies current market state:
   0 = Trending/Low-vol   1 = Ranging/Med-vol   2 = High-vol/Risk-off
        │
        ▼
3. Model Inference (parallel)
   Each loaded model outputs [P(SELL), P(HOLD), P(BUY)]
   e.g. PPO=[0.05, 0.20, 0.75]  SAC=[0.55, 0.30, 0.15]  ...
        │
        ▼
4. Weighted Ensemble Vote (CAIRO Aggregation)
   Each model votes for its highest-probability action.
   Votes are tallied weighted by that model's historical Sharpe.
   Winner = action. Confidence = weighted agreement level.
   Tie (50/50 split) → HOLD
        │
        ▼
5. VIX Overlay (USA500 only)
   If VIX > threshold: scale down position size
   If VIX term structure inverted (backwardation): scale to 25%
        │
        ▼
6. Risk Layers (applied in order — any can override to HOLD)
   a) EquityCurveTrend: is the system currently losing? → reduce/pause
   b) CircuitBreaker:   spread spike? news event? → force HOLD
   c) CorrelationFilter: two correlated symbols both open? → reduce smaller
        │
        ▼
7. Position Sizing
   base_lot × ATR_scaler × vote_confidence × ECT_factor × prop_factor
   Capped at max 2.0× base lot
        │
        ▼
8. PropFirmGuard (final gate)
   Checks daily loss and total drawdown against prop firm limits.
   If limit breached → override to HOLD, send Telegram alert
        │
        ▼
Response to MT5: {"action": "BUY", "lot": 0.04, "confidence": 1.73}
```

---

## 7. Risk Management — The Four Safety Layers

These run every bar *after* the models vote, in order. Any layer can override the signal to HOLD.

### Layer A — PropFirmGuard
Protects the prop challenge account from rule violations.

| Trigger | Action |
|---|---|
| Daily loss > 4.5% | All signals → HOLD until next trading day |
| Total drawdown > 9% | All signals → HOLD permanently (challenge failed) |
| Daily loss > 3% | Halve all lot sizes (soft warning before hard stop) |
| Profit ≥ 8% | Telegram alert: challenge target reached |

### Layer B — EquityCurveTrend (ECT)
Detects when the *strategy itself* is underperforming, before any prop limit is hit.

It tracks two exponential moving averages of the account equity:
- **Fast EMA** (10 bars) — reacts quickly
- **Slow EMA** (30 bars) — baseline trend

| State | Condition | Effect |
|---|---|---|
| **NORMAL** | Fast EMA ≥ Slow EMA | Full position sizing |
| **CAUTION** | Fast EMA < Slow EMA | Halve all lot sizes |
| **PAUSED** | Equity Z-score < −2σ | No new trades for 24 bars |

The idea: if the equity curve is trending down, the models have drifted from the regime they were trained on. Reduce exposure and wait for conditions to normalise.

### Layer C — Circuit Breaker
Three independent tripwires that pause trading during abnormal market conditions:

| Tripwire | Trigger | Duration |
|---|---|---|
| **Spread spike** | Current spread > 5× 100-bar rolling median | 15 bars |
| **VIX backwardation** | Spot VIX > 6-month VIX future (USA500 only) | Until normalised |
| **News window** | Within ±30 min of high-impact economic event | Duration of window |

Example: NFP (Non-Farm Payroll) is scheduled at 13:30 UTC. From 13:00–14:00 UTC, all USA500 signals are forced to HOLD, regardless of what the models say.

### Layer D — Correlation Filter
Prevents accidentally doubling a position through correlated instruments.

USA500 (S&P 500) and US30 (Dow Jones) are often 95% correlated. If both are simultaneously signalling BUY, holding full size in both is effectively 2× leverage. The filter:
1. Computes rolling 20-bar Pearson correlation between all active symbols
2. If |correlation| > 0.70 AND both positions are in the same direction: scales down the smaller position by `(1 − |correlation|)` factor

Example: USA500 BUY lot=1.2, US30 BUY lot=0.9, correlation=0.92 → US30 scaled to `0.9 × (1 − 0.92) = 0.072`

---

## 8. Training Pipeline

This runs on your **local machine**. The VPS does NOT run training.

### Overview

Training has three phases:
1. **Walk-forward backtest** — train each of the 7 models across all historical folds
2. **Optimisation** — find the best model subset and position sizing
3. **Live fold** — retrain the selected models on the most recent 12 months (this is what the VPS actually uses)

### Phase 1 — Walk-Forward Training

For each symbol and each model, run the corresponding training script. Each script trains on every fold automatically and saves fold results + probability outputs.

```bash
# From the project root. Repeat for each symbol (EURUSD, USA500, XAUUSD)
# and each model (ppo, impala, appo, sac, tqc, gbm, dt)

REINFORCE_SYMBOL=EURUSD REINFORCE_TIMEFRAME=15M \
  conda run -n reinforce_rl python parallel/train_ppo.py

REINFORCE_SYMBOL=EURUSD REINFORCE_TIMEFRAME=15M \
  conda run -n reinforce_rl python parallel/train_sac.py

# ... repeat for all models and symbols
```

Each training run adds folds to:
```
parallel/data/{SYMBOL}/15M/{MODEL}/results.csv    ← Sharpe, MaxDD, WinRate per fold
parallel/data/{SYMBOL}/15M/{MODEL}/probs/         ← probability outputs per fold
```

Training is slow (hours to days). It does NOT need to be rerun unless:
- The feature set changes (new indicators added)
- You add a new symbol
- A model consistently degrades and you want to reset it

**Current fold counts** (how many walk-forward folds have been completed):

| Model | EURUSD | USA500 | XAUUSD |
|---|---|---|---|
| PPO | 54 | 18 | 25 |
| IMPALA | 25 | 24 | 21 |
| APPO | 30 | 24 | 13 |
| SAC | 28 | 41 | 29 |
| TQC | 26 | 12 | 25 |
| GBM | 124 | 52 | 75 |
| DT | — | 76 | — |

> **Priority:** USA500 TQC has only 12 folds — the most critical gap to fill. Run `train_tqc.py` for USA500 first.

### Phase 2 — Optimisation

After training, run the optimiser to find the best ensemble configuration:

```bash
# Run for each symbol
REINFORCE_SYMBOL=EURUSD REINFORCE_TIMEFRAME=15M \
  conda run -n reinforce_rl python parallel/optimize_ensemble.py

REINFORCE_SYMBOL=USA500 REINFORCE_TIMEFRAME=15M \
  conda run -n reinforce_rl python parallel/optimize_ensemble.py

REINFORCE_SYMBOL=XAUUSD REINFORCE_TIMEFRAME=15M \
  conda run -n reinforce_rl python parallel/optimize_ensemble.py
```

This does two things:

**Part 1 — Subset search:** Tests every possible combination of models (e.g. PPO alone, PPO+SAC, PPO+SAC+TQC, all 7, etc.) and ranks them by ensemble Sharpe. Output: `subset_results.csv`.

**Part 2 — Sizing optimisation (Optuna):** Finds the optimal `base_lot` and vote confidence table for the chosen subset, subject to the constraint that max drawdown ≤ 4.5%. Output: `sizing_results.json`.

The `sizing_results.json` also now includes `model_weights` — how much each model's vote counts relative to the others, derived from their historical Sharpe on common folds.

### Phase 3 — Live Fold Training

This trains the selected models on the *most recent* 12 months of data, producing the checkpoints that will actually run on the VPS.

```bash
bash deploy/1_train_all_symbols.sh
```

This script runs for all three symbols automatically:
1. `multi_live/02_build_live_fold.py` — pulls recent data from DB, computes features, fits scaler
2. `multi_live/03_train_live_fold.py` — trains the selected models for that symbol
3. Copies `sizing_results.json` (with model weights) into the live_fold directory

Output for each symbol:
```
multi_live/data/{SYMBOL}/15m/live_fold/
├── splits.json              ← feature list + scaler path
├── scalers/scaler.pkl       ← normalises features before model input
├── sizing_results.json      ← subset + lot + model weights
└── checkpoints/
    ├── ppo_policy/          ← trained PPO
    ├── impala_policy/       ← trained IMPALA
    ├── tqc_model.zip        ← trained TQC
    └── gbm_model.txt        ← trained GBM
```

### Viewing Results — The Tearsheet

After training and running `cairo_aggregate.py`, generate an HTML performance report:

```bash
REINFORCE_SYMBOL=USA500 REINFORCE_TIMEFRAME=15M \
  conda run -n reinforce_rl python parallel/tearsheet.py --format html
```

Opens `parallel/data/USA500/15M/CAIRO/tearsheet.html` in your browser. Contains:
- Full equity curve + rolling Sharpe/Sortino
- Bootstrap 95% confidence interval on Sharpe ("is this real edge or luck?")
- Win rate by hour of day and day of week
- Per-model contribution to PnL
- Cost attribution (how much of gross PnL is lost to spread + commission)

---

## 9. The Optimisation Layer

### Why Model Selection Matters

Not all models should have equal votes. Including a redundant model (one that always agrees with another) wastes voting power and does not improve accuracy. Including a model that is individually weak but independent can *improve* ensemble accuracy because its disagreements carry information.

**Example from real EURUSD data:**
- APPO: mean Sharpe = 19.0, but votes identically to PPO 78% of the time → excluded
- SAC: mean Sharpe = 8.4, but only agrees with PPO 49% of the time → included

SAC is kept despite weak individual performance because when PPO and SAC *both* agree on BUY, that is a far stronger signal than PPO+APPO agreeing (APPO almost always agrees with PPO regardless).

This is exactly how Modern Portfolio Theory works: a low-return, low-correlation asset can improve portfolio Sharpe beyond what a higher-return but correlated asset can achieve.

### Model Weights

The selected models do not vote with equal weight. Weights are derived from historical fold Sharpe:

```
EURUSD weights (25 intersection folds):
  PPO    → 30.8%  (highest Sharpe: 19.1)
  IMPALA → 29.9%  (Sharpe: 18.5)
  TQC    → 25.7%  (Sharpe: 15.9)
  SAC    → 13.6%  (Sharpe:  8.4 — low but retained for diversification)

USA500 weights (recent 12 folds per model — no temporal overlap):
  PPO    → 24.5%   IMPALA → 24.0%   TQC → 20.3%
  GBM    → 18.4%   SAC    → 12.7%

XAUUSD weights (21 intersection folds):
  IMPALA → 38.6%   PPO → 37.0%   SAC → 24.4%
```

These weights are stored in `sizing_results.json` and automatically loaded by the live engine at startup. A model with higher weight contributes more strongly to the final decision.

---

## 10. The Live Engine

### What It Does

`multi_live/04_live_engine.py` is the production server. It:
- Loads trained checkpoints for all three symbols at startup
- Listens on port 9999 for HTTP POST requests from MT5
- Processes each incoming price bar in < 200ms
- Returns a JSON signal to MT5

### Starting the Engine

```bash
# On the VPS, in a tmux session:
bash ~/reinforce_live/deploy/vps_start_engine.sh

# Or with systemd (auto-restarts if it crashes):
systemctl --user start cairo_live
```

### What the Engine Logs

Every bar is logged to `multi_live/data/logs/`. Example log line:

```
[EURUSD] bar=202604150900 regime=0 action=BUY lot=0.042
         models={PPO:BUY, IMPALA:BUY, TQC:BUY, SAC:HOLD}
         cairo_weight=1.73 ect=NORMAL cb=CLEAR prop_dd=2.1%
```

### MT5 Communication

The MT5 Expert Advisor sends a POST request to `http://localhost:9999/bar` every time a new 15-minute bar closes:

```json
{
  "symbol": "EURUSD",
  "time": 1714982400,
  "open": 1.07432,
  "high": 1.07501,
  "low": 1.07389,
  "close": 1.07465,
  "volume": 12847,
  "spread": 0.00015
}
```

The engine returns:

```json
{
  "action": "BUY",
  "lot": 0.042,
  "confidence": 1.73,
  "regime": 0,
  "models": {"PPO": "BUY", "IMPALA": "BUY", "TQC": "BUY", "SAC": "HOLD"}
}
```

### Single-Symbol vs Multi-Symbol Engine

| | Single (`live/04_live_engine.py`) | Multi (`multi_live/04_live_engine.py`) |
|---|---|---|
| Port | 9998 | 9999 |
| Symbols | One (set via `REINFORCE_SYMBOL`) | EURUSD + USA500 + XAUUSD simultaneously |
| Correlation filter | Records updates only (one symbol) | Full cross-symbol filter active |
| Use case | Testing a single instrument | Production: all three instruments |

**For production, always use the multi-asset engine.**

---

## 11. Deploying to VPS

### What the VPS Needs (and Does NOT Need)

**Needs:**
- The live engine code (`multi_live/`)
- The risk module code (`live/equity_curve_trader.py`, `circuit_breaker.py`, `correlation_filter.py`)
- `config.py`
- Model checkpoints for each symbol
- Python environment (conda)
- A `.env` file with your database and Telegram credentials

**Does NOT need:**
- The `parallel/` directory (training code and all fold data — gigabytes)
- GPU (inference is CPU-only)
- The full training environment

### First-Time VPS Setup

Run this once on a fresh VPS:

```bash
# 1. Copy setup script to VPS
scp /home/milad/Desktop/Reinforce/deploy/vps_first_time_setup.sh user@YOUR_VPS_IP:~/

# 2. SSH in and run it
ssh user@YOUR_VPS_IP
bash ~/vps_first_time_setup.sh
# Takes ~10 minutes. Installs Miniforge, creates cairo_live env, sets up systemd.

# 3. Edit credentials
nano ~/reinforce_live/.env
# Fill in: DB_PASSWORD, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
```

### Transferring Checkpoints (After Training Locally)

```bash
# On LOCAL machine — package checkpoints:
cd /home/milad/Desktop/Reinforce
bash deploy/2_package_checkpoints.sh
# Produces: /tmp/cairo_checkpoints_YYYYMMDD.tar.gz

# Transfer to VPS:
scp /tmp/cairo_checkpoints_YYYYMMDD.tar.gz user@YOUR_VPS_IP:~/

# On VPS — unpack and restart engine:
ssh user@YOUR_VPS_IP
cd ~/reinforce_live
bash deploy/vps_update_checkpoints.sh ~/cairo_checkpoints_YYYYMMDD.tar.gz
```

The update script stops the engine, unpacks new checkpoints (including updated engine code), verifies all three symbols have checkpoints, and restarts automatically.

### Verifying the Engine Is Running

```bash
# Check service status
systemctl --user status cairo_live

# Watch live logs
journalctl --user -u cairo_live -f

# Or if using tmux:
tmux attach -t cairo
```

Expected startup output:
```
[INIT] CAIRO Multi-Asset Engine starting...
  Active symbols: EURUSD, USA500, XAUUSD
  Timeframe: 15m  |  HTTP port: 9999
[EURUSD] Loading live fold metadata...
[EURUSD] Fold weights: IMPALA=0.299, PPO=0.308, SAC=0.136, TQC=0.257
[EURUSD] ✓ PPO loaded (RLlib)
[EURUSD] ✓ IMPALA loaded (RLlib)
[EURUSD] ✓ TQC loaded (SB3)
[EURUSD] ✓ SAC loaded (RLlib)
[USA500] Loading live fold metadata...
...
[HTTP] Server listening on 0.0.0.0:9999
```

---

## 12. Monthly Rollover

Every month, the models need to be retrained on the newest data. Markets evolve, and a model trained only on 2022–2024 data may not capture 2026 market microstructure.

### What Rollover Does

`multi_live/05_rollover.py` runs automatically (triggered by the engine's internal scheduler) on the 1st of each month at 00:05 UTC. It:

1. Downloads the past month's tick data from Dukascopy
2. Imports ticks into PostgreSQL (`ohlc_1min` table)
3. Rebuilds the live fold (sliding 12-month window now includes the new month)
4. Retrains all selected models on the new fold
5. Reloads models in-process (no restart required)
6. Sends Telegram notification

### Manual Rollover (If Automatic Fails)

```bash
# On LOCAL machine — full retrain pipeline:
bash deploy/1_train_all_symbols.sh    # ~2-4 hours

# Package and send:
bash deploy/2_package_checkpoints.sh
scp /tmp/cairo_checkpoints_YYYYMMDD.tar.gz user@VPS_IP:~/

# On VPS:
bash ~/reinforce_live/deploy/vps_update_checkpoints.sh ~/cairo_checkpoints_YYYYMMDD.tar.gz
```

---

## 13. Project File Map

```
Reinforce/
│
├── config.py                    ← ALL global constants (edit this, not individual scripts)
├── GUIDE.md                     ← this document
│
├── parallel/                    ← BACKTEST TRAINING PIPELINE (local only)
│   ├── train_ppo.py             ← trains PPO across all walk-forward folds
│   ├── train_impala.py          ← trains IMPALA
│   ├── train_appo.py            ← trains APPO
│   ├── train_sac.py             ← trains SAC
│   ├── train_tqc.py             ← trains TQC
│   ├── train_gbm.py             ← trains GBM
│   ├── train_dt.py              ← trains Decision Transformer
│   ├── optimize_ensemble.py     ← finds best subset + sizing → sizing_results.json
│   ├── cairo_aggregate.py       ← combines fold probs → ensemble results + charts
│   ├── tearsheet.py             ← generates HTML performance report
│   ├── regime_detector.py       ← HMM regime detection (trains + predicts)
│   ├── common.py                ← shared utilities, feature computation, aggregation fns
│   └── core/
│       └── ensemble.py          ← BMAVoter, RegimeConditionalVoter, HedgeVoter classes
│
├── multi_live/                  ← LIVE ENGINE — multi-symbol (runs on VPS)
│   ├── 02_build_live_fold.py    ← pull DB data, compute features, fit scaler
│   ├── 03_train_live_fold.py    ← train selected models on live fold
│   ├── 04_live_engine.py        ← HTTP server — main production process (port 9999)
│   ├── 05_rollover.py           ← monthly: fetch new ticks, rebuild fold, retrain
│   ├── common.py                ← (does not exist — imports from parallel/common.py)
│   ├── prop_firm_guard.py       ← PropFirmGuard class
│   ├── position_sizer.py        ← Kelly + vol-target sizing
│   ├── portfolio_coordinator.py ← cross-symbol position coordination
│   ├── news_calendar.py         ← ForexFactory calendar integration
│   ├── telegram_notifier.py     ← Telegram alerts
│   ├── vix_overlay.py           ← VIX fetch + scaling (USA500)
│   ├── trade_logger.py          ← logs executed trades to DB
│   └── _train_*_live.py         ← sub-scripts called by 03_train_live_fold.py
│
├── live/                        ← LIVE ENGINE — single-symbol (port 9998)
│   ├── 04_live_engine.py        ← single-symbol version (testing/backup)
│   ├── equity_curve_trader.py   ← ECT: EMA-based equity curve state machine
│   ├── circuit_breaker.py       ← CB: spread spike + VIX + news halts
│   ├── correlation_filter.py    ← CF: cross-symbol position correlation scaling
│   └── [mirrors of multi_live/ support files]
│
├── deploy/                      ← DEPLOYMENT SCRIPTS
│   ├── 1_train_all_symbols.sh   ← LOCAL: build + train live folds for all 3 symbols
│   ├── 2_package_checkpoints.sh ← LOCAL: creates tar.gz ready for scp
│   ├── vps_first_time_setup.sh  ← VPS: one-time conda + env + systemd setup
│   ├── vps_update_checkpoints.sh ← VPS: unpack new checkpoints, restart engine
│   └── vps_start_engine.sh      ← VPS: start engine in tmux session
│
├── scripts/                     ← utility scripts
│   ├── import_historical_ticks.py  ← bulk import of historical data
│   └── refresh_news_calendar.py    ← update ForexFactory calendar cache
│
└── tests/                       ← unit tests
    └── conftest.py
```

---

## 14. Configuration Reference

All configuration is centralised in `config.py`. **Never hard-code numbers in scripts — edit config.py only.**

### Trading Instruments

| Symbol | Description | Base Lot | Contract Size |
|---|---|---|---|
| EURUSD | Euro / US Dollar | 0.03 | 100,000 units |
| USA500 | S&P 500 CFD | 2.8 | 1 contract |
| XAUUSD | Gold / US Dollar | 0.02 | 100 troy oz |

### Walk-Forward Parameters

```python
TRAIN_MONTHS = 12   # train on the past 12 months
TEST_MONTHS  = 1    # test on the next 1 month
STEP_MONTHS  = 1    # advance window by 1 month each fold
```

### Prop Firm Limits

```python
PROP_DAILY_LOSS_LIMIT = 0.045  # hard stop at 4.5% daily loss (rule is 5%)
PROP_TOTAL_DD_LIMIT   = 0.09   # hard stop at 9% total DD (rule is 10%)
PROP_PROFIT_TARGET    = 0.08   # Telegram alert at 8% profit
PROP_LOT_SOFT_THRESH  = 0.03   # halve lots when daily loss > 3%
```

### Equity-Curve Trading (ECT)

```python
ECT_FAST          = 10    # fast EMA window (bars)
ECT_SLOW          = 30    # slow EMA window (bars)
ECT_ZSCORE        = -2.0  # pause when equity Z-score < -2σ
ECT_PAUSE_BARS    = 24    # bars to halt after Z trigger (24 bars = 6 hours at 15M)
ECT_CAUTION_SCALE = 0.5   # lot multiplier in CAUTION state
```

### Circuit Breaker

```python
CB_SPREAD_MULT     = 5.0   # spike = spread > 5× rolling median
CB_SPREAD_BARS     = 15    # bars to suspend after spike
CB_NEWS_WINDOW_MIN = 30    # ±30 minutes around high-impact events
```

### Correlation Filter

```python
CORR_THRESHOLD = 0.70  # scale down when |ρ| > 0.70
CORR_LOOKBACK  = 20    # rolling window (bars) for correlation
```

### Position Sizing

```python
R_MAX           = 0.005  # max fractional risk per bar (0.5% of account)
KELLY_FRACTION  = 0.5    # use half-Kelly for safety
VOL_TARGET_ANNUAL = 0.15  # target 15% annualised portfolio volatility
LOT_SCALE_MAX   = 2.0    # never scale a position above 2× base lot
WARMUP_BARS     = 300    # minimum bars of history before any signal is emitted
```

### Environment Variables (`.env` file)

```bash
# Database
DB_USER=postgres
DB_PASSWORD=YOUR_PASSWORD
DB_HOST=localhost
DB_PORT=5432
DB_NAME=forex_rl

# Telegram (optional)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Prop challenge
PROP_CHALLENGE_BALANCE=0     # 0 = auto-detect from first equity update
PROP_ENABLED=1

# Active instruments
REINFORCE_ACTIVE_SYMBOLS=EURUSD,USA500,XAUUSD
REINFORCE_TIMEFRAME=15m
```

---

## 15. Common Tasks — Quick Reference

### Train more folds for a specific model and symbol
```bash
REINFORCE_SYMBOL=USA500 REINFORCE_TIMEFRAME=15M \
  conda run -n reinforce_rl python parallel/train_tqc.py
```

### Re-run the ensemble optimiser (after adding folds)
```bash
REINFORCE_SYMBOL=EURUSD REINFORCE_TIMEFRAME=15M \
  conda run -n reinforce_rl python parallel/optimize_ensemble.py
```

### Generate a tearsheet for a symbol
```bash
REINFORCE_SYMBOL=USA500 REINFORCE_TIMEFRAME=15M \
  conda run -n reinforce_rl python parallel/tearsheet.py --format html
```

### Run the full ensemble backtest (all available folds)
```bash
REINFORCE_SYMBOL=EURUSD REINFORCE_TIMEFRAME=15M \
  conda run -n reinforce_rl python parallel/cairo_aggregate.py
```

### Build and package new checkpoints (after adding folds + re-optimising)
```bash
cd /home/milad/Desktop/Reinforce
bash deploy/1_train_all_symbols.sh      # ~2-4 hours
bash deploy/2_package_checkpoints.sh    # creates /tmp/cairo_checkpoints_YYYYMMDD.tar.gz
```

### Update VPS with new checkpoints
```bash
scp /tmp/cairo_checkpoints_YYYYMMDD.tar.gz user@VPS_IP:~/
ssh user@VPS_IP
cd ~/reinforce_live && bash deploy/vps_update_checkpoints.sh ~/cairo_checkpoints_YYYYMMDD.tar.gz
```

### Check engine health on VPS
```bash
systemctl --user status cairo_live
journalctl --user -u cairo_live -n 100
```

### Manually trigger rollover on VPS
```bash
ssh user@VPS_IP
cd ~/reinforce_live
REINFORCE_ACTIVE_SYMBOLS=EURUSD,USA500,XAUUSD REINFORCE_TIMEFRAME=15m \
  conda run -n cairo_live python multi_live/05_rollover.py
```

### Test that MT5 can reach the engine
```bash
# On VPS — should return {"status": "ok"}
curl http://localhost:9999/health
```

### Smoke-test a single symbol with fake bars (no MT5 needed)
```bash
REINFORCE_SYMBOL=USA500 REINFORCE_SMOKE_TEST=1 \
  conda run -n reinforce_rl python multi_live/04_live_engine.py
```

---

## Glossary

| Term | Meaning |
|---|---|
| **Bar** | One candlestick — OHLC (Open, High, Low, Close) for a fixed time period (here: 15 minutes) |
| **Fold** | One walk-forward training/test cycle (train 12 months, test 1 month) |
| **CAIRO weight** | The position-size confidence multiplier from ensemble voting (1.0 to ~2.0) |
| **Checkpoint** | The saved state of a trained model — loaded by the engine to make predictions |
| **Scaler** | A normaliser fitted on training data that maps raw feature values to a standardised range |
| **Prop firm** | A proprietary trading firm that gives you their money to trade after you pass a performance challenge |
| **MaxDD** | Maximum Drawdown — the largest peak-to-trough drop in account equity |
| **Sharpe ratio** | Return divided by risk (standard deviation). Higher = better risk-adjusted performance |
| **OOS** | Out-of-Sample — data the model has never seen during training (the test period) |
| **HMM** | Hidden Markov Model — a statistical model for detecting market regime |
| **ETA** | Confidence multiplier for the CAIRO vote; higher ETA = more aggressive scaling |
| **Kelly** | A formula that determines optimal bet size based on win rate and odds |
| **ATR** | Average True Range — a measure of market volatility used in position sizing |
| **ECT** | Equity-Curve Trading — reducing position size when the equity curve is trending down |
| **VIX** | Volatility Index — measures expected market volatility for the S&P 500 |
| **BMA** | Bayesian Model Averaging — updating model weights based on performance after each fold |
