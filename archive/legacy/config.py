"""
config.py — Single source of truth for project-wide configuration.

Loads credentials from .env (never committed).
Copy .env.example → .env and fill in your values.

This file is derived from CAIRO's config.py. All CAIRO constants are preserved
unchanged. RR-specific constants are appended with the RR_ prefix.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (this file's directory)
load_dotenv(Path(__file__).parent / ".env")

# ── Database ──────────────────────────────────────────────────────────────────
_db_user = os.environ.get("DB_USER", "postgres")
_db_password = os.environ.get("DB_PASSWORD", "")
_db_host = os.environ.get("DB_HOST", "localhost")
_db_port = os.environ.get("DB_PORT", "5432")
_db_name = os.environ.get("DB_NAME", "forex_rl")

# SQLAlchemy-style URL (used by parallel/common.py and cairo_visualizer.py)
DB_URL = (
    f"postgresql+psycopg2://{_db_user}:{_db_password}"
    f"@{_db_host}:{_db_port}/{_db_name}"
)

# psycopg2 dict-style config (used by live scripts)
DB_CONFIG = {
    "host": _db_host,
    "port": int(_db_port),
    "user": _db_user,
    "password": _db_password,
    "dbname": _db_name,
}

# Plain postgres:// URL (used by update_db.py subprocess calls)
DB_URL_PLAIN = (
    f"postgresql://{_db_user}:{_db_password}"
    f"@{_db_host}:{_db_port}/{_db_name}"
)

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()

# ── Trading instrument ─────────────────────────────────────────────────────────
# Active symbol — overridden by REINFORCE_SYMBOL env var at runtime.
SYMBOL = os.environ.get("REINFORCE_SYMBOL", "EURUSD").upper()

# ── Walk-forward window sizes (months) ────────────────────────────────────────
TRAIN_MONTHS = 12
TEST_MONTHS  = 1
STEP_MONTHS  = 1

# ── CAIRO ensemble ────────────────────────────────────────────────────────────
ETA = 0.5           # confidence multiplier: weight = 1 + ETA * (votes_for_winner - 1)

# ── Position sizing & costs ───────────────────────────────────────────────────
R_MAX            = 0.005     # max fractional risk per bar (0.5%)
COMMISSION_PRICE = 0.00005   # $5 per lot = 0.5 pips for EURUSD
LOT_SCALE_MAX    = 2.0       # cap on ATR-scaled position multiplier

# ── Live trading / CAIRO monetary mode ───────────────────────────────────────
START_BALANCE = 10_000.0    # starting account balance ($)

BASE_LOT_MAP = {
    "EURUSD":  0.03,
    "GBPUSD":  0.1,
    "USDJPY":  0.1,
    "XAUUSD":  0.02,
    "USA500":  0.5,
    "US30":    0.5,
}


def get_base_lot(symbol: str | None = None) -> float:
    """Return the base lot for the given symbol (defaults to active SYMBOL)."""
    sym = (symbol or SYMBOL).upper()
    return BASE_LOT_MAP.get(sym, 0.1)


# Backward-compatible alias
BASE_LOT = get_base_lot(SYMBOL)

# ── Live engine ───────────────────────────────────────────────────────────────
WARMUP_BARS  = 300
MAX_LOOKBACK = 220

# ── Prop firm challenge limits ────────────────────────────────────────────────
PROP_CHALLENGE_BALANCE = float(os.environ.get("PROP_CHALLENGE_BALANCE", "0"))
PROP_DAILY_LOSS_LIMIT  = 0.045
PROP_TOTAL_DD_LIMIT    = 0.09
PROP_PROFIT_TARGET     = 0.08
PROP_LOT_SOFT_THRESH   = 0.03
PROP_ENABLED           = os.environ.get("PROP_ENABLED", "1") != "0"

# ── News event trading halt ───────────────────────────────────────────────────
NEWS_BLACKOUT_BEFORE_MIN = 5
NEWS_BLACKOUT_AFTER_MIN  = 5

# ── Equity-Curve Trading (ECT) ────────────────────────────────────────────────
ECT_FAST          = int(os.environ.get("ECT_FAST",          "10"))
ECT_SLOW          = int(os.environ.get("ECT_SLOW",          "30"))
ECT_ZSCORE        = float(os.environ.get("ECT_ZSCORE",      "-2.0"))
ECT_PAUSE_BARS    = int(os.environ.get("ECT_PAUSE_BARS",    "24"))
ECT_CAUTION_SCALE = float(os.environ.get("ECT_CAUTION_SCALE","0.5"))
ECT_ENABLED       = os.environ.get("ECT_ENABLED", "1") == "1"

# ── Cross-asset Correlation Filter ────────────────────────────────────────────
CORR_THRESHOLD     = float(os.environ.get("CORR_THRESHOLD",    "0.70"))
CORR_LOOKBACK      = int(os.environ.get("CORR_LOOKBACK",       "20"))
CORR_FILTER_ENABLED = os.environ.get("CORR_FILTER_ENABLED", "1") == "1"

# ── Circuit Breaker ───────────────────────────────────────────────────────────
CB_SPREAD_MULT     = float(os.environ.get("CB_SPREAD_MULT",    "5.0"))
CB_SPREAD_BARS     = int(os.environ.get("CB_SPREAD_BARS",      "15"))
CB_NEWS_WINDOW_MIN = int(os.environ.get("CB_NEWS_WINDOW_MIN",  "30"))
CB_ENABLED         = os.environ.get("CB_ENABLED", "1") == "1"

# ── Hold guard — minimum bars between directional (BUY ↔ SELL) flips ──────────
HOLD_MIN_BARS = int(os.environ.get(
    "REINFORCE_HOLD_MIN_BARS",
    os.environ.get("REINFORCE_MIN_BARS_BETWEEN_FLIPS", "4")
))

# ── Dynamic position sizing (Kelly + vol targeting) ───────────────────────────
KELLY_FRACTION     = 0.5
VOL_TARGET_ANNUAL  = 0.15
VOL_LOOKBACK_BARS  = 20
KELLY_WIN_LOOKBACK = 50

# ── Per-symbol instrument parameters ─────────────────────────────────────────
SYMBOL_PARAMS: dict[str, dict] = {
    "EURUSD": {"spread_fallback": 0.00015, "bars_per_day": 24, "commission": 0.00005, "pip_size": 0.0001, "contract_size": 100_000},
    "GBPUSD": {"spread_fallback": 0.00018, "bars_per_day": 24, "commission": 0.00005, "pip_size": 0.0001, "contract_size": 100_000},
    "USDJPY": {"spread_fallback": 0.015,   "bars_per_day": 24, "commission": 0.05,    "pip_size": 0.01,   "contract_size": 100_000},
    "XAUUSD": {"spread_fallback": 0.35,    "bars_per_day": 24, "commission": 0.30,    "pip_size": 0.1,    "contract_size": 100},
    "USA500": {"spread_fallback": 0.40,    "bars_per_day": 8,  "commission": 0.10,    "pip_size": 0.1,    "contract_size": 1},
    "US30":   {"spread_fallback": 3.0,     "bars_per_day": 8,  "commission": 0.50,    "pip_size": 1.0,    "contract_size": 1},
}
_DEFAULT_SYMBOL_PARAMS = {"spread_fallback": 0.00015, "bars_per_day": 24, "commission": 0.00005, "pip_size": 0.0001, "contract_size": 100_000}

# ── Internal → MT5 terminal symbol name mapping ──────────────────────────────
# Some brokers use different symbol names in MT5 (e.g., US500.c vs USA500).
# Used by config_generator.py to write the correct Symbol= line in the .ini.
# The bridge normalizes the reverse mapping (MT5 → internal) in server_socket.py.
RR_MT5_SYMBOL_MAP: dict[str, str] = {
    "USA500":  "US500.c",
}

# Custom MT5 symbols holding Dukascopy-imported tick history. Used for
# reconciliation runs when the broker feed lacks local tick/M1 history for a
# month (e.g. US500.c January). Same internal symbol, different MT5 symbol.
RR_MT5_CUSTOM_SYMBOL_MAP: dict[str, str] = {
    "USA500":  "US500-tick",
}


def get_symbol_params(symbol: str | None = None) -> dict:
    """Return per-symbol instrument parameters for the given symbol."""
    sym = (symbol or SYMBOL).upper()
    return SYMBOL_PARAMS.get(sym, _DEFAULT_SYMBOL_PARAMS)


# ═══════════════════════════════════════════════════════════════════════════════
# RR-SPECIFIC CONSTANTS (not present in CAIRO config.py)
# ═══════════════════════════════════════════════════════════════════════════════

# ── RR directory structure ────────────────────────────────────────────────────
RR_CHECKPOINT_BASE_DIR = ROOT_DIR / "data" / "checkpoints"
RR_RESULTS_DIR         = ROOT_DIR / "data" / "results"
RR_QUEUE_DIR           = ROOT_DIR / "data" / "queue"
RR_LOGS_DIR            = ROOT_DIR / "data" / "logs"
RR_CONFIGS_DIR         = ROOT_DIR / "configs"
RR_TICK_COVERAGE_PATH  = RR_QUEUE_DIR / "tick_coverage.json"

# ── MT5 Strategy Tester parallelisation ───────────────────────────────────────
RR_TERMINAL_COUNT      = int(os.environ.get("RR_TERMINAL_COUNT",      "3"))
RR_BRIDGE_PORT_START   = int(os.environ.get("RR_BRIDGE_PORT_START",   "9998"))
RR_TEST_MODEL          = int(os.environ.get("RR_TEST_MODEL",          "0"))     # 0=Every tick
RR_TEST_DEPOSIT        = int(os.environ.get("RR_TEST_DEPOSIT",        "10000"))
RR_TEST_LEVERAGE       = int(os.environ.get("RR_TEST_LEVERAGE",       "100"))

# ── EA / DLL paths ────────────────────────────────────────────────────────────
RR_EA_DIR         = ROOT_DIR / "ea"
RR_DLL_NAME       = "bridge_client.dll"
RR_EA_MASTER      = "MULTI_CAIRO_EA.mq5"
RR_EA_TEST_EX5    = "MULTI_CAIRO_EA_TEST.ex5"

# ── Bridge modes ──────────────────────────────────────────────────────────────
# When True, PropFirmGuard uses virtual account limits (tester mode).
# When False, PropFirmGuard applies real prop firm limits (production).
RR_TESTER_MODE = os.environ.get("RR_TESTER_MODE", "1") == "1"

# ── Timeframe ─────────────────────────────────────────────────────────────────
# REINFORCE_TIMEFRAME:   Python training timeframe ("15m", "1h", "5m", "1m")
# RR_MT5_PERIOD:          MT5 Strategy Tester Period value ("M15", "H1", "M5", "M1")
# These MUST stay in sync. All CAIRO training uses 15m; the tester must use M15.
# Changing REINFORCE_TIMEFRAME requires re-training all checkpoints.
REINFORCE_TIMEFRAME = os.environ.get("REINFORCE_TIMEFRAME", "15m").lower()
_MT5_PERIOD_MAP = {"15m": "M15", "1h": "H1", "5m": "M5", "1m": "M1", "15min": "M15"}
RR_MT5_PERIOD = _MT5_PERIOD_MAP.get(REINFORCE_TIMEFRAME, "M15")
