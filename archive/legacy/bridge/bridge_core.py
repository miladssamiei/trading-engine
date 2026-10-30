"""
bridge_core.py — Shared bridge logic: model loading, feature computation,
ensemble voting, risk layer application, and signal generation.

Extracted from CAIRO's multi_live/04_live_engine.py (Phase 1 refactor).
This module has no HTTP dependencies — it is imported by both
server_http.py (production) and server_socket.py (Strategy Tester).
"""

from __future__ import annotations

import collections
import gc
import json
import joblib
import os
import time

import numpy as np
import pandas as pd
import pandas_ta as ta

try:
    from cairo_aggregation import (
        cairo_aggregate,
        cairo_aggregate_regime,
        cairo_aggregate_weighted,
        cairo_aggregate_soft_weighted,
    )
except ImportError:
    cairo_aggregate = None
    cairo_aggregate_regime = None
    cairo_aggregate_weighted = None
    cairo_aggregate_soft_weighted = None

from config import (
    ETA,
    HOLD_MIN_BARS,
    MAX_LOOKBACK,
    WARMUP_BARS,
)
from pathlib import Path


# ── Module-level configuration ────────────────────────────────────────────────
ACTIVE_SYMBOLS = [
    s.strip().upper()
    for s in os.environ.get("REINFORCE_ACTIVE_SYMBOLS", "EURUSD,USA500,XAUUSD").split(",")
    if s.strip()
]
MIN_BARS_BETWEEN_FLIPS = HOLD_MIN_BARS

ALGOS     = ["PPO", "IMPALA", "APPO", "SAC"]
ALGOS_SB3 = ["TQC"]
ALGOS_GBM = ["GBM"]
ALGOS_DT  = ["DT"]


def _log(msg: str) -> None:
    """Minimal logger — print with flush for immediate output."""
    print(msg, flush=True)


# ── Timeframe-dependent annualisation factor for HV_20 ───────────────────────
_ACTIVE_TIMEFRAME = os.environ.get("REINFORCE_TIMEFRAME", "15m").lower()
_BARS_PER_YEAR_FEAT: int = (
    252 * 24 * 60       if _ACTIVE_TIMEFRAME == "1m"  else
    252 * 24 * 60 // 5  if _ACTIVE_TIMEFRAME == "5m"  else
    252 * 14            if _ACTIVE_TIMEFRAME == "1h"  else
    252 * 24
)


# ── Shared column-name helpers ────────────────────────────────────────────────


def gc_col(df, *patterns):
    """Fuzzy column-name lookup: return first column starting with any pattern."""
    for pat in patterns:
        hits = [c for c in df.columns if c.lower().startswith(pat.lower())]
        if hits:
            return df[hits[0]]
    raise KeyError(f"No column starting with {patterns!r}")


# ── Feature computation ───────────────────────────────────────────────────────


def compute_features_df(df_1h: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Compute all 86 indicators on a 1H OHLC DataFrame.

    Extracted verbatim from CAIRO's multi_live/04_live_engine.py.
    """
    df_1h = df_1h.tail(MAX_LOOKBACK)
    O = df_1h["open"]
    H = df_1h["high"]
    L = df_1h["low"]
    C = df_1h["close"]
    feat = pd.DataFrame(index=df_1h.index)

    # Trend
    for n in [5, 10, 20, 50, 100, 200]:
        feat[f"SMA_{n}"] = ta.sma(C, length=n)
        feat[f"EMA_{n}"] = ta.ema(C, length=n)
    feat["WMA_20"]  = ta.wma(C, length=20)
    feat["HMA_20"]  = ta.hma(C, length=20)
    feat["KAMA_10"] = ta.kama(C, length=10)
    feat["DEMA_20"] = ta.dema(C, length=20)
    feat["TEMA_20"] = ta.tema(C, length=20)
    feat["T3_5"]    = ta.t3(C, length=5)
    feat["ALMA_20"] = ta.alma(C, length=20)
    _psar = ta.psar(H, L, C)
    feat["PSAR_VAL"] = gc_col(_psar, "PSARl").fillna(gc_col(_psar, "PSARs"))
    feat["PSAR_DIR"] = np.where(C > feat["PSAR_VAL"], 1.0, -1.0)
    _ichi, _ = ta.ichimoku(H, L, C, lookahead=False)
    feat["ICH_TENKAN"]   = gc_col(_ichi, "ITS")
    feat["ICH_KIJUN"]    = gc_col(_ichi, "IKS")
    feat["ICH_SENKOU_A"] = gc_col(_ichi, "ISA")
    feat["ICH_SENKOU_B"] = gc_col(_ichi, "ISB")
    _adx14 = ta.adx(H, L, C, length=14)
    feat["ADX_14"]   = gc_col(_adx14, "ADX_14")
    feat["DI_PLUS"]  = gc_col(_adx14, "DMP_14")
    feat["DI_MINUS"] = gc_col(_adx14, "DMN_14")
    feat["ADX_25"]   = gc_col(ta.adx(H, L, C, length=25), "ADX_25")
    _aroon = ta.aroon(H, L, length=25)
    feat["AROON_UP"]   = gc_col(_aroon, "AROONU")
    feat["AROON_DOWN"] = gc_col(_aroon, "AROOND")
    feat["AROON_OSC"]  = gc_col(_aroon, "AROONOSC")
    feat["LINREG_SLOPE"] = ta.linreg(C, length=14, slope=True)
    _st = ta.supertrend(H, L, C, length=10, multiplier=3.0)
    feat["SUPERT_VAL"] = gc_col(_st, "SUPERT_")
    feat["SUPERT_DIR"] = gc_col(_st, "SUPERTd")

    # Momentum
    for n in [7, 14, 21]:
        feat[f"RSI_{n}"] = ta.rsi(C, length=n)
    _macd = ta.macd(C, fast=12, slow=26, signal=9)
    feat["MACD"]        = gc_col(_macd, "MACD_")
    feat["MACD_SIGNAL"] = gc_col(_macd, "MACDs")
    feat["MACD_HIST"]   = gc_col(_macd, "MACDh")
    _stoch = ta.stoch(H, L, C, k=14, d=3)
    feat["STOCH_K"] = gc_col(_stoch, "STOCHk")
    feat["STOCH_D"] = gc_col(_stoch, "STOCHd")
    feat["STOCHRSI_K"] = gc_col(ta.stochrsi(C, length=14), "STOCHRSIk")
    feat["CCI_20"]   = ta.cci(H, L, C, length=20)
    feat["WILLR_14"] = ta.willr(H, L, C, length=14)
    feat["ROC_10"]   = ta.roc(C, length=10)
    feat["MOM_10"]   = ta.mom(C, length=10)
    feat["UO"]       = ta.uo(H, L, C, fast=7, medium=14, slow=28)
    feat["CMO_14"]   = ta.cmo(C, length=14)
    feat["DPO_20"]   = ta.dpo(C, length=20).squeeze()
    feat["FISHER"]   = gc_col(ta.fisher(H, L, length=9), "FISHERT_")
    feat["AO"]       = ta.ao(H, L)
    feat["APO"]      = ta.apo(C, fast=12, slow=26)
    feat["TRIX"]     = gc_col(ta.trix(C, length=15), "TRIX_")
    feat["COPPOCK"]  = ta.coppock(C)
    feat["PPO"]      = gc_col(ta.ppo(C, fast=12, slow=26), "PPO_")
    _ema13 = ta.ema(C, length=13)
    feat["ELDER_BULL"] = H - _ema13
    feat["ELDER_BEAR"] = L - _ema13

    # Volatility
    feat["ATR_7"]   = ta.atr(H, L, C, length=7)
    feat["ATR_14"]  = ta.atr(H, L, C, length=14)
    feat["NATR_14"] = ta.natr(H, L, C, length=14)
    _bb = ta.bbands(C, length=20, std=2.0)
    feat["BB_UPPER"] = gc_col(_bb, "BBU")
    feat["BB_LOWER"] = gc_col(_bb, "BBL")
    feat["BB_WIDTH"] = gc_col(_bb, "BBB")
    feat["BB_PCT"]   = gc_col(_bb, "BBP")
    _kc = ta.kc(H, L, C, length=20, scalar=2)
    feat["KC_UPPER"] = gc_col(_kc, "KCUe")
    feat["KC_LOWER"] = gc_col(_kc, "KCLe")
    _dc = ta.donchian(H, L, length=20)
    feat["DC_UPPER"] = gc_col(_dc, "DCU")
    feat["DC_LOWER"] = gc_col(_dc, "DCL")
    _log_ret = np.log(C / C.shift(1))
    feat["HV_20"] = _log_ret.rolling(20).std() * np.sqrt(_BARS_PER_YEAR_FEAT)
    _rv5  = _log_ret.rolling(5).std()
    _rv20 = _log_ret.rolling(20).std().replace(0, np.nan)
    feat["RVR"]   = (_rv5 / _rv20).clip(0, 10)
    _hl_ema = ta.ema(H - L, length=10)
    feat["CHAIKV"] = (
        (_hl_ema - _hl_ema.shift(10)) / _hl_ema.shift(10).replace(0, np.nan) * 100
    )

    # Price action
    feat["LOG_RETURN"] = np.log(C / C.shift(1))
    feat["RETURN_1P"]  = C.pct_change(1)
    feat["HL_RANGE"]   = (H - L) / C
    feat["BODY"]       = (C - O).abs() / C
    feat["UPPER_WICK"] = (H - np.maximum(O, C)) / C
    feat["LOWER_WICK"] = (np.minimum(O, C) - L) / C
    feat["GAP"]        = (O - C.shift(1)) / C.shift(1)
    PP = (H + L + C) / 3
    feat["PIVOT"] = PP
    feat["R1"]    = 2 * PP - L
    feat["S1"]    = 2 * PP - H
    feat["R2"]    = PP + (H - L)
    feat["S2"]    = PP - (H - L)
    _ha_close = (O + H + L + C) / 4
    _ha_open  = (O.shift(1) + C.shift(1)) / 2
    feat["HA_CLOSE"] = _ha_close
    feat["HA_BODY"]  = _ha_close - _ha_open

    if os.environ.get("USE_REGIME_FEATURES", "1") == "1":
        try:
            from regime_detector import get_detector as _get_detector
            _rd = _get_detector(auto_load=True)
            if _rd is not None:
                onehot = _rd.regime_onehot(df_1h)
                feat["REGIME_0"] = onehot[0]
                feat["REGIME_1"] = onehot[1]
                feat["REGIME_2"] = onehot[2]
        except Exception:
            feat["REGIME_0"] = feat["REGIME_1"] = feat["REGIME_2"] = 0.0

    if os.environ.get("USE_MACRO_FEATURES", "0") == "1":
        try:
            from macro_features import add_macro_cols as _add_macro
            feat = _add_macro(feat)
        except Exception:
            from macro_features import macro_feature_names as _mnms
            for _c in _mnms():
                if _c not in feat.columns:
                    feat[_c] = 0.0

    return feat[feature_cols].fillna(0.0)


# ── Model loading and inference ───────────────────────────────────────────────


class LiveModels:
    """Loads and manages all trained algo objects for a single symbol.

    Extracted from CAIRO's multi_live/04_live_engine.py.
    """

    def __init__(self, symbol: str, fold_dir: Path):
        self.symbol       = symbol
        self.fold_dir     = fold_dir
        self.algos:        dict = {}
        self._algo_types:  dict = {}
        self.scaler        = None
        self.splits        = None
        self.feature_cols: list = []
        self.model_weights: dict = {}
        self.base_lot:     float = 0.01
        self._cairo_warned     = False

    def load(self):
        _log(f"  [{self.symbol}] Loading live fold metadata...")
        splits_path = self.fold_dir / "splits.json"
        if not splits_path.exists():
            raise FileNotFoundError(
                f"{splits_path} not found. "
                f"Run 02_build_live_fold.py and 03_train_live_fold.py for {self.symbol} first."
            )
        self.splits       = json.loads(splits_path.read_text())
        self.feature_cols = self.splits["feature_cols"]

        _log(f"  [{self.symbol}] Loading scaler...")
        scaler_path = Path(self.splits["scaler_pkl"])
        if not scaler_path.exists():
            scaler_path = self.fold_dir / "scalers" / scaler_path.name
        self.scaler = joblib.load(scaler_path)

        # --- Model weight loading (fallback chain) ---
        fold_id = self.splits.get("fold_id")
        fold_weights_path = (
            Path(__file__).parent.parent / "engine_v2" / "fold" / "folds"
            / str(fold_id) / "cairo_weights.json"
        )

        self.model_weights = {}
        loaded_from_fold = False

        if fold_weights_path.exists():
            try:
                w_dict = json.loads(fold_weights_path.read_text())
                self.model_weights = {k.upper(): v for k, v in w_dict.items()
                                      if k.lower() != "comment"}
                _log(f"  [{self.symbol}] [Cairo] Loaded ensemble weights from engine_v2 fold config: "
                     f"{list(self.model_weights.keys())}")
                loaded_from_fold = True
            except Exception as e:
                _log(f"  [{self.symbol}] [WARN] Failed to load cairo_weights.json: {e}")

        if not loaded_from_fold:
            port_dir = (
                Path(__file__).parent.parent / "parallel" / "data"
                / "PORTFOLIO" / _ACTIVE_TIMEFRAME.upper()
            )
            port_cfg_path = port_dir / "portfolio_sizing.json"
            if port_cfg_path.exists():
                try:
                    _sz = json.loads(port_cfg_path.read_text())
                    if "symbols" in _sz and self.symbol in _sz["symbols"]:
                        sym_cfg = _sz["symbols"][self.symbol]
                        optimal_subset = sym_cfg["models"]
                        w_dict = sym_cfg.get("model_weights", {})
                        self.model_weights = {algo: w_dict.get(algo, 1.0)
                                              for algo in optimal_subset}
                        self.base_lot = float(sym_cfg.get("base_lot", self.base_lot))
                        _log(f"  [{self.symbol}] Portfolio optimal subset loaded: {optimal_subset}"
                             + f" | base_lot={self.base_lot}")
                        _log(f"  [{self.symbol}] Portfolio optimal subset loaded: {optimal_subset}")
                        _log(f"  [{self.symbol}] Loaded weights: "
                             + ", ".join(f"{k}={v:.3f}" for k, v in self.model_weights.items()))
                except Exception as e:
                    _log(f"  [{self.symbol}] Failed to parse portfolio_sizing.json: {e}")

        if not self.model_weights:
            sizing_path = self.fold_dir / "sizing_results.json"
            if sizing_path.exists():
                _sz = json.loads(sizing_path.read_text())
                self.model_weights = _sz.get("model_weights", {})
                self.base_lot = float(_sz.get("base_lot", self.base_lot))
                if self.model_weights:
                    _log(f"  [{self.symbol}] Fold weights: "
                         + ", ".join(f"{k}={v:.3f}" for k, v in self.model_weights.items())
                         + f" | base_lot={self.base_lot}")
            else:
                _log(f"  [{self.symbol}] sizing_results.json not found — using uniform vote")

        # --- Ray init + RLlib model loading ---
        import ray
        from ray.rllib.policy.policy import Policy

        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                num_cpus=1, num_gpus=0,
                include_dashboard=False, log_to_driver=False,
                _temp_dir="/tmp/ray_multi_live",
            )

        for algo in ALGOS:
            if self.model_weights and algo not in self.model_weights:
                continue
            policy_dir = (
                self.fold_dir / "checkpoints"
                / f"{algo.lower()}_policy" / "policies" / "default_policy"
            )
            if not policy_dir.exists():
                _log(f"  [{self.symbol}] WARNING: {algo} not found at {policy_dir}")
                continue
            try:
                policy = Policy.from_checkpoint(str(policy_dir))
                self.algos[algo] = policy
                self._algo_types[algo] = "rllib"
                _log(f"  [{self.symbol}] [OK] {algo} loaded (RLlib)")
            except Exception as e:
                _log(f"  [{self.symbol}] [FAIL] {algo} load failed: {e}")

        for algo in ALGOS_SB3:
            if self.model_weights and algo not in self.model_weights:
                continue
            model_path = self.fold_dir / "checkpoints" / f"{algo.lower()}_model.zip"
            if not model_path.exists():
                _log(f"  [{self.symbol}] INFO: {algo} not found — skipping")
                continue
            try:
                from sb3_contrib import TQC
                model = TQC.load(str(model_path))
                self.algos[algo] = model
                self._algo_types[algo] = "sb3"
                _log(f"  [{self.symbol}] [OK] {algo} loaded (SB3)")
            except Exception as e:
                _log(f"  [{self.symbol}] [FAIL] {algo} load failed: {e}")

        for algo in ALGOS_GBM:
            if self.model_weights and algo not in self.model_weights:
                continue
            model_path = self.fold_dir / "checkpoints" / f"{algo.lower()}_model.txt"
            if not model_path.exists():
                _log(f"  [{self.symbol}] INFO: {algo} not found — skipping")
                continue
            try:
                import lightgbm as lgb
                model = lgb.Booster(model_file=str(model_path))
                self.algos[algo] = model
                self._algo_types[algo] = "gbm"
                _log(f"  [{self.symbol}] [OK] {algo} loaded (LightGBM)")
            except Exception as e:
                _log(f"  [{self.symbol}] [FAIL] {algo} load failed: {e}")

        for algo in ALGOS_DT:
            if self.model_weights and algo not in self.model_weights:
                continue
            model_path = self.fold_dir / "checkpoints" / f"{algo.lower()}_model.pt"
            if not model_path.exists():
                continue
            try:
                import sys as _sys
                _reinforce_parallel = (
                    Path(__file__).parent.parent.parent
                    / "Reinforce" / "Reinforce_Windows_20260613" / "parallel"
                )
                if _reinforce_parallel.exists() and str(_reinforce_parallel) not in _sys.path:
                    _sys.path.insert(0, str(_reinforce_parallel))
                import torch
                from train_dt import DecisionTransformer
                ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
                cfg  = ckpt["cfg"]
                n_feat = ckpt.get("n_features", len(self.feature_cols))
                dt_model = DecisionTransformer(n_features=n_feat, cfg=cfg)
                dt_model.load_state_dict(ckpt["model"])
                dt_model.eval()
                self.algos[algo] = dt_model
                self._algo_types[algo] = "dt"
                self._dt_context = {
                    "K":       cfg.get("K", 20),
                    "states":  [],
                    "actions": [],
                    "rtg":     [],
                    "step":    0,
                }
                _log(f"  [{self.symbol}] [OK] {algo} loaded (DT K={cfg.get('K', 20)})")
            except Exception as e:
                _log(f"  [{self.symbol}] [FAIL] {algo} load failed: {e}")

        if not self.algos:
            raise RuntimeError(
                f"No models loaded for {self.symbol}. Train the live fold first."
            )

        if self.model_weights:
            missing = [a for a in self.model_weights if a not in self.algos]
            if missing:
                _log(f"  [{self.symbol}] WARNING: portfolio weights include missing models: {missing}. "
                     f"Loaded: {list(self.algos.keys())}. "
                     f"This WILL diverge from 08_simulate_live.py / tester_bridge_parallel.py.")

        fold_id_env = os.environ.get("REINFORCE_FOLD_ID")
        fold_id_disk = self.splits.get("fold_id") if self.splits else None
        if fold_id_env and fold_id_disk and str(fold_id_env) != str(fold_id_disk):
            _log(f"  [{self.symbol}] WARNING: REINFORCE_FOLD_ID={fold_id_env} requested, "
                 f"but live_fold/splits.json is for fold {fold_id_disk}. "
                 f"Run set_historical_fold.py first to switch historical folds.")

    # ── Inference helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _infer_rllib(model, obs_scaled: np.ndarray) -> tuple[int, np.ndarray]:
        action, _, info = model.compute_single_action(obs_scaled, explore=False)
        logits = info.get("action_dist_inputs")
        if logits is not None:
            lp = np.array(logits, dtype=np.float32)
            lp -= lp.max()
            p  = np.exp(lp); p /= p.sum()
        else:
            p = np.zeros(3, dtype=np.float32); p[int(action)] = 1.0
        return int(action), p

    @staticmethod
    def _infer_sb3(model, obs_scaled: np.ndarray) -> tuple[int, np.ndarray]:
        def _cont_to_discrete(cont_action: float) -> int:
            if cont_action < -0.33:
                return 0   # SELL
            if cont_action > 0.33:
                return 2   # BUY
            return 1       # HOLD

        det_arr, _ = model.predict(obs_scaled.reshape(1, -1), deterministic=True)
        action = _cont_to_discrete(float(np.asarray(det_arr).flatten()[0]))

        n_samples = 32   # matches tqc_infer_probs default in parallel/train_tqc.py
        obs_batch = np.repeat(obs_scaled.reshape(1, -1), n_samples, axis=0)
        stoch_arr, _ = model.predict(obs_batch, deterministic=False)
        counts = np.zeros(3, dtype=np.int32)
        for val in np.asarray(stoch_arr).flatten():
            counts[_cont_to_discrete(float(val))] += 1
        if counts.sum() == 0:
            p = np.zeros(3, dtype=np.float32)
            p[action] = 1.0
        else:
            p = (counts / counts.sum()).astype(np.float32)
        return action, p

    @staticmethod
    def _infer_gbm(model, obs_scaled: np.ndarray) -> tuple[int, np.ndarray]:
        p_2d = model.predict(obs_scaled.reshape(1, -1))
        p = p_2d[0].astype(np.float32) if p_2d.ndim == 2 else p_2d.astype(np.float32)
        return int(p.argmax()), p

    def _infer_dt(self, model, obs_scaled: np.ndarray) -> tuple[int, np.ndarray]:
        import torch
        ctx = self._dt_context
        K   = ctx["K"]
        ctx["states"].append(obs_scaled.copy())
        if not ctx["actions"]:
            ctx["actions"].append(1)
        ctx["step"] += 1
        for key in ("states", "actions"):
            if len(ctx[key]) > K:
                ctx[key] = ctx[key][-K:]
        pad         = K - len(ctx["states"])
        states_arr  = np.zeros((K, len(obs_scaled)), dtype=np.float32)
        actions_arr = np.ones(K, dtype=np.int64)
        # Constant target RTG of 0.08 at every context position — matches
        # train_dt.infer_probs (ctx_rtg = np.full(K, target_rtg), target_rtg=0.08).
        # The previous 0.08/(K-len) conditioning diverged from the Python
        # evaluator and produced different DT actions.
        rtg_arr     = np.full(K, 0.08, dtype=np.float32)
        ts_arr      = np.zeros(K, dtype=np.int64)
        for j, s in enumerate(ctx["states"]):  states_arr[pad + j]  = s
        for j, a in enumerate(ctx["actions"]): actions_arr[pad + j] = a
        for j in range(len(ctx["states"])):
            ts_arr[pad + j] = min(ctx["step"] - len(ctx["states"]) + j,
                                  model.cfg.get("max_timestep", 2000))
        with torch.no_grad():
            logits = model(
                torch.tensor(states_arr).unsqueeze(0),
                torch.tensor(actions_arr).unsqueeze(0),
                torch.tensor(rtg_arr).unsqueeze(0),
                torch.tensor(ts_arr).unsqueeze(0),
            )
            p = torch.softmax(logits[0, -1, :], dim=0).numpy().astype(np.float32)
        action = int(p.argmax())
        ctx["actions"].append(action)
        if len(ctx["actions"]) > K:
            ctx["actions"] = ctx["actions"][-K:]
        return action, p

    def predict(self, obs_raw: np.ndarray, regime_label: int = 1) -> tuple[str, dict]:
        """Scale obs_raw, run all models, CAIRO-aggregate, return (action_str, details)."""
        obs_scaled = np.clip(
            self.scaler.transform(obs_raw.reshape(1, -1)), -10, 10
        ).astype(np.float32)[0]

        _infer_dispatch = {
            "rllib": self._infer_rllib,
            "sb3":   self._infer_sb3,
            "gbm":   self._infer_gbm,
            "dt":    self._infer_dt,
        }
        action_map      = {0: "SELL", 1: "HOLD", 2: "BUY"}
        probs_dict:      dict[str, np.ndarray] = {}
        model_votes_raw: dict[str, str]        = {}

        for algo, model_obj in self.algos.items():
            infer_fn = _infer_dispatch.get(self._algo_types.get(algo, "rllib"))
            if infer_fn is None:
                continue

            algo_type = self._algo_types.get(algo, "rllib")
            expected_dim = len(obs_scaled)
            if algo_type == "gbm":
                expected_dim = model_obj.num_feature()
            elif algo_type == "sb3":
                expected_dim = model_obj.observation_space.shape[0]
            elif algo_type == "rllib":
                expected_dim = model_obj.observation_space.shape[0]
            elif algo_type == "dt":
                expected_dim = model_obj.state_embed.in_features

            model_obs = obs_scaled
            if len(obs_scaled) != expected_dim:
                if len(obs_scaled) > expected_dim:
                    model_obs = obs_scaled[:expected_dim]
                else:
                    model_obs = np.pad(obs_scaled, (0, expected_dim - len(obs_scaled)))

            try:
                action, p             = infer_fn(model_obj, model_obs)
                probs_dict[algo]      = p.reshape(1, 3)
                model_votes_raw[algo] = action_map.get(action, "HOLD")
            except Exception as e:
                _log(f"  [{self.symbol}/{algo}] Infer error: {e}")

        if not probs_dict:
            return "HOLD", {"reason": "no_models"}

        use_regime = os.environ.get("USE_REGIME_FEATURES", "1") == "1"
        try:
            if self.model_weights and cairo_aggregate_soft_weighted is not None:
                actions, weights, contributors = cairo_aggregate_soft_weighted(
                    probs_dict, self.model_weights)
            elif use_regime and cairo_aggregate_regime is not None:
                actions, weights, contributors = cairo_aggregate_regime(
                    probs_dict, regime_label=regime_label)
            elif cairo_aggregate is not None:
                actions, weights, contributors = cairo_aggregate(probs_dict)
            else:
                raise RuntimeError("No CAIRO aggregation function available")
        except Exception as e:
            if not self._cairo_warned:
                _log(f"[{self.symbol}] CAIRO error: {e} — fallback majority vote "
                     f"(suppressing further CAIRO warnings)")
                self._cairo_warned = True
            counts = {0: 0, 1: 0, 2: 0}
            for p in probs_dict.values():
                counts[int(p[0].argmax())] += 1
            max_c  = max(counts.values())
            tops   = [a for a, c in counts.items() if c == max_c]
            winner = tops[0] if len(tops) == 1 else 1
            actions      = np.array([winner], dtype=np.int32)
            weights      = np.array([1.0 + ETA * max(0, counts[winner] - 1)],
                                    dtype=np.float32)
            contributors = list(probs_dict.keys())

        final_action = int(actions[0])
        cairo_weight = float(weights[0])
        action_str   = action_map.get(final_action, "HOLD")
        winner_votes = sum(1 for p in probs_dict.values()
                           if int(p[0].argmax()) == final_action)

        # Pooled soft-weighted ensemble distribution + per-model probs for
        # decision logging (diagnostics). Never affects the action itself.
        try:
            _algos_l = list(probs_dict.keys())
            _raw_w = np.array([self.model_weights.get(a, 1.0) for a in _algos_l],
                              dtype=np.float64)
            _raw_w = np.maximum(_raw_w, 0.0)
            _w = _raw_w / _raw_w.sum() if _raw_w.sum() > 0 else np.ones(len(_algos_l)) / len(_algos_l)
            _pooled = np.zeros(3, dtype=np.float64)
            for _i, _a in enumerate(_algos_l):
                _pooled += _w[_i] * probs_dict[_a][0].astype(np.float64)
            ens_probs = [round(float(x), 6) for x in _pooled]
            model_probs = {a: [round(float(x), 6) for x in probs_dict[a][0]]
                           for a in _algos_l}
        except Exception:
            ens_probs = None
            model_probs = None

        return action_str, {
            "cairo_action": action_str,
            "cairo_weight": cairo_weight,
            "winner_votes": winner_votes,
            "n_models":     len(probs_dict),
            "model_votes":  model_votes_raw,
            "used_algos":   contributors,
            "regime":       regime_label if use_regime else None,
            "ens_probs":    ens_probs,
            "model_probs":  model_probs,
        }

    def reload(self):
        self.algos.clear()
        gc.collect()
        self.load()


# ── Bar validation and buffering ──────────────────────────────────────────────


class BarValidator:
    """Validates incoming bars. Rejects price/spread spikes, zero-range, duplicate timestamps.

    Per-instrument floors are derived from 1st-percentile of rolling 100-bar std.
    """

    _MIN_SIGMA: dict[str, float] = {
        "EURUSD": 0.00048,
        "GBPUSD": 0.00055,
        "USDJPY": 0.050,
        "USA500": 2.70,
        "US30":   8.0,
        "XAUUSD": 1.40,
    }
    _DEFAULT_MIN_SIGMA = 0.001

    _MAX_SPREAD_ABS: dict[str, float] = {
        "EURUSD": 0.004,
        "GBPUSD": 0.005,
        "USDJPY": 0.5,
        "USA500": 2.0,
        "US30":   5.0,
        "XAUUSD": 10.0,
    }

    def __init__(self, symbol: str = "",
                 lookback: int = 100,
                 price_sigma_threshold: float = 5.0,
                 spread_multiplier_max: float = 10.0):
        self._symbol  = symbol.upper()
        self._prices  = collections.deque(maxlen=lookback)
        self._spreads = collections.deque(maxlen=lookback)
        self._last_ts: float | None = None
        self._last_bar_recv: float  = time.time()
        self._sigma_thresh          = price_sigma_threshold
        self._spread_max            = spread_multiplier_max
        self._rejection_counts: dict[str, int] = {}
        self._min_sigma     = self._MIN_SIGMA.get(self._symbol, self._DEFAULT_MIN_SIGMA)
        self._max_spread_abs = self._MAX_SPREAD_ABS.get(self._symbol, None)

    def validate(self, bar: dict) -> tuple[bool, str]:
        ts = float(bar.get("timestamp", 0))
        if ts and ts == self._last_ts:
            return self._reject("duplicate_timestamp")
        try:
            high   = float(bar["high"])
            low    = float(bar["low"])
            price  = float(bar["close"])
            spread = float(bar.get("spread", bar.get("avg_spread", 0.0)))
        except (KeyError, TypeError, ValueError) as e:
            return self._reject(f"malformed_bar:{e}")
        if high == low:
            return self._reject("zero_range_bar")
        if len(self._prices) >= 10:
            arr = np.asarray(self._prices)
            mu  = arr.mean()
            sigma = max(arr.std(), self._min_sigma)
            if abs(price - mu) > self._sigma_thresh * sigma:
                return self._reject(
                    f"price_spike:{price:.4f}_mu:{mu:.4f}_sigma:{sigma:.6f}")
        if spread > 0 and len(self._spreads) >= 10:
            med = float(np.median(self._spreads))
            if med > 0 and spread > self._spread_max * med:
                abs_cap = self._max_spread_abs
                if abs_cap is None or spread > abs_cap:
                    return self._reject(
                        f"spread_spike:{spread:.6f}_median:{med:.6f}")
        if ts:
            self._last_ts = ts
        self._prices.append(price)
        if spread > 0:
            self._spreads.append(spread)
        self._last_bar_recv = time.time()
        return True, ""

    def heartbeat_ok(self, bar_interval_s: int = 3600) -> bool:
        return (time.time() - self._last_bar_recv) < 2 * bar_interval_s

    def _reject(self, reason: str) -> tuple[bool, str]:
        self._rejection_counts[reason] = self._rejection_counts.get(reason, 0) + 1
        return False, reason


class BarBuffer:
    """Rolling bar buffer per symbol. Converts Bid to Mid prices."""

    MAX_BARS = 500
    _COLS = ["open", "high", "low", "close", "avg_spread"]

    def __init__(self):
        self.bars: pd.DataFrame = pd.DataFrame(columns=self._COLS).astype("float64")

    def append(self, bar: dict):
        ts  = pd.Timestamp(bar["timestamp"], unit="s", tz="UTC")
        spread = float(bar.get("spread", bar.get("avg_spread", 0.0001)))
        half_spread = spread * 0.5
        row = pd.DataFrame([{
            "open":       float(bar.get("open", bar.get("close", 1.0))) + half_spread,
            "high":       float(bar["high"]) + half_spread,
            "low":        float(bar["low"]) + half_spread,
            "close":      float(bar["close"]) + half_spread,
            "avg_spread": spread,
        }], index=[ts])
        self.bars = (
            pd.concat([self.bars, row])
              [~pd.concat([self.bars, row]).index.duplicated(keep="last")]
              .sort_index()
              .tail(self.MAX_BARS)
              .astype("float64")
        )

    def ready(self):
        return len(self.bars) >= WARMUP_BARS

    def __len__(self):
        return len(self.bars)


# ── Signal generation pipeline ────────────────────────────────────────────────


def process_bar_core(
    bar: dict,
    models: LiveModels,
    buffer: BarBuffer,
    validator: BarValidator,
    feature_cols: list,
    prev_action: str = "HOLD",
    bars_since_flip: int = 0,
) -> tuple[str, dict, float, int]:
    """Core signal pipeline: validate → buffer → features → model predict.

    Returns (action_str, details_dict, elapsed_ms, new_bars_since_flip).

    Risk layers (ECT, CircuitBreaker, PropFirmGuard), position sizing,
    and side effects (logging, notifications) are handled by the caller.
    """
    # 1. Bar validation
    valid, reject_reason = validator.validate(bar)
    if not valid:
        if validator._rejection_counts.get("_suppress_warned"):
            pass
        else:
            _log(f"[BarValidator] [{models.symbol}] Rejected: {reject_reason} "
                 f"(suppressing further rejection logs)")
            validator._rejection_counts["_suppress_warned"] = 1
        return "HOLD", {"reason": reject_reason, "guard_reason": "bar_rejected"}, 0.0, bars_since_flip

    # 2. Buffer append + warmup check
    buffer.append(bar)
    if not buffer.ready():
        n = len(buffer)
        if n == 1 or n % 50 == 0 or n == WARMUP_BARS:
            _log(f"[Bar] [{models.symbol}] Warmup: {n}/{WARMUP_BARS} — HOLD")
        return "HOLD", {"reason": "warmup", "warmup_bars": f"{len(buffer)}/{WARMUP_BARS}"}, 0.0, bars_since_flip

    # 3. Feature computation
    t0 = time.time()
    feat_df = compute_features_df(buffer.bars, feature_cols)
    obs_raw = feat_df.iloc[-1].values.astype(np.float32)

    # 4. Regime detection
    regime_label = 1
    if os.environ.get("USE_REGIME_FEATURES", "1") == "1":
        try:
            from regime_detector import get_detector as _get_rd
            _rd = _get_rd(auto_load=False)
            if _rd is not None:
                regime_label = _rd.predict_regime(buffer.bars)
        except Exception:
            pass

    # 5. Model prediction + CAIRO aggregation
    action_str, details = models.predict(obs_raw, regime_label=regime_label)
    elapsed_ms = (time.time() - t0) * 1000

    # 6. Hold persistence filter (anti-overtrading)
    if action_str in ("BUY", "SELL") and action_str != prev_action:
        if prev_action in ("BUY", "SELL") and bars_since_flip < MIN_BARS_BETWEEN_FLIPS:
            _log(f"[HoldGuard] [{models.symbol}] Preventing flip to {action_str}, "
                 f"holding {prev_action} ({bars_since_flip}/{MIN_BARS_BETWEEN_FLIPS} bars)")
            action_str = prev_action
        else:
            bars_since_flip = 0

    if action_str == prev_action:
        bars_since_flip += 1

    return action_str, details, elapsed_ms, bars_since_flip
