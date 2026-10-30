"""
config_generator.py — Generate Strategy Tester .ini configs for one (symbol, fold) pair.

Produces two files:
  1. tester_config.ini — MT5 Strategy Tester configuration
  2. ea_params.ini     — EA input parameters (bridge port, symbol, checkpoint, etc.)

CP_UTF8 ENCODING NOTE:
  Both .ini files are written in UTF-8 (WITHOUT BOM — the terminal and EA
  both default to the system ANSI code page for file reads).  Since all our
  parameter names and values are ASCII (no non-ASCII characters in symbol
  names, file paths, or numeric values), this is safe.
  
  If future parameters include non-ASCII characters, the EA MUST use
  FileReadArray + CharArrayToString(..., CP_UTF8) instead of FileReadString.
  This is the same CP_ACP / CP_UTF8 issue documented in PROGRESS.md
  § "CP_ACP JSON corruption fix (2026-08-05)".
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import (
    RR_BRIDGE_PORT_START,
    RR_EA_TEST_EX5,
    RR_MT5_PERIOD,
    RR_TEST_DEPOSIT,
    RR_TEST_LEVERAGE,
    RR_TEST_MODEL,
    RR_MT5_SYMBOL_MAP,
    RR_MT5_CUSTOM_SYMBOL_MAP,
)


def _log(msg: str) -> None:
    print(f"[ConfigGen] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# MT5 Strategy Tester .ini (consumed by terminal64.exe /config:<path>)
# ═══════════════════════════════════════════════════════════════════════════

TESTER_INI_TEMPLATE = """[Common]
Login=
Password=
Server=
EnableDDE=0
EnableNews=0

[Tester]
Expert={ea_name}
Symbol={symbol}
Period={period}
Model={model}
Optimization=0
FromDate={from_date}
ToDate={to_date}
ForwardMode=0
Deposit={deposit}
Currency={currency}
Leverage={leverage}
ExecutionMode=0
OptimizationCriterion=0
VisualMode=0
FrameMode=0
Report={html_report_path}
TestReport={xml_report_path}
ReplaceReport=1
ShutdownTerminal=1
"""


# ═══════════════════════════════════════════════════════════════════════════
# EA input parameters .ini (read by the EA via OnTesterInit / input vars)
# ═══════════════════════════════════════════════════════════════════════════

EA_PARAMS_TEMPLATE = """[Common]
; EA input parameters for fold validation
; Generated: {generated_at}
; Symbol:    {symbol}
; Fold:      {fold_id}
; Port:      {bridge_port}

[Tester]
BridgePort={bridge_port}
symbol={symbol}
checkpoint_path={checkpoint_path}
tick_coverage_pct={tick_coverage_pct}

[Bridge]
alive_timeout_s=30
max_retry_per_bar=3
"""


# ═══════════════════════════════════════════════════════════════════════════

class ConfigGenerator:
    """Generates tester config + EA params for one (symbol, fold) pair."""

    def __init__(self, output_dir: str = ""):
        self._output_dir = Path(output_dir) if output_dir else ROOT_DIR / "configs"

    def generate(
        self,
        symbol: str,
        fold_id: str,
        checkpoint_path: str,
        bridge_port: int,
        from_date: str = "2025-01-01",
        to_date: str = "2026-01-01",
        period: str = RR_MT5_PERIOD,
        deposit: int = RR_TEST_DEPOSIT,
        leverage: int = RR_TEST_LEVERAGE,
        model: int = RR_TEST_MODEL,
        currency: str = "USD",
        tick_coverage_pct: float = 100.0,
        ea_name: str = "",
        ea_params: dict | None = None,
        data_source: str = "broker_live",
    ) -> dict[str, Path]:
        """Generate both .ini files. Returns {"tester_config": Path, "ea_params": Path}."""
        ea_params = ea_params or {}

        self._output_dir.mkdir(parents=True, exist_ok=True)

        safe_fold_id = fold_id.replace("/", "_").replace("\\", "_")

        # ── Strategy Tester config ────────────────────────────────────
        actual_ea = ea_name or RR_EA_TEST_EX5.replace(".ex5", "")
        if data_source == "custom_dukascopy_tick":
            mt5_symbol = RR_MT5_CUSTOM_SYMBOL_MAP.get(symbol, symbol)
        else:
            mt5_symbol = RR_MT5_SYMBOL_MAP.get(symbol, symbol)
        # Build EA input parameter lines (each on its own line under [Tester])
        # Note: MT5 batch .ini supports individual key=value lines for certain params
        # (e.g., BridgePort). For other EA inputs, use the EA's own auto-detection.
        extra_params_lines = [f"{key}={val}" for key, val in ea_params.items()]
        extra_params_block = "\n".join(extra_params_lines)
        html_report_path = str(
            self._output_dir / f"report_{symbol}_{safe_fold_id}.html"
        )
        xml_report_path = str(
            self._output_dir / f"report_{symbol}_{safe_fold_id}.xml"
        )
        tester_ini = TESTER_INI_TEMPLATE.format(
            ea_name=actual_ea,
            symbol=mt5_symbol,
            period=period,
            model=model,
            from_date=from_date,
            to_date=to_date,
            deposit=deposit,
            currency=currency,
            leverage=leverage,
            html_report_path=html_report_path,
            xml_report_path=xml_report_path,
        )
        # Append EA input parameters at the START of [Tester] section
        if extra_params_block:
            tester_ini = tester_ini.replace(
                "[Tester]\nExpert=",
                "[Tester]\n" + extra_params_block.rstrip("\n") + "\nExpert=",
            )
        tester_path = self._write_utf8(
            tester_ini,
            f"tester_{symbol}_{safe_fold_id}.ini",
        )

        # ── EA params ─────────────────────────────────────────────────
        ea_params = EA_PARAMS_TEMPLATE.format(
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            symbol=symbol,
            fold_id=fold_id,
            bridge_port=bridge_port,
            checkpoint_path=checkpoint_path,
            tick_coverage_pct=f"{tick_coverage_pct:.1f}",
        )
        ea_params_path = self._write_utf8(
            ea_params,
            f"ea_params_{symbol}_{safe_fold_id}.ini",
        )

        _log(f"Generated: {tester_path.name}, {ea_params_path.name}")
        return {
            "tester_config": tester_path,
            "ea_params": ea_params_path,
        }

    def _write_utf8(self, content: str, filename: str) -> Path:
        """Write content as UTF-8 (NO BOM — terminal expects plain ASCII/UTF-8).

        All parameter values in our templates are ASCII-safe.
        If non-ASCII values are ever added, this method must be changed
        to write with BOM or the EA reader must use explicit CP_UTF8 decoding.
        """
        path = self._output_dir / filename
        encoded = content.encode("utf-8")
        # Verify roundtrip: all bytes must be ASCII (0x00-0x7F)
        non_ascii = [b for b in encoded if b > 0x7F]
        if non_ascii:
            _log(
                f"WARNING: {filename} contains {len(non_ascii)} non-ASCII byte(s). "
                f"The EA must use CP_UTF8 decoding for this file. "
                f"First non-ASCII at byte: {non_ascii[0]:#x}"
            )
        path.write_bytes(encoded)
        return path


# ── Self-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("=== ConfigGenerator self-test ===")
    tmp = Path(tempfile.mkdtemp())
    gen = ConfigGenerator(output_dir=str(tmp))

    result = gen.generate(
        symbol="EURUSD",
        fold_id="202606",
        checkpoint_path=str(
            ROOT_DIR / "data" / "checkpoints" / "EURUSD" / "fold_202606"
        ),
        bridge_port=9998,
        from_date="2025-06-01",
        to_date="2026-06-01",
        tick_coverage_pct=100.0,
    )

    for key, path in result.items():
        print(f"\n{'='*60}")
        print(f"{key}: {path}")
        print(f"{'='*60}")
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        print(text)

        # Verify: all bytes are ASCII
        non_ascii = [b for b in raw if b > 0x7F]
        print(f"[VERIFY] Non-ASCII bytes: {len(non_ascii)} (0 = safe for CP_ACP readers)")

        # Verify: no BOM
        has_bom = raw[:3] == b"\xef\xbb\xbf"
        print(f"[VERIFY] UTF-8 BOM: {has_bom} (False = terminal-compatible)")

    print("\n=== Done ===")
