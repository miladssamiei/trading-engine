"""Archive and audit FTMO's public US500.cash specification without login."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ENDPOINT = "https://ftmo.com/wp-json/ftmo/symbols"
SYMBOL = "US500.cash"
ALLOWED_HOSTS = {"ftmo.com", "www.ftmo.com"}
MAX_RESPONSE_BYTES = 5_000_000
SECRET_KEY = re.compile(
    r"(?:password|passwd|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|"
    r"authorization|cookie|account[_-]?(?:number|login))",
    re.IGNORECASE,
)

PUBLIC_FIELDS = (
    "id", "assetClass", "region", "code", "displayCode", "name",
    "contractSize", "marginPercent", "initialMargin", "marginCurrency",
    "profitCurrency", "marginCalculation", "profitCalculation", "swapLong",
    "swapShort", "swapType", "aliases", "commission", "commissionType",
    "active", "start", "end", "leverageStandard", "leverageSwing", "digits",
    "usdCommission", "spreadConversionToUsd", "pipCommissionPerLot",
    "pipConversion", "maxTradeVolume", "maxTotalVolume",
    "isExpiringContract", "expiringContractInstrument", "tickerGroup",
)

NON_NULL_FIELDS = (
    "id", "assetClass", "code", "name", "contractSize", "marginPercent",
    "marginCurrency", "profitCurrency", "marginCalculation",
    "profitCalculation", "swapLong", "swapShort", "swapType", "commission",
    "commissionType", "active", "start", "end", "leverageStandard",
    "leverageSwing", "digits", "maxTradeVolume", "isExpiringContract",
)

UNRESOLVED = (
    "account fingerprint and exact assigned server",
    "current account balance, equity and phase",
    "MT5 tick size, tick value profit and tick value loss",
    "minimum volume and volume step",
    "stops level, freeze level and filling modes",
    "rollover multiplier schedule",
    "historical bid/ask observations and spread distribution",
    "realized fills, slippage and all-in cost reconciliation",
    "independent account and rule reconciliation",
)


@dataclass(frozen=True)
class Retrieval:
    raw: bytes
    requested_url: str
    final_url: str
    status: int
    content_type: str
    retrieved_at: datetime
    headers: dict[str, str]


def parse_utc(value: str | datetime, label: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{label} is not valid ISO-8601") from exc
    else:
        raise ValueError(f"{label} must be an ISO-8601 string")
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_official_url(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        raise ValueError("URL must use HTTPS on an allowlisted ftmo.com host")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("URL must not contain credentials or a fragment")


def contains_secret_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(SECRET_KEY.search(str(key)) or contains_secret_key(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(contains_secret_key(item) for item in value)
    return False


def finite_number(value: Any, *, positive: bool = False,
                  nonnegative: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    if not math.isfinite(number):
        return False
    if positive and number <= 0:
        return False
    if nonnegative and number < 0:
        return False
    return True


def nonfinite_paths(value: Any, prefix: str = "symbol") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(nonfinite_paths(item, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(nonfinite_paths(item, f"{prefix}[{index}]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            found.append(prefix)
    return found


def fetch_official(url: str = ENDPOINT, timeout: float = 30.0) -> Retrieval:
    validate_official_url(url)
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "RR-FTMO-public-evidence/1.0",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        final_url = response.geturl()
        validate_official_url(final_url)
        status = int(response.status)
        content_type = response.headers.get_content_type()
        raw = response.read(MAX_RESPONSE_BYTES + 1)
        headers = {
            key: response.headers.get(key)
            for key in ("Content-Type", "Content-Length", "Date", "ETag", "Last-Modified")
            if response.headers.get(key) is not None
        }
    if status != 200:
        raise ValueError(f"official endpoint returned HTTP {status}")
    if content_type != "application/json":
        raise ValueError(f"official endpoint returned {content_type}, not application/json")
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("official response exceeds the size limit")
    return Retrieval(
        raw=raw,
        requested_url=url,
        final_url=final_url,
        status=status,
        content_type=content_type,
        retrieved_at=datetime.now(timezone.utc),
        headers=headers,
    )


def decode_payload(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("official response is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("official response root must be an object")
    if contains_secret_key(payload):
        raise ValueError("secret-like response key detected; response refused")
    return payload


def protocol_gate(project_root: Path | None = None) -> tuple[str, list[str]]:
    root = project_root or Path(__file__).resolve().parents[1]
    folder = root / "data" / "results" / "ftmo_public_evidence_v1"
    spec_path = folder / "SPEC_FTMO_PUBLIC_EVIDENCE.txt"
    gate_path = folder / "GATE_FTMO_PUBLIC_EVIDENCE.txt"
    errors: list[str] = []
    if not spec_path.is_file() or not gate_path.is_file():
        return "FAIL", ["SPEC or GATE is missing"]
    spec_lines = [line for line in spec_path.read_text(encoding="utf-8").splitlines()
                  if line.startswith("P-")]
    gate_lines = gate_path.read_text(encoding="utf-8").splitlines()
    if spec_lines != gate_lines:
        errors.append("GATE is not the verbatim ordered extraction of SPEC P- lines")
    return ("PASS" if not errors else "FAIL"), errors


def analyze_payload(payload: dict[str, Any], as_of: datetime,
                    symbol_code: str = SYMBOL) -> tuple[dict[str, Any] | None,
                                                        list[str], list[str]]:
    errors: list[str] = []
    unavailable: list[str] = []
    data = payload.get("data")
    if not isinstance(data, dict):
        return None, ["data must be an object"], unavailable
    symbols = data.get("symbols")
    hours = data.get("tradingHours")
    offset = (data.get("platformTimeOffset") or {}).get("UTC") \
        if isinstance(data.get("platformTimeOffset"), dict) else None
    if not isinstance(symbols, list):
        errors.append("data.symbols must be an array")
        symbols = []
    if not isinstance(hours, list):
        errors.append("data.tradingHours must be an array")
        hours = []
    if not finite_number(offset):
        errors.append("data.platformTimeOffset.UTC must be finite numeric")

    selected = [item for item in symbols
                if isinstance(item, dict) and item.get("code") == symbol_code]
    if len(selected) != 1:
        errors.append(f"expected exactly one {symbol_code} record, found {len(selected)}")
        return None, errors, unavailable
    symbol = selected[0]
    for path in nonfinite_paths(symbol):
        errors.append(f"{path} must be finite")
    for field in PUBLIC_FIELDS:
        if field not in symbol:
            errors.append(f"selected symbol is missing field {field}")
        elif symbol[field] is None:
            unavailable.append(field)
    for field in NON_NULL_FIELDS:
        if symbol.get(field) is None:
            errors.append(f"selected symbol requires non-null {field}")

    for field in ("contractSize", "marginPercent", "leverageStandard",
                  "leverageSwing", "digits"):
        if not finite_number(symbol.get(field), positive=True):
            errors.append(f"{field} must be finite and positive")
    for field in ("commission", "maxTradeVolume"):
        if not finite_number(symbol.get(field), nonnegative=True):
            errors.append(f"{field} must be finite and nonnegative")
    for field in ("swapLong", "swapShort"):
        if not finite_number(symbol.get(field)):
            errors.append(f"{field} must be finite numeric")
    if not isinstance(symbol.get("active"), bool):
        errors.append("active must be boolean")
    if not isinstance(symbol.get("isExpiringContract"), bool):
        errors.append("isExpiringContract must be boolean")

    start = end = None
    try:
        start = parse_utc(symbol.get("start"), "symbol start")
        end = parse_utc(symbol.get("end"), "symbol end")
        if start >= end:
            errors.append("symbol active interval is empty or reversed")
        if not (start <= as_of <= end) or symbol.get("active") is not True:
            errors.append("selected contract is not active at as-of")
    except ValueError as exc:
        errors.append(str(exc))

    symbol_id = symbol.get("id")
    selected_hours: list[dict[str, Any]] = []
    for index, row in enumerate(hours):
        if not isinstance(row, dict):
            errors.append(f"tradingHours row {index} must be an object")
            continue
        if row.get("symbolCode") == symbol_code and row.get("symbolId") != symbol_id:
            errors.append(f"tradingHours row {index} has inconsistent symbolId")
        if row.get("symbolId") != symbol_id:
            continue
        if row.get("symbolCode") != symbol_code:
            errors.append(f"tradingHours row {index} has inconsistent symbolCode")
        try:
            row_start = parse_utc(row.get("start"), f"tradingHours row {index} start")
            row_end = parse_utc(row.get("end"), f"tradingHours row {index} end")
            if row_start >= row_end:
                errors.append(f"tradingHours row {index} is empty or reversed")
            selected_hours.append(dict(row))
        except ValueError as exc:
            errors.append(str(exc))
    if not selected_hours:
        errors.append("no trading-hour rows found for selected symbol")

    schedule_current = False
    for row in selected_hours:
        try:
            row_start = parse_utc(row["start"], "trading-hour start")
            row_end = parse_utc(row["end"], "trading-hour end")
        except (KeyError, ValueError):
            continue
        distance = min(abs((row_start.date() - as_of.date()).days),
                       abs((row_end.date() - as_of.date()).days))
        if row_start.date() == as_of.date() or row_end.date() == as_of.date() \
                or distance <= 7:
            schedule_current = True
    if not schedule_current:
        errors.append("published trading-hour schedule is stale at as-of")

    evidence = {
        "schema_version": 1,
        "as_of_utc": iso_z(as_of),
        "platform_time_offset_utc_hours": offset,
        "symbol": {field: symbol.get(field) for field in PUBLIC_FIELDS},
        "trading_hours": selected_hours,
        "supported_public_facts": [
            "public contract identity and active interval",
            "contract size and profit/margin currencies",
            "Standard and Swing leverage and margin percentage",
            "published swap long/short values and swap unit",
            "published commission value and unit",
            "current published weekly trading intervals",
        ],
        "nullable_or_unavailable_public_fields": sorted(unavailable),
        "unresolved_required_evidence": list(UNRESOLVED),
        "interpretation_limits": [
            "digits do not prove MT5 tick size or tick value",
            "swap points are not converted to account currency",
            "maximum volume does not prove minimum volume or volume step",
            "zero commission does not establish spread, slippage or all-in cost",
            "weekly intervals do not establish historical schedule continuity",
        ],
    }
    return evidence, sorted(set(errors)), sorted(set(unavailable))


def build_audit(retrieval: Retrieval, payload: dict[str, Any], as_of: datetime,
                project_root: Path | None = None) -> tuple[dict[str, Any],
                                                           dict[str, Any] | None]:
    errors: list[str] = []
    p_i1, protocol_errors = protocol_gate(project_root)
    errors.extend(protocol_errors)

    source_errors: list[str] = []
    try:
        validate_official_url(retrieval.requested_url)
        validate_official_url(retrieval.final_url)
    except ValueError as exc:
        source_errors.append(str(exc))
    source_ok = (retrieval.status == 200 and retrieval.content_type == "application/json"
                 and len(retrieval.raw) <= MAX_RESPONSE_BYTES
                 and retrieval.retrieved_at <= as_of)
    if retrieval.retrieved_at > as_of:
        source_errors.append("retrieval time is after as-of")
    if retrieval.status != 200:
        source_errors.append(f"HTTP status is {retrieval.status}")
    if retrieval.content_type != "application/json":
        source_errors.append("response media type is not application/json")
    if len(retrieval.raw) > MAX_RESPONSE_BYTES:
        source_errors.append("response exceeds size limit")
    errors.extend(source_errors)

    evidence, semantic_errors, unavailable = analyze_payload(payload, as_of)
    errors.extend(semantic_errors)
    p_s1 = "PASS" if source_ok and not source_errors else "FAIL"
    p_s2 = "PASS" if evidence is not None and not semantic_errors else "FAIL"
    p_s3 = "PASS" if evidence is not None and not semantic_errors else "FAIL"
    p_c1 = "PASS" if evidence is not None and not semantic_errors else "FAIL"
    core_pass = all(value == "PASS" for value in (p_i1, p_s1, p_s2, p_s3, p_c1))
    gates = {
        "P-I1": p_i1,
        "P-S1": p_s1,
        "P-S2": p_s2,
        "P-S3": p_s3,
        "P-C1": p_c1,
        "P-L1": "INCOMPLETE",
        "P-F1": "PARTIAL" if core_pass else "FAIL",
    }
    audit = {
        "schema_version": 1,
        "status": "PARTIAL" if core_pass else "FAIL",
        "downstream_model_allowed": False,
        "live_execution_allowed": False,
        "as_of_utc": iso_z(as_of),
        "retrieved_at_utc": iso_z(retrieval.retrieved_at),
        "response_sha256": hashlib.sha256(retrieval.raw).hexdigest(),
        "response_bytes": len(retrieval.raw),
        "gates": gates,
        "errors": sorted(set(errors)),
        "nullable_or_unavailable_public_fields": unavailable,
        "unresolved_required_evidence": list(UNRESOLVED),
    }
    return audit, evidence


def retrieval_metadata(retrieval: Retrieval) -> dict[str, Any]:
    return {
        "requested_url": retrieval.requested_url,
        "final_url": retrieval.final_url,
        "http_status": retrieval.status,
        "content_type": retrieval.content_type,
        "retrieved_at_utc": iso_z(retrieval.retrieved_at),
        "response_bytes": len(retrieval.raw),
        "response_sha256": hashlib.sha256(retrieval.raw).hexdigest(),
        "selected_response_headers": retrieval.headers,
        "authentication_used": False,
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--as-of")
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    if output.exists():
        print("output directory already exists", file=sys.stderr)
        return 1
    try:
        retrieval = fetch_official(args.endpoint, args.timeout)
        as_of = parse_utc(args.as_of, "as-of") if args.as_of \
            else retrieval.retrieved_at
        payload = decode_payload(retrieval.raw)
        audit, evidence = build_audit(retrieval, payload, as_of)
        output.mkdir(parents=True)
        raw_path = output / "raw_symbols_response.json"
        raw_path.write_bytes(retrieval.raw)
        (output / "raw_symbols_response.json.sha256").write_text(
            hashlib.sha256(retrieval.raw).hexdigest() + "\n", encoding="ascii")
        write_json(output / "retrieval_metadata.json", retrieval_metadata(retrieval))
        if evidence is not None:
            evidence["provenance"] = retrieval_metadata(retrieval)
            write_json(output / "public_symbol_evidence.json", evidence)
        write_json(output / "audit.json", audit)
        print(json.dumps({
            "status": audit["status"],
            "downstream_model_allowed": False,
            "live_execution_allowed": False,
        }))
        return 2 if audit["status"] == "PARTIAL" else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
