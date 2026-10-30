import hashlib
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "evaluate"))
import ftmo_public_evidence_v1 as public


UTC = timezone.utc
AS_OF = datetime(2026, 9, 11, 12, tzinfo=UTC)


def fixture_payload():
    symbol = {field: None for field in public.PUBLIC_FIELDS}
    symbol.update({
        "id": 75, "assetClass": "Cash CFD", "region": "global",
        "code": "US500.cash", "displayCode": "US500.cash", "name": "S&P 500",
        "contractSize": 1, "marginPercent": 2, "marginCurrency": "USD",
        "profitCurrency": "USD", "marginCalculation": "cfd_leverage",
        "profitCalculation": "cfd", "swapLong": -130.92,
        "swapShort": -29.24, "swapType": "points", "aliases": "sp500",
        "commission": 0, "commissionType": "percent", "active": True,
        "start": "1969-12-31T23:00:00Z", "end": "9999-12-31T22:59:59Z",
        "leverageStandard": 50, "leverageSwing": 15, "digits": 2,
        "usdCommission": 0, "spreadConversionToUsd": 1,
        "pipCommissionPerLot": 0, "pipConversion": 1,
        "maxTradeVolume": 1000, "maxTotalVolume": 0,
        "isExpiringContract": False,
        "tickerGroup": {"code": "global_indices"},
    })
    return {"data": {
        "symbols": [symbol],
        "tradingHours": [{
            "symbolCode": "US500.cash", "symbolId": 75,
            "start": "2026-09-10T22:05:00Z", "end": "2026-09-11T20:50:00Z",
        }],
        "platformTimeOffset": {"UTC": 3},
    }}


def retrieval(payload=None, **changes):
    raw = json.dumps(payload or fixture_payload()).encode()
    values = dict(
        raw=raw, requested_url=public.ENDPOINT, final_url=public.ENDPOINT,
        status=200, content_type="application/json",
        retrieved_at=datetime(2026, 9, 11, 11, tzinfo=UTC), headers={},
    )
    values.update(changes)
    return public.Retrieval(**values)


class PublicEvidenceTests(unittest.TestCase):
    def test_preserves_values_units_hours_and_limits(self):
        evidence, errors, unavailable = public.analyze_payload(fixture_payload(), AS_OF)
        self.assertFalse(errors)
        self.assertEqual(evidence["symbol"]["swapLong"], -130.92)
        self.assertEqual(evidence["symbol"]["swapType"], "points")
        self.assertEqual(evidence["symbol"]["leverageSwing"], 15)
        self.assertEqual(len(evidence["trading_hours"]), 1)
        self.assertIn("initialMargin", unavailable)
        self.assertIn("digits do not prove MT5 tick size or tick value",
                      evidence["interpretation_limits"])

    def test_duplicate_symbol_and_inconsistent_hours_fail(self):
        payload = fixture_payload()
        payload["data"]["symbols"].append(dict(payload["data"]["symbols"][0]))
        self.assertIn("expected exactly one US500.cash record, found 2",
                      public.analyze_payload(payload, AS_OF)[1])
        payload = fixture_payload()
        payload["data"]["tradingHours"][0]["symbolCode"] = "other"
        self.assertTrue(any("inconsistent symbolCode" in item
                            for item in public.analyze_payload(payload, AS_OF)[1]))
        payload = fixture_payload()
        payload["data"]["tradingHours"][0]["symbolId"] = 999
        self.assertTrue(any("inconsistent symbolId" in item
                            for item in public.analyze_payload(payload, AS_OF)[1]))

    def test_stale_schedule_and_inactive_contract_fail_currentness(self):
        payload = fixture_payload()
        payload["data"]["tradingHours"][0].update(
            start="2025-01-01T00:00:00Z", end="2025-01-01T01:00:00Z")
        self.assertTrue(any("schedule is stale" in item
                            for item in public.analyze_payload(payload, AS_OF)[1]))
        payload = fixture_payload()
        payload["data"]["symbols"][0]["active"] = False
        self.assertTrue(any("not active at as-of" in item
                            for item in public.analyze_payload(payload, AS_OF)[1]))

    def test_nonfinite_and_boolean_numeric_values_fail(self):
        payload = fixture_payload()
        payload["data"]["symbols"][0]["contractSize"] = float("nan")
        self.assertTrue(any("contractSize" in item
                            for item in public.analyze_payload(payload, AS_OF)[1]))
        payload = fixture_payload()
        payload["data"]["platformTimeOffset"]["UTC"] = True
        self.assertTrue(any("platformTimeOffset" in item
                            for item in public.analyze_payload(payload, AS_OF)[1]))

    def test_secret_keys_and_unofficial_urls_are_refused(self):
        with self.assertRaisesRegex(ValueError, "secret-like"):
            public.decode_payload(json.dumps({"password": "x"}).encode())
        for value in ("http://ftmo.com/x", "https://example.com/x",
                      "https://user:pass@ftmo.com/x"):
            with self.assertRaises(ValueError):
                public.validate_official_url(value)

    def test_build_audit_is_partial_and_never_authorizes_downstream(self):
        item = retrieval()
        audit, evidence = public.build_audit(item, fixture_payload(), AS_OF)
        self.assertEqual(audit["status"], "PARTIAL")
        self.assertEqual(audit["gates"]["P-L1"], "INCOMPLETE")
        self.assertEqual(audit["gates"]["P-F1"], "PARTIAL")
        self.assertFalse(audit["downstream_model_allowed"])
        self.assertFalse(audit["live_execution_allowed"])
        self.assertIsNotNone(evidence)

    def test_future_retrieval_fails_provenance(self):
        item = retrieval(retrieved_at=datetime(2026, 9, 12, tzinfo=UTC))
        audit, _ = public.build_audit(item, fixture_payload(), AS_OF)
        self.assertEqual(audit["status"], "FAIL")
        self.assertEqual(audit["gates"]["P-S1"], "FAIL")

    def test_cli_archives_exact_bytes_hash_and_protects_output(self):
        item = retrieval()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "evidence"
            with patch.object(public, "fetch_official", return_value=item):
                code = public.main(["--output-dir", str(output), "--as-of",
                                    "2026-09-11T12:00:00Z"])
            self.assertEqual(code, 2)
            self.assertEqual((output / "raw_symbols_response.json").read_bytes(), item.raw)
            expected = hashlib.sha256(item.raw).hexdigest()
            self.assertEqual((output / "raw_symbols_response.json.sha256")
                             .read_text().strip(), expected)
            audit = json.loads((output / "audit.json").read_text())
            self.assertEqual(audit["response_sha256"], expected)
            self.assertEqual(public.main(["--output-dir", str(output)]), 1)

    def test_secret_response_is_rejected_before_output_creation(self):
        item = retrieval(payload={"api_key": "dummy-secret"})
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "evidence"
            with patch.object(public, "fetch_official", return_value=item):
                self.assertEqual(public.main(["--output-dir", str(output)]), 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
