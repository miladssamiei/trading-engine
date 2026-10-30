import csv
import json
import sys
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "evaluate"))
import ftmo10k_qualification_v1 as q

UTC = timezone.utc
AS_OF = datetime(2026, 9, 11, tzinfo=UTC)


def manifest():
    return {"identity": {
        "account_fingerprint": None, "server": None, "broker": "FTMO",
        "platform": "MT5", "currency": "USD", "initial_balance": 10000,
        "program": "2-Step", "account_type": "Swing", "phase": "Challenge",
    }}


def snapshot(kind="synthetic_fixture"):
    return {
        "schema_version": 1,
        "provenance": {"source_kind": kind, "source": "sanitized export",
                       "received_at": "2026-01-02T00:00:01Z"},
        "account": {
            "account_fingerprint": "fp", "server": "srv", "broker": "FTMO",
            "platform": "MT5", "currency": "USD", "initial_balance": 10000,
            "program": "2-Step", "account_type": "Swing", "phase": "Challenge",
            "captured_at": "2026-01-02T00:00:00Z", "balance": 10000,
            "equity": 10000,
        },
        "symbol": {
            "name": "US500.cash", "contract_id": "contract-v1",
            "quote_currency": "USD", "profit_currency": "USD",
            "tick_size": "0.1", "tick_value_profit": "0.1",
            "tick_value_loss": "0.1", "contract_size": 1,
            "volume_min": "0.01", "volume_max": 100, "volume_step": "0.01",
            "margin_initial": 0, "stops_level": 0, "freeze_level": 0,
            "filling_modes": ["IOC"],
            "trading_sessions": [{"weekday": i, "open": "00:00", "close": "23:59"}
                                 for i in range(5)],
            "swap_mode": "points", "swap_long": -1, "swap_short": 1,
            "rollover_multipliers": {str(i): (3 if i == 2 else 1) for i in range(7)},
        },
    }


def write_quotes(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def quote_row(observed="2026-01-02T14:35:00Z", received="2026-01-02T14:35:01Z",
              bid="100", ask="100.1", kind="synthetic_fixture"):
    return {"timestamp_utc": observed, "received_at_utc": received,
            "bid": bid, "ask": ask, "symbol": "US500.cash",
            "contract_id": "contract-v1", "source_kind": kind}


class QualificationTests(unittest.TestCase):
    def test_missing_and_synthetic_evidence_stay_incomplete(self):
        self.assertEqual(q.audit(manifest(), as_of=AS_OF)["status"], "INCOMPLETE")
        errors, missing = q.validate_snapshot(snapshot(), manifest(), AS_OF)
        self.assertFalse(errors)
        self.assertFalse(missing)
        result = q.audit(manifest(), snapshot(), as_of=AS_OF)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertFalse(result["downstream_model_allowed"])
        self.assertIn("expected identity.server", result["missing_evidence"])

    def test_missing_null_and_malformed_sections_are_handled(self):
        item = snapshot(); item["provenance"] = None
        self.assertEqual(q.audit(manifest(), item, as_of=AS_OF)["status"], "INCOMPLETE")
        item = snapshot(); item["provenance"] = []
        result = q.audit(manifest(), item, as_of=AS_OF)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["gates"]["G-D2"], "FAIL")

    def test_identity_mismatch_and_future_snapshot_fail(self):
        item = snapshot(); item["account"]["currency"] = "EUR"
        result = q.audit(manifest(), item, as_of=AS_OF)
        self.assertEqual(result["gates"]["G-D1"], "FAIL")
        item = snapshot(); item["account"]["captured_at"] = "2099-01-01T00:00:00Z"
        item["provenance"]["received_at"] = "2099-01-01T00:00:01Z"
        self.assertEqual(q.audit(manifest(), item, as_of=AS_OF)["status"], "FAIL")

    def test_missing_metadata_is_incomplete_and_invalid_metadata_fails(self):
        item = snapshot(); item["symbol"]["volume_min"] = None
        self.assertEqual(q.audit(manifest(), item, as_of=AS_OF)["status"], "INCOMPLETE")
        item = snapshot(); item["symbol"]["trading_sessions"] = []
        self.assertEqual(q.audit(manifest(), item, as_of=AS_OF)["status"], "INCOMPLETE")
        item = snapshot(); item["symbol"]["tick_size"] = float("nan")
        self.assertEqual(q.audit(manifest(), item, as_of=AS_OF)["status"], "FAIL")

    def test_invalid_quote_never_enters_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "quotes.csv"
            write_quotes(path, [quote_row(bid="invalid")])
            result = q.audit(manifest(), snapshot(), path, AS_OF)
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["gates"]["G-D3"], "FAIL")
            self.assertEqual(result["quote_summary"]["count"], 0)

    def test_quote_time_order_asof_and_even_median(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "quotes.csv"
            rows = [quote_row(ask="101"),
                    quote_row("2026-01-05T14:35:00Z", "2026-01-05T14:35:03Z",
                              ask="103")]
            write_quotes(path, rows)
            errors, observations = q.validate_quotes(path, snapshot(), AS_OF)
            self.assertFalse(errors); self.assertEqual(len(observations), 2)
            result = q.audit(manifest(), snapshot(), path, AS_OF)
            self.assertEqual(result["quote_summary"]["spread"]["median"], 2.0)
            self.assertEqual(result["quote_summary"]["receipt_latency_seconds"]["median"], 2.0)
            rows[1]["received_at_utc"] = "2099-01-01T00:00:00Z"
            write_quotes(path, rows)
            self.assertTrue(q.validate_quotes(path, snapshot(), AS_OF)[0])

    def test_rollover_cross_midnight_is_one_following_prague_date(self):
        before = datetime(2026, 1, 2, 22, 56, tzinfo=UTC)
        after = datetime(2026, 1, 2, 23, 4, tzinfo=UTC)
        expected = date(2026, 1, 3)
        self.assertEqual(q._window(before, "rollover_prague"), (True, expected))
        self.assertEqual(q._window(after, "rollover_prague"), (True, expected))

    def test_cli_rejects_credentials_before_archive_and_protects_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest()), encoding="utf-8")
            quotes = root / "quotes.csv"; row = quote_row(); row["login"] = "secret"
            write_quotes(quotes, [row]); output = root / "new-output"
            self.assertEqual(q.main(["--manifest", str(manifest_path), "--quotes",
                                     str(quotes), "--output-dir", str(output)]), 1)
            self.assertFalse(output.exists())
            output.mkdir()
            self.assertEqual(q.main(["--manifest", str(manifest_path),
                                     "--output-dir", str(output)]), 1)

    def test_cli_archives_fixed_names_and_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); manifest_path = root / "audit.json"
            snapshot_path = root / "also-audit.json"
            manifest_path.write_text(json.dumps(manifest()), encoding="utf-8")
            snapshot_path.write_text(json.dumps(snapshot()), encoding="utf-8")
            output = root / "output"
            code = q.main(["--manifest", str(manifest_path), "--snapshot",
                           str(snapshot_path), "--output-dir", str(output),
                           "--as-of", "2026-09-11T00:00:00Z"])
            self.assertEqual(code, 2)
            for name in ("raw_manifest.json", "raw_snapshot.json", "audit.json",
                         "raw_manifest.json.sha256", "raw_snapshot.json.sha256"):
                self.assertTrue((output / name).is_file())
            audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["as_of"], "2026-09-11T00:00:00Z")


if __name__ == "__main__": unittest.main()
