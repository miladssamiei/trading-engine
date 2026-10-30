import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from evaluate import execution_safe_model4_v1 as mod


class ExecutionSafeTests(unittest.TestCase):
    def test_reason_priority_parser_understands_numeric_mql5_reasons(self):
        text = ("EXEC_SAFE_EXIT strategy=ORB deal_reason=4 reason=stop_loss\n"
                "EXEC_SAFE_EXIT strategy=SP2L deal_reason=5 reason=time_exit\n")
        result = mod.parse_reason_priority_logs(text)
        self.assertEqual(result["checked"], 2)
        self.assertEqual(len(result["violations"]), 1)

    def test_batch_result_list_success_contract(self):
        rows = [{"test_passed": True, "outputs": {"trades.csv": "x"}}]
        self.assertTrue(rows and all(x.get("test_passed") and x.get("outputs") for x in rows))

    def test_ini_is_model4_and_strategy_specific(self):
        with tempfile.TemporaryDirectory() as td:
            p = mod.make_ini("orb", "2021", Path(td))
            text = p.read_text()
            self.assertIn("Expert=ORB_ExecutionSafe_v1", text)
            self.assertIn("Period=M5", text)
            self.assertIn("Model=4", text)
            self.assertIn("FromDate=2021-01-01", text)
            report_line = next(line for line in text.splitlines() if line.startswith("Report="))
            report = Path(report_line.split("=", 1)[1])
            self.assertTrue(report.is_absolute())
            self.assertEqual(report.parent, (Path(td) / "orb" / "batches" / "year_2021").resolve())

    def test_agent_selection_requires_both_files_in_mql5_files(self):
        with tempfile.TemporaryDirectory() as td:
            a1, a2 = Path(td) / "a1", Path(td) / "a2"
            for a in (a1, a2):
                (a / "MQL5" / "Files").mkdir(parents=True)
            (a1 / "MQL5" / "Files" / "trades.csv").write_text("x")
            (a1 / "MQL5" / "Files" / "orb_days.csv").write_text("x")
            (a2 / "MQL5" / "Files" / "trades.csv").write_text("x")
            agent, files = mod.select_agent_outputs([a1, a2], ("trades.csv", "orb_days.csv"))
            self.assertEqual(agent, a1)
            self.assertEqual(set(files), {"trades.csv", "orb_days.csv"})
            self.assertTrue(all("MQL5" in str(p) and "Files" in str(p) for p in files.values()))

    def test_blocked_entry_and_same_day_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "trades.csv"
            p.write_text("ticket;open_time;close_time;type;open_price;profit;exit_reason\n"
                         "1;2021.01.18 14:00;2021.01.18 15:00;BUY;1;2.0;target\n"
                         "2;2021.01.19 14:00;2021.01.20 15:00;BUY;1;-1.0;time_exit\n")
            d = mod.parse_trades(p)
            # The metric requires a common session-day universe, not market quotes.
            # Supply synthetic dates rather than reading the author's local export.
            common = pd.DataFrame({
                "ny_date": ["20210118", "20210119", "20210120"],
                "prev_close": [1, 1, 1], "session_open": [1, 1, 1],
            })
            with patch.object(mod.pd, "read_csv", return_value=common):
                m = mod.stream_metrics(d)
            self.assertEqual(m["blocked_entries"], 1)
            self.assertEqual(m["later_close_ny"], 1)
            self.assertEqual(m["negative_duration"], 0)
            self.assertEqual(m["exit_reasons"]["target"], 1)

    def test_stitch_renumbers_tickets_without_touching_frozen_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for year in mod.WINDOWS:
                b = root / "orb" / "batches" / f"year_{year}"
                b.mkdir(parents=True)
                b.joinpath("trades.csv").write_text("ticket;open_time;close_time;type;open_price;profit;exit_reason\n"
                    f"99;{year}.01.02 14:00;{year}.01.02 15:00;BUY;1;1;target\n")
                b.joinpath("orb_days.csv").write_text("ny_date;orders_placed\n20210102;1\n")
            out = mod.stitch("orb", root)
            d = pd.read_csv(out["trades_path"], sep=";")
            self.assertEqual(d.ticket.tolist(), list(range(1, 7)))
            self.assertEqual(out["trades"], 6)

    def test_kill_processes_targets_only_named_images(self):
        calls = []
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        def fake(cmd, **kwargs):
            calls.append(cmd)
            return R()
        with patch.object(mod.subprocess, "run", side_effect=fake):
            result = mod.kill_processes()
        self.assertEqual([c[-1] for c in calls], ["terminal64.exe", "metatester64.exe"])
        self.assertEqual(set(result), {"terminal64.exe", "metatester64.exe"})

    def test_static_audit_records_new_source_corrections(self):
        result = mod.static_audit()
        self.assertTrue(result["X-S1"])
        self.assertTrue(result["X-S2"])
        self.assertTrue(all(v["grid_normalization"] for v in result["new_source_checks"].values()))
        self.assertTrue(all(v["sl_tp_priority"] for v in result["new_source_checks"].values()))

    def test_audit_before_batches_fails_closed_and_disables_execution(self):
        with tempfile.TemporaryDirectory() as td:
            result = mod.audit(Path(td))
            self.assertEqual(result["status"], "FAIL")
            self.assertFalse(result["gates"]["X-M1"])
            self.assertFalse(result["downstream_model_allowed"])
            self.assertFalse(result["live_execution_allowed"])
            self.assertTrue((Path(td) / "audit.json").exists())


if __name__ == "__main__":
    unittest.main()
