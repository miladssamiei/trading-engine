import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from evaluate.har_risk_scaling_model4_v1 import (
    add_multipliers,
    bootstrap_sharpe_probability,
    build_cash_rv,
    build_portfolio,
    fit_har_forecasts,
    load_trades,
    metrics,
    multiplier,
    qlike,
)


class HarRiskScalingTests(unittest.TestCase):
    def test_cash_rv_uses_open_then_closes(self):
        day = pd.Timestamp("2021-01-05", tz="America/New_York")
        # A valid cash session has 78 five-minute bars.  Keep the path
        # deterministic so the expected log-return sum is independent of
        # implementation details.
        ts = pd.date_range(day + pd.Timedelta(hours=9, minutes=30), periods=78, freq="5min")
        close = np.linspace(100.0, 101.0, len(ts))
        m5 = pd.DataFrame({"ts": ts, "ny_date": ts.normalize(), "wall": ts.hour * 60 + ts.minute,
                           "open": close, "high": close + 0.1, "low": close - 0.1, "close": close})
        rv = build_cash_rv(m5, [day])
        expected = np.square(np.diff(np.log(np.r_[100.0, close]))).sum()
        self.assertAlmostEqual(float(rv.iloc[0].rv), expected)
        self.assertEqual(int(rv.iloc[0].n_bars), 78)

    def test_har_cutoff_and_multiplier_bounds(self):
        dates = pd.date_range("2021-01-01", periods=320, freq="D", tz="America/New_York")
        vals = pd.Series(0.0001 + (np.arange(320) % 17) * 0.000001, index=dates)
        f = add_multipliers(fit_har_forecasts(vals))
        self.assertEqual(int(f.iloc[0].training_rows), 252)
        self.assertTrue((f[["har_multiplier", "b1_multiplier", "b22_multiplier"]] >= 0.25).all().all())
        self.assertTrue((f[["har_multiplier", "b1_multiplier", "b22_multiplier"]] <= 1.0).all().all())
        self.assertAlmostEqual(multiplier(1.0, 100.0), 0.25)
        self.assertAlmostEqual(multiplier(1.0, 0.1), 1.0)

    def test_portfolio_cost_is_incremental_and_scaled(self):
        d = pd.date_range("2024-01-01", periods=2, tz="America/New_York")
        f = pd.DataFrame({"har_multiplier": [0.5, 1.0], "b1_multiplier": [1.0, 1.0],
                          "b22_multiplier": [1.0, 1.0]}, index=d)
        orb = pd.DataFrame({"open_date": [d[0]], "profit": [10.0]})
        sp = pd.DataFrame({"open_date": [d[0], d[0]], "profit": [4.0, -2.0]})
        p = build_portfolio(f, orb, sp, d, spread_increment=0.255)
        # Base portfolio = .5*(10+2)=6; HAR = .5*6 - 3 trades*.5*.5*.255
        self.assertAlmostEqual(float(p.iloc[0].har_pnl), 2.80875)
        self.assertAlmostEqual(float(p.iloc[1].har_pnl), 0.0)

    def test_metrics_and_qlike(self):
        d = pd.date_range("2024-01-01", periods=4, tz="America/New_York")
        x = pd.Series([1.0, -2.0, 3.0, 0.0], index=d)
        m = metrics(x)
        self.assertEqual(m["total_pnl"], 2.0)
        self.assertEqual(m["worst_day"], -2.0)
        self.assertEqual(m["trade_days"], 3)
        self.assertAlmostEqual(qlike(pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0])), 0.0)

    def test_bootstrap_is_deterministic_and_paired(self):
        d = pd.date_range("2024-01-01", periods=60, tz="America/New_York")
        a = pd.Series(np.sin(np.arange(60) / 4) + 0.5, index=d)
        b = pd.Series(np.sin(np.arange(60) / 4), index=d)
        p1 = bootstrap_sharpe_probability(a, b, n=200, block=5, seed=20260911)
        p2 = bootstrap_sharpe_probability(a, b, n=200, block=5, seed=20260911)
        self.assertEqual(p1, p2)
        self.assertGreaterEqual(p1, 0.0)
        self.assertLessEqual(p1, 1.0)

    def test_equal_minute_trade_is_allowed_but_overnight_is_reported(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trades.csv"
            pd.DataFrame({
                "open_time": ["2024.01.02 15:00", "2024.01.02 20:00"],
                "close_time": ["2024.01.02 15:00", "2024.01.03 15:00"],
                "profit": [1.0, -1.0],
            }).to_csv(path, sep=";", index=False)
            trades = load_trades(path, strict=False)
            self.assertNotIn("negative trade duration", trades.attrs["validation_errors"])
            self.assertEqual(trades.attrs["validation_counts"]["equal_minute_timestamps"], 1)
            self.assertEqual(trades.attrs["validation_counts"]["overnight"], 1)
            self.assertIn("overnight trade", trades.attrs["validation_errors"])


if __name__ == "__main__":
    unittest.main()
