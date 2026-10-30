import unittest
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from evaluate.ftmo_account_replay_v1 import *

UTC = timezone.utc
NOW = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)


def identity(phase="Challenge"):
    values = {"account_fingerprint": "fp", "server": "FTMO-Demo",
              "broker": "FTMO", "platform": "MT5", "currency": "USD",
              "initial_balance": 10000, "program": "2-Step",
              "account_type": "Swing", "phase": phase}
    return attest_identity(values, values, independently_reviewed=True)


class ReplayTests(unittest.TestCase):
    def snap(self, **changes):
        values = {"equity": Decimal("10200"), "midnight_balance": Decimal("10200"),
                  "weekly_anchor": Decimal("10200"), "strategy_hwm": Decimal("10200"),
                  "observed_at": NOW, "anchor_at": prague_midnight(NOW),
                  "identity": identity(), "existing_risk": Decimal(0),
                  "history_complete": True, "exposure_complete": True}
        values.update(changes)
        return RiskSnapshot(**values)

    def allowance(self, snapshot=None, **kwargs):
        return risk_allowance(snapshot or self.snap(), as_of=NOW, **kwargs)

    def test_fixed_daily_allowance_static_floor_and_equality(self):
        status = firm_status(9700, 10200, currency="USD")
        self.assertEqual(status["daily_floor"], Decimal("9700"))
        self.assertTrue(status["no_headroom"])
        self.assertFalse(status["daily_breach"])
        self.assertEqual(status["total_floor"], Decimal("9000"))
        self.assertEqual(self.allowance(), Decimal("10"))
        self.assertEqual(self.allowance(self.snap(equity=9700)), Decimal(0))
        # A $200 floating loss already present in equity leaves $300 of the
        # firm's fixed $500 allowance; it is not reset or subtracted twice.
        floating = firm_status(9800, 10000, currency="USD")
        self.assertEqual(Decimal("9800") - floating["daily_floor"], Decimal("300"))

    def test_internal_limits_and_all_account_exposure(self):
        self.assertEqual(risk_allowance_from_exposure(
            self.snap(), [10], [20], exposure_complete=True, as_of=NOW), Decimal("10"))
        self.assertEqual(self.allowance(self.snap(existing_risk=50)), Decimal(0))
        with self.assertRaises(RiskRefusal):
            risk_allowance_from_exposure(self.snap(), [10], [20], as_of=NOW)
        with self.assertRaises(ValidationError):
            risk_allowance_from_exposure(self.snap(), [-1], [],
                                         exposure_complete=True, as_of=NOW)
        with self.assertRaises(RiskRefusal): self.allowance(self.snap(existing_risk=None))

    def test_identity_history_freshness_and_currency_are_required(self):
        with self.assertRaises(RiskRefusal): self.allowance(self.snap(identity=None))
        with self.assertRaises(RiskRefusal): self.allowance(self.snap(history_complete=False))
        with self.assertRaises(RiskRefusal): self.allowance(self.snap(exposure_complete=False))
        with self.assertRaises(RiskRefusal):
            self.allowance(self.snap(observed_at=NOW - timedelta(minutes=1)))
        values = dict(identity().__dict__); values["currency"] = "EUR"
        with self.assertRaises(RiskRefusal): IdentityAttestation(**values)

    def test_anchor_exact_and_reconstructed_without_laundering(self):
        reset = prague_midnight(NOW)
        exact = BalanceObservation(reset, Decimal("10100"), exact_midnight=True)
        self.assertEqual(daily_anchor([exact], NOW), Decimal("10100"))
        prior = reset - timedelta(hours=1)
        event = LedgerEvent(prior + timedelta(minutes=30), Decimal("-100"), "deal-1")
        self.assertEqual(daily_anchor([], NOW, [event], Decimal("10200"),
                                      prior_at=prior, coverage_end=reset,
                                      history_complete=True), Decimal("10100"))
        with self.assertRaises(RiskRefusal):
            daily_anchor([], NOW, [event], Decimal("10200"), prior_at=prior,
                         coverage_end=reset, history_complete=False)
        with self.assertRaises(ValidationError):
            daily_anchor([], NOW, [event, event], Decimal("10200"), prior_at=prior,
                         coverage_end=reset, history_complete=True)
        with self.assertRaises(RiskRefusal): daily_anchor([], NOW)

    def test_prague_dst_and_us_europe_transition_weeks(self):
        self.assertEqual(prague_midnight(datetime(2026, 3, 30, 1, tzinfo=UTC)),
                         datetime(2026, 3, 29, 22, tzinfo=UTC))
        self.assertEqual(prague_midnight(datetime(2026, 10, 26, 1, tzinfo=UTC)),
                         datetime(2026, 10, 25, 23, tzinfo=UTC))
        self.assertEqual(prague_midnight(datetime(2026, 3, 15, 23, tzinfo=UTC)),
                         datetime(2026, 3, 15, 23, tzinfo=UTC))

    def test_targets_require_absolute_balance_four_dates_and_flat_account(self):
        days = [date(2026, 1, day) for day in range(1, 5)]
        kwargs = dict(open_positions=0, pending_orders=0, known_breach=False,
                      history_complete=True)
        self.assertTrue(qualification_complete(11000, "Challenge", days,
                                                identity=identity(), **kwargs))
        self.assertTrue(qualification_complete(10500, "Verification", days,
                                                identity=identity("Verification"), **kwargs))
        self.assertFalse(qualification_complete(1000, "Challenge", days,
                                                 identity=identity(), **kwargs))
        self.assertFalse(qualification_complete(11000, "Challenge", days[:3],
                                                 identity=identity(), **kwargs))
        self.assertFalse(qualification_complete(11000, "Challenge",
                                                 [days[0]] * 4,
                                                 identity=identity(), **kwargs))
        self.assertFalse(qualification_complete(
            11000, "Challenge", days, open_positions=1, pending_orders=0,
            known_breach=False, history_complete=True, identity=identity()))

    def test_sizing_rounds_down_and_below_minimum_is_zero(self):
        reviewed = identity()
        self.assertEqual(size_lots(25, 7, ".1", ".1", 5,
                                   identity=reviewed), Decimal("3.5"))
        self.assertEqual(size_lots("0.1", 7, ".1", ".1", 5,
                                   identity=reviewed), Decimal(0))
        self.assertEqual(size_lots(100, 7, ".3", ".1", "1.05",
                                   identity=reviewed), Decimal("0.9"))
        with self.assertRaises(RiskRefusal): size_lots(25, 7, ".1", ".1", 5)

    def test_cost_helpers_count_spread_once(self):
        self.assertEqual(midpoint_spread_cost(1, 3, currency="USD"), Decimal("2"))
        self.assertEqual(actual_fill_cost(2, 1, 4, currency="USD"), Decimal("7"))
        self.assertEqual(risk_per_lot("0.01", "2", "0.10", "1",
                                     currency="USD"), Decimal("21"))
        self.assertEqual(actual_fill_cost(2, 1, -4, currency="USD"), Decimal("-1"))
        with self.assertRaises(RiskRefusal): midpoint_spread_cost(1, 3, currency="EUR")

    def test_state_namespace_corruption_age_and_reset(self):
        payload = {"midnight_balance": "10000",
                   "anchor_at": prague_midnight(NOW).isoformat()}
        raw = serialize_state(payload, server="srv", account_fingerprint="fp",
                              policy_version=POLICY_VERSION, asof_utc=NOW)
        self.assertEqual(deserialize_state(raw, server="srv", account_fingerprint="fp",
                                           policy_version=POLICY_VERSION, now_utc=NOW,
                                           max_age_seconds=1)["midnight_balance"], "10000")
        with self.assertRaises(RiskRefusal):
            deserialize_state(raw, server="other", account_fingerprint="fp",
                              policy_version=POLICY_VERSION, now_utc=NOW)
        with self.assertRaises(RiskRefusal):
            deserialize_state(raw, server="srv", account_fingerprint="fp",
                              policy_version=POLICY_VERSION,
                              now_utc=datetime(2026, 1, 16, tzinfo=UTC))
        with self.assertRaises(RiskRefusal):
            deserialize_state("[]", server="srv", account_fingerprint="fp",
                              policy_version=POLICY_VERSION, now_utc=NOW)

    def test_nonfinite_boolean_and_negative_inputs_fail(self):
        for value in (True, float("nan"), float("inf")):
            with self.assertRaises(ValidationError): dec(value)
        with self.assertRaises(ValidationError):
            risk_per_lot("0", "1", "1", currency="USD")
        with self.assertRaises(ValidationError):
            self.allowance(additional_liquidation_reserve=-1)


if __name__ == "__main__": unittest.main()
