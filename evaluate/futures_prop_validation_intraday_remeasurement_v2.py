"""Run V2 intraday mechanical corrections in their own immutable study."""
from pathlib import Path

import futures_prop_validation_intraday as study

study.OUT = Path(__file__).resolve().parents[1] / "data/results/futures_prop_validation_intraday_remeasurement_v2"
study.SPEC_NAME = "SPEC_FUTURES_INTRADAY_REMEASUREMENT_V2.txt"
study.GATE_NAME = "GATE_FUTURES_INTRADAY_REMEASUREMENT_V2.txt"
study.REPORT_NAME = "REPORT_FUTURES_INTRADAY_REMEASUREMENT_V2.txt"

if __name__ == "__main__":
    study.main()
