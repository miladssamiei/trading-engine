"""Run corrected intraday study in a new immutable artifact directory."""
from pathlib import Path

import futures_prop_validation_intraday as study

study.OUT = Path(__file__).resolve().parents[1] / "data/results/futures_prop_validation_intraday_remeasurement_v1"
study.SPEC_NAME = "SPEC_FUTURES_INTRADAY_REMEASUREMENT_V1.txt"
study.GATE_NAME = "GATE_FUTURES_INTRADAY_REMEASUREMENT_V1.txt"
study.REPORT_NAME = "REPORT_FUTURES_INTRADAY_REMEASUREMENT_V1.txt"

if __name__ == "__main__":
    study.main()
