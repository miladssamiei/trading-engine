"""Run the frozen Edris engine's synthetic execution diagnostics, without data.

These cases check mechanics, not profitability. No terminal or network is used.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate.edris_sweep_divergence_ib import check_examples


def main():
    check_examples()
    print("PASS: synthetic stop/target, spread, gap, pivot, RSI and DST checks.")
    print("Synthetic cases only; no historical performance is estimated.")


if __name__ == "__main__":
    main()
