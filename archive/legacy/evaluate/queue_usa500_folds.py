"""Queue USA500 fold MT5 validation jobs.

Supports a configurable fold list, data source (broker_live vs
custom_dukascopy_tick), and tester model. Uses fold_test_window for the
full-month test window (matching the Python evaluator).

Usage (4-fold custom Dukascopy-tick run):
    python queue_usa500_folds.py --folds 202601,202602,202603,202604 \
        --data-source custom_dukascopy_tick --tester-model 0
"""
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Milad\Desktop\RR")
sys.path.insert(0, str(ROOT))

from evaluate.job_queue import JobQueue
from evaluate.orchestrator import fold_test_window

SUBSET_ARTIFACTS = {
    "GBM": "checkpoints/gbm_model.txt",
    "TQC": "checkpoints/tqc_model.zip",
    "DT":  "checkpoints/dt_model.pt",
}

BASE = ROOT / "data" / "checkpoints" / "USA500"


def fold_complete(fold_dir: Path) -> bool:
    if not (fold_dir / "splits.json").exists():
        return False
    if not (fold_dir / "sizing_results.json").exists():
        return False
    if not list((fold_dir / "scalers").glob("*.pkl")):
        return False
    for rel in SUBSET_ARTIFACTS.values():
        if not (fold_dir / rel).exists():
            return False
    return True


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="202601,202602,202603,202604,202605,202606")
    ap.add_argument("--data-source", default="broker_live",
                    choices=["broker_live", "custom_dukascopy_tick"])
    ap.add_argument("--tester-model", type=int, default=0)
    args = ap.parse_args()

    folds = [f.strip() for f in args.folds.split(",") if f.strip()]

    jq = JobQueue()
    db = jq.connect()

    queued = 0
    for fold_id in folds:
        fd = BASE / f"fold_{fold_id}"
        if not fd.exists() or not fold_complete(fd):
            print(f"  fold_{fold_id}: INCOMPLETE — skipping")
            continue

        from_d, to_d = fold_test_window(fold_id)
        db.execute(
            """INSERT INTO jobs
               (symbol, fold_id, checkpoint_path, status, tick_coverage_pct,
                from_date, to_date, tester_model, data_source)
               VALUES (?, ?, ?, 'queued', 100.0, ?, ?, ?, ?)""",
            ("USA500", fold_id, str(fd), from_d, to_d,
             args.tester_model, args.data_source),
        )
        print(f"  fold_{fold_id}: QUEUED ({from_d} -> {to_d}) "
              f"model={args.tester_model} data_source={args.data_source}")
        queued += 1

    db.commit()
    print(f"\nQueued {queued} new job(s)")
    print(f"Queue summary: {jq.status_summary()}")
    jq.close()


if __name__ == "__main__":
    main()
