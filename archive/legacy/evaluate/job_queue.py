"""
job_queue.py — SQLite-backed job queue for fold testing.

Schema matches plan §Phase 5 exactly. Supports the orchestrator's main loop:
  - Discover checkpoints and populate queue
  - Assign jobs to terminals (queued → running)
  - Collect results (running → completed / failed)
  - Retry logic (failed → queued, up to MAX_RETRIES)
  - Permanent failure gate (permanently_failed)
"""

from __future__ import annotations

import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


def _log(msg: str) -> None:
    print(f"[JobQueue] {msg}", flush=True)


MAX_RETRIES = 3

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol           TEXT    NOT NULL,
    fold_id          TEXT    NOT NULL,
    checkpoint_path  TEXT    NOT NULL,
    status           TEXT    NOT NULL DEFAULT 'queued',
    retry_count      INTEGER NOT NULL DEFAULT 0,
    terminal_id      TEXT,
    bridge_port      INTEGER,
    started_at       TEXT,
    completed_at     TEXT,
    sharpe           REAL,
    max_dd           REAL,
    win_rate         REAL,
    trade_count      INTEGER,
    pnl              REAL,
    tick_coverage_pct REAL,
    error_message    TEXT,
    from_date        TEXT,
    to_date          TEXT,
    tester_model     INTEGER,
    data_source      TEXT
);

CREATE INDEX IF NOT EXISTS idx_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_symbol_fold ON jobs(symbol, fold_id);
"""

MIGRATION_SQL = """
ALTER TABLE jobs ADD COLUMN from_date TEXT;
ALTER TABLE jobs ADD COLUMN to_date TEXT;
"""


class JobQueue:
    """SQLite-backed job queue for fold testing pipeline."""

    VALID_STATUSES = {"queued", "running", "completed", "failed",
                      "permanently_failed"}

    def __init__(self, db_path: str | Path = ""):
        self.db_path = str(
            Path(db_path) if db_path else ROOT_DIR / "data" / "queue" / "jobs.db"
        )
        self._conn: sqlite3.Connection | None = None

    # ── Connection management ───────────────────────────────────────────

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(SCHEMA_SQL)
            # Migrate older DBs that predate the from_date/to_date columns
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(jobs)")}
            if "from_date" not in cols or "to_date" not in cols:
                try:
                    if "from_date" not in cols:
                        self._conn.execute("ALTER TABLE jobs ADD COLUMN from_date TEXT")
                    if "to_date" not in cols:
                        self._conn.execute("ALTER TABLE jobs ADD COLUMN to_date TEXT")
                except sqlite3.OperationalError:
                    pass
            if "tester_model" not in cols:
                try:
                    self._conn.execute("ALTER TABLE jobs ADD COLUMN tester_model INTEGER")
                except sqlite3.OperationalError:
                    pass
            if "data_source" not in cols:
                try:
                    self._conn.execute("ALTER TABLE jobs ADD COLUMN data_source TEXT")
                except sqlite3.OperationalError:
                    pass
            self._conn.commit()
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    # ── Populate ────────────────────────────────────────────────────────

    def discover_checkpoints(self, base_dir: str | Path = "") -> list[dict]:
        """Scan data/checkpoints/ for (symbol, fold_id, checkpoint_path) tuples.

        Returns list of dicts: {symbol, fold_id, checkpoint_path}
        """
        base = Path(base_dir) if base_dir else ROOT_DIR / "data" / "checkpoints"
        results = []
        if not base.exists():
            return results
        for symbol_dir in sorted(base.iterdir()):
            if not symbol_dir.is_dir():
                continue
            symbol = symbol_dir.name
            for fold_dir in sorted(symbol_dir.iterdir()):
                if not fold_dir.is_dir() or not fold_dir.name.startswith("fold_"):
                    continue
                fold_id = fold_dir.name.replace("fold_", "")
                results.append({
                    "symbol": symbol,
                    "fold_id": fold_id,
                    "checkpoint_path": str(fold_dir),
                })
        return results

    def populate(self, base_dir: str | Path = "",
                 tick_coverage: dict | None = None) -> int:
        """Discover all checkpoints and insert queued jobs for unprocessed folds.

        Skips folds that already have any job entry (queued/running/completed/failed).
        Returns count of newly inserted jobs.
        """
        db = self.connect()
        folders = self.discover_checkpoints(base_dir)
        tick_coverage = tick_coverage or {}
        count = 0

        for f in folders:
            existing = db.execute(
                "SELECT COUNT(*) FROM jobs WHERE symbol=? AND fold_id=?",
                (f["symbol"], f["fold_id"]),
            ).fetchone()[0]
            if existing > 0:
                continue

            key = f"{f['symbol']}_{f['fold_id']}"
            cov = tick_coverage.get(key, tick_coverage.get(f["fold_id"], 100.0))

            db.execute(
                """INSERT INTO jobs
                   (symbol, fold_id, checkpoint_path, tick_coverage_pct)
                   VALUES (?, ?, ?, ?)""",
                (f["symbol"], f["fold_id"], f["checkpoint_path"], cov),
            )
            count += 1

        db.commit()
        _log(f"Populated {count} new job(s)")
        return count

    # ── Queries ─────────────────────────────────────────────────────────

    def count_by_status(self, status: str) -> int:
        db = self.connect()
        row = db.execute(
            "SELECT COUNT(*) FROM jobs WHERE status=?",
            (status,),
        ).fetchone()
        return row[0] if row else 0

    def next_queued(self) -> dict | None:
        """Get the next queued job (FIFO by job_id). Returns None if empty."""
        db = self.connect()
        row = db.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY job_id LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def pending_count(self) -> int:
        return self.count_by_status("queued")

    def running_count(self) -> int:
        return self.count_by_status("running")

    def get_job(self, job_id: int) -> dict | None:
        db = self.connect()
        row = db.execute(
            "SELECT * FROM jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_jobs_by_terminal(self, terminal_id: str) -> list[dict]:
        db = self.connect()
        rows = db.execute(
            "SELECT * FROM jobs WHERE terminal_id=? ORDER BY job_id",
            (terminal_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def status_summary(self) -> dict[str, int]:
        db = self.connect()
        rows = db.execute(
            "SELECT status, COUNT(*) as cnt FROM jobs GROUP BY status"
        ).fetchall()
        return {r["status"]: r["cnt"] for r in rows}

    # ── State transitions ───────────────────────────────────────────────

    def start_job(self, job_id: int, terminal_id: str,
                  bridge_port: int) -> bool:
        """Mark job as running, assign to terminal."""
        db = self.connect()
        row = db.execute(
            "SELECT status FROM jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if not row or row[0] != "queued":
            return False
        now = datetime.now(timezone.utc).isoformat()
        db.execute(
            """UPDATE jobs SET status='running', terminal_id=?,
               bridge_port=?, started_at=? WHERE job_id=?""",
            (terminal_id, bridge_port, now, job_id),
        )
        db.commit()
        return True

    def complete_job(self, job_id: int, sharpe: float = 0.0,
                     max_dd: float = 0.0, win_rate: float = 0.0,
                     trade_count: int = 0, pnl: float = 0.0) -> bool:
        """Mark job as completed with results."""
        db = self.connect()
        now = datetime.now(timezone.utc).isoformat()
        db.execute(
            """UPDATE jobs SET status='completed', completed_at=?,
               sharpe=?, max_dd=?, win_rate=?, trade_count=?, pnl=?
               WHERE job_id=?""",
            (now, sharpe, max_dd, win_rate, trade_count, pnl, job_id),
        )
        db.commit()
        _log(f"Job {job_id}: completed (Sharpe={sharpe:.2f})")
        return True

    def fail_job(self, job_id: int, error: str = "",
                 permanent: bool = False) -> bool:
        """Mark job as failed. If retry_count < MAX_RETRIES, re-queue."""
        db = self.connect()
        row = db.execute(
            "SELECT retry_count FROM jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if not row:
            return False

        retries = row[0]
        if permanent or retries >= MAX_RETRIES:
            now = datetime.now(timezone.utc).isoformat()
            db.execute(
                """UPDATE jobs SET status='permanently_failed',
                   completed_at=?, error_message=? WHERE job_id=?""",
                (now, error, job_id),
            )
            _log(f"Job {job_id}: permanently_failed ({error})")
        else:
            db.execute(
                """UPDATE jobs SET status='queued',
                   retry_count=retry_count+1, terminal_id=NULL,
                   bridge_port=NULL, started_at=NULL, error_message=?
                   WHERE job_id=?""",
                (error, job_id),
            )
            _log(f"Job {job_id}: failed, retry {retries+1}/{MAX_RETRIES}")

        db.commit()
        return True

    def reset_job(self, job_id: int) -> bool:
        """Reset a job back to queued (manual recovery)."""
        db = self.connect()
        db.execute(
            """UPDATE jobs SET status='queued', retry_count=0,
               terminal_id=NULL, bridge_port=NULL, started_at=NULL,
               completed_at=NULL, error_message=NULL
               WHERE job_id=?""",
            (job_id,),
        )
        db.commit()
        return True


# ── Self-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    print("=== JobQueue self-test ===")

    tmp = Path(tempfile.mkdtemp())
    db_path = tmp / "test_jobs.db"
    jq = JobQueue(db_path)

    print(f"\n[1] Database: {db_path}")
    jq.connect()

    # Insert mock jobs
    db = jq._conn
    db.execute(
        """INSERT INTO jobs (symbol, fold_id, checkpoint_path, status, tick_coverage_pct)
           VALUES ('EURUSD', '202601', '/tmp/ckpt/EURUSD/fold_202601', 'queued', 100.0)"""
    )
    db.execute(
        """INSERT INTO jobs (symbol, fold_id, checkpoint_path, status, tick_coverage_pct)
           VALUES ('EURUSD', '202602', '/tmp/ckpt/EURUSD/fold_202602', 'queued', 95.5)"""
    )
    db.execute(
        """INSERT INTO jobs (symbol, fold_id, checkpoint_path, status, tick_coverage_pct)
           VALUES ('USA500', '202601', '/tmp/ckpt/USA500/fold_202601', 'queued', 100.0)"""
    )
    db.commit()

    print(f"\n[2] Initial state:")
    summary = jq.status_summary()
    for s, cnt in summary.items():
        print(f"    {s}: {cnt}")

    # Next queued
    next_job = jq.next_queued()
    print(f"\n[3] Next queued: {next_job['symbol']} fold_{next_job['fold_id']}")

    # Start it
    ok = jq.start_job(next_job["job_id"], "terminal_1", 9998)
    print(f"\n[4] start_job on terminal_1: {'OK' if ok else 'FAIL'}")
    running = jq.get_job(next_job["job_id"])
    print(f"    Status: {running['status']}, Terminal: {running['terminal_id']}")

    # Complete it
    jq.complete_job(next_job["job_id"], sharpe=2.35, max_dd=0.12, win_rate=55.0,
                    trade_count=150, pnl=5000.0)
    completed = jq.get_job(next_job["job_id"])
    print(f"\n[5] Completed: status={completed['status']}, Sharpe={completed['sharpe']}")

    # Fail + retry
    job2 = jq.next_queued()
    jq.start_job(job2["job_id"], "terminal_1", 9999)
    jq.fail_job(job2["job_id"], "bridge timeout")
    failed = jq.get_job(job2["job_id"])
    print(f"\n[6] Failed+retry: status={failed['status']}, retries={failed['retry_count']}")

    # Permanently fail
    jq.start_job(job2["job_id"], "terminal_1", 9999)
    jq.fail_job(job2["job_id"], "bridge timeout", permanent=True)
    perm = jq.get_job(job2["job_id"])
    print(f"\n[7] Permanent: status={perm['status']}, error={perm['error_message']}")

    # Final summary
    print(f"\n[8] Final summary: {jq.status_summary()}")
    print(f"    Pending: {jq.pending_count()}, Running: {jq.running_count()}")

    jq.close()
    print("\n=== Done ===")
