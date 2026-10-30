"""Protect the public export boundary and the original workspace."""
from pathlib import Path
import tempfile

import pytest

from tools.prepare_publication import audit, candidates, credential_findings, export


def test_allowlist_keeps_both_systems_but_excludes_runtime_and_scratch():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for name in ("evaluate/alpha.py", "evaluate/orchestrator.py", "bridge/core.py",
                     "ea/MULTI_CAIRO_EA.mq5", "ea/ORB_Strategy.mq5", "ea/ORB_Strategy.ex5",
                     "ea/embed/orb_child.mqh",
                     "data/results/study/SPEC_A.txt", "data/results/study/tape.csv",
                     "temp_secret.py", ".env", "data/queue/jobs.db"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture\n", encoding="utf-8")
        selected = {p.as_posix() for p in candidates(root)}
        assert "evaluate/alpha.py" in selected
        assert "archive/legacy/evaluate/orchestrator.py" in selected
        assert "archive/legacy/bridge/core.py" in selected
        assert "archive/legacy/ea/MULTI_CAIRO_EA.mq5" in selected
        assert "ea/ORB_Strategy.mq5" in selected
        assert "ea/embed/orb_child.mqh" in selected
        assert "data/results/study/SPEC_A.txt" in selected
        assert not any(p.endswith((".csv", ".db", ".ex5")) for p in selected)
        assert "temp_secret.py" not in selected and ".env" not in selected


def test_nonempty_destination_is_never_cleared():
    with tempfile.TemporaryDirectory() as td:
        root, target = Path(td) / "source", Path(td) / "target"
        root.mkdir()
        target.mkdir()
        marker = target / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with pytest.raises(ValueError, match="absent or empty"):
            export(root, target)
        assert marker.read_text() == "keep"


def test_nested_destination_is_rejected_before_creation():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with pytest.raises(ValueError, match="separate"):
            export(root, root / "nested")
        assert not (root / "nested").exists()


def test_credential_screen_checks_values_without_reporting_them():
    value = "sensitive" + "-fixture-value"
    assert credential_findings('password = "' + value + '"')
    assert not credential_findings('password = "your password here"')
    assert not credential_findings('password = os.getenv("DB_PASSWORD", "")')
    assert credential_findings("login=" + str(123 * 10001))
    assert credential_findings("gh" + "p_" + "a" * 36)


def test_audit_rejects_quotes_and_finds_broken_new_doc_links():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "prices.csv").write_text("time,close\n", encoding="utf-8")
        (root / "README.md").write_text("[missing](missing.md)\n", encoding="utf-8")
        errors = audit(root)
        assert any("Unexpected file type" in e for e in errors)
        assert any("Broken link" in e for e in errors)
