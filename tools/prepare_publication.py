"""Prepare a source-only publication without modifying the research workspace.

Uses an explicit allowlist, refuses a nonempty destination and audits the result.
No git, terminal, market-data or network operations are performed by this tool.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LEGACY_EVALUATORS = {
    "bridge_manager", "config_generator", "dll_preflight", "job_queue",
    "orchestrator", "queue_usa500_folds", "result_collector", "terminal_manager",
}
LEGACY_TESTS = {
    "smoke_roundtrip.py", "test_bridge_signals.py", "test_dll_protocol.py",
    "test_signals_near_end.py", "test_single_job_match_port.py",
    "test_terminal_manager_basic.py",
}
OFFLINE_TESTS = {
    "conftest.py", "test_execution_safe_model4_v1.py",
    "test_ftmo10k_qualification_v1.py", "test_ftmo_account_replay_v1.py",
    "test_ftmo_public_evidence_v1.py", "test_har_risk_scaling_model4_v1.py",
    "test_edris_execution_examples.py", "test_publication.py",
}
ROOT_FILES = {
    "README.md", "CONTRIBUTING.md", "SECURITY.md", "pyproject.toml",
    "requirements.txt", ".gitignore", ".gitattributes",
}
CODE_EXTENSIONS = {".py", ".mq5", ".mqh", ".c", ".h", ".ps1"}
TEXT_EXTENSIONS = CODE_EXTENSIONS | {".md", ".txt", ".toml", ".yml", ".yaml", ".json", ".ini"}
FORBIDDEN_PARTS = {
    ".kilo", ".dukascopy-cache", "graphify-out", "node_modules", ".venv",
    "venv", "checkpoints", "terminals", "compile_logs", "temp_test", "~",
}
IGNORED_RUNTIME_PARTS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
TOKEN_PATTERN = re.compile(
    r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|"
    r"AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)
# Values, not mere mentions of configuration keys. Never report matched values.
CREDENTIAL_PATTERN = re.compile(
    r'''(?im)["']?(?:password|passwd|api_key|api_secret|access_token|auth_token|db_password)["']?\s*[:=]\s*["']([^"'\r\n]{4,})["']'''
)
LOGIN_PATTERN = re.compile(r'''(?im)["']?(?:login|account_number|account_login)["']?\s*[:=]\s*["']?\d{5,}''')
PLACEHOLDER = re.compile(r"(?i)^(?:your[ _-]|example|placeholder|dummy|test[ _-]|fake|<|\$|\*|REDACTED|os\.)")
MAX_BYTES = 5 * 1024 * 1024


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def credential_findings(text: str) -> list[str]:
    findings = []
    if TOKEN_PATTERN.search(text):
        findings.append("token/private-key pattern")
    if LOGIN_PATTERN.search(text):
        findings.append("potential account identifier")
    for match in CREDENTIAL_PATTERN.finditer(text):
        if not PLACEHOLDER.search(match.group(1)):
            findings.append("potential literal credential")
            break
    return findings


def candidates(root: Path) -> dict[Path, Path]:
    """Map release-relative destination to original source; no scratch traversal."""
    mapped: dict[Path, Path] = {}

    def add(source: Path, dest: Path | None = None):
        if source.is_file() and not source.is_symlink():
            rel = dest or source.relative_to(root)
            if rel in mapped and mapped[rel] != source:
                raise ValueError(f"Publication path collision: {rel}")
            mapped[rel] = source

    for name in ROOT_FILES:
        add(root / name)
    for directory in ("docs", "examples", "tools", ".github/workflows"):
        for source in (root / directory).rglob("*"):
            if source.suffix.lower() in TEXT_EXTENSIONS and not any(
                part in IGNORED_RUNTIME_PARTS for part in source.relative_to(root).parts
            ):
                add(source)
    for name in ("data/README.md", "configs/README.md", "archive/legacy/README.md",
                 "archive/experiments/README.md"):
        add(root / name)
    for source in (root / "configs/examples").glob("*.ini"):
        add(source)
    for source in (root / "evaluate").glob("*.py"):
        dest = (Path("archive/legacy/evaluate") / source.name
                if source.stem in LEGACY_EVALUATORS else None)
        add(source, dest)
    for directory in ("ea", "Include", "scripts", "deploy/infrastructure"):
        for source in (root / directory).rglob("*"):
            if any(p in IGNORED_RUNTIME_PARTS or p == "compile_logs" for p in source.relative_to(root).parts):
                continue
            if source.suffix.lower() not in CODE_EXTENSIONS | {".md", ".json", ".txt"}:
                continue
            if source.stem.startswith("AuditTrivial") or source.stem == "ScanHistory":
                continue
            dest = (Path("archive/legacy/ea") / source.name
                    if source.name in {"MULTI_CAIRO_EA.mq5", "transport_dll.mqh", "test_bridge_client.mq5"}
                    else None)
            add(source, dest)
    for name in OFFLINE_TESTS:
        add(root / "tests" / name)
    for name in LEGACY_TESTS:
        add(root / "tests" / name, Path("archive/legacy/tests") / name)
    for directory in ("bridge", "train", "live", "optimizer"):
        for source in (root / directory).rglob("*"):
            if source.suffix.lower() in CODE_EXTENSIONS:
                add(source, Path("archive/legacy") / source.relative_to(root))
    for name in ("config.py", "cairo_aggregation.py", "deploy/compile_dll.ps1"):
        add(root / name, Path("archive/legacy") / name)
    for name in ("GUIDE.md", "project_info__1.md", "project_info__2.md"):
        add(root / name, Path("archive/legacy/docs") / name)
    add(root / "evaluate/__init__.py", Path("archive/legacy/evaluate/__init__.py"))
    add(root / "SP2L_Advanced_Bot.py", Path("archive/experiments/SP2L_Advanced_Bot.py"))
    for study in sorted((root / "data/results").iterdir()):
        if study.is_dir():
            for source in sorted(study.iterdir()):
                if source.suffix.lower() in {".txt", ".md"} and (
                    source.name.upper().startswith(("SPEC", "GATE", "REPORT"))
                    or source.name.upper().startswith("REVALIDATION_REPORT")
                    or source.name == "FROZEN_PARENT_HASHES.txt"
                ):
                    add(source)
    return mapped


def study_index(root: Path) -> None:
    lines = ["# Published study records", "",
             "Original frozen text records included in this source release. Full market tapes,",
             "runtime ledgers and terminal output are not distributed. Read each report's",
             "costs, data dates, tick model, gates and limitations before interpreting it.", "",
             "A listed study is not necessarily passing, independently validated or runnable",
             "without additional local data and configuration.", "",
             "| Study | Published records |", "| --- | --- |"]
    for study in sorted((root / "data/results").iterdir()):
        files = sorted(p for p in study.iterdir() if p.is_file())
        if files:
            links = ", ".join(f"[{p.name}](../{p.relative_to(root).as_posix()})" for p in files)
            lines.append(f"| `{study.name}` | {links} |")
    (root / "docs/STUDIES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(root: Path, verify_manifest: bool = True) -> list[str]:
    errors = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(p in IGNORED_RUNTIME_PARTS or p.endswith(".egg-info") for p in rel.parts):
            continue
        if path.is_symlink():
            errors.append(f"Symbolic link: {rel.as_posix()}")
            continue
        if not path.is_file():
            continue
        if any(p in FORBIDDEN_PARTS for p in rel.parts) or path.name.startswith(".env") or path.name.startswith("temp_"):
            errors.append(f"Forbidden path: {rel.as_posix()}")
        if path.suffix.lower() not in TEXT_EXTENSIONS and path.name not in {".gitignore", ".gitattributes"}:
            errors.append(f"Unexpected file type: {rel.as_posix()}")
            continue
        if path.stat().st_size > MAX_BYTES:
            errors.append(f"File exceeds 5 MiB: {rel.as_posix()}")
            continue
        try:
            text = read_text(path)
        except UnicodeError:
            errors.append(f"Unrecognized text encoding: {rel.as_posix()}")
            continue
        errors += [f"{finding}: {rel.as_posix()}" for finding in credential_findings(text)]
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=rel.as_posix())
            except SyntaxError as exc:
                errors.append(f"Python syntax: {rel.as_posix()}:{exc.lineno}")
        # New documentation links are checked; archived/frozen links retain historical context.
        if path.suffix == ".md" and (rel.parts[0] in {"docs", "configs"} or len(rel.parts) == 1):
            stripped = re.sub(r"```.*?```", "", text, flags=re.S)
            for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^\s)]+)\)", stripped):
                if re.match(r"[a-z]+:|#", target, re.I):
                    continue
                local = unquote(target.split("#", 1)[0])
                if local and not (path.parent / local).exists():
                    errors.append(f"Broken link: {rel.as_posix()} -> {local}")
    manifest = root / "PUBLICATION_MANIFEST.json"
    if verify_manifest and manifest.exists():
        entries = json.loads(manifest.read_text(encoding="utf-8"))["files"]
        for entry in entries:
            path = root / entry["path"]
            if not path.is_file() or sha256(path) != entry["sha256"]:
                errors.append(f"Manifest mismatch: {entry['path']}")
    return errors


def write_manifest(root: Path, excluded: list) -> dict:
    manifest = {
        "format": 1,
        "scope": "Source and frozen text release; local market data and old Git history excluded",
        "excluded_records": excluded,
        "files": [
            {"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "PUBLICATION_MANIFEST.json"
            and not any(part in IGNORED_RUNTIME_PARTS or part.endswith(".egg-info")
                        for part in p.relative_to(root).parts)
        ],
    }
    (root / "PUBLICATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def export(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Publication output must be separate from the source tree")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Publication output must be absent or empty; nothing was removed")
    selected = candidates(source)
    excluded = []
    # Inspect before copying. Frozen text with a potential private value is omitted whole.
    for rel, original in list(selected.items()):
        findings = credential_findings(read_text(original))
        if findings:
            if rel.parts[:2] == ("data", "results") or rel.parts[:3] == ("archive", "legacy", "docs"):
                excluded.append({"path": rel.as_posix(), "reason": findings})
                del selected[rel]
            else:
                raise ValueError(f"Review required before publication: {rel.as_posix()} ({', '.join(findings)})")
    output.mkdir(parents=True, exist_ok=True)
    for rel, original in sorted(selected.items()):
        dest = output / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, dest)
    (output / "data/results").mkdir(parents=True, exist_ok=True)
    study_index(output)
    errors = audit(output, verify_manifest=False)
    if errors:
        raise ValueError("Publication audit failed; output retained for review:\n" + "\n".join(errors))
    manifest = write_manifest(output, excluded)
    return {"files": len(manifest["files"]), "bytes": sum(e["bytes"] for e in manifest["files"]),
            "excluded_records": len(excluded), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--audit-only", type=Path, metavar="ROOT")
    group.add_argument("--refresh-manifest", type=Path, metavar="ROOT")
    args = parser.parse_args()
    if args.audit_only or args.refresh_manifest:
        target = (args.audit_only or args.refresh_manifest).resolve()
        errors = audit(target, verify_manifest=not bool(args.refresh_manifest))
        if errors:
            parser.exit(1, "\n".join(errors) + "\n")
        if args.refresh_manifest:
            existing = target / "PUBLICATION_MANIFEST.json"
            if not existing.is_file():
                parser.exit(1, "Manifest refresh requires an existing prepared publication\n")
            previous = json.loads(existing.read_text(encoding="utf-8"))
            write_manifest(target, previous["excluded_records"])
        print("PASS: publication paths, text, syntax, links and available manifest hashes")
    else:
        try:
            print(json.dumps(export(ROOT, args.output), indent=2))
        except ValueError as exc:
            parser.exit(1, str(exc) + "\n")


if __name__ == "__main__":
    main()
