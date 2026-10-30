# Publication policy

The development workspace contains both source and years of machine-local
research output. The public repository is a deliberately selected source release
with independent initial Git history. Existing local Git history is preserved.

## Included

- Active evaluation Python sources and native MQL5 source/includes.
- Shared execution/guard diagnostics and credential-free tester examples.
- Offline tests and a synthetic execution example.
- Original top-level SPEC/GATE/REPORT text records that pass the publication checks.
- CAIRO source, orchestration, historical documentation and integration tests in
  `archive/legacy/`, preserving its internal relative layout.
- The standalone exploratory Python bot in `archive/experiments/`.

## Excluded

Raw or aggregated market quote tapes, account exports/snapshots, full runtime
ledgers, trained weights, databases, terminal folders, generated configurations,
compiled EAs/DLLs, logs, IDE state, knowledge-graph output, cache/worktree folders
and personal `temp_*.py` scratch scripts are not published.

Historical text documents are copied without rewriting their contents. A record
flagged for a potential credential is excluded as a whole, with the exclusion
listed in the publication manifest. Do not silently redact and relabel it as
the original artifact. Local originals remain untouched.

## Build and audit

From the development workspace:

```powershell
python tools/prepare_publication.py --output C:\path\to\empty\trading-engine
```

The output must be outside the source tree and absent or empty. The tool never
recursively clears a directory, edits frozen sources or pushes to a remote.
It writes a file/hash manifest and generates the public study index.

From the prepared publication:

```powershell
python tools/prepare_publication.py --audit-only .
python -m pytest -q
python examples/bracket_replay.py
```

After a reviewed source/documentation change inside the prepared public
repository, refresh its file/hash snapshot before committing:

```powershell
python tools/prepare_publication.py --refresh-manifest .
```

Refresh first audits contents and requires an existing publication manifest;
it is not a way to alter frozen historical artifacts or bypass review.

The audit checks forbidden paths/types, file size, credential patterns,
Python syntax, local Markdown links and manifest hashes. The allowlist and
scanner are defense in depth, not a guarantee that every sensitive fact can be
recognized automatically. Review the prepared Git diff before publishing.

## Preservation and maintenance

The exporter does not remove local datasets or rewrite research history. Updates
to frozen studies require new artifacts. A public source correction must say
which measured results it changes; documentation cleanup alone cannot upgrade
a research verdict. Choose and add an explicit license before describing this
project as open source.
