# CAIRO: historical ML ensemble and execution bridge

This archive preserves the earlier CAIRO system separately from the active
strategy-research program. It is retained for code provenance and analysis of
the gap between offline predictions and executable trading results.

The **complete original learning/training framework** is now published as
**[Reinforce / CAIRO](https://github.com/miladssamiei/reinforce)**. Start there for
the original idea, seven model families, feature pipeline, ensemble methods and
inference architecture. This archive retains the later RR bridge/orchestration
snapshot; it is not the full foundation project.

## Components

| Path | Purpose |
| --- | --- |
| `bridge/` | Shared model/feature logic, HTTP/socket servers, C DLL source |
| `evaluate/` | Job queue, terminal orchestration, generated tester configurations and result collection |
| `ea/` | `MULTI_CAIRO_EA.mq5`, DLL transport include and bridge test EA |
| `config.py` | Environment-based model, database and execution configuration |
| `cairo_aggregation.py` | Ensemble aggregation utilities |
| `deploy/compile_dll.ps1` | Historical bridge build helper |
| `tests/` | Historical integration scripts; excluded from normal offline test discovery |
| `docs/` | Original GUIDE and project snapshots, with original historical claims |

`train/`, `live/` and `optimizer/` preserve the available scaffolding. The public
repository is not a full distribution of the separately published Reinforce project.
Checkpoint files, training datasets, PostgreSQL databases and compiled DLLs are
not included. External model code and packages referenced by these modules must
be supplied through the foundation project and a compatible local environment;
installing the active project's core dependencies does
not make this archive runnable.

## Historical verdict

Offline Python Sharpes around 15–20 did not transfer to the six-fold MT5 bridge
evaluation (approximately -6.6 to +4.4). The project recorded a **no-go** verdict
and moved to explicit signal/execution reconciliation. These numbers are a
diagnostic record, not a claim of an available profitable model.

## Working with the archive

The internal relative layout is retained, because legacy modules import `config`,
`bridge` and `evaluate` from their own root. Start any deliberate investigation
from `archive/legacy/` in a separate environment; do not add it to the active
project's Python path. The historical tests can start terminals or require
models and external services, so they are not run by the repository CI.

Read the original guide as historical documentation. Paths, deployment commands,
credentials configuration and dependencies need local review before use. Source
publication does not imply successful live deployment or broker compatibility.
