# The foundation: Reinforce / CAIRO

## Original idea and implementation

**[Reinforce / CAIRO](https://github.com/miladssamiei/reinforce)** is the original
project from which Trading Engine developed. CAIRO means Collective AI
Reinforced Intelligence for trading Outcomes. Its central idea was to treat
different learning algorithms as contributors to a collective trading decision,
with a standardized action interface and separate risk/execution responsibilities.

The original implementation covered the full path from locally supplied market
data to trained policies, saved predictions, ensemble decisions and a MetaTrader
interface. Its primary instruments were EURUSD, USA500 and XAUUSD.

## The seven model families

| Family | Role in the original framework |
| --- | --- |
| PPO | Policy-gradient trading agent |
| IMPALA | Actor-learner training architecture |
| APPO | Asynchronous proximal policy optimization |
| SAC | Entropy-regularized actor-critic agent |
| TQC | Distributional continuous-action agent adapted to trading actions |
| LightGBM | Supervised next-bar classifier with a spread-based HOLD region |
| Decision Transformer | Return-conditioned sequence model trained on saved trajectories |

The model families share SELL/HOLD/BUY semantics, but not one learning objective.
The original research asked whether their differences could improve collective
decisions. It did not establish that agreement represents independent evidence
or guarantees a profitable trade.

The [model guide](https://github.com/miladssamiei/reinforce/blob/main/docs/MODELS.md)
describes the implementations and links primary algorithm references.

## More than the bridge archive

Reinforce contains primary and earlier comparison training pipelines, rolling
walk-forward folds, technical and optional contextual features, fitted-artifact
interfaces, ensemble research, single-/multi-symbol services and a modular engine
refactor. It also contains original risk, sizing, coordination, rollover and
terminal-side source components.

The uniform ensemble's explicit HOLD tie rule, agreement multiplier and
alternative weighted/regime-aware paths are documented in its
[ensemble guide](https://github.com/miladssamiei/reinforce/blob/main/docs/ENSEMBLES.md).
Different inference paths are not presumed equivalent: their feature contracts,
timeframe defaults, input adaptation and vote semantics need reconciliation.

This repository's `archive/legacy/` retains a **later RR integration snapshot**:
the Python-to-MT5 bridge, orchestration, queue and terminal management helpers,
configuration and corresponding historical tests. It does not contain the
complete original training project. Those parts are now available in the
separate foundation repository.

## Why Trading Engine followed

The checkpoint-to-MT5 investigation recorded a major gap between offline
metrics and tester execution. That led to the strategy-research program here:
explicit causal signal rules, executable bid/ask prices, intrabar exits,
SPEC/GATE/REPORT records, tick-model audits and account-risk experiments.

Preserving the original architecture and its detailed documentation is compatible
with preserving the later negative integration verdict. A compelling design,
correct software mechanics and a demonstrated trading edge are separate claims.

## Where to look

| Question | Authoritative starting point |
| --- | --- |
| What was the original idea and full framework? | [Reinforce README](https://github.com/miladssamiei/reinforce#readme) |
| How were models trained and combined? | [Reinforce architecture](https://github.com/miladssamiei/reinforce/blob/main/docs/ARCHITECTURE.md), [training](https://github.com/miladssamiei/reinforce/blob/main/docs/TRAINING.md) |
| What assumptions and implementation differences remain? | [Reinforce research status](https://github.com/miladssamiei/reinforce/blob/main/docs/RESEARCH_STATUS.md) |
| What later bridge source is retained here? | [CAIRO integration archive](../archive/legacy/README.md) |
| What does later execution evidence show? | [Selected results](RESULTS.md) and [frozen study index](STUDIES.md) |
| How does the active research protocol work? | [Research protocol](RESEARCH_PROTOCOL.md) |

Both public repositories are source releases. Local data, trained weights,
account exports and runtime artifacts remain outside GitHub; independent full
reproduction requires the matching authorized inputs and environment.
