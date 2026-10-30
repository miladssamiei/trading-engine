# Selected results and evidence limits

These are historical research records, not current broker terms or a promise of
returns. Consult the original documents in the [study index](STUDIES.md). The
public source release omits tapes and many generated ledgers, so a published
report is not itself an independently reproduced result.

## ORB and SP2L: real-tick baselines

The individually validated original implementations used **Model=0 generated
ticks**. The later real-tick checks were materially weaker:

| Baseline | Closed trades | Net native-unit total | Daily annualized Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
| ORB, Model=4 | 437 | +114.91 | 0.0745 | 7.92% |
| SP2L, Model=4 | 1,349 | +138.65 | 0.133, common day universe | 5.13% |

SP2L's 2025 net result was -341.68. These figures use the historical study's
cost, capital, lot and day-universe conventions. They do not establish a robust
current prop-trading edge. See `tick_model_revalidation` and
`tick_model_revalidation_children` in the study index.

## Regime rotation

Offline Phase 1 recorded Sharpe 0.720; execution Phase 2 failed under both tick
models. A D1 history-source difference changed the 2025-05 regime label, rather
than being resolved by changing synthetic ticks to real ticks. Preserve the
failed verdict; do not quote the offline number as an executed strategy result.

## Macro Gold and crypto

The financed Crypto+Gold proxy study recorded 2,275 closed trades, approximately
+0.09691 net R/trade and pooled bootstrap probability 0.0119 under its declared
cost/financing scenario. That is conditional Phase-1 evidence. It does not
establish matching live cash fills, a firm-rule-complete equity path or full EA
parity. Current financing extrapolated over history is not historical swap data.

Gold macro EA parity investigations matched 328 entry signals/initial stops in
the v1.02 comparison, but recorded two 0.02-unit Engine B ratchet differences
against a 0.01 tick tolerance; G3 failed. Independent cash/fill reconstruction
remained unverified. Matching entries is insufficient to claim identical trading.

## EURUSD grid

The unrestricted grid's closed-basket win rate hid extreme floating exposure:
633 closed winning baskets, worst floating loss around 28,232.85 base-lot pip
equivalents, maximum depth 11 and total size approximately 171 times the base.
Profit-only exits make the closed-basket win rate particularly misleading.

A separately declared risk-control study froze development selection before
validation. Its selected 0.02-lot / 72-hour restart case produced **-455.36 EUR**
over 238 validation baskets, expectancy -1.91328 EUR and bootstrap probability
0.7890. The adopted verdict was **NOT_ADOPTED** despite no modeled official
daily/static rule breach. Loss limits reduce exposure; they do not create edge.

## Edris H4 sweep / M30 divergence / inside bar

All six cells failed their declared Phase-1 gates:

| Asset | Entry arm | Trades | Net R/trade | R Sharpe | MBB probability | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| US500 | All hours | 177 | -0.03172 | -0.01842 | 0.5646 | FAIL |
| US500 | Killzones | 15 | +0.33333 | +0.17078 | 0.0904 | FAIL |
| USTECH | All hours | 206 | +0.00956 | +0.00549 | 0.4370 | FAIL |
| USTECH | Killzones | 23 | -0.47826 | -0.34722 | 0.8859 | FAIL |
| XAUUSD | All hours | 168 | -0.11905 | -0.07160 | 0.8071 | FAIL |
| XAUUSD | Killzones | 34 | +0.05882 | +0.03284 | 0.4370 | FAIL |

USTECH all-hours has positive mean R but negative native-unit expectancy;
XAUUSD all-hours has the reverse. Both fail the two-part expectancy gate. The
killzone arms also fail the sample-size gate. No cell advances to MT5 validation.
These bootstrap probabilities are conditional on researched history and are not
adjusted for the full project-wide search.

## CAIRO

The old ML bridge did not reproduce its offline performance in MT5 and received
a no-go verdict. Its source is preserved in the [legacy archive](../archive/legacy/README.md),
not promoted as an available trained trading product.

## Reading any positive finding

Check sample size, cost units, data provenance, synthetic versus real execution,
year cohorts, selection history, terminal censoring, overnight financing,
concurrency and whether account rules were checked on marked equity. No single
win rate, Sharpe, bootstrap number or Monte Carlo pass rate answers all of these.
