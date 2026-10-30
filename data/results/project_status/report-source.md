# Research source record — intraday mechanic deep research

**Date:** 2026-09-06  
**Audience:** the US500/XAUUSD/USA30 strategy-research project.  
**Scope:** evidence synthesis and triage for one new intraday mechanical mechanism or a cost-floor improvement. No data, signals, code, or tester activity.  
**Canonical user-facing report:** INTRADAY_DEEP_RESEARCH_2026-09-06.md.

## Executive answer

No new intraday SPEC is authorized by the evidence currently available. The only conditional lead is a signed U.S. macro-surprise response in XAUUSD. Its reported total reaction is large relative to the $0.41 CFD cost, but the evidence includes the immediate information jump and does not demonstrate an executable return after a known entry latency. A verified as-known-at-the-time event/consensus/timestamp dataset is a prerequisite, not an implementation detail.

Full GC is the only examined futures route whose provisional published one-tick cost is below half of the XAUUSD CFD floor. ES and YM do not roughly halve the US500/USA30 floors, so they do not change the decision to defer a futures migration for the two marginal Model-4 survivors.

## Assumptions and decision rules

- Current project floors are treated as fixed measurements: US500 0.51 points, XAUUSD $0.41/oz, USA30 2.282 points, all round trip.
- The required sample convention is at least 150 unique trade dates over approximately five to six years.
- A reaction at the release time cannot be credited to a strategy that needs first to observe the released data and submit an order.
- An unavailable exchange order book cannot be reconstructed from CFD OHLCV, tick count, or broker tick volume.
- Published research is an existence proof, never a substitute for current walk-forward evidence. McLean-Pontiff-style decay is a conservative robustness warning, not a mechanical parameter adjustment.

## Claim-to-source ledger

| Material claim | Source and quality | Scope / access note | Confidence and limitation |
|---|---|---|---|
| One-standard-deviation positive NFP, retail-sales, and durable-goods surprises had first-five-minute gold-futures coefficients of -0.3050%, -0.0783%, and -0.0644% | [Impact of macroeconomic news on metal futures — Elder, Miao, Ramchander, Journal of Banking & Finance, 2012](https://mountainscholar.org/bitstream/10217/206884/1/Miao_H_BanFin_2012.pdf) [PR] | Tick-level metal-futures study, 2002–2008; live-checked full text | High for the reported historical coefficients. It measures a contemporaneous event window, not delayed trade expectancy. |
| Gold’s largest macro-announcement movement is immediate and subsides rapidly, often within roughly 90 seconds | [The importance of belief dispersion in the response of gold futures to macroeconomic announcements — Smales and Yang, International Review of Financial Analysis, 2015](https://espace.curtin.edu.au/bitstream/handle/20.500.11937/40677/227313.pdf?sequence=4) [PR] | Short-window gold-futures analysis; live-checked author manuscript | High for the timing limitation; this weakens, rather than validates, a fixed delayed entry. |
| Three fixed release families can nominally provide 36 observations/year | [Employment Situation schedule — BLS](https://www.bls.gov/schedule/news_release/empsit.htm), [Retail Trade schedule — Census](https://www.census.gov/retail/release_schedule.html), [Durable Goods schedule — Census](https://www.census.gov/manufacturing/m3/release_schedule.html) [EX] | Official schedules; live-checked | High for scheduled frequency. Actual usable unique dates still require an audit of overlaps/missing records. |
| MT5 calendar data have values/forecast fields, but strategy-test calendar calls are unavailable and a historical value-arrival timestamp is not supplied as a testable event clock | [CalendarValueHistory — MQL5 Reference](https://www.mql5.com/en/docs/calendar/calendarvaluehistory), [Economic calendar chapter — MQL5 AlgoBook](https://www.mql5.com/en/book/advanced/calendar) [EX] | Official platform documentation; live-checked | High for the platform limitation. An external, archived event source could resolve it. |
| Auction imbalances move individual stocks; reported price effects are expressed in each stock’s own daily-spread units | [Closing auction immediate market impact, price drift, and transaction cost of trading — NYSE, 2023](https://www.nyse.com/data-insights/closing-auction-immediate-market-impact-price-drift-and-transaction-cost-of-trading) [EX] | NYSE research on individual Russell 1000 names; live-checked | High for the stock-level result. No direct conversion to US500 CFD points or post-message index expectancy. |
| Historical auction/depth data products exist, subject to acquisition/licensing | [NYSE historical TAQ integrated feed](https://www.nyse.com/market-data/historical/taq-integrated-feed) and [NYSE auction imbalances](https://www.nyse.com/data-products/catalog/imbalances) [EX] | Primary exchange product descriptions; live-checked | High for availability categories; price, exact historical fields, and constituent coverage must be quoted before a project begins. |
| OFI/queue imbalance results require exchange-book inputs and are not a constructible CFD OHLCV proxy; OFI evidence is primarily contemporaneous | [The price impact of order flow imbalance — Cont, Kukanov, Stoikov](https://arxiv.org/html/1011.6402v3) [WP]; [Queue imbalance as a one-tick-ahead price predictor in a limit order book — Gould, Bonart](https://arxiv.org/html/1512.03492v1) [WP] | Research manuscripts; live-checked | High for data dependency and contemporaneous caveat. No cost-clearing forward magnitude for a valid CFD proxy was found. |
| Modern overnight ES drift was near zero in the cited 2021–2025 review window | [The disappearing overnight drift — Federal Reserve Bank of New York, 2026](https://libertystreeteconomics.newyorkfed.org/2026/07/the-disappearing-overnight-drift/) [EX] | Official research commentary; live-checked | Moderate-to-high as a modern contextual check; overnight is already a closed project family. |
| Ironbeam API/data access requires a funded account and is governed by market-data/activity conditions | [Ironbeam API access](https://www.ironbeam.com/knowledge-base/ironbeam-api-access/) and [Ironbeam API Docs](https://docs.ironbeamapi.com/) [EX] | Broker documentation; live-checked | High for access-condition limitation. Do not infer a free, ten-year research archive. |
| Provisional futures cost inputs: broker promotion, current exchange fees, and NFA assessment | [Ironbeam futures promotion](https://www.ironbeam.com/futures-account-promo/), [Ironbeam round-turn costs](https://www.ironbeam.com/futures-round-turn-cost/), [Ironbeam commission FAQ](https://www.ironbeam.com/knowledge-base/what-are-your-commission-rates/), [CME fee schedule, effective 2026-04-01](https://www.cmegroup.com/company/files/cme-fee-schedule-2026-04-01.pdf), [CBOT fee schedule](https://www.cmegroup.com/company/files/cbot-fee-schedule-2026-04-01.pdf), [NFA assessment fees](https://www.nfa.futures.org/faqs/members/nfa-assessment-fees.html) [EX] | Primary vendor/regulatory material; live-checked | High for stated inputs on review date. Clearing/routing, platform/data, spread, and slippage remain unquoted or variable. |
| Contract point values/ticks support the price-unit comparison | [Micro E-mini S&P contract specs](https://www.cmegroup.com/markets/equities/sp/micro-e-mini-sandp-500.contractSpecs.html), [Micro E-mini FAQ](https://www.cmegroup.com/articles/faqs/micro-e-mini-equity-index-futures-frequently-asked-questions.html), [E-mini Dow contract specs](https://www.cmegroup.com/markets/equities/dow-jones/e-mini-dow.contractSpecs.html), [CME gold education](https://www.cmegroup.com/education/courses/event-contracts-underlying-markets/product-gold) [EX] | CME primary specifications; live-checked | High. Broker execution and margin remain separate matters. |
| CME history can extend to 2010 for some schemas/products, but genuine MBO began only in 2016–2017 | [Databento CME-history extension](https://databento.com/blog/CME-history-extended-to-2010) [PP]; [CME MBO FAQ](https://www.cmegroup.com/articles/faqs/market-by-order-mbo.html?redirect=%2Feducation%2Fmarket-by-order-mbo.html) [EX] | Provider and exchange documentation; live-checked | High for chronology. A separate purchase/onboarding audit is necessary for exact product coverage and quality. |
| Published anomaly returns fall out of sample and after publication | [Does Academic Research Destroy Stock Return Predictability? — McLean and Pontiff, Journal of Finance, 2016](https://doi.org/10.1111/jofi.12365) [PR] | Broad cross-sectional anomaly evidence | High as a generic caution; not a direct estimate for macro-event trading. |

## Derived calculations retained for audit

At $2,500/oz, the historical five-minute coefficients convert as follows:

- NFP: 0.003050 × 2,500 = $7.625, or 18.6 XAUUSD CFD spread floors.
- Retail sales: 0.000783 × 2,500 = $1.9575, or 4.8 floors.
- Durable goods: 0.000644 × 2,500 = $1.6100, or 3.9 floors.

These are explicitly total release-window moves, not estimated trade P&L.

The futures lower-bound formula is one tick plus two sides of published known fees divided by the contract point value. With a one-tick market and no unquoted clearing/routing/slippage:

- ES: 0.25 + 2 × ($0.99 + $1.38 + $0.01)/$50 = 0.3452 index points.
- MES: 0.25 + 2 × ($0.39 + $0.35 + $0.01)/$5 = 0.5500 index points.
- GC: $0.10 + 2 × ($0.99 + $1.65 + $0.01)/$100 = $0.1530/oz.
- MGC: at least $0.10 + 2 × ($0.70 + $0.01)/$10 = $0.2420/oz, before an applicable broker commission.
- YM: 1 + 2 × ($0.99 + $1.38 + $0.01)/$5 = 1.9520 points.
- MYM: 1 + 2 × ($0.39 + $0.35 + $0.01)/$0.50 = 4.0000 points.

## Research stop rationale

The material questions have primary evidence or an explicit evidence limitation:

- The only cost-clearing new mechanism lacks a verifiable post-entry residual and timestamped inputs.
- L2/auction mechanisms lack the required data and a transferable index-cost magnitude.
- Futures can reduce cost for full GC but do not rescue US500/USA30 marginal survivors.

Further broad searching is unlikely to change the immediate decision. The next useful work is a bounded data-provenance qualification for the Rank-1 event dataset, not another pattern search.
