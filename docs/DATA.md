# Data conventions and reproducibility

## Inputs

Market tapes are excluded from Git. Obtain data under the provider's terms and
keep a local source manifest with provider, symbol, bid/ask semantics, export
version, coverage, row count and SHA-256. `GC`/`ES`/`NQ` labels in a script can
refer to proxies; verify the underlying data rather than assuming exchange fills.

Typical M1 schema is `time,open,high,low,close,tick_volume`. `time` is UTC Unix
seconds. Each timestamp is the **bar open**, not its completion. Some loaders
need only OHLC; volume-based strategies require meaningful tick-volume input.
MT5 trade/day exports often use semicolons and `YYYY.MM.DD HH:MM` UTC times.

Validate monotonic timestamps, duplicates, positive/finite OHLC, high/low
geometry, gaps and exact bar duration. A loader should reject ambiguous units
rather than silently infer them from price magnitude.

## Clocks and grids

- Source clock: UTC epoch.
- Session clock: `America/New_York`, DST-aware.
- Forex/CFD daily anchor: 17:00 New York where specified.
- Some H4 studies: NY-wall buckets `[00,04)`, `[04,08)`, etc. DST transition
  buckets can span three or five real hours.
- Account daily-loss reset: its own study/account clock, often Europe/Prague.

Continuous grids and observed-only grids are both present in historical studies.
Read the SPEC: fieldwise filling OHLC is not equivalent to inserting a flat
prior-close bar. Never use a synthetic row as evidence of an executable entry.

The original US500 protocol builds M15 from the verified M5 export where
required; an M1 export can omit no-tick minutes. A study explicitly requesting
M1 aggregation or carried-row removal documents a separate data definition.

## Carried rows

Some exports contain synthetic `O=H=L=C=prior_close` rows. Recent studies remove
them before aggregation. That predicate can also remove genuine unchanged flat
observations; retain counts and disclose the ambiguity. Do not apply this
transformation retroactively to frozen studies with another definition.

## Coverage

Names such as `2020_2026` describe intended study periods, not a promise of full
coverage. For example, the Edris study's supplied US500 tape ends on 2026-04-30,
USTECH on 2026-09-23 and XAUUSD on 2026-09-01. Its report records actual ranges.
Pre-sample history may be warmup only. Incomplete years and terminal open trades
must be reported separately.

## Costs

Price units, pips, lots, point values and contract sizes are separate quantities.
A one-unit Gold move cannot be interpreted as account dollars without volume
and contract size. Round-trip fixed floors are typically embedded once through
the ask side. Percentage commissions and swap debits need separate cash/price
conversion. Current swap snapshots extrapolated backward are sensitivity
scenarios, not historical financing observations.

Historical costs differ intentionally. US500 studies include 0.51, 0.55 or 0.8
point assumptions; Gold studies include 0.41 and 0.585 unit floors. These are
frozen scenarios, not current executable quotes. Check each SPEC and report.

## Reproducing a study

1. Read its frozen SPEC, GATE and REPORT, including failure verdict and limitations.
2. Supply the matching tape and verify source hashes where available.
3. Inspect imports, paths, symbol definitions, timezone and original package versions.
4. Freeze a **new** replication specification/output directory. Do not remove or
   overwrite an existing report to bypass an evaluator's safety check.
5. Run synthetic cases, then the full evaluation, preserving new hashes and ledgers.
6. Reconcile persisted artifacts independently before interpreting results.

The public release contains source and text evidence. Omitted tapes, account
snapshots, complete ledgers and terminal caches limit independent historical
verification until equivalent inputs are provided locally.
