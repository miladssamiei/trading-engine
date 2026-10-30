# Read-only FTMO MT5 quote capture

Use this only with a terminal already connected to your own FTMO Challenge or Free Trial. It reads the current account, one broker symbol's specifications and the latest bid/ask quote; it has no order functions and takes no login or password. The script refuses a non-FTMO server, a server that differs from the exact name you supply, and a custom/imported symbol.

Run with the project Python and the exact terminal executable and symbol shown in Market Watch. Example (replace the placeholders, including the output directory):

```powershell
& 'C:\Users\Milad\miniconda3\envs\reinforce_rl\python.exe' scripts\ftmo_quote_capture.py --terminal 'C:\PATH\TO\FTMO\terminal64.exe' --expected-server 'YOUR-FTMO-SERVER' --symbol 'XAUUSD' --alias 'my-challenge' --server-utc-offset-hours 3 --output-dir 'data\ftmo_capture_001' --minutes 60
```

Use a plain, non-secret alias. Never put an account number, password, token or private key in the command or files. Run a separate new output directory for each symbol and capture period. Keep the terminal connected; `Ctrl+C` stops capture and writes its summary. The script samples the latest tick when polled, so it does not claim to retain every market tick. It ignores quotes more than five seconds old. A captured quote's `received_at_utc` is the local UTC time after the terminal API returns that quote; a historical tick export cannot supply this forward receipt time. The offset must be checked for each capture: [FTMO says its MT5 server uses GMT+2 with daylight saving time](https://ftmo.com/en/faq/what-are-the-account-specifications/), which was +3 hours on 2026-09-25. The recorder rejects rows that do not fit the supplied offset and stores the raw broker timestamp endpoints and conversion rule in its summary. Do not use one fixed offset across a daylight saving clock change.

The output contains `snapshot_partial.json`, `quotes.csv` and `capture_summary.json`. The snapshot deliberately leaves initial balance, program/phase confirmation, trading sessions, fill modes, rollover multipliers, swap units and quote currency unresolved where the Python API cannot establish them. Review those fields against the terminal's specification and FTMO account view before using the existing [input schema](../data/results/ftmo10k_qualification_v1/INPUT_SCHEMA.md). A single short capture does not meet the importer’s 20-weekday coverage or independent provenance checks; multiple sessions need a reviewed chronological merge. Actual order fills and slippage remain separate evidence.
