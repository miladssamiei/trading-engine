# MT5 setup and reconciliation

## Requirements

Native compilation/testing requires a local Windows MetaTrader 5 installation
and MetaEditor. The optional Python `mt5` extra is needed only by tools that use
the terminal API. The CAIRO DLL bridge is separate archived software; active
native EAs do not require it unless their own source explicitly says otherwise.

Before running a study, inspect its terminal-path constants and configure your
own data folder, expert names, symbol store and output paths. Historical scripts
can retain the original author's Windows paths. Do not copy local terminal
folders, account credentials or generated tester `.ini` files into Git.

## Compile

Copy the selected `.mq5` and its required `.mqh` files into your local MQL5 tree.
Inspect `#include` directives for the expected relative locations. Typical
PowerShell invocation:

```powershell
$editor = 'C:\Program Files\MetaTrader 5\metaeditor64.exe'
$expert = (Resolve-Path '.\ea\MacroTrendEnsemble_FTMO.mq5').Path
Start-Process -FilePath $editor -ArgumentList "/compile:`"$expert`" /log" -Wait -WindowStyle Hidden
```

Inspect the UTF-16 compile log. The offline CI does not assert a clean MQL5
compile. Broker/terminal builds, symbol metadata and includes must be checked
in the actual installation. Compiled files and logs remain local.

## Tester model

| Setting | Meaning | Evidence limitation |
| --- | --- | --- |
| `Model=0` | Generated ticks from minute-bar history | Synthetic intrabar order and spread; not a real-tick replication |
| `Model=4` | Every tick based on available real ticks | Requires imported coverage; confirm actual range in tester logs |

The original ORB and SP2L synthetic reconciliations are not substitutes for the
later Model=4 records. Check `generating based on real ticks`, tick coverage,
bar/tick counts and successful completion in the tester log.

The original environment also used a secondary `US500.c` D1 series for regime
labels. Changing its source can alter labels even when the execution symbol's
ticks match. Offline/disconnected tester modes can make secondary history
unavailable. Resolve label-source parity before interpreting strategy differences.

## Batch operations

Start from the [credential-free template](../configs/examples/model4_orb.ini).
Configure the required history and EA inputs locally. Use a dedicated tester
terminal and `ShutdownTerminal=1`; a stale running terminal/tester can swallow
configuration requests or lock outputs. Close only the processes belonging to
your controlled test session. Historical automation that kills processes must
be reviewed before use alongside another terminal.

Tester CSVs are written to agent sandboxes. Archive each batch before starting
the next and record the EA/source/config/data hashes. Do not stitch output by
filename alone: verify strategy, date range, schema and ticket renumbering.

## Reconciliation checklist

Compare signal timestamps, completed-bar inputs, direction, eligibility, volume,
entry price side, initial stop, each ratchet, exit reason, fill price, financing
and account cash arithmetic. Signal parity does not establish fill/cash parity.
Review rejected orders and minimum-lot overrides rather than counting only fills.

Stops and prices must respect the symbol's **tick size**, not merely decimal
digits. Minimum lots can exceed intended cash risk; check actual risk after
volume-step rounding. Shared account exposure and guard latches need account-wide
testing across all EA instances.

Use demo/tester environments for a separately authorized deployment. This
repository's presence on GitHub is not an instruction to place live orders.
