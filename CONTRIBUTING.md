# Contributing

Read [research protocol](docs/RESEARCH_PROTOCOL.md) before proposing a strategy
change. A negative result is a complete contribution.

1. Open an issue describing the mechanism, data and intended scope.
2. Freeze a new plain-text SPEC and its verbatim GATE extraction before writing
   or running a new evaluation. Declare costs, clocks, exclusions and stopping rules.
3. Implement causal signals and adverse-first execution. Preserve failed findings.
4. Run the offline tests and the relevant synthetic boundary checks.
5. Submit code, a reproducible method, and a report that separates exploratory
   proxy findings from independent walk-forward and MT5 execution evidence.

Never overwrite a frozen SPEC/GATE/REPORT or the frozen ORB/SP2L EA sources.
Use a new versioned study directory for corrections. Generated controller EAs
must be changed through their generator. Do not commit raw market data, broker
credentials, account exports, compiled binaries, model weights, caches or logs.

The default tests use temporary or synthetic inputs and must not start terminals,
connect to trading accounts, submit orders or download market history.

No open-source license has been selected. Discuss licensing with the project
owner before contributing code that depends on a specific redistribution license.
