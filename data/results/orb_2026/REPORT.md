# ORB EA vs Python reference -- reconciliation report

US500-tick, 2026-01-02 -> 2026-04-22 -- generated 2026-08-23T01:58:14 by evaluate/orb_reconcile.py

## 1. MT5 agent log -- startup line and throughput

Offset line, quoted verbatim from `mt5_agent_utf8.log`:

```
CS	0	23:43:58.945	ORB_Strategy (US500-tick,M5)	2026.01.01 00:00:00   ORB NY offset now: -5 h -> server 2026.01.01 00:00 = NY 2025.12.31 19:00
```

* the line appears 2 times (identical): the agent log holds a discarded first ORB pass and the recorded pass
* recorded pass throughput, verbatim: `CS	0	23:54:30.732	Tester	US500-tick,M5: 7820903 ticks, 22369 bars generated. Test passed in 0:00:01.016 (including ticks preprocessing 0:00:00.203).`
* ticks processed: 7,820,903 (~7.8 million, consistent with the expected ~7.8M); bars generated: 22,369
* final balance 10,243.32 USD = 10,000.00 deposit + 243.32 net PnL, which equals the mt5_trades.csv profit sum (243.32) exactly
* the discarded first pass sent stop orders at unrounded 4-decimal prices (e.g. `buy stop ... at 6904.5413`); all 820 placements were rejected (802 `Invalid price` retcode 10015, 18 `Invalid stops` -- the US500-tick tick grid is 0.01); it produced no trades and was superseded by the recorded pass, which rounds levels to the 0.01 grid
* recorded pass CSV writes: 2026.04.30 23:59:59   Trade CSV written: trades.csv (27 trades); 2026.04.30 23:59:59   Day CSV written: orb_days.csv (102 days)

## 2. Test parameters

| Parameter | Value |
|---|---|
| Model | 0 (every tick, tick-precise backtest) |
| Timeframe | M5 |
| Symbol | US500-tick (tick grid 0.01) |
| Window | 2026-01-02 -> 2026-04-22 |
| Ticks | 7,820,903 (~7.8M) |
| Strategy | ORB: opening range 09:30-10:00 NY, stop orders 0.15% beyond the OR, TP 0.8%, SL = opposite OR side, gap filter 1.0%, entries 10:00-10:30 NY, force-close 16:00 NY |
| Sides | MT5 `ORB_Strategy.ex5` vs Python `evaluate/orb_reference.py` (M5 bid bars) |

## 3. Metrics -- Python vs MT5

| Metric | Python | MT5 |
|---|---|---|
| Day rows (NY calendar dates) | 102 | 102 |
| Trading days with session data (84 expected) | 84 | 84 |
| Days skipped by gap filter | 9 (10.7%) | 9 (10.7%) |
| Total trades | 27 | 27 |
| Win rate (profit > 0) | 63.0% (17/27) | 63.0% (17/27) |
| Total PnL ($) | 249.09 | 243.32 (= account PnL from log) |
| Daily Sharpe (per-day PnL over all 84 session days, zero on no-trade days, x sqrt(252)) | 2.81 | 2.75 |
| Max drawdown of cumulative PnL ($) | 84.11 | 84.28 |

PnL delta: Python - MT5 = +5.77 (2.3% of Python PnL) -- spread/slippage on fills.

## 4. Agreement metrics

(a) Day agreement, decision fields (skipped, skip_reason, orders_placed, entry_direction) identical: **102/102 (100.0%)**
    Day full-field agreement (all compared fields within tolerance): 18/102 (17.6%) -- every failing day fails only on price/level deltas or exit-reason labels (see section 5); decision fields agree on all 102 days
(b) Trade match rate: **27/27 (100.0%)** (matched / max(n_py=27, n_mt5=27))
(c) Direction agreement within matched: **27/27 (100.0%)** (matching requires same type)
(d) Price agreement within matched at tolerances: entry <= 1.5 pts: **27/27 (100.0%)**; SL <= 0.05 pts: 4/27 (14.8%); TP <= 0.05 pts: 2/27 (7.4%); exit <= 1.5 pts: **27/27 (100.0%)**; exit time <= 15 min: **27/27 (100.0%)**
(e) Exit-reason agreement within matched: raw exact **23/27 (85.2%)**; mapped (MT5 'other' reclassified when |close-tp| or |close-sl| <= 0.5): **27/27 (100.0%)**
(f) OVERALL reconciliation agreement (matched AND entry price AND exit price AND exit time AND mapped reason within tolerance): **27/27 (100.0%)**

## 5. Day-level diff

* day rows: Python 102, MT5 102; ny_date sets identical: **yes** (only-py: none; only-mt5: none; common: 102)
* skipped flag + skip_reason: identical on all 102 days (9 gap_filter days on both sides)
* or_bars / orders_placed / entry_direction: identical on all 102 days
* entry_time / exit_time deltas: max 5 min / 4 min (tolerance 10 min) -- no violations
* days with at least one flagged field: **84 of 102** (500 flagged field instances total); the 18 clean days are the inert weekend/holiday rows
* level-delta profile over the 84 session-data days (|Python - MT5|, pts):

| Field | mean | p95 | max | > 0.02 tol |
|---|---|---|---|---|
| prev_close | 0.252 | 0.575 | 0.916 | 79/84 |
| session_open | 0.206 | 0.503 | 0.560 | 82/84 |
| range_high | 0.209 | 0.517 | 0.599 | 77/84 |
| range_low | 0.206 | 0.516 | 0.559 | 82/84 |
| buy_level | 0.204 | 0.517 | 0.596 | 69/75 |
| sell_level | 0.196 | 0.494 | 0.555 | 73/75 |

Median |level delta| across all fields: 0.174 pts. These deltas are dominated by the different price sources (Python M5 bid bars vs MT5 native tick data) with a <= 0.005 pt contribution from MT5's 0.01 tick-grid rounding of levels; they stay far inside the 1.5 pt fill tolerance and do not change any trading decision.

Breakdown of the 500 flagged instances: session_open: 82, range_low: 82, prev_close: 79, range_high: 77, sell_level: 73, buy_level: 69, gap_pct: 34, exit_reason: 4

Every day-level discrepancy (py value vs mt5 value):

- 20260102: prev_close 6845.1420 vs 6845.1850 (delta 0.0430 > 0.02); session_open 6881.2510 vs 6881.0240 (delta 0.2270 > 0.02); range_high 6894.4540 vs 6894.2000 (delta 0.2540 > 0.02); range_low 6867.0360 vs 6866.8200 (delta 0.2160 > 0.02); buy_level 6904.7957 vs 6904.5400 (delta 0.2557 > 0.02); sell_level 6856.7354 vs 6856.5200 (delta 0.2154 > 0.02)
- 20260105: prev_close 6857.0330 vs 6856.8500 (delta 0.1830 > 0.02); session_open 6892.1990 vs 6891.9180 (delta 0.2810 > 0.02); range_high 6905.2450 vs 6904.9100 (delta 0.3350 > 0.02); range_low 6890.5480 vs 6890.2600 (delta 0.2880 > 0.02); buy_level 6915.6029 vs 6915.2700 (delta 0.3329 > 0.02); sell_level 6880.2122 vs 6879.9200 (delta 0.2922 > 0.02)
- 20260106: prev_close 6903.5390 vs 6903.3500 (delta 0.1890 > 0.02); session_open 6904.7330 vs 6904.6090 (delta 0.1240 > 0.02); range_high 6927.8420 vs 6927.6500 (delta 0.1920 > 0.02); range_low 6903.5420 vs 6903.4000 (delta 0.1420 > 0.02); buy_level 6938.2338 vs 6938.0400 (delta 0.1938 > 0.02); sell_level 6893.1867 vs 6893.0400 (delta 0.1467 > 0.02)
- 20260107: gap_pct 0.0448 vs 0.0517 (delta 0.0069 > 0.005); prev_close 6944.3390 vs 6943.9000 (delta 0.4390 > 0.02); session_open 6947.4510 vs 6947.4890 (delta 0.0380 > 0.02); range_low 6942.3390 vs 6942.3800 (delta 0.0410 > 0.02); sell_level 6931.9255 vs 6931.9700 (delta 0.0445 > 0.02)
- 20260108: session_open 6912.8510 vs 6913.0790 (delta 0.2280 > 0.02); range_high 6918.0510 vs 6918.2700 (delta 0.2190 > 0.02); range_low 6900.1480 vs 6900.3800 (delta 0.2320 > 0.02); buy_level 6928.4281 vs 6928.6500 (delta 0.2219 > 0.02); sell_level 6889.7978 vs 6890.0300 (delta 0.2322 > 0.02)
- 20260109: prev_close 6919.4990 vs 6919.6500 (delta 0.1510 > 0.02); session_open 6930.7480 vs 6930.5690 (delta 0.1790 > 0.02); range_high 6948.4480 vs 6948.2300 (delta 0.2180 > 0.02); range_low 6929.8390 vs 6929.6600 (delta 0.1790 > 0.02); buy_level 6958.8707 vs 6958.6500 (delta 0.2207 > 0.02); sell_level 6919.4442 vs 6919.2700 (delta 0.1742 > 0.02); exit_reason 'stop_loss' vs 'other'
- 20260112: prev_close 6965.3510 vs 6964.9000 (delta 0.4510 > 0.02); session_open 6930.3420 vs 6930.0420 (delta 0.3000 > 0.02); range_high 6963.9450 vs 6963.6700 (delta 0.2750 > 0.02); range_low 6929.9450 vs 6929.6500 (delta 0.2950 > 0.02); buy_level 6974.3909 vs 6974.1200 (delta 0.2709 > 0.02); sell_level 6919.5501 vs 6919.2600 (delta 0.2901 > 0.02)
- 20260113: gap_pct 0.0430 vs 0.0550 (delta 0.0120 > 0.005); prev_close 6977.0540 vs 6976.5800 (delta 0.4740 > 0.02); session_open 6980.0510 vs 6980.4180 (delta 0.3670 > 0.02); range_high 6985.6510 vs 6985.9900 (delta 0.3390 > 0.02); range_low 6967.0510 vs 6967.4000 (delta 0.3490 > 0.02); buy_level 6996.1295 vs 6996.4700 (delta 0.3405 > 0.02); sell_level 6956.6004 vs 6956.9500 (delta 0.3496 > 0.02)
- 20260114: prev_close 6963.6510 vs 6964.1400 (delta 0.4890 > 0.02); session_open 6933.6390 vs 6933.7880 (delta 0.1490 > 0.02); range_high 6941.3480 vs 6941.4700 (delta 0.1220 > 0.02); range_low 6911.2330 vs 6911.3800 (delta 0.1470 > 0.02); buy_level 6951.7600 vs 6951.8800 (delta 0.1200 > 0.02); sell_level 6900.8662 vs 6901.0100 (delta 0.1438 > 0.02)
- 20260115: prev_close 6927.1450 vs 6927.1800 (delta 0.0350 > 0.02); session_open 6970.6540 vs 6970.3690 (delta 0.2850 > 0.02); range_high 6971.2360 vs 6970.9400 (delta 0.2960 > 0.02); range_low 6956.4360 vs 6956.1600 (delta 0.2760 > 0.02); buy_level 6981.6929 vs 6981.4000 (delta 0.2929 > 0.02); sell_level 6946.0013 vs 6945.7300 (delta 0.2713 > 0.02)
- 20260116: prev_close 6943.5360 vs 6943.2600 (delta 0.2760 > 0.02); session_open 6960.7990 vs 6960.7460 (delta 0.0530 > 0.02); range_high 6966.3480 vs 6966.2900 (delta 0.0580 > 0.02); buy_level 6976.7975 vs 6976.7400 (delta 0.0575 > 0.02)
- 20260119: prev_close 6939.8510 vs 6940.0500 (delta 0.1990 > 0.02); session_open 6860.4450 vs 6860.9490 (delta 0.5040 > 0.02); range_high 6864.3480 vs 6864.8400 (delta 0.4920 > 0.02); range_low 6852.0450 vs 6852.5800 (delta 0.5350 > 0.02)
- 20260120: prev_close 6939.8510 vs 6940.0500 (delta 0.1990 > 0.02); session_open 6837.9330 vs 6838.4630 (delta 0.5300 > 0.02); range_high 6856.3330 vs 6856.8500 (delta 0.5170 > 0.02); range_low 6837.5450 vs 6838.0800 (delta 0.5350 > 0.02)
- 20260121: gap_pct 0.3391 vs 0.3276 (delta 0.0115 > 0.005); prev_close 6796.2540 vs 6797.1700 (delta 0.9160 > 0.02); session_open 6819.2990 vs 6819.4380 (delta 0.1390 > 0.02); range_high 6850.1480 vs 6850.2400 (delta 0.0920 > 0.02); range_low 6812.6330 vs 6812.7700 (delta 0.1370 > 0.02); buy_level 6860.4232 vs 6860.5200 (delta 0.0968 > 0.02); sell_level 6802.4141 vs 6802.5500 (delta 0.1359 > 0.02)
- 20260122: prev_close 6874.4510 vs 6874.6500 (delta 0.1990 > 0.02); session_open 6921.9360 vs 6922.0380 (delta 0.1020 > 0.02); range_high 6929.9360 vs 6930.0000 (delta 0.0640 > 0.02); range_low 6896.9330 vs 6897.0000 (delta 0.0670 > 0.02); buy_level 6940.3309 vs 6940.4000 (delta 0.0691 > 0.02); sell_level 6886.5876 vs 6886.6500 (delta 0.0624 > 0.02)
- 20260123: prev_close 6910.5450 vs 6910.6200 (delta 0.0750 > 0.02); session_open 6902.7450 vs 6903.0860 (delta 0.3410 > 0.02); range_high 6910.8390 vs 6911.0500 (delta 0.2110 > 0.02); range_low 6894.4390 vs 6894.7600 (delta 0.3210 > 0.02); buy_level 6921.2053 vs 6921.4200 (delta 0.2147 > 0.02); sell_level 6884.0973 vs 6884.4200 (delta 0.3227 > 0.02)
- 20260126: prev_close 6914.6450 vs 6914.5500 (delta 0.0950 > 0.02); session_open 6927.1420 vs 6927.0850 (delta 0.0570 > 0.02); range_high 6949.1540 vs 6949.0900 (delta 0.0640 > 0.02); range_low 6921.8330 vs 6921.7800 (delta 0.0530 > 0.02); buy_level 6959.5777 vs 6959.5100 (delta 0.0677 > 0.02); sell_level 6911.4503 vs 6911.4000 (delta 0.0503 > 0.02)
- 20260127: gap_pct 0.1955 vs 0.1897 (delta 0.0058 > 0.005); prev_close 6950.6420 vs 6950.5500 (delta 0.0920 > 0.02); session_open 6964.2330 vs 6963.7350 (delta 0.4980 > 0.02); range_high 6977.7990 vs 6977.2000 (delta 0.5990 > 0.02); range_low 6958.3330 vs 6957.8200 (delta 0.5130 > 0.02); buy_level 6988.2657 vs 6987.6700 (delta 0.5957 > 0.02); sell_level 6947.8955 vs 6947.3800 (delta 0.5155 > 0.02)
- 20260128: gap_pct 0.2177 vs 0.2229 (delta 0.0052 > 0.005); prev_close 6977.8390 vs 6977.2600 (delta 0.5790 > 0.02); session_open 6993.0330 vs 6992.8120 (delta 0.2210 > 0.02); range_high 7001.6990 vs 7001.4600 (delta 0.2390 > 0.02); range_low 6992.0420 vs 6991.8200 (delta 0.2220 > 0.02); buy_level 7012.2015 vs 7011.9600 (delta 0.2415 > 0.02); sell_level 6981.5539 vs 6981.3300 (delta 0.2239 > 0.02)
- 20260129: prev_close 6977.7330 vs 6977.2800 (delta 0.4530 > 0.02); session_open 6985.7390 vs 6985.6260 (delta 0.1130 > 0.02); range_high 6992.8990 vs 6992.7800 (delta 0.1190 > 0.02); range_low 6937.5330 vs 6937.4800 (delta 0.0530 > 0.02); buy_level 7003.3883 vs 7003.2700 (delta 0.1183 > 0.02); sell_level 6927.1267 vs 6927.0700 (delta 0.0567 > 0.02)
- 20260130: prev_close 6966.0540 vs 6965.9200 (delta 0.1340 > 0.02); session_open 6939.9450 vs 6939.5310 (delta 0.4140 > 0.02); range_high 6959.0540 vs 6958.5900 (delta 0.4640 > 0.02); range_low 6935.0330 vs 6934.5900 (delta 0.4430 > 0.02); buy_level 6969.4926 vs 6969.0300 (delta 0.4626 > 0.02); sell_level 6924.6305 vs 6924.1900 (delta 0.4405 > 0.02)
- 20260202: prev_close 6938.6480 vs 6937.8800 (delta 0.7680 > 0.02); session_open 6918.4510 vs 6917.9190 (delta 0.5320 > 0.02); range_high 6960.7450 vs 6960.1900 (delta 0.5550 > 0.02); range_low 6916.5450 vs 6916.0100 (delta 0.5350 > 0.02); buy_level 6971.1861 vs 6970.6300 (delta 0.5561 > 0.02); sell_level 6906.1702 vs 6905.6400 (delta 0.5302 > 0.02)
- 20260203: gap_pct 0.1146 vs 0.1208 (delta 0.0062 > 0.005); prev_close 6976.3510 vs 6975.8000 (delta 0.5510 > 0.02); session_open 6984.3450 vs 6984.2290 (delta 0.1160 > 0.02); range_high 6991.7480 vs 6991.6000 (delta 0.1480 > 0.02); range_low 6955.3360 vs 6955.2200 (delta 0.1160 > 0.02); buy_level 7002.2356 vs 7002.0900 (delta 0.1456 > 0.02); sell_level 6944.9030 vs 6944.7900 (delta 0.1130 > 0.02)
- 20260204: prev_close 6917.1360 vs 6917.0900 (delta 0.0460 > 0.02); range_low 6900.4360 vs 6900.5000 (delta 0.0640 > 0.02); sell_level 6890.0853 vs 6890.1500 (delta 0.0647 > 0.02)
- 20260205: prev_close 6883.4330 vs 6883.2500 (delta 0.1830 > 0.02); session_open 6830.0450 vs 6829.9130 (delta 0.1320 > 0.02); range_high 6858.5510 vs 6858.4300 (delta 0.1210 > 0.02); range_low 6809.3480 vs 6809.2400 (delta 0.1080 > 0.02); buy_level 6868.8388 vs 6868.7200 (delta 0.1188 > 0.02); sell_level 6799.1340 vs 6799.0300 (delta 0.1040 > 0.02)
- 20260206: prev_close 6798.2420 vs 6798.1900 (delta 0.0520 > 0.02); session_open 6835.8990 vs 6836.0090 (delta 0.1100 > 0.02); range_low 6830.1510 vs 6830.2600 (delta 0.1090 > 0.02); sell_level 6819.9058 vs 6820.0100 (delta 0.1042 > 0.02)
- 20260209: gap_pct 0.2738 vs 0.2643 (delta 0.0095 > 0.005); prev_close 6934.4330 vs 6934.2500 (delta 0.1830 > 0.02); session_open 6915.4450 vs 6915.9210 (delta 0.4760 > 0.02); range_high 6937.1390 vs 6937.6400 (delta 0.5010 > 0.02); range_low 6904.2510 vs 6904.7400 (delta 0.4890 > 0.02); buy_level 6947.5447 vs 6948.0500 (delta 0.5053 > 0.02); sell_level 6893.8946 vs 6894.3800 (delta 0.4854 > 0.02)
- 20260210: prev_close 6963.5480 vs 6963.9700 (delta 0.4220 > 0.02); session_open 6969.3450 vs 6969.4250 (delta 0.0800 > 0.02); range_high 6981.6540 vs 6981.7300 (delta 0.0760 > 0.02); range_low 6955.1330 vs 6955.2200 (delta 0.0870 > 0.02); buy_level 6992.1265 vs 6992.2000 (delta 0.0735 > 0.02); sell_level 6944.7003 vs 6944.7900 (delta 0.0897 > 0.02)
- 20260211: gap_pct 0.6110 vs 0.6053 (delta 0.0057 > 0.005); prev_close 6942.4330 vs 6942.6800 (delta 0.2470 > 0.02); session_open 6984.8510 vs 6984.7050 (delta 0.1460 > 0.02); range_high 6991.9480 vs 6991.7800 (delta 0.1680 > 0.02); range_low 6958.2360 vs 6958.0900 (delta 0.1460 > 0.02); buy_level 7002.4359 vs 7002.2700 (delta 0.1659 > 0.02); sell_level 6947.7986 vs 6947.6500 (delta 0.1486 > 0.02)
- 20260212: prev_close 6942.7540 vs 6942.7800 (delta 0.0260 > 0.02); session_open 6961.3990 vs 6961.4610 (delta 0.0620 > 0.02); range_high 6973.6990 vs 6973.7400 (delta 0.0410 > 0.02); range_low 6943.5360 vs 6943.6500 (delta 0.1140 > 0.02); buy_level 6984.1595 vs 6984.2000 (delta 0.0405 > 0.02); sell_level 6933.1207 vs 6933.2300 (delta 0.1093 > 0.02)
- 20260213: gap_pct 0.0779 vs 0.0728 (delta 0.0051 > 0.005); prev_close 6829.5360 vs 6829.7100 (delta 0.1740 > 0.02); session_open 6834.8540 vs 6834.6840 (delta 0.1700 > 0.02); range_high 6846.8480 vs 6846.6600 (delta 0.1880 > 0.02); range_low 6793.8420 vs 6793.6800 (delta 0.1620 > 0.02); buy_level 6857.1183 vs 6856.9300 (delta 0.1883 > 0.02); sell_level 6783.6512 vs 6783.4900 (delta 0.1612 > 0.02)
- 20260216: gap_pct 0.1405 vs 0.1485 (delta 0.0080 > 0.005); prev_close 6834.6390 vs 6834.1100 (delta 0.5290 > 0.02); session_open 6844.2390 vs 6844.2590 (delta 0.0200 > 0.02); range_low 6831.1420 vs 6831.2100 (delta 0.0680 > 0.02); sell_level 6820.8953 vs 6820.9600 (delta 0.0647 > 0.02)
- 20260217: gap_pct 0.2193 vs 0.2123 (delta 0.0070 > 0.005); prev_close 6834.6390 vs 6834.1100 (delta 0.5290 > 0.02); session_open 6819.6480 vs 6819.6020 (delta 0.0460 > 0.02); range_high 6845.7330 vs 6845.6900 (delta 0.0430 > 0.02); range_low 6782.8420 vs 6782.8800 (delta 0.0380 > 0.02); buy_level 6856.0016 vs 6855.9600 (delta 0.0416 > 0.02); sell_level 6772.6677 vs 6772.7100 (delta 0.0423 > 0.02)
- 20260218: gap_pct 0.1594 vs 0.1536 (delta 0.0058 > 0.005); prev_close 6845.5390 vs 6845.4600 (delta 0.0790 > 0.02); session_open 6856.4540 vs 6855.9760 (delta 0.4780 > 0.02); range_high 6881.6990 vs 6881.1700 (delta 0.5290 > 0.02); range_low 6845.5450 vs 6845.0700 (delta 0.4750 > 0.02); buy_level 6892.0215 vs 6891.4900 (delta 0.5315 > 0.02); sell_level 6835.2767 vs 6834.8000 (delta 0.4767 > 0.02)
- 20260219: gap_pct 0.3403 vs 0.3352 (delta 0.0051 > 0.005); prev_close 6877.4390 vs 6877.0400 (delta 0.3990 > 0.02); session_open 6854.0330 vs 6853.9890 (delta 0.0440 > 0.02); range_high 6874.6420 vs 6874.5700 (delta 0.0720 > 0.02); range_low 6841.6360 vs 6841.6000 (delta 0.0360 > 0.02); buy_level 6884.9540 vs 6884.8800 (delta 0.0740 > 0.02); sell_level 6831.3735 vs 6831.3400 (delta 0.0335 > 0.02)
- 20260220: gap_pct 0.3556 vs 0.3615 (delta 0.0059 > 0.005); prev_close 6862.6390 vs 6862.6700 (delta 0.0310 > 0.02); session_open 6838.2360 vs 6837.8610 (delta 0.3750 > 0.02); range_high 6870.1480 vs 6869.7400 (delta 0.4080 > 0.02); range_low 6833.8390 vs 6833.4700 (delta 0.3690 > 0.02); buy_level 6880.4532 vs 6880.0400 (delta 0.4132 > 0.02); sell_level 6823.5882 vs 6823.2200 (delta 0.3682 > 0.02)
- 20260223: prev_close 6911.6450 vs 6911.2200 (delta 0.4250 > 0.02); session_open 6894.7450 vs 6894.6080 (delta 0.1370 > 0.02); range_high 6916.6480 vs 6916.4900 (delta 0.1580 > 0.02); range_low 6879.2450 vs 6879.1600 (delta 0.0850 > 0.02); buy_level 6927.0230 vs 6926.8600 (delta 0.1630 > 0.02); sell_level 6868.9261 vs 6868.8400 (delta 0.0861 > 0.02)
- 20260224: session_open 6834.6360 vs 6834.8890 (delta 0.2530 > 0.02); range_high 6870.7540 vs 6871.0000 (delta 0.2460 > 0.02); range_low 6815.9450 vs 6816.2200 (delta 0.2750 > 0.02); buy_level 6881.0601 vs 6881.3100 (delta 0.2499 > 0.02); sell_level 6805.7211 vs 6806.0000 (delta 0.2789 > 0.02)
- 20260225: gap_pct 0.4049 vs 0.3974 (delta 0.0075 > 0.005); prev_close 6889.7480 vs 6889.9800 (delta 0.2320 > 0.02); session_open 6917.6480 vs 6917.3620 (delta 0.2860 > 0.02); range_high 6936.5990 vs 6936.2300 (delta 0.3690 > 0.02); range_low 6917.1360 vs 6916.8500 (delta 0.2860 > 0.02); buy_level 6947.0039 vs 6946.6300 (delta 0.3739 > 0.02); sell_level 6906.7603 vs 6906.4700 (delta 0.2903 > 0.02)
- 20260226: gap_pct 0.0433 vs 0.0495 (delta 0.0062 > 0.005); prev_close 6946.2390 vs 6945.7600 (delta 0.4790 > 0.02); session_open 6949.2450 vs 6949.2010 (delta 0.0440 > 0.02); range_high 6949.2450 vs 6949.2010 (delta 0.0440 > 0.02); range_low 6919.9480 vs 6919.9200 (delta 0.0280 > 0.02); buy_level 6959.6689 vs 6959.6200 (delta 0.0489 > 0.02); sell_level 6909.5681 vs 6909.5400 (delta 0.0281 > 0.02)
- 20260227: prev_close 6909.4540 vs 6909.2800 (delta 0.1740 > 0.02); session_open 6845.3450 vs 6845.2240 (delta 0.1210 > 0.02); range_high 6856.9480 vs 6856.8200 (delta 0.1280 > 0.02); range_low 6831.1330 vs 6831.0200 (delta 0.1130 > 0.02); buy_level 6867.2334 vs 6867.1100 (delta 0.1234 > 0.02); sell_level 6820.8863 vs 6820.7700 (delta 0.1163 > 0.02)
- 20260302: prev_close 6878.0450 vs 6877.8900 (delta 0.1550 > 0.02); session_open 6801.0990 vs 6801.2410 (delta 0.1420 > 0.02); range_high 6848.1330 vs 6848.2100 (delta 0.0770 > 0.02); range_low 6794.1420 vs 6794.2900 (delta 0.1480 > 0.02)
- 20260303: prev_close 6878.3480 vs 6878.3000 (delta 0.0480 > 0.02); session_open 6763.8480 vs 6763.9260 (delta 0.0780 > 0.02); range_high 6785.0510 vs 6785.1200 (delta 0.0690 > 0.02); range_low 6733.1390 vs 6733.2800 (delta 0.1410 > 0.02)
- 20260304: session_open 6829.4480 vs 6829.1460 (delta 0.3020 > 0.02); range_high 6849.1390 vs 6848.8300 (delta 0.3090 > 0.02); range_low 6810.5510 vs 6810.2200 (delta 0.3310 > 0.02); buy_level 6859.4127 vs 6859.1000 (delta 0.3127 > 0.02); sell_level 6800.3352 vs 6800.0000 (delta 0.3352 > 0.02)
- 20260305: prev_close 6868.1330 vs 6867.6600 (delta 0.4730 > 0.02); session_open 6834.8450 vs 6834.5190 (delta 0.3260 > 0.02); range_high 6864.9540 vs 6864.5700 (delta 0.3840 > 0.02); range_low 6832.1510 vs 6831.8200 (delta 0.3310 > 0.02); buy_level 6875.2514 vs 6874.8700 (delta 0.3814 > 0.02); sell_level 6821.9028 vs 6821.5700 (delta 0.3328 > 0.02)
- 20260306: prev_close 6827.8330 vs 6827.4900 (delta 0.3430 > 0.02); session_open 6746.8540 vs 6746.6660 (delta 0.1880 > 0.02); range_high 6752.5390 vs 6752.3400 (delta 0.1990 > 0.02); range_low 6710.1330 vs 6709.9900 (delta 0.1430 > 0.02)
- 20260309: gap_pct 0.9496 vs 0.9404 (delta 0.0092 > 0.005); prev_close 6738.8360 vs 6738.5100 (delta 0.3260 > 0.02); session_open 6674.8450 vs 6675.1410 (delta 0.2960 > 0.02); range_high 6687.7480 vs 6688.0400 (delta 0.2920 > 0.02); range_low 6638.7330 vs 6639.0400 (delta 0.3070 > 0.02); buy_level 6697.7796 vs 6698.0700 (delta 0.2904 > 0.02); sell_level 6628.7749 vs 6629.0800 (delta 0.3051 > 0.02)
- 20260310: gap_pct 0.0811 vs 0.0882 (delta 0.0071 > 0.005); prev_close 6794.6480 vs 6795.0200 (delta 0.3720 > 0.02); session_open 6789.1360 vs 6789.0240 (delta 0.1120 > 0.02); range_high 6799.2510 vs 6799.1100 (delta 0.1410 > 0.02); range_low 6766.3990 vs 6766.3200 (delta 0.0790 > 0.02); buy_level 6809.4499 vs 6809.3100 (delta 0.1399 > 0.02); sell_level 6756.2494 vs 6756.1700 (delta 0.0794 > 0.02)
- 20260311: gap_pct 0.1027 vs 0.1079 (delta 0.0052 > 0.005); prev_close 6781.0330 vs 6780.7700 (delta 0.2630 > 0.02); session_open 6787.9990 vs 6788.0880 (delta 0.0890 > 0.02); range_high 6796.3540 vs 6796.4300 (delta 0.0760 > 0.02); range_low 6772.5360 vs 6772.6500 (delta 0.1140 > 0.02); buy_level 6806.5485 vs 6806.6200 (delta 0.0715 > 0.02); sell_level 6762.3772 vs 6762.4900 (delta 0.1128 > 0.02)
- 20260312: prev_close 6773.2420 vs 6773.1400 (delta 0.1020 > 0.02); session_open 6721.6510 vs 6721.5030 (delta 0.1480 > 0.02); range_high 6726.7990 vs 6726.6600 (delta 0.1390 > 0.02); range_low 6692.3330 vs 6692.2400 (delta 0.0930 > 0.02); buy_level 6736.8892 vs 6736.7500 (delta 0.1392 > 0.02); sell_level 6682.2945 vs 6682.2000 (delta 0.0945 > 0.02)
- 20260313: prev_close 6671.2990 vs 6671.3200 (delta 0.0210 > 0.02); session_open 6703.3420 vs 6703.3960 (delta 0.0540 > 0.02); range_high 6733.2390 vs 6733.2800 (delta 0.0410 > 0.02); range_low 6700.1390 vs 6700.2000 (delta 0.0610 > 0.02); buy_level 6743.3389 vs 6743.3800 (delta 0.0411 > 0.02); sell_level 6690.0888 vs 6690.1500 (delta 0.0612 > 0.02); exit_reason 'take_profit' vs 'other'
- 20260316: prev_close 6632.3480 vs 6632.4300 (delta 0.0820 > 0.02); session_open 6693.3450 vs 6693.1510 (delta 0.1940 > 0.02); range_high 6718.3540 vs 6718.1200 (delta 0.2340 > 0.02); range_low 6686.6510 vs 6686.4600 (delta 0.1910 > 0.02); buy_level 6728.4315 vs 6728.2000 (delta 0.2315 > 0.02); sell_level 6676.6210 vs 6676.4300 (delta 0.1910 > 0.02)
- 20260317: prev_close 6699.1330 vs 6698.9800 (delta 0.1530 > 0.02); session_open 6733.5540 vs 6733.6850 (delta 0.1310 > 0.02); range_high 6754.1360 vs 6754.2900 (delta 0.1540 > 0.02); range_low 6730.3480 vs 6730.4800 (delta 0.1320 > 0.02); buy_level 6764.2672 vs 6764.4200 (delta 0.1528 > 0.02); sell_level 6720.2525 vs 6720.3800 (delta 0.1275 > 0.02)
- 20260318: gap_pct 0.3783 vs 0.3856 (delta 0.0073 > 0.005); prev_close 6717.5450 vs 6717.8400 (delta 0.2950 > 0.02); session_open 6692.1330 vs 6691.9390 (delta 0.1940 > 0.02); range_high 6705.6990 vs 6705.4600 (delta 0.2390 > 0.02); range_low 6687.7390 vs 6687.5400 (delta 0.1990 > 0.02); buy_level 6715.7575 vs 6715.5200 (delta 0.2375 > 0.02); sell_level 6677.7074 vs 6677.5100 (delta 0.1974 > 0.02)
- 20260319: prev_close 6623.9540 vs 6623.7600 (delta 0.1940 > 0.02); session_open 6575.2510 vs 6575.0410 (delta 0.2100 > 0.02); range_high 6596.8480 vs 6596.6700 (delta 0.1780 > 0.02); range_low 6558.8330 vs 6558.6300 (delta 0.2030 > 0.02); buy_level 6606.7433 vs 6606.5700 (delta 0.1733 > 0.02); sell_level 6548.9948 vs 6548.7900 (delta 0.2048 > 0.02)
- 20260320: gap_pct 0.2208 vs 0.2144 (delta 0.0064 > 0.005); prev_close 6605.8330 vs 6605.4800 (delta 0.3530 > 0.02); session_open 6591.2450 vs 6591.3190 (delta 0.0740 > 0.02); range_high 6591.3450 vs 6591.4900 (delta 0.1450 > 0.02); range_low 6555.5390 vs 6555.6700 (delta 0.1310 > 0.02); buy_level 6601.2320 vs 6601.3800 (delta 0.1480 > 0.02); sell_level 6545.7057 vs 6545.8400 (delta 0.1343 > 0.02)
- 20260323: prev_close 6507.6390 vs 6507.4700 (delta 0.1690 > 0.02); session_open 6607.3360 vs 6607.3640 (delta 0.0280 > 0.02); range_low 6588.8390 vs 6588.8800 (delta 0.0410 > 0.02)
- 20260324: range_high 6554.7510 vs 6554.7200 (delta 0.0310 > 0.02); range_low 6523.6450 vs 6523.6800 (delta 0.0350 > 0.02); buy_level 6564.5831 vs 6564.5500 (delta 0.0331 > 0.02); sell_level 6513.8595 vs 6513.8900 (delta 0.0305 > 0.02)
- 20260325: gap_pct 0.8370 vs 0.8306 (delta 0.0064 > 0.005); prev_close 6558.2480 vs 6558.2700 (delta 0.0220 > 0.02); session_open 6613.1390 vs 6612.7440 (delta 0.3950 > 0.02); range_high 6635.8480 vs 6635.4400 (delta 0.4080 > 0.02); range_low 6588.8990 vs 6588.5300 (delta 0.3690 > 0.02); buy_level 6645.8018 vs 6645.3900 (delta 0.4118 > 0.02); sell_level 6579.0157 vs 6578.6500 (delta 0.3657 > 0.02)
- 20260326: prev_close 6593.0330 vs 6592.6500 (delta 0.3830 > 0.02); session_open 6544.2390 vs 6543.9340 (delta 0.3050 > 0.02); range_high 6571.5510 vs 6571.2200 (delta 0.3310 > 0.02); range_low 6531.2330 vs 6530.9640 (delta 0.2690 > 0.02); buy_level 6581.4083 vs 6581.0800 (delta 0.3283 > 0.02); sell_level 6521.4362 vs 6521.1700 (delta 0.2662 > 0.02)
- 20260327: prev_close 6477.1540 vs 6476.9700 (delta 0.1840 > 0.02); session_open 6449.2360 vs 6449.2940 (delta 0.0580 > 0.02); range_high 6451.0480 vs 6451.1000 (delta 0.0520 > 0.02); range_low 6409.5390 vs 6409.6100 (delta 0.0710 > 0.02); buy_level 6460.7246 vs 6460.7800 (delta 0.0554 > 0.02); sell_level 6399.9247 vs 6400.0000 (delta 0.0753 > 0.02)
- 20260330: prev_close 6364.8390 vs 6364.9900 (delta 0.1510 > 0.02); session_open 6423.1540 vs 6423.0890 (delta 0.0650 > 0.02); range_high 6427.2360 vs 6427.1600 (delta 0.0760 > 0.02); range_low 6369.7450 vs 6369.6900 (delta 0.0550 > 0.02); buy_level 6436.8769 vs 6436.8000 (delta 0.0769 > 0.02); sell_level 6360.1904 vs 6360.1400 (delta 0.0504 > 0.02)
- 20260331: gap_pct 1.0953 vs 1.1003 (delta 0.0050 > 0.005); prev_close 6344.2480 vs 6344.3300 (delta 0.0820 > 0.02); session_open 6413.7360 vs 6414.1360 (delta 0.4000 > 0.02); range_high 6450.9990 vs 6451.3600 (delta 0.3610 > 0.02); range_low 6406.9360 vs 6407.3600 (delta 0.4240 > 0.02)
- 20260401: gap_pct 0.5334 vs 0.5278 (delta 0.0056 > 0.005); prev_close 6527.7330 vs 6528.0600 (delta 0.3270 > 0.02); session_open 6562.5510 vs 6562.5140 (delta 0.0370 > 0.02); range_high 6585.2540 vs 6585.2000 (delta 0.0540 > 0.02); range_low 6554.2540 vs 6554.2300 (delta 0.0240 > 0.02); buy_level 6595.1319 vs 6595.0800 (delta 0.0519 > 0.02); sell_level 6544.4226 vs 6544.4000 (delta 0.0226 > 0.02)
- 20260402: gap_pct 1.3442 vs 1.3384 (delta 0.0058 > 0.005); prev_close 6575.2330 vs 6575.1300 (delta 0.1030 > 0.02); session_open 6486.8510 vs 6487.1290 (delta 0.2780 > 0.02); range_high 6522.7360 vs 6522.9500 (delta 0.2140 > 0.02); range_low 6474.3420 vs 6474.6100 (delta 0.2680 > 0.02)
- 20260406: gap_pct 0.0245 vs 0.0170 (delta 0.0075 > 0.005); prev_close 6582.6420 vs 6582.8600 (delta 0.2180 > 0.02); session_open 6584.2540 vs 6583.9820 (delta 0.2720 > 0.02); range_high 6608.7390 vs 6608.4600 (delta 0.2790 > 0.02); range_low 6580.0360 vs 6579.8000 (delta 0.2360 > 0.02); buy_level 6618.6521 vs 6618.3700 (delta 0.2821 > 0.02); sell_level 6570.1659 vs 6569.9300 (delta 0.2359 > 0.02); exit_reason 'stop_loss' vs 'other'
- 20260407: gap_pct 0.3443 vs 0.3375 (delta 0.0068 > 0.005); prev_close 6612.1990 vs 6612.0100 (delta 0.1890 > 0.02); session_open 6589.4330 vs 6589.6920 (delta 0.2590 > 0.02); range_high 6598.8420 vs 6599.0900 (delta 0.2480 > 0.02); range_low 6563.1990 vs 6563.5200 (delta 0.3210 > 0.02); buy_level 6608.7403 vs 6608.9900 (delta 0.2497 > 0.02); sell_level 6553.3542 vs 6553.6700 (delta 0.3158 > 0.02); exit_reason 'stop_loss' vs 'other'
- 20260408: gap_pct 2.5535 vs 2.5404 (delta 0.0131 > 0.005); prev_close 6618.3450 vs 6618.8300 (delta 0.4850 > 0.02); session_open 6787.3450 vs 6786.9720 (delta 0.3730 > 0.02); range_high 6794.9510 vs 6794.5700 (delta 0.3810 > 0.02); range_low 6763.7330 vs 6763.3900 (delta 0.3430 > 0.02)
- 20260409: gap_pct 0.1708 vs 0.1530 (delta 0.0178 > 0.005); prev_close 6784.6390 vs 6783.9600 (delta 0.6790 > 0.02); session_open 6773.0540 vs 6773.5810 (delta 0.5270 > 0.02); range_high 6781.4510 vs 6781.9700 (delta 0.5190 > 0.02); range_low 6768.4330 vs 6768.9500 (delta 0.5170 > 0.02); buy_level 6791.6232 vs 6792.1400 (delta 0.5168 > 0.02); sell_level 6758.2804 vs 6758.8000 (delta 0.5196 > 0.02)
- 20260410: gap_pct 0.1785 vs 0.1679 (delta 0.0106 > 0.005); prev_close 6824.4540 vs 6825.0700 (delta 0.6160 > 0.02); session_open 6836.6360 vs 6836.5290 (delta 0.1070 > 0.02); range_high 6842.7360 vs 6842.6000 (delta 0.1360 > 0.02); range_low 6825.2420 vs 6825.1600 (delta 0.0820 > 0.02); buy_level 6853.0001 vs 6852.8600 (delta 0.1401 > 0.02); sell_level 6815.0041 vs 6814.9200 (delta 0.0841 > 0.02)
- 20260413: prev_close 6815.4990 vs 6815.2000 (delta 0.2990 > 0.02); session_open 6796.5450 vs 6796.3720 (delta 0.1730 > 0.02); range_high 6820.6990 vs 6820.5300 (delta 0.1690 > 0.02); range_low 6788.8330 vs 6788.6700 (delta 0.1630 > 0.02); buy_level 6830.9300 vs 6830.7600 (delta 0.1700 > 0.02); sell_level 6778.6498 vs 6778.4900 (delta 0.1598 > 0.02)
- 20260414: prev_close 6885.1330 vs 6885.1600 (delta 0.0270 > 0.02); session_open 6899.7360 vs 6899.8410 (delta 0.1050 > 0.02); range_high 6923.0990 vs 6923.1700 (delta 0.0710 > 0.02); range_low 6899.7360 vs 6899.8410 (delta 0.1050 > 0.02); buy_level 6933.4836 vs 6933.5500 (delta 0.0664 > 0.02); sell_level 6889.3864 vs 6889.4900 (delta 0.1036 > 0.02)
- 20260415: session_open 6976.0360 vs 6975.8790 (delta 0.1570 > 0.02); range_high 6986.3990 vs 6986.2300 (delta 0.1690 > 0.02); range_low 6965.7510 vs 6965.6100 (delta 0.1410 > 0.02); buy_level 6996.8786 vs 6996.7100 (delta 0.1686 > 0.02); sell_level 6955.3024 vs 6955.1600 (delta 0.1424 > 0.02)
- 20260416: prev_close 7022.0420 vs 7021.6900 (delta 0.3520 > 0.02); session_open 7034.0510 vs 7033.9290 (delta 0.1220 > 0.02); range_high 7036.4420 vs 7036.3100 (delta 0.1320 > 0.02); range_low 7008.8360 vs 7008.7200 (delta 0.1160 > 0.02); buy_level 7046.9967 vs 7046.8600 (delta 0.1367 > 0.02); sell_level 6998.3227 vs 6998.2100 (delta 0.1127 > 0.02)
- 20260417: prev_close 7038.4420 vs 7038.1000 (delta 0.3420 > 0.02); session_open 7084.7990 vs 7084.4360 (delta 0.3630 > 0.02); range_high 7107.2510 vs 7106.8700 (delta 0.3810 > 0.02); range_low 7081.2540 vs 7080.9100 (delta 0.3440 > 0.02); buy_level 7117.9119 vs 7117.5300 (delta 0.3819 > 0.02); sell_level 7070.6321 vs 7070.2900 (delta 0.3421 > 0.02)
- 20260420: gap_pct 0.1683 vs 0.1605 (delta 0.0078 > 0.005); prev_close 7123.8420 vs 7123.4500 (delta 0.3920 > 0.02); session_open 7111.8510 vs 7112.0150 (delta 0.1640 > 0.02); range_high 7122.7540 vs 7122.9500 (delta 0.1960 > 0.02); range_low 7105.7330 vs 7105.9300 (delta 0.1970 > 0.02); buy_level 7133.4381 vs 7133.6300 (delta 0.1919 > 0.02); sell_level 7095.0744 vs 7095.2700 (delta 0.1956 > 0.02)
- 20260421: prev_close 7112.0360 vs 7112.1200 (delta 0.0840 > 0.02); session_open 7125.2540 vs 7125.0550 (delta 0.1990 > 0.02); range_high 7132.2420 vs 7132.0100 (delta 0.2320 > 0.02); range_low 7111.0480 vs 7110.8500 (delta 0.1980 > 0.02); buy_level 7142.9404 vs 7142.7100 (delta 0.2304 > 0.02); sell_level 7100.3814 vs 7100.1800 (delta 0.2014 > 0.02)
- 20260422: prev_close 7064.2480 vs 7064.1400 (delta 0.1080 > 0.02); session_open 7114.9450 vs 7114.8980 (delta 0.0470 > 0.02)
- 20260423: prev_close 7135.4720 vs 7135.4000 (delta 0.0720 > 0.02); session_open 7117.9980 vs 7118.0850 (delta 0.0870 > 0.02); range_high 7136.9830 vs 7137.0300 (delta 0.0470 > 0.02); range_low 7116.1160 vs 7116.2000 (delta 0.0840 > 0.02); buy_level 7147.6885 vs 7147.7400 (delta 0.0515 > 0.02); sell_level 7105.4418 vs 7105.5300 (delta 0.0882 > 0.02)
- 20260424: prev_close 7107.4950 vs 7107.6500 (delta 0.1550 > 0.02); session_open 7130.2450 vs 7130.5130 (delta 0.2680 > 0.02); range_high 7134.1160 vs 7134.3800 (delta 0.2640 > 0.02); range_low 7112.8780 vs 7113.1800 (delta 0.3020 > 0.02); buy_level 7144.8172 vs 7145.0800 (delta 0.2628 > 0.02); sell_level 7102.2087 vs 7102.5100 (delta 0.3013 > 0.02)
- 20260427: gap_pct 0.1063 vs 0.1132 (delta 0.0069 > 0.005); prev_close 7162.6940 vs 7162.9400 (delta 0.2460 > 0.02); session_open 7155.0830 vs 7154.8330 (delta 0.2500 > 0.02); range_high 7166.6510 vs 7166.3600 (delta 0.2910 > 0.02); range_low 7149.1990 vs 7148.9300 (delta 0.2690 > 0.02); buy_level 7177.4010 vs 7177.1100 (delta 0.2910 > 0.02); sell_level 7138.4752 vs 7138.2100 (delta 0.2652 > 0.02)
- 20260428: gap_pct 0.4774 vs 0.4636 (delta 0.0138 > 0.005); prev_close 7175.1510 vs 7174.7200 (delta 0.4310 > 0.02); session_open 7140.8990 vs 7141.4590 (delta 0.5600 > 0.02); range_high 7152.3330 vs 7152.8500 (delta 0.5170 > 0.02); range_low 7130.5510 vs 7131.1100 (delta 0.5590 > 0.02); buy_level 7163.0615 vs 7163.5800 (delta 0.5185 > 0.02); sell_level 7119.8552 vs 7120.4100 (delta 0.5548 > 0.02)
- 20260429: gap_pct 0.1105 vs 0.1161 (delta 0.0056 > 0.005); prev_close 7141.0390 vs 7141.4700 (delta 0.4310 > 0.02); session_open 7133.1510 vs 7133.1790 (delta 0.0280 > 0.02); range_low 7119.7390 vs 7119.8400 (delta 0.1010 > 0.02); sell_level 7109.0594 vs 7109.1600 (delta 0.1006 > 0.02)
- 20260430: prev_close 7138.3360 vs 7138.5900 (delta 0.2540 > 0.02); session_open 7170.9480 vs 7171.2520 (delta 0.3040 > 0.02); range_high 7171.0990 vs 7171.4000 (delta 0.3010 > 0.02); range_low 7130.8330 vs 7131.1700 (delta 0.3370 > 0.02); buy_level 7181.8556 vs 7182.1600 (delta 0.3044 > 0.02); sell_level 7120.1368 vs 7120.4700 (delta 0.3332 > 0.02)

## 6. Trade-level diff

* trades: Python 27, MT5 27; matched one-to-one **27** (same type, same UTC open date, |open_time delta| <= 10 min, greedy nearest)
* unmatched Python: none
* unmatched MT5: none
* entry window 10:00-10:30 NY (15:00-15:30 UTC before 2026-03-08, 14:00-14:30 UTC after): violations Python **0**, MT5 **0**

Matched-pair delta statistics (|MT5 - Python|):

| Delta | mean | p95 | max | tolerance | within |
|---|---|---|---|---|---|
| entry price (pts) | 0.212 | 0.494 | 0.526 | <= 1.5 pts | 27/27 (100.0%) |
| stop loss (pts) | 0.225 | 0.485 | 0.535 | <= 0.05 pts | 4/27 (14.8%) |
| take profit (pts) | 0.216 | 0.528 | 0.556 | <= 0.05 pts | 2/27 (7.4%) |
| exit price (pts) | 0.271 | 0.591 | 0.802 | <= 1.5 pts | 27/27 (100.0%) |
| exit time (min) | 0.889 | 3.700 | 4.000 | <= 15 min | 27/27 (100.0%) |

SL/TP deltas track the day-level OR level deltas (SL = opposite OR side, TP = f(entry level)), so they exceed the tight 0.05 pt tolerance on most pairs while entry/exit fills stay well inside the 1.5 pt tolerance.

Per-trade table for outliers (pairs failing any tolerance; entry/exit deltas are MT5 - Python signed, `*` marks a tolerance breach):

| py# | mt5# | dir | date | entry d | SL d | TP d | exit d | exit t (min) | reason (py / mt5 -> mapped) | fails |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | SELL | 01-02 | -0.31 | 0.25* | 0.21* | +0.42 | 0 | session_end / session_end -> session_end | SL, TP |
| 2 | 4 | SELL | 01-09 | -0.23 | 0.22* | 0.17* | +0.02 | 2 | stop_loss / other -> stop_loss | SL, TP, REASON_RAW |
| 3 | 6 | SELL | 01-13 | +0.25 | 0.34* | 0.34* | +0.80 | 0 | session_end / session_end -> session_end | SL, TP |
| 4 | 8 | BUY | 01-23 | +0.23 | 0.32* | 0.22* | -0.09 | 0 | session_end / session_end -> session_end | SL, TP |
| 5 | 10 | SELL | 01-29 | -0.11 | 0.12* | 0.06* | -0.07 | 2 | take_profit / take_profit -> take_profit | SL, TP |
| 6 | 12 | BUY | 02-02 | -0.53 | 0.53* | 0.56* | -0.41 | 0 | session_end / session_end -> session_end | SL, TP |
| 7 | 14 | SELL | 02-03 | -0.16 | 0.15* | 0.11* | -0.15 | 3 | take_profit / take_profit -> take_profit | SL, TP |
| 8 | 16 | SELL | 02-04 | +0.01 | 0.01 | 0.07* | +0.23 | 0 | session_end / session_end -> session_end | TP |
| 9 | 18 | SELL | 02-05 | -0.13 | 0.12* | 0.10* | +0.18 | 0 | session_end / session_end -> session_end | SL, TP |
| 10 | 20 | BUY | 02-09 | +0.53 | 0.49* | 0.50* | +0.46 | 0 | session_end / session_end -> session_end | SL, TP |
| 11 | 22 | SELL | 02-11 | -0.17 | 0.17* | 0.15* | +0.52 | 0 | session_end / session_end -> session_end | SL, TP |
| 12 | 24 | BUY | 02-18 | -0.42 | 0.48* | 0.54* | -0.20 | 0 | session_end / session_end -> session_end | SL, TP |
| 13 | 26 | BUY | 02-20 | -0.10 | 0.37* | 0.42* | -0.38 | 0 | session_end / session_end -> session_end | SL, TP |
| 14 | 28 | SELL | 02-23 | -0.11 | 0.16* | 0.08* | +0.49 | 0 | session_end / session_end -> session_end | SL, TP |
| 16 | 32 | BUY | 02-27 | +0.01 | 0.11* | 0.12* | -0.14 | 0 | session_end / session_end -> session_end | SL, TP |
| 17 | 34 | BUY | 03-04 | -0.28 | 0.33* | 0.32* | -0.62 | 0 | session_end / session_end -> session_end | SL, TP |
| 18 | 36 | SELL | 03-05 | -0.33 | 0.38* | 0.33* | +0.12 | 0 | session_end / session_end -> session_end | SL, TP |
| 19 | 38 | BUY | 03-11 | +0.15 | 0.11* | 0.07* | +0.06 | 4 | stop_loss / stop_loss -> stop_loss | SL, TP |
| 20 | 40 | SELL | 03-13 | +0.00 | 0.04 | 0.06* | -0.06 | 4 | take_profit / other -> take_profit | TP, REASON_RAW |
| 21 | 42 | SELL | 03-18 | -0.20 | 0.24* | 0.20* | -0.22 | 3 | take_profit / take_profit -> take_profit | SL, TP |
| 22 | 44 | SELL | 03-20 | +0.06 | 0.14* | 0.13* | +0.10 | 3 | take_profit / take_profit -> take_profit | SL, TP |
| 23 | 46 | BUY | 04-06 | -0.25 | 0.24* | 0.28* | -0.35 | 3 | stop_loss / other -> stop_loss | SL, TP, REASON_RAW |
| 24 | 48 | SELL | 04-07 | +0.32 | 0.25* | 0.31* | +0.40 | 0 | stop_loss / other -> stop_loss | SL, TP, REASON_RAW |
| 25 | 50 | BUY | 04-14 | +0.16 | 0.10* | 0.07* | +0.07 | 0 | session_end / session_end -> session_end | SL, TP |
| 26 | 52 | BUY | 04-17 | -0.33 | 0.34* | 0.39* | -0.42 | 0 | session_end / session_end -> session_end | SL, TP |

Entry-price bias (MT5 - Python, signed pts):

* BUY entries (n=12): mean **-0.060** (sd 0.312, min -0.526, max +0.525)
* SELL entries (n=15): mean -0.087 (sd 0.187, min -0.333, max +0.316)
* decomposition for BUY: MT5 fills sit +0.084 pts above their own buy level while MT5 buy levels sit -0.144 pts vs Python levels, so the raw bias nets to ~-0.06; expected +spread (~+0.39 pts, average bid/ask spread in the log's stop-order lines) only if levels were identical -- the level-source delta offsets the ask-side fill premium

## 7. Anomaly flags

| # | Check | Result | Details |
|---|---|---|---|
| 1 | Skip rate > 20% | PASS | 10.7% (9/84 session days) on both sides, identical gap_filter dates |
| 2 | Total trades < 20 or > 90 | PASS | Python 27, MT5 27 |
| 3 | Overall agreement < 90% | PASS | 100.0% |
| 4 | Unmatched trades or day discrepancies | FLAG | unmatched trades: 0 both sides; day discrepancies: 84/102 days (500 field instances -- price/level deltas > 0.02 on session days, 34 gap_pct deltas > 0.005, 4 exit_reason mismatches); decision fields identical on all 102 days; all instances attributable to the documented artifacts (section 8) |
| 5 | Systematic BUY entry-price bias (MT5 - Python) | PASS | mean -0.060 pts (n=12, sd 0.312) -- not systematically one-sided; MT5 fills ~+0.08 above own level vs spread ~0.39; offset by level-source delta -0.14 |
| 6 | Trades outside the 10:00-10:30 NY entry window | PASS | 0 Python / 0 MT5 violations |
| 7 | MT5 exit reason 'other' count | FLAG (benign) | 4/27 (14.8%) MT5 exits labelled 'other'; all reclassify to the Python reason under the +/-0.5 slippage-invariant mapping (mapped agreement 27/27 (100.0%)) |

## 8. Known artifacts

1. **18 `ORB force-close failed retcode=10015` self-healing retries at 16:00 NY closes**: every session_end close (18 trades) logs one failed force-close attempt followed by a successful retry within the same minute; all close_times still match the 16:00 NY bar (21:00 UTC winter / 20:00 UTC after DST) and agree with Python.
2. **MT5 levels rounded to the 0.01 tick grid vs unrounded Python levels**: rounding contributes <= 0.005 pts; the larger observed level deltas (mean 0.213 pts) come from the different price sources (MT5 native ticks vs Python M5 bid bars). The discarded first pass in the agent log shows the pre-fix behaviour: 820 stop placements rejected (`Invalid price` / `Invalid stops`) because unrounded 4-decimal prices violate the tick grid.
3. **4 MT5 exits labelled `other`** (server-side SL/TP closes the EA did not tag): resolved by the mapped comparison, 27/27 (100.0%) agreement.

## 9. Conclusions

* All 27 Python trades matched one-to-one to MT5 trades (same direction, same day, entry times within 5 min); no unmatched trades on either side.
* Day-level trading decisions are identical on all 102 NY dates (skips, order placement, entry direction); overall reconciliation agreement **100.0%**.
* Residual differences are fully explained: price/level deltas from the two price sources (all inside fill tolerances), 4 'other' exit labels (mapped comparison resolves all), 18 self-healing force-close retries, and fill slippage producing a +5.77 USD PnL difference (2.3%).
* Performance is equivalent: Sharpe 2.81 vs 2.75, max drawdown 84.11 vs 84.28 USD, win rates 63.0% vs 63.0%.
* The MT5 EA reproduces the Python reference implementation for this window; the two FLAG anomalies (day-level price deltas, 'other' exit labels) are documented artifacts with no impact on trading decisions or overall agreement.
