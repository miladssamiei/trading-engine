//+------------------------------------------------------------------+
//|                                          TestPropGuardBody.mqh   |
//|  Shared self-test body for Include/PropFirmGuard.mqh             |
//|  (SPEC_EXECUTION_INFRASTRUCTURE section 5).                      |
//|                                                                   |
//|  Consumed VERBATIM by two wrappers so the exact same test runs   |
//|  both ways with one body of logic:                               |
//|    scripts/TestPropGuard.mq5      -> OnStart()      (demo chart) |
//|    deploy/infrastructure/         -> OnInit()       (headless    |
//|    TestPropGuardRunner.mq5                           tester run) |
//|  Print markers are load-bearing:                                  |
//|  deploy/infrastructure/verify_propguard_live.py regex-parses     |
//|  them from the Experts/agent logs.                                |
//+------------------------------------------------------------------+
#property copyright "RR project"
#property strict

#include "..\Include\PropFirmGuard.mqh"

void TestPropGuard_RunAll()
{
   Print("=== TestPropGuard start ===");

   PG_TestReset();
   PG_Init("TEST", 26082999);

   int passed = 0;
   int failed = 0;

   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   PrintFormat("account equity=%.2f balance=%.2f",
               eq, AccountInfoDouble(ACCOUNT_BALANCE));

   Print("--- TEST 1: forced 3.6% daily drawdown ---");
   PG_TestForceDailyBreach();
   double eod = GlobalVariableGet(PG_GV_EOD_BAL);
   PrintFormat("synthetic EOD_BAL=%.2f -> simulated daily loss %.2f%%",
               eod, (eod - eq) / eod * 100.0);
   bool okTick = PG_OnTick();
   bool latched = PG_IsBreached();
   if(!okTick && latched)
   {
      Print("TEST 1 PASS: PG_OnTick=false, breach latched, "
            "PANIC_FLATTEN executed");
      passed++;
   }
   else
   {
      PrintFormat("TEST 1 FAIL: okTick=%s latched=%s",
                  okTick ? "true" : "false", latched ? "yes" : "no");
      failed++;
   }
   bool okEntry = PG_EntryAllowed(_Symbol, "TEST");
   if(!okEntry)
   {
      Print("TEST 1b PASS: entry refused while latched");
      passed++;
   }
   else
   {
      Print("TEST 1b FAIL: entry allowed while latched");
      failed++;
   }

   Print("--- TEST 2: forced 8.1% trailing drawdown ---");
   PG_TestReset();
   PG_Init("TEST", 26082999);
   PG_TestForceTrailingBreach();
   double hwm = GlobalVariableGet(PG_GV_HWM);
   PrintFormat("synthetic HWM=%.2f -> simulated trailing DD %.2f%%",
               hwm, (hwm - eq) / hwm * 100.0);
   okTick = PG_OnTick();
   latched = PG_IsBreached();
   if(!okTick && latched)
   {
      Print("TEST 2 PASS: PG_OnTick=false, breach latched, "
            "PANIC_FLATTEN executed");
      passed++;
   }
   else
   {
      PrintFormat("TEST 2 FAIL: okTick=%s latched=%s",
                  okTick ? "true" : "false", latched ? "yes" : "no");
      failed++;
   }

   Print("--- TEST 3: spread filter (informational, live feed) ---");
   PG_TestReset();
   PG_Init("TEST", 26082999);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   PrintFormat("current spread=%.3f limit=%.2f -> %s",
               ask - bid, PG_SpreadLimit(_Symbol),
               PG_SpreadOK(_Symbol) ? "OK" : "would block");

   Print("--- TEST 4: deviation points ---");
   int dev = PG_DeviationPoints(_Symbol);
   bool devOk = (dev >= 1);
   PrintFormat("deviation=%d points (point=%s)",
               dev, DoubleToString(SymbolInfoDouble(_Symbol,
                                                    SYMBOL_POINT), 8));
   if(devOk)
   {
      Print("TEST 4 PASS");
      passed++;
   }
   else
   {
      Print("TEST 4 FAIL");
      failed++;
   }

   PG_TestReset();
   PrintFormat("=== TestPropGuard done: %d passed, %d failed; "
               "guard state reset ===", passed, failed);
}
//+------------------------------------------------------------------+
