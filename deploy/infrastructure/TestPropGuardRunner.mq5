//+------------------------------------------------------------------+
//|                                       TestPropGuardRunner.mq5    |
//|  Headless runner for the PropFirmGuard self-test.                |
//|                                                                   |
//|  Includes scripts/TestPropGuardBody.mqh - the EXACT test body    |
//|  also used by scripts/TestPropGuard.mq5 (no logic duplication) - |
//|  and invokes it from OnInit, allowing the same self-test to      |
//|  execute in the MT5 Strategy Tester via a /config run (scripts   |
//|  cannot run in the tester; EAs can). Print markers are           |
//|  identical to the script path.                                   |
//|                                                                   |
//|  Driven by deploy/infrastructure/verify_propguard_live.py        |
//|  --tester-run; verdicts print to the tester agent log.           |
//+------------------------------------------------------------------+
#property copyright "RR project"
#property version   "1.00"

#include "..\..\scripts\TestPropGuardBody.mqh"

int OnInit()
{
   TestPropGuard_RunAll();
   return(INIT_SUCCEEDED);
}
//+------------------------------------------------------------------+
