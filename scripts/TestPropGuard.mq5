//+------------------------------------------------------------------+
//|                                              TestPropGuard.mq5   |
//|  Synthetic self-test for Include/PropFirmGuard.mqh               |
//|  (SPEC_EXECUTION_INFRASTRUCTURE section 5).                      |
//|                                                                   |
//|  Forces a 3.6% daily drawdown and an 8.1% trailing drawdown by    |
//|  planting synthetic EOD/HWM anchors in the terminal global        |
//|  variables, then verifies that PG_OnTick() latches the breach,    |
//|  runs PANIC_FLATTEN, and refuses further trading. Run on a demo   |
//|  chart; verdicts print to the Experts log. State is reset at the  |
//|  end so live EAs are unaffected.                                  |
//|                                                                   |
//|  The test body lives in scripts/TestPropGuardBody.mqh and is     |
//|  shared with deploy/infrastructure/TestPropGuardRunner.mq5       |
//|  (headless Strategy-Tester path) - one body, two wrappers.       |
//+------------------------------------------------------------------+
#property copyright "RR project"
#property version   "1.01"
#property script_show_inputs
#include "TestPropGuardBody.mqh"

void OnStart()
{
   TestPropGuard_RunAll();
}
//+------------------------------------------------------------------+
