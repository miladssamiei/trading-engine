//+------------------------------------------------------------------+
//|                                            test_bridge_client.mq5 |
//|                              RR Phase 2.6b — Full roundtrip       |
//+------------------------------------------------------------------+
#property copyright "RR"
#property version   "2.01"
#property description "Sandbox DLL validation — BridgePing + BridgeSendReceive with real bar"

#import "bridge_client.dll"
int BridgePing(int port);
int BridgeSendReceiveRaw(int port, uchar &request[], uchar &response[], int response_size);
#import

#define TEST_PORT 19998

//+------------------------------------------------------------------+
int OnInit()
{
   Print("=== test_bridge_client.mq5 v2.01 — OnInit ===");
   
   // Test 1: Ping
   int pingResult = BridgePing(TEST_PORT);
   Print("BridgePing(", TEST_PORT, ") = ", pingResult, " (expected 1)");
   if (pingResult != 1) { Print("FAIL"); return(INIT_FAILED); }
   Print("PASS");
   
   // Test 2: SendReceive with real bar via uchar[] (avoids CP_ACP encoding)
   string barJson = "{\"symbol\":\"EURUSD\","
                     "\"open\":1.14260,"
                     "\"high\":1.14275,"
                     "\"low\":1.14255,"
                     "\"close\":1.14275,"
                     "\"spread\":0.00015,"
                     "\"timestamp\":1754400000}";
   
   // Convert MQL5 Unicode string to UTF-8 bytes for DLL
   uchar request[];
   StringToCharArray(barJson, request, 0, WHOLE_ARRAY, CP_UTF8);
   
   Print("BridgeSendReceive request (CP_UTF8, ", ArraySize(request), " bytes): ", barJson);
   
   uchar response[4096];
   int n = BridgeSendReceiveRaw(TEST_PORT, request, response, 4096);
   
   string raw = CharArrayToString(response, 0, n, CP_UTF8);
   Print("BridgeSendReceive returned ", n, " bytes");
   Print("RAW RESPONSE: ", raw);
   
   if (n > 0 && StringLen(raw) > 0 && StringFind(raw, "\"action\"") >= 0)
      Print("PASS — Valid JSON response with action field received inside sandbox.");
   else
      Print("FAIL");
   
   Print("=== Roundtrip test complete ===");
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason) { Print("=== Deinit (reason=", reason, ") ==="); }
void OnTick() { }
//+------------------------------------------------------------------+
