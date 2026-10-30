//+------------------------------------------------------------------+
//|                                              transport_dll.mqh     |
//|                          RR — DLL socket transport for MT5 Tester  |
//|                                                                     |
//| Wraps bridge_client.dll imports with transparent CP_UTF8 encoding.  |
//| Callers pass/recv MQL5 strings normally — all encoding is internal. |
//|                                                                     |
//| CP_ACP JSON CORRUPTION NOTE (2026-08-05):                          |
//|   MQL5's `string` -> `const char*` marshaling goes through the      |
//|   system ANSI code page (CP_ACP / CP1252), which corrupts JSON      |
//|   containing double-quote and brace characters.  This module uses    |
//|   BridgeSendReceiveRaw with CP_UTF8 byte arrays exclusively.         |
//|   See PROGRESS.md § "CP_ACP JSON corruption fix" for full analysis. |
//+------------------------------------------------------------------+
#property copyright "RR"
#property version   "1.00"
#property description "DLL socket transport — same interface as transport_http.mqh"

// ── DLL imports (Raw variant ONLY — never use string-based imports) ──
#import "bridge_client.dll"
int BridgePing(int port);
int BridgeSendReceiveRaw(int port, uchar &request[], uchar &response[], int response_size);
#import

// ── Public API (identical signatures to transport_http.mqh) ──────────

//+------------------------------------------------------------------+
//| Check if the bridge server is alive on 127.0.0.1:<port>.          |
//| Returns 1 if alive, 0 otherwise.                                  |
//+------------------------------------------------------------------+
int Bridge_IsAlive(int port)
{
   return BridgePing(port);
}

//+------------------------------------------------------------------+
//| Send a bar JSON string to the bridge and receive a signal JSON.    |
//|                                                                    |
//|   barJson       — MQL5 string (Unicode), converted to CP_UTF8     |
//|   responseJson  — [out] MQL5 string, decoded from CP_UTF8 bytes   |
//|                                                                    |
//| Returns the number of response bytes, 0 on error.                  |
//+------------------------------------------------------------------+
int Bridge_GetSignal(int port, string barJson, string &responseJson)
{
   // 1. Encode request as UTF-8 bytes
   uchar request[];
   StringToCharArray(barJson, request, 0, WHOLE_ARRAY, CP_UTF8);

   // 2. Ensure newline terminator (DLL appends one, but be explicit)
   bool hasNewline = false;
   int  reqLen = ArraySize(request);
   if (reqLen > 0 && request[reqLen - 1] == '\n')
      hasNewline = true;

   if (!hasNewline)
   {
      ArrayResize(request, reqLen + 1);
      request[reqLen] = '\n';
   }

   // 3. Call DLL — Raw variant, encoding-transparent
   uchar response[4096];
   int   n = BridgeSendReceiveRaw(port, request, response, 4096);

   if (n <= 0)
   {
      responseJson = "";
      return 0;
   }

   // 4. Decode response bytes to MQL5 Unicode string (CP_UTF8)
   responseJson = CharArrayToString(response, 0, n, CP_UTF8);
   return n;
}
//+------------------------------------------------------------------+
