//+------------------------------------------------------------------+
//|                                                MULTI_CAIRO_EA.mq5 |
//|                              RR — Real trading EA for fold testing |
//|                                                                     |
//| On each new H1 bar, calls Bridge_GetSignal() from transport_dll.mqh |
//| and places real orders via OrderSend() based on the signal.          |
//|                                                                     |
//| Compile: opens MetaEditor, open this file, press F7                  |
//+------------------------------------------------------------------+
#property copyright "RR"
#property version   "2.00"
#property description "Real trading EA — bridge signal + OrderSend execution"

#import "bridge_client.dll"
int BridgePing(int port);
int BridgeSendReceiveRaw(int port, uchar &request[], uchar &response[], int response_size);
#import

// ── Inputs ───────────────────────────────────────────────────────────────
input int      BridgePort = 9998;  // MUST match orchestrator port
input double   LotSize    = 0.01;
input int      MagicNumber = 25080601;

// ── Globals ──────────────────────────────────────────────────────────────
string g_currentAction  = "HOLD";
int    g_barsReceived   = 0;
int    g_tradeCount     = 0;
bool   g_bridgeAlive    = false;
double g_lotSize        = 0.01;

// ── CP_UTF8-safe bridge wrappers (inline, avoids include encoding issues) ──

int Bridge_IsAlive(int port)
{
   return BridgePing(port);
}

// ── Get base_lot from bridge via ping-parsed response ───────────────────
double Bridge_GetBaseLot(int port)
{
   uchar request[];
   StringToCharArray("{\"ping\":1}", request, 0, WHOLE_ARRAY, CP_UTF8);

   uchar response[256];
   int n = BridgeSendReceiveRaw(port, request, response, 256);
   if (n <= 0)
      return 0.0;

   string resp = CharArrayToString(response, 0, n, CP_UTF8);
   string lotStr = ExtractJsonString(resp, "base_lot");
   if (lotStr == "")
      return 0.0;

   return StringToDouble(lotStr);
}

int Bridge_GetSignal(int port, string barJson, string &responseJson)
{
   uchar request[];
   StringToCharArray(barJson, request, 0, WHOLE_ARRAY, CP_UTF8);

   // Ensure newline terminator
   int reqLen = ArraySize(request);
   if (reqLen > 0 && request[reqLen - 1] != '\n')
   {
      ArrayResize(request, reqLen + 1);
      request[reqLen] = '\n';
   }

   uchar response[4096];
   int n = BridgeSendReceiveRaw(port, request, response, 4096);

   if (n <= 0)
   {
      responseJson = "";
      return 0;
   }

   responseJson = CharArrayToString(response, 0, n, CP_UTF8);
   return n;
}

// ── Helper: extract JSON string value (minimal parser, no external deps) ──
string ExtractJsonString(string json, string key)
{
   string search = "\"" + key + "\"";
   int pos = StringFind(json, search);
   if (pos < 0) return "";

   // Find the colon after the key
   int colon = StringFind(json, ":", pos);
   if (colon < 0) return "";

   // Skip whitespace after colon
   int valStart = colon + 1;
   while (valStart < StringLen(json))
   {
      ushort c = StringGetCharacter(json, valStart);
      if (c != ' ' && c != '\t')
         break;
      valStart++;
   }

   ushort firstChar = StringGetCharacter(json, valStart);

   // String value: "..." → read until closing quote
   if (firstChar == '"')
   {
      int qEnd = StringFind(json, "\"", valStart + 1);
      if (qEnd < 0) return "";
      return StringSubstr(json, valStart + 1, qEnd - valStart - 1);
   }

   // Numeric/boolean/null value → read until , or }
   int vEnd = valStart;
   while (vEnd < StringLen(json))
   {
      ushort c = StringGetCharacter(json, vEnd);
      if (c == ',' || c == '}')
         break;
      vEnd++;
   }
   return StringSubstr(json, valStart, vEnd - valStart);
}

// ── Helper: build bar JSON from current MT5 bar data ─────────────────────
string BuildBarJson()
{
   // Send the just-completed bar (index 1) to the bridge
   int barIdx     = 1;  // previous completed bar
   double open    = iOpen(_Symbol, PERIOD_CURRENT, barIdx);
   double high    = iHigh(_Symbol, PERIOD_CURRENT, barIdx);
   double low     = iLow(_Symbol, PERIOD_CURRENT, barIdx);
   double close   = iClose(_Symbol, PERIOD_CURRENT, barIdx);
   double ask     = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid     = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double spread  = ask - bid;
   long   ts      = (long)iTime(_Symbol, PERIOD_CURRENT, barIdx);

   return StringFormat(
      "{\"symbol\":\"%s\",\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"spread\":%.6f,\"timestamp\":%lld}",
      _Symbol, open, high, low, close, spread, ts
   );
}

// ── Helper: round a raw price to the symbol's tick grid ─────────────────
// Custom symbols (e.g. US500-tick) can carry 4-digit Dukascopy prices while
// their tick_size is 0.01 (2-digit broker grid). Off-grid prices are
// rejected by OrderSend with error 4756 "Invalid price", so round first.
double NormalizePriceToTick(double raw)
{
   double tickSz = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   if (tickSz <= 0.0)
      tickSz = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if (tickSz <= 0.0)
      return raw;
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   return NormalizeDouble(MathRound(raw / tickSz) * tickSz, digits);
}

// ── Helper: close all positions for this symbol ──────────────────────────
void CloseAllPositions()
{
   for (int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if (!PositionSelectByTicket(PositionGetTicket(i)))
         continue;
      if (PositionGetString(POSITION_SYMBOL) != _Symbol)
         continue;
      if (PositionGetInteger(POSITION_MAGIC) != MagicNumber)
         continue;

      MqlTradeRequest req;
      MqlTradeResult  res;
      ZeroMemory(req);
      ZeroMemory(res);

      req.action   = TRADE_ACTION_DEAL;
      req.symbol   = _Symbol;
      req.volume   = PositionGetDouble(POSITION_VOLUME);
      req.type     = (PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
                     ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
      req.position = PositionGetTicket(i);
      req.price    = (req.type == ORDER_TYPE_SELL)
                     ? NormalizePriceToTick(SymbolInfoDouble(_Symbol, SYMBOL_BID))
                     : NormalizePriceToTick(SymbolInfoDouble(_Symbol, SYMBOL_ASK));
      req.deviation = 50;
      req.magic    = MagicNumber;
      req.comment  = "RR_close";

      OrderSend(req, res);
   }
}

// ── Helper: open a new position ──────────────────────────────────────────
void OpenPosition(string direction)
{
   MqlTradeRequest req;
   MqlTradeResult  res;
   ZeroMemory(req);
   ZeroMemory(res);

   req.action   = TRADE_ACTION_DEAL;
   req.symbol   = _Symbol;
   req.volume   = g_lotSize;
   req.type     = (direction == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   req.price    = (direction == "BUY")
                  ? NormalizePriceToTick(SymbolInfoDouble(_Symbol, SYMBOL_ASK))
                  : NormalizePriceToTick(SymbolInfoDouble(_Symbol, SYMBOL_BID));
   req.deviation = 50;
   req.magic    = MagicNumber;
   req.comment  = "RR_" + direction;

   if (OrderSend(req, res))
   {
      if (res.retcode == TRADE_RETCODE_DONE || res.retcode == TRADE_RETCODE_DONE_PARTIAL)
      {
         g_tradeCount++;
         Print("Trade #", g_tradeCount, " | ",
                direction, " ", g_lotSize, " ",
               _Symbol, " @ ", req.price,
               " | ticket=", res.order);
      }
      else
      {
         Print("OrderSend failed: retcode=", res.retcode, " ", res.comment);
      }
   }
   else
   {
      Print("OrderSend error: ", GetLastError());
   }
}

// ── Core: new bar handler ────────────────────────────────────────────────
void OnNewBar()
{
   g_barsReceived++;

   // Build bar JSON and send to bridge
   string barJson     = BuildBarJson();
   string responseJson;

   int n = Bridge_GetSignal(BridgePort, barJson, responseJson);
   if (n <= 0)
   {
      // Bridge not responding — don't change position
      if (g_barsReceived % 50 == 0)
         Print("Bridge not responding (bars=", g_barsReceived, ")");
      return;
   }

   // Parse the response
   string action = ExtractJsonString(responseJson, "action");
   if (action == "")
   {
      Print("Failed to parse action from response: ", responseJson);
      return;
   }

   // Log every 50 bars
   if (g_barsReceived % 50 == 0)
   {
      string weight = ExtractJsonString(responseJson, "cairo_weight");
      string votes  = ExtractJsonString(responseJson, "winner_votes");
      Print("Bar ", g_barsReceived, " | action=", action,
            " | weight=", weight, " | votes=", votes,
            " | bridge_n=", n);
   }

   // Only trade on BUY or SELL signals
   if (action != "BUY" && action != "SELL")
   {
      if (action == "HOLD" && g_currentAction != "HOLD")
      {
         // Signal flipped to HOLD — close existing position
         CloseAllPositions();
         g_currentAction = "HOLD";
      }
      return;
   }

   // Signal is BUY or SELL
   if (action == g_currentAction)
      return; // same signal, hold position

   // Signal changed — close existing, open new
   if (g_currentAction != "HOLD")
      CloseAllPositions();

   OpenPosition(action);
   g_currentAction = action;
}

// ═══════════════════════════════════════════════════════════════════════════
// Event handlers
// ═══════════════════════════════════════════════════════════════════════════

int OnInit()
{
   Print("=== MULTI_CAIRO_EA v1.0 ===");
   Print("Symbol: ", _Symbol, "  Period: ", EnumToString(PERIOD_CURRENT));
   Print("Bridge port: ", BridgePort);

   // Initial bridge health check
   g_bridgeAlive = (Bridge_IsAlive(BridgePort) == 1);
   Print("Bridge alive: ", g_bridgeAlive);

   // ── Lot size: query bridge for checkpoint's base_lot ──────────────────
   if (g_bridgeAlive)
   {
      double bridgeBaseLot = Bridge_GetBaseLot(BridgePort);
      if (bridgeBaseLot > 0.0)
      {
         g_lotSize = bridgeBaseLot;
         Print("  Bridge base_lot: ", bridgeBaseLot);
      }
      else
      {
         Print("  Bridge base_lot not returned — using default");
      }
   }

   // Clamp to broker's minimum tradeable volume
   double minVol  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double volStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double brokerMin = MathMax(minVol, volStep);
   if (g_lotSize < brokerMin)
   {
      Print("  Clamping lot: ", g_lotSize, " -> ", brokerMin,
            " (broker min = ", minVol, ", step = ", volStep, ")");
      g_lotSize = brokerMin;
   }
   Print("  Trading lot size: ", g_lotSize);

   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   Print("=== Deinit (reason=", reason, ") ===");
   Print("Total bars received: ", g_barsReceived);
   Print("Total trades placed: ", g_tradeCount);
   Print("Final action:        ", g_currentAction);

   // Close all positions on exit
   CloseAllPositions();

      // ── Write structured trade CSV (plan §3.3) ────────────────────────
      // Columns: ticket; open_time; close_time; type; lot; open_price;
      //           close_price; swap; commission; profit
      // Semicolon delimited (robust across MT5 locales vs tab/semicolon
      // FileWrite auto-delimiter behaviour)
      string csvName = "trades.csv";
      int handle = FileOpen(csvName, FILE_WRITE | FILE_TXT | FILE_ANSI);
      if (handle != INVALID_HANDLE)
      {
         FileWriteString(handle,
            "ticket;open_time;close_time;symbol;type;lot;open_price;close_price;swap;commission;profit\r\n",
            StringLen("ticket;open_time;close_time;symbol;type;lot;open_price;close_price;swap;commission;profit\r\n"));

         if (!HistorySelect(0, TimeCurrent()))
         {
            Print("HistorySelect failed");
            FileClose(handle);
            return;
         }

         int totalDeals = HistoryDealsTotal();

         for (int i = 0; i < totalDeals; i++)
         {
            ulong ticket = HistoryDealGetTicket(i);
            if (ticket == 0)
               continue;

            int dealMagic = (int)HistoryDealGetInteger(ticket, DEAL_MAGIC);
            if (dealMagic != MagicNumber)
               continue;

            ENUM_DEAL_ENTRY entryType = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(ticket, DEAL_ENTRY);
            if (entryType != DEAL_ENTRY_OUT)
               continue;

            long positionID = HistoryDealGetInteger(ticket, DEAL_POSITION_ID);

            // Find matching entry deal (also filtered by MagicNumber)
            ulong entryTicket = 0;
            double openPrice = 0;
            datetime openTime = 0;
            int    entryDealType = 0;
            string dealSymbol = HistoryDealGetString(ticket, DEAL_SYMBOL);
            for (int j = 0; j < totalDeals; j++)
            {
               ulong dt = HistoryDealGetTicket(j);
               if (dt == 0 || dt == ticket)
                  continue;
               if ((int)HistoryDealGetInteger(dt, DEAL_MAGIC) != MagicNumber)
                  continue;
               if ((ENUM_DEAL_ENTRY)HistoryDealGetInteger(dt, DEAL_ENTRY) != DEAL_ENTRY_IN)
                  continue;
               if (HistoryDealGetInteger(dt, DEAL_POSITION_ID) == positionID)
               {
                  entryTicket = dt;
                  openPrice = HistoryDealGetDouble(dt, DEAL_PRICE);
                  openTime = (datetime)HistoryDealGetInteger(dt, DEAL_TIME);
                  entryDealType = (int)HistoryDealGetInteger(dt, DEAL_TYPE);
                  break;
               }
            }

            datetime closeTime  = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
            double   closePrice = HistoryDealGetDouble(ticket, DEAL_PRICE);
            double   volume     = HistoryDealGetDouble(ticket, DEAL_VOLUME);
            double   swapVal    = HistoryDealGetDouble(ticket, DEAL_SWAP);
            double   commVal    = HistoryDealGetDouble(ticket, DEAL_COMMISSION);
            double   profitVal  = HistoryDealGetDouble(ticket, DEAL_PROFIT);

            string dir = (entryDealType == DEAL_TYPE_BUY) ? "BUY" : "SELL";

            string line = IntegerToString(ticket) + ";" +
                          TimeToString(openTime) + ";" +
                          TimeToString(closeTime) + ";" +
                          dealSymbol + ";" +
                          dir + ";" +
                          DoubleToString(volume, 2) + ";" +
                          DoubleToString(openPrice) + ";" +
                          DoubleToString(closePrice) + ";" +
                          DoubleToString(swapVal, 2) + ";" +
                          DoubleToString(commVal, 2) + ";" +
                          DoubleToString(profitVal, 2) + "\r\n";
            FileWriteString(handle, line, StringLen(line));
         }

         FileClose(handle);
         Print("Trade CSV written: ", csvName, " (", totalDeals, " deals)");
      }
      else
      {
         Print("WARNING: Could not open trades.csv for writing. Error=", GetLastError());
      }
}

void OnTick()
{
   // Only act on new bar
   static datetime lastBarTime = 0;
   datetime currentBar = iTime(_Symbol, PERIOD_CURRENT, 0);
   if (currentBar == lastBarTime)
      return;
   lastBarTime = currentBar;

   OnNewBar();
}

//+------------------------------------------------------------------+
