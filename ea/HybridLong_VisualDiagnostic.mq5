//+------------------------------------------------------------------+
//|                                   HybridLong_VisualDiagnostic.mq5 |
//| Visual diagnostic for the youtube_hybrid_scalping (#22/#25) ATR   |
//| band-touch Long model: live lower band, entry drift/slippage,     |
//| live spread, floating PnL/MAE and FTMO 5%/10% trackers (visual    |
//| only). Fully standalone (no PropFirmGuard include). Run in the    |
//| Strategy Tester visual mode or on a live Demo chart, US500 M5.    |
//+------------------------------------------------------------------+
#property copyright "RR project"
#property version   "1.00"
#property strict

input double FixedLots      = 1.0;
input long   InpMagicNumber = 26091701;

#define HLVD_PREFIX    "HLVD_"
#define HLVD_ATR_P     14
#define HLVD_BAND_K    3.1
#define HLVD_TP_K      2.0
#define HLVD_SL_K      1.5
#define HLVD_DAILY_PCT 5.0
#define HLVD_MAX_PCT   10.0
#define HLVD_LINE_BARS 12

int      g_hATR         = INVALID_HANDLE;
datetime g_lastBar      = 0;
datetime g_lastEntryBar = 0;
datetime g_failedBar    = 0;
double   g_initEquity   = 0.0;
double   g_eodAnchor    = 0.0;
double   g_eodDate      = 0.0;
bool     g_haveOpen     = false;
bool     g_codeManaged  = false;
ulong    g_ticket       = 0;
double   g_theoretical  = 0.0;
double   g_fill         = 0.0;
double   g_sigATR       = 0.0;
double   g_tpLevel      = 0.0;
double   g_slLevel      = 0.0;
datetime g_entryTime    = 0;
double   g_floatPnl     = 0.0;
double   g_floatMae     = 0.0;
double   g_lastClosed   = 0.0;
int      g_trades       = 0;

//+------------------------------------------------------------------+
string Px(const double p)
{
   return DoubleToString(p, _Digits);
}

//+------------------------------------------------------------------+
void DrawTrend(const string name, const datetime t1, const double p1,
               const datetime t2, const double p2, const color c,
               const ENUM_LINE_STYLE st, const int w)
{
   if(ObjectFind(0, name) >= 0)
      return;
   if(!ObjectCreate(0, name, OBJ_TREND, 0, t1, p1, t2, p2))
      return;
   ObjectSetInteger(0, name, OBJPROP_RAY_LEFT, false);
   ObjectSetInteger(0, name, OBJPROP_RAY_RIGHT, false);
   ObjectSetInteger(0, name, OBJPROP_COLOR, c);
   ObjectSetInteger(0, name, OBJPROP_STYLE, st);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, w);
   ObjectSetInteger(0, name, OBJPROP_BACK, false);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
}

//+------------------------------------------------------------------+
void DrawBandSegment(const datetime t0, const double band)
{
   string name = HLVD_PREFIX + "BAND_" + IntegerToString((long)t0);
   DrawTrend(name, t0, band, t0 + PeriodSeconds(PERIOD_M5), band,
             clrSilver, STYLE_DOT, 1);
}

//+------------------------------------------------------------------+
void DrawArrowAt(const string name, const datetime t, const double p,
                 const int code, const color c)
{
   if(ObjectFind(0, name) >= 0)
      return;
   if(!ObjectCreate(0, name, OBJ_ARROW, 0, t, p))
      return;
   ObjectSetInteger(0, name, OBJPROP_ARROWCODE, code);
   ObjectSetInteger(0, name, OBJPROP_COLOR, c);
   ObjectSetInteger(0, name, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
   ObjectSetInteger(0, name, OBJPROP_HIDDEN, true);
}

//+------------------------------------------------------------------+
void DrawEntryObjects()
{
   string id = IntegerToString((long)g_entryTime);
   datetime t2 = g_entryTime + HLVD_LINE_BARS * PeriodSeconds(PERIOD_M5);
   DrawTrend(HLVD_PREFIX + "ENTRY_THEO_" + id, g_entryTime, g_theoretical,
             t2, g_theoretical, clrRed, STYLE_SOLID, 2);
   DrawTrend(HLVD_PREFIX + "ENTRY_FILL_" + id, g_entryTime, g_fill,
             t2, g_fill, clrBlue, STYLE_SOLID, 2);
   DrawTrend(HLVD_PREFIX + "SLIP_GAP_" + id, g_entryTime, g_theoretical,
             g_entryTime, g_fill, clrRed, STYLE_SOLID, 2);
   DrawTrend(HLVD_PREFIX + "TARGET_" + id, g_entryTime, g_tpLevel,
             t2, g_tpLevel, clrGreen, STYLE_DASH, 1);
   DrawTrend(HLVD_PREFIX + "STOP_" + id, g_entryTime, g_slLevel,
             t2, g_slLevel, clrOrange, STYLE_DASH, 1);
   DrawArrowAt(HLVD_PREFIX + "FILL_ARROW_" + id, g_entryTime, g_fill, 241, clrBlue);
   DrawArrowAt(HLVD_PREFIX + "THEO_ARROW_" + id, g_entryTime, g_theoretical, 242, clrRed);
}

//+------------------------------------------------------------------+
bool AttachStops()
{
   MqlTradeRequest req;
   MqlTradeResult  res;
   ZeroMemory(req);
   ZeroMemory(res);
   req.action   = TRADE_ACTION_SLTP;
   req.symbol   = _Symbol;
   req.position = g_ticket;
   req.tp       = g_tpLevel;
   req.sl       = g_slLevel;
   if(!OrderSend(req, res))
      return false;
   return (res.retcode == TRADE_RETCODE_DONE);
}

//+------------------------------------------------------------------+
bool TryOpen(const double band, const double atr1)
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(ask <= 0.0)
      return false;
   MqlTradeRequest req;
   MqlTradeResult  res;
   ZeroMemory(req);
   ZeroMemory(res);
   req.action    = TRADE_ACTION_DEAL;
   req.symbol    = _Symbol;
   req.volume    = FixedLots;
   req.type      = ORDER_TYPE_BUY;
   req.price     = NormalizeDouble(ask, _Digits);
   req.deviation = 20;
   req.magic     = (ulong)InpMagicNumber;
   req.comment   = "HLVD entry";
   if(!OrderSend(req, res))
   {
      PrintFormat("HLVD open OrderSend failed ret=%u", res.retcode);
      return false;
   }
   if(res.retcode != TRADE_RETCODE_DONE &&
      res.retcode != TRADE_RETCODE_DONE_PARTIAL &&
      res.retcode != TRADE_RETCODE_PLACED)
   {
      PrintFormat("HLVD open rejected ret=%u", res.retcode);
      return false;
   }
   ulong tk = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong t = PositionGetTicket(i);
      if(t == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
         PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
      {
         tk = t;
         break;
      }
   }
   if(tk == 0)
   {
      Print("HLVD open FATAL: filled but no position found");
      return false;
   }
   g_ticket      = tk;
   g_theoretical = band;
   g_fill        = (res.price > 0.0 ? res.price : ask);
   g_sigATR      = atr1;
   g_tpLevel     = NormalizeDouble(g_fill + HLVD_TP_K * atr1, _Digits);
   g_slLevel     = NormalizeDouble(g_fill - HLVD_SL_K * atr1, _Digits);
   g_entryTime   = TimeCurrent();
   g_floatPnl    = 0.0;
   g_floatMae    = 0.0;
   g_haveOpen    = true;
   g_codeManaged = !AttachStops();
   g_trades++;
   DrawEntryObjects();
   PrintFormat("HLVD OPEN t=%I64u theo=%s fill=%s drift=%+.1f pts tp=%s sl=%s code_managed=%s",
               g_ticket, Px(g_theoretical), Px(g_fill),
               (g_fill - g_theoretical) / _Point, Px(g_tpLevel), Px(g_slLevel),
               g_codeManaged ? "yes" : "no");
   return true;
}

//+------------------------------------------------------------------+
bool ClosePos()
{
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   MqlTradeRequest req;
   MqlTradeResult  res;
   ZeroMemory(req);
   ZeroMemory(res);
   req.action    = TRADE_ACTION_DEAL;
   req.symbol    = _Symbol;
   req.volume    = PositionGetDouble(POSITION_VOLUME);
   req.type      = ORDER_TYPE_SELL;
   req.position  = g_ticket;
   req.price     = NormalizeDouble(bid, _Digits);
   req.deviation = 20;
   req.magic     = (ulong)InpMagicNumber;
   req.comment   = "HLVD exit";
   if(!OrderSend(req, res))
   {
      PrintFormat("HLVD close OrderSend failed ret=%u", res.retcode);
      return false;
   }
   return (res.retcode == TRADE_RETCODE_DONE ||
           res.retcode == TRADE_RETCODE_DONE_PARTIAL ||
           res.retcode == TRADE_RETCODE_PLACED);
}

//+------------------------------------------------------------------+
void FinalizeClosed()
{
   double p = 0.0;
   if(HistorySelectByPosition(g_ticket))
   {
      int n = HistoryDealsTotal();
      for(int i = 0; i < n; i++)
      {
         ulong d = HistoryDealGetTicket(i);
         if(d == 0)
            continue;
         p += HistoryDealGetDouble(d, DEAL_PROFIT)
             + HistoryDealGetDouble(d, DEAL_SWAP)
             + HistoryDealGetDouble(d, DEAL_COMMISSION);
      }
   }
   g_lastClosed  = p;
   g_haveOpen    = false;
   g_codeManaged = false;
   PrintFormat("HLVD CLOSED t=%I64u pnl=$%.2f | theo=%s fill=%s drift=%+.1f pts | MAE=$%.2f",
               g_ticket, p, Px(g_theoretical), Px(g_fill),
               (g_fill - g_theoretical) / _Point, g_floatMae);
}

//+------------------------------------------------------------------+
void EodRoll()
{
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   double today = dt.year * 10000.0 + dt.mon * 100.0 + dt.day;
   if(today != g_eodDate)
   {
      g_eodDate  = today;
      g_eodAnchor = AccountInfoDouble(ACCOUNT_BALANCE);
   }
}

//+------------------------------------------------------------------+
string BuildDashboard(const double bid, const double ask, const double atr0,
                      const double atr1, const double band)
{
   double spreadPts = (ask > 0.0 && bid > 0.0) ? (ask - bid) / _Point : 0.0;
   double equity   = AccountInfoDouble(ACCOUNT_EQUITY);
   double balance  = AccountInfoDouble(ACCOUNT_BALANCE);
   double dailyPct = (g_eodAnchor > 0.0) ? (g_eodAnchor - equity) / g_eodAnchor * 100.0 : 0.0;
   double maxPct   = (g_initEquity > 0.0) ? (g_initEquity - equity) / g_initEquity * 100.0 : 0.0;
   double floorUsd = g_initEquity * (1.0 - HLVD_MAX_PCT / 100.0);

   string s = "";
   s += StringFormat("--- LIVE MARKET DATA (%s, computed on M5) ---\n", _Symbol);
   s += StringFormat("Bid %s | Ask %s | Live Spread %.1f pts\n",
                     Px(bid), Px(ask), spreadPts);
   s += StringFormat("ATR(%d) current %s | signal-bar %s\n",
                     HLVD_ATR_P, Px(atr0), Px(atr1));
   s += StringFormat("Lower Band = Close[1] - %.1f*ATR[1] = %s\n",
                     HLVD_BAND_K, Px(band));
   s += "--- EXECUTION DIAGNOSTICS ---\n";
   if(g_trades > 0)
   {
      s += StringFormat("Theoretical Entry (band at signal) %s\n", Px(g_theoretical));
      s += StringFormat("Actual Market Fill (Ask received)  %s\n", Px(g_fill));
      s += StringFormat("ENTRY DRIFT / SLIPPAGE: %+.1f pts\n",
                        (g_fill - g_theoretical) / _Point);
   }
   else
      s += "No entry fired yet (waiting for band breach)\n";
   if(g_haveOpen)
      s += StringFormat("Floating PnL $%.2f | Floating MAE $%.2f | Ticket %I64u%s\n",
                        g_floatPnl, g_floatMae, g_ticket,
                        g_codeManaged ? " | exits code-managed" : "");
   else
      s += StringFormat("Flat | Trades so far %d | Last closed PnL $%.2f\n",
                        g_trades, g_lastClosed);
   s += "--- FTMO TRACKERS (visual only, no enforcement) ---\n";
   s += StringFormat("Day anchor $%.2f | Equity $%.2f | Balance $%.2f\n",
                     g_eodAnchor, equity, balance);
   s += StringFormat("Daily loss %.2f%% of %.2f%%%s | Max loss %.2f%% of %.2f%%%s\n",
                     dailyPct, HLVD_DAILY_PCT,
                     dailyPct >= HLVD_DAILY_PCT ? "  <DAILY BREACH>" : "",
                     maxPct, HLVD_MAX_PCT,
                     maxPct >= HLVD_MAX_PCT ? "  <MAX BREACH>" : "");
   s += StringFormat("Initial $%.2f | 10%% floor $%.2f | cushion $%.2f\n",
                     g_initEquity, floorUsd, equity - floorUsd);
   return s;
}

//+------------------------------------------------------------------+
int OnInit()
{
   g_hATR = iATR(_Symbol, PERIOD_M5, HLVD_ATR_P);
   if(g_hATR == INVALID_HANDLE)
   {
      Print("HLVD FATAL: cannot create ATR handle");
      return INIT_FAILED;
   }
   ObjectsDeleteAll(0, HLVD_PREFIX);
   g_initEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   g_eodAnchor  = AccountInfoDouble(ACCOUNT_BALANCE);
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   g_eodDate      = dt.year * 10000.0 + dt.mon * 100.0 + dt.day;
   g_lastBar      = 0;
   g_lastEntryBar = 0;
   g_failedBar    = 0;
   g_haveOpen     = false;
   g_codeManaged  = false;
   g_ticket       = 0;
   g_theoretical  = 0.0;
   g_fill         = 0.0;
   g_floatPnl     = 0.0;
   g_floatMae     = 0.0;
   g_lastClosed   = 0.0;
   g_trades       = 0;
   PrintFormat("HLVD init: %s lots=%.2f magic=%I64d init_equity=%.2f (visual diagnostic; "
               "band=%.1f*ATR, TP=%.1f*ATR, SL=%.1f*ATR off actual fill)",
               _Symbol, FixedLots, InpMagicNumber, g_initEquity,
               HLVD_BAND_K, HLVD_TP_K, HLVD_SL_K);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   ObjectsDeleteAll(0, HLVD_PREFIX);
   Comment("");
}

//+------------------------------------------------------------------+
void OnTick()
{
   double atrArr[];
   ArraySetAsSeries(atrArr, true);
   if(CopyBuffer(g_hATR, 0, 0, 2, atrArr) < 2)
      return;
   double atr0 = atrArr[0];
   double atr1 = atrArr[1];
   double close1 = iClose(_Symbol, PERIOD_M5, 1);
   if(atr1 <= 0.0 || close1 <= 0.0)
      return;
   double band = close1 - HLVD_BAND_K * atr1;

   datetime t0 = iTime(_Symbol, PERIOD_M5, 0);
   if(t0 > 0 && t0 != g_lastBar)
   {
      g_lastBar = t0;
      DrawBandSegment(t0, band);
   }

   EodRoll();

   if(g_haveOpen)
   {
      if(PositionSelectByTicket(g_ticket))
      {
         g_floatPnl = PositionGetDouble(POSITION_PROFIT)
                      + PositionGetDouble(POSITION_SWAP);
         if(g_floatPnl < g_floatMae)
            g_floatMae = g_floatPnl;
         if(g_codeManaged)
         {
            double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            if(bid > 0.0 && (bid >= g_tpLevel || bid <= g_slLevel))
            {
               if(ClosePos())
                  FinalizeClosed();
            }
         }
      }
      else
         FinalizeClosed();
   }

   if(!g_haveOpen && t0 > 0 && t0 != g_lastEntryBar && band > 0.0)
   {
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double low0 = iLow(_Symbol, PERIOD_M5, 0);
      double trig = (low0 > 0.0) ? MathMin(ask, low0) : ask;
      if(ask > 0.0 && trig <= band)
      {
         if(TryOpen(band, atr1))
            g_lastEntryBar = t0;
         else if(g_failedBar != t0)
         {
            g_failedBar = t0;
            PrintFormat("HLVD entry failed on %s; will retry next tick",
                        TimeToString(t0, TIME_DATE | TIME_MINUTES));
         }
      }
   }

   Comment(BuildDashboard(SymbolInfoDouble(_Symbol, SYMBOL_BID),
                          SymbolInfoDouble(_Symbol, SYMBOL_ASK),
                          atr0, atr1, band));
}
//+------------------------------------------------------------------+
