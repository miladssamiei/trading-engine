//+------------------------------------------------------------------+
//|                                      ORB_ExecutionSafe_v1.mq5   |
//|      Opening Range Breakout — native, self-contained EA          |
//|      No DLLs, no sockets, no Python bridge, no ML.               |
//+------------------------------------------------------------------+
#property copyright "RR project"
#property version   "1.01"
#property description "ORB execution-safe revalidation copy on US500-tick"
//
// STRATEGY (published rule, exact):
//   Session: 09:30-16:00 New York time (regular trading hours).
//   Opening range (OR): high/low of M5 bars opening 09:30..09:55 NY.
//   Long entry:  buy-stop 0.15% above OR high.  Short entry: sell-stop
//   0.15% below OR low.  Stop loss: opposite side of the OR.
//   Take profit: 0.8% from entry.  Forced close at 16:00 NY.
//   Filter: skip the day if the overnight gap (session open vs prior
//   session 16:00 close) exceeds 1% in either direction.
//   One position at a time; pending orders cancelled at 10:30 NY if
//   untriggered; max one trade per day (OCO entries).
//
// KNOWN SIMPLIFICATION: the published IVTS (implied volatility term
//   structure) filter is intentionally OMITTED — no options data exists
//   for this instrument.
//
// KNOWN SIMPLIFICATION: fixed lot sizing — the volatility-adjusted
//   position sizing overlay is deferred until the base strategy
//   reconciles against the Python reference.
//
// TIMEZONE MODEL (verified):
//   US500-tick bar timestamps are RAW UTC. Evidence: diag_202603
//   bridge bar_ts 1772406000 == 2026-03-01 23:00 UTC, price-matched to
//   the Dukascopy DB reference. TimeCurrent() is therefore UTC during
//   backtests. New York = UTC-5 (EST) / UTC-4 (EDT); US DST begins the
//   2nd Sunday of March 07:00 UTC, ends the 1st Sunday of November
//   06:00 UTC. For Jan-Apr 2026 the switch is 2026-03-08 07:00 UTC:
//   09:30 NY = 14:30 UTC before the switch, 13:30 UTC after.
//
// NOTE: MT5 batch /config .ini cannot override double input parameters
//   (see PROGRESS.md 3.14) — the input defaults below ARE the strategy
//   spec. Int inputs can be overridden via the [Tester] ini section.
//
// OUTPUT FILES (written to the tester agent MQL5\Files sandbox):
//   trades.csv  ticket;open_time;close_time;symbol;type;lot;open_price;
//               stop_loss;take_profit;close_price;exit_reason;swap;
//               commission;profit
//   orb_days.csv ny_date;prev_close;session_open;gap_pct;skipped;
//               skip_reason;range_high;range_low;or_bars;buy_level;
//               sell_level;orders_placed;entry_direction;entry_time;
//               exit_time;exit_reason
//+------------------------------------------------------------------+

#include <Trade\Trade.mqh>

// ── Inputs (defaults ARE the strategy spec) ─────────────────────────
input double InpRangeOffsetPct   = 0.15;     // Entry offset above/below opening range (%)
input double InpTakeProfitPct    = 0.8;      // Take-profit distance from entry (%)
input double InpGapFilterPct     = 1.0;      // Overnight gap skip threshold (%)
input double InpLotSize          = 1.0;      // Fixed lot size (clamped to symbol min/step)
input int    InpSessionOpenHour  = 9;        // NY session open hour
input int    InpSessionOpenMin   = 30;       // NY session open minute
input int    InpRangeEndHour     = 10;       // NY opening-range end hour
input int    InpRangeEndMin      = 0;        // NY opening-range end minute
input int    InpOrderCancelHour  = 10;       // NY pending-order cancel hour
input int    InpOrderCancelMin   = 30;       // NY pending-order cancel minute
input int    InpSessionCloseHour = 16;       // NY forced-close hour
input int    InpSessionCloseMin  = 0;        // NY forced-close minute
input long   InpMagicNumber      = 26082211; // Execution-safe artifact magic

// ── Globals ─────────────────────────────────────────────────────────
CTrade   g_trade;
double   g_lot = 1.0;
datetime g_lastM5Bar = 0;

int      g_nyDate = 0;            // current NY date, YYYYMMDD
double   g_rangeHigh = 0.0, g_rangeLow = 0.0;
int      g_orBars = 0;
bool     g_orHasOpenBar = false;  // the 09:30 bar is part of the OR
bool     g_gapChecked = false, g_skipped = false;
string   g_skipReason = "";
bool     g_ordersPlaced = false, g_entryDone = false, g_cancelDone = false;
ulong    g_buyStopTicket = 0, g_sellStopTicket = 0;
double   g_buyLevel = 0.0, g_sellLevel = 0.0;
double   g_prevClose = 0.0, g_sessionOpen = 0.0, g_gapPct = 0.0;
string   g_entryDir = "";
datetime g_entryTime = 0, g_exitTime = 0;
string   g_exitReasonDay = "";
double   g_posSL = 0.0, g_posTP = 0.0;
bool     g_forceClose = false;    // remains set until the owned position is absent

struct DayRecord
{
   int      nyDate;
   double   prevClose, sessionOpen, gapPct;
   bool     skipped;
   string   skipReason;
   double   rangeHigh, rangeLow;
   int      orBars;
   double   buyLevel, sellLevel;
   bool     ordersPlaced;
   string   entryDirection;
   datetime entryTime, exitTime;
   string   exitReason;
};
DayRecord g_days[];

long     g_exitPosIds[];
string   g_exitReasons[];

//+------------------------------------------------------------------+
//| Time / DST helpers (server time == UTC on US500-tick)             |
//+------------------------------------------------------------------+
datetime NthSundayUtc(const int year, const int month, const int nth, const int hourUtc)
{
   MqlDateTime dt;
   dt.year = year; dt.mon = month; dt.day = 1;
   dt.hour = hourUtc; dt.min = 0; dt.sec = 0;
   datetime first = StructToTime(dt);
   MqlDateTime fd;
   TimeToStruct(first, fd);
   int daysToSunday = (7 - fd.day_of_week) % 7;   // 0 if the 1st is a Sunday
   dt.day = 1 + daysToSunday + 7 * (nth - 1);
   return StructToTime(dt);
}

// NY offset in hours for a UTC timestamp: -4 (EDT) or -5 (EST)
int NyOffsetHours(const datetime utc)
{
   MqlDateTime dt;
   TimeToStruct(utc, dt);
   datetime dstStart = NthSundayUtc(dt.year, 3, 2, 7);    // 2nd Sunday March 07:00 UTC
   datetime dstEnd   = NthSundayUtc(dt.year, 11, 1, 6);   // 1st Sunday Nov   06:00 UTC
   if(utc >= dstStart && utc < dstEnd)
      return -4;
   return -5;
}

// NY wall clock -> UTC (DST transitions never fall inside trading hours)
datetime NyToUtc(const int y, const int mo, const int d, const int h, const int mi)
{
   MqlDateTime dt;
   dt.year = y; dt.mon = mo; dt.day = d; dt.hour = h; dt.min = mi; dt.sec = 0;
   datetime naive = StructToTime(dt);
   return (datetime)((long)naive - (long)NyOffsetHours(naive) * 3600);
}

// NY date (YYYYMMDD) of a UTC timestamp
int NyDateOf(const datetime utc)
{
   MqlDateTime dt;
   TimeToStruct((datetime)((long)utc + (long)NyOffsetHours(utc) * 3600), dt);
   return dt.year * 10000 + dt.mon * 100 + dt.day;
}

datetime NyToUtcOfDate(const int nyDate, const int h, const int mi)
{
   return NyToUtc(nyDate / 10000, (nyDate / 100) % 100, nyDate % 100, h, mi);
}

// Fixed historical-tape safety block from execution_safe_model4_v1.
// Position management remains active on these dates; only new entries stop.
bool IsBlockedSessionDate(const int nyDate)
{
   switch(nyDate)
   {
      case 20210118: case 20210215: case 20210531: case 20210705:
      case 20210906: case 20211125: case 20211126:
      case 20220117: case 20220221: case 20220530: case 20220620:
      case 20220704: case 20220905: case 20221124: case 20221125:
      case 20230116: case 20230220: case 20230529: case 20230619:
      case 20230703: case 20230904: case 20231123: case 20231124:
      case 20240115: case 20240219: case 20240527: case 20240619:
      case 20240703: case 20240704: case 20240902: case 20241128:
      case 20241129: case 20241224:
      case 20250120: case 20250217: case 20250526: case 20250619:
      case 20250703: case 20250704: case 20250901: case 20251127:
      case 20251128: case 20251224:
      case 20260119: case 20260216:
         return true;
   }
   return false;
}

string TsStr(const datetime t)
{
   return TimeToString(t, TIME_DATE | TIME_MINUTES);
}

double NormPrice(const double p)
{
   double tickSz = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   int    digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   if(tickSz <= 0.0)
      tickSz = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(tickSz <= 0.0)
      return NormalizeDouble(p, digits);
   return NormalizeDouble(MathRound(p / tickSz) * tickSz, digits);
}

//+------------------------------------------------------------------+
//| Day bookkeeping                                                   |
//+------------------------------------------------------------------+
int DayIndex(const int nyDate)
{
   for(int i = ArraySize(g_days) - 1; i >= 0; i--)
      if(g_days[i].nyDate == nyDate)
         return i;
   return -1;
}

void StartNewDay(const int nyDate)
{
   int n = ArraySize(g_days);
   ArrayResize(g_days, n + 1);
   g_days[n].nyDate       = nyDate;
   g_days[n].prevClose    = 0.0;
   g_days[n].sessionOpen  = 0.0;
   g_days[n].gapPct       = 0.0;
   g_days[n].skipped      = false;
   g_days[n].skipReason   = "";
   g_days[n].rangeHigh    = 0.0;
   g_days[n].rangeLow     = 0.0;
   g_days[n].orBars       = 0;
   g_days[n].buyLevel     = 0.0;
   g_days[n].sellLevel    = 0.0;
   g_days[n].ordersPlaced = false;
   g_days[n].entryDirection = "";
   g_days[n].entryTime    = 0;
   g_days[n].exitTime     = 0;
   g_days[n].exitReason   = "";

   g_rangeHigh = 0.0;    g_rangeLow = 0.0;
   g_orBars = 0;         g_orHasOpenBar = false;
   g_gapChecked = false; g_skipped = false; g_skipReason = "";
   g_ordersPlaced = false; g_entryDone = false; g_cancelDone = false;
   g_buyStopTicket = 0;  g_sellStopTicket = 0;
   g_buyLevel = 0.0;     g_sellLevel = 0.0;
   g_prevClose = 0.0;    g_sessionOpen = 0.0; g_gapPct = 0.0;
   g_entryDir = "";      g_entryTime = 0; g_exitTime = 0; g_exitReasonDay = "";
   g_posSL = 0.0;        g_posTP = 0.0; g_forceClose = false;
}

void SyncDayRecord()
{
   int di = DayIndex(g_nyDate);
   if(di < 0)
      return;
   g_days[di].prevClose      = g_prevClose;
   g_days[di].sessionOpen    = g_sessionOpen;
   g_days[di].gapPct         = g_gapPct;
   g_days[di].skipped        = g_skipped;
   g_days[di].skipReason     = g_skipReason;
   g_days[di].rangeHigh      = g_rangeHigh;
   g_days[di].rangeLow       = g_rangeLow;
   g_days[di].orBars         = g_orBars;
   g_days[di].buyLevel       = g_buyLevel;
   g_days[di].sellLevel      = g_sellLevel;
   g_days[di].ordersPlaced   = g_ordersPlaced;
   g_days[di].entryDirection = g_entryDir;
   g_days[di].entryTime      = g_entryTime;
   g_days[di].exitTime       = g_exitTime;
   g_days[di].exitReason     = g_exitReasonDay;
}

void AppendExitReason(const long posId, const string reason)
{
   int n = ArraySize(g_exitPosIds);
   ArrayResize(g_exitPosIds, n + 1);
   ArrayResize(g_exitReasons, n + 1);
   g_exitPosIds[n]   = posId;
   g_exitReasons[n]  = reason;
}

string FindExitReason(const long posId)
{
   for(int i = ArraySize(g_exitPosIds) - 1; i >= 0; i--)
      if(g_exitPosIds[i] == posId)
         return g_exitReasons[i];
   return "";
}

//+------------------------------------------------------------------+
//| Prior session close: M5 bar opening exactly 15:55 NY, most recent |
//| prior date that has one (search back up to 10 calendar days)      |
//+------------------------------------------------------------------+
bool GetPriorSessionClose(const int nyDate, double &closeOut, int &priorNyDateOut)
{
   int pd = nyDate;
   for(int back = 0; back < 10; back++)
   {
      // step one calendar day back (build noon naive UTC, minus a day)
      int y = pd / 10000, mo = (pd / 100) % 100, d = pd % 100;
      MqlDateTime dt;
      dt.year = y; dt.mon = mo; dt.day = d; dt.hour = 12; dt.min = 0; dt.sec = 0;
      datetime prev = StructToTime(dt) - 86400;
      TimeToStruct(prev, dt);
      pd = dt.year * 10000 + dt.mon * 100 + dt.day;

      datetime t1555 = NyToUtc(dt.year, dt.mon, dt.day, 15, 55);
      MqlRates rates[];
      int n = CopyRates(_Symbol, PERIOD_M5, t1555 - 60, t1555 + 60, rates);
      if(n <= 0)
         continue;
      for(int i = 0; i < n; i++)
      {
         if(rates[i].time == t1555)
         {
            closeOut        = rates[i].close;
            priorNyDateOut  = pd;
            return true;
         }
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| Position / order helpers                                          |
//+------------------------------------------------------------------+
ulong OurPositionTicket()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0)
         continue;
      if((long)PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
         return tk;
   }
   return 0;
}

bool PendingAlive(const ulong ticket)
{
   if(ticket == 0)
      return false;
   if(!OrderSelect(ticket))
      return false;   // filled / expired / deleted
   return ((ENUM_ORDER_STATE)OrderGetInteger(ORDER_STATE) == ORDER_STATE_PLACED);
}

void CancelPendings(const string why)
{
   if(PendingAlive(g_buyStopTicket))
   {
      if(!g_trade.OrderDelete(g_buyStopTicket))
         Print("ORB cancel buy-stop failed (", why, ") retcode=", g_trade.ResultRetcode());
      else
         Print("ORB cancelled buy-stop (", why, ")");
   }
   if(PendingAlive(g_sellStopTicket))
   {
      if(!g_trade.OrderDelete(g_sellStopTicket))
         Print("ORB cancel sell-stop failed (", why, ") retcode=", g_trade.ResultRetcode());
      else
         Print("ORB cancelled sell-stop (", why, ")");
   }
   g_buyStopTicket = 0;
   g_sellStopTicket = 0;
}
//+------------------------------------------------------------------+
//| Expert initialization                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   g_trade.SetExpertMagicNumber((ulong)InpMagicNumber);
   g_trade.SetDeviationInPoints(50);

   double vmin  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double vstep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double lot   = InpLotSize;
   if(vstep > 0)
      lot = MathFloor(lot / vstep + 0.5) * vstep;
   if(vmin > 0 && lot < vmin)
      lot = vmin;
   g_lot = lot;

   Print("=== ORB_Strategy v1.0 init ===");
   PrintFormat("ORB inputs: offset=%.4f%% tp=%.4f%% gap=%.4f%% lot=%.2f session=%02d:%02d-%02d:%02d NY range_end=%02d:%02d cancel=%02d:%02d magic=%I64d",
               InpRangeOffsetPct, InpTakeProfitPct, InpGapFilterPct, InpLotSize,
               InpSessionOpenHour, InpSessionOpenMin, InpSessionCloseHour, InpSessionCloseMin,
               InpRangeEndHour, InpRangeEndMin, InpOrderCancelHour, InpOrderCancelMin, InpMagicNumber);

   int off = NyOffsetHours(TimeCurrent());
   PrintFormat("ORB NY offset now: %+d h -> server %s = NY %s",
               off, TsStr(TimeCurrent()), TsStr((datetime)((long)TimeCurrent() + (long)off * 3600)));

   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   PrintFormat("ORB DST boundaries %d: start=%s end=%s (UTC)",
               dt.year, TsStr(NthSundayUtc(dt.year, 3, 2, 7)), TsStr(NthSundayUtc(dt.year, 11, 1, 6)));

   int nd = NyDateOf(TimeCurrent());
   PrintFormat("ORB session map today: open=%s range_end=%s cancel=%s close=%s (UTC)",
               TsStr(NyToUtcOfDate(nd, InpSessionOpenHour, InpSessionOpenMin)),
               TsStr(NyToUtcOfDate(nd, InpRangeEndHour, InpRangeEndMin)),
               TsStr(NyToUtcOfDate(nd, InpOrderCancelHour, InpOrderCancelMin)),
               TsStr(NyToUtcOfDate(nd, InpSessionCloseHour, InpSessionCloseMin)));
   PrintFormat("ORB lot: input=%.2f -> trading=%.2f (min=%.2f step=%.2f digits=%d point=%s)",
               InpLotSize, g_lot, vmin, vstep,
               (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS), DoubleToString(_Point, 8));

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Grid-safe explicit close. CTrade::PositionClose submits raw bid/  |
//| ask, which is rejected on this custom symbol when the quote is    |
//| finer than its 0.01 trade grid.                                  |
//+------------------------------------------------------------------+
bool ClosePositionGrid(const ulong ticket, const string why)
{
   ResetLastError();
   if(!PositionSelectByTicket(ticket))
   {
      int errMissing = GetLastError();
      PrintFormat("EXEC_SAFE_CLOSE close attempt strategy=ORB ticket=%I64d why=%s sent=false retcode=0 error=%d accepted=0 comment=position absent",
                  ticket, why, errMissing);
      return false;
   }

   string symbol = PositionGetString(POSITION_SYMBOL);
   long ptype = PositionGetInteger(POSITION_TYPE);
   double volume = PositionGetDouble(POSITION_VOLUME);
   double raw = (ptype == POSITION_TYPE_BUY)
                ? SymbolInfoDouble(symbol, SYMBOL_BID)
                : SymbolInfoDouble(symbol, SYMBOL_ASK);
   double price = NormPrice(raw);

   MqlTradeRequest req;
   MqlTradeResult  res;
   ZeroMemory(req);
   ZeroMemory(res);
   req.action    = TRADE_ACTION_DEAL;
   req.symbol    = symbol;
   req.volume    = volume;
   req.type      = (ptype == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   req.position  = ticket;
   req.price     = price;
   req.deviation = 50;
   req.magic     = (ulong)InpMagicNumber;
   req.comment   = "ORB execution-safe close";

   ResetLastError();
   bool sent = OrderSend(req, res);
   int localError = GetLastError();
   bool accepted = sent && (res.retcode == TRADE_RETCODE_DONE
                            || res.retcode == TRADE_RETCODE_DONE_PARTIAL
                            || res.retcode == TRADE_RETCODE_PLACED);
   PrintFormat("EXEC_SAFE_CLOSE close attempt strategy=ORB ticket=%I64d why=%s raw=%s price=%s sent=%s retcode=%u error=%d accepted=%d comment=%s",
               ticket, why, DoubleToString(raw), DoubleToString(price),
               sent ? "true" : "false", res.retcode, localError,
               accepted ? 1 : 0, res.comment);
   return accepted;
}

//+------------------------------------------------------------------+
//| 16:00 NY forced close — checked on every tick                     |
//+------------------------------------------------------------------+
void ManageForceClose()
{
   ulong ticket = OurPositionTicket();
   if(ticket == 0)
   {
      g_forceClose = false;
      return;
   }
   datetime now = TimeCurrent();
   if(NyDateOf(now) != g_nyDate)
   {
      g_forceClose = true;
      ClosePositionGrid(ticket, "past_midnight");
      return;
   }
   datetime closeUtc = NyToUtcOfDate(g_nyDate, InpSessionCloseHour, InpSessionCloseMin);
   if(now >= closeUtc)
   {
      g_forceClose = true;
      ClosePositionGrid(ticket, "session_end");
   }
}

//+------------------------------------------------------------------+
//| Order placement at the first M5 bar >= 10:00 NY                   |
//+------------------------------------------------------------------+
void PlaceOrders()
{
   double buyLevel  = NormPrice(g_rangeHigh * (1.0 + InpRangeOffsetPct / 100.0));
   double sellLevel = NormPrice(g_rangeLow  * (1.0 - InpRangeOffsetPct / 100.0));
   double buySL     = NormPrice(g_rangeLow);
   double sellSL    = NormPrice(g_rangeHigh);
   double buyTP     = NormPrice(buyLevel  * (1.0 + InpTakeProfitPct / 100.0));
   double sellTP    = NormPrice(sellLevel * (1.0 - InpTakeProfitPct / 100.0));
   g_buyLevel  = buyLevel;
   g_sellLevel = sellLevel;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double stopsDist = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
   datetime expiry = NyToUtcOfDate(g_nyDate, InpOrderCancelHour, InpOrderCancelMin);

   bool buyBeyond  = (ask >= buyLevel  - stopsDist);
   bool sellBeyond = (bid <= sellLevel + stopsDist);
   bool anySuccess = false;

   if(buyBeyond && sellBeyond)
      Print("ORB WARNING: both sides beyond level at placement (ask=", DoubleToString(ask),
            " bid=", DoubleToString(bid), ") — taking BUY side");

   if(buyBeyond)
   {
      if(g_trade.Buy(g_lot, _Symbol, 0.0, buySL, buyTP, "ORB market buy"))
      {
         anySuccess = true;
         PrintFormat("ORB %d BUY at market @ %s sl=%s tp=%s",
                     g_nyDate, DoubleToString(g_trade.ResultPrice()),
                     DoubleToString(buySL), DoubleToString(buyTP));
      }
      else
      {
         Print("ORB market buy failed retcode=", g_trade.ResultRetcode());
         if(g_trade.Buy(g_lot, _Symbol, 0.0, buySL, buyTP, "ORB market buy retry"))
            anySuccess = true;
         else
            Print("ORB market buy retry failed retcode=", g_trade.ResultRetcode());
      }
   }
   else if(sellBeyond)
   {
      if(g_trade.Sell(g_lot, _Symbol, 0.0, sellSL, sellTP, "ORB market sell"))
      {
         anySuccess = true;
         PrintFormat("ORB %d SELL at market @ %s sl=%s tp=%s",
                     g_nyDate, DoubleToString(g_trade.ResultPrice()),
                     DoubleToString(sellSL), DoubleToString(sellTP));
      }
      else
      {
         Print("ORB market sell failed retcode=", g_trade.ResultRetcode());
         if(g_trade.Sell(g_lot, _Symbol, 0.0, sellSL, sellTP, "ORB market sell retry"))
            anySuccess = true;
         else
            Print("ORB market sell retry failed retcode=", g_trade.ResultRetcode());
      }
   }
   else
   {
      bool okB = g_trade.BuyStop(g_lot, buyLevel, _Symbol, buySL, buyTP,
                                 ORDER_TIME_SPECIFIED, expiry, "ORB buy stop");
      uint rcB = g_trade.ResultRetcode();
      if((!okB || (rcB != TRADE_RETCODE_DONE && rcB != TRADE_RETCODE_PLACED))
         && rcB == TRADE_RETCODE_INVALID_EXPIRATION)
      {
         okB = g_trade.BuyStop(g_lot, buyLevel, _Symbol, buySL, buyTP,
                               ORDER_TIME_GTC, 0, "ORB buy stop");
         rcB = g_trade.ResultRetcode();
      }
      if(okB && (rcB == TRADE_RETCODE_DONE || rcB == TRADE_RETCODE_PLACED))
      {
         anySuccess = true;
         g_buyStopTicket = g_trade.ResultOrder();
      }
      else
         Print("ORB buy-stop failed retcode=", rcB);

      bool okS = g_trade.SellStop(g_lot, sellLevel, _Symbol, sellSL, sellTP,
                                  ORDER_TIME_SPECIFIED, expiry, "ORB sell stop");
      uint rcS = g_trade.ResultRetcode();
      if((!okS || (rcS != TRADE_RETCODE_DONE && rcS != TRADE_RETCODE_PLACED))
         && rcS == TRADE_RETCODE_INVALID_EXPIRATION)
      {
         okS = g_trade.SellStop(g_lot, sellLevel, _Symbol, sellSL, sellTP,
                                ORDER_TIME_GTC, 0, "ORB sell stop");
         rcS = g_trade.ResultRetcode();
      }
      if(okS && (rcS == TRADE_RETCODE_DONE || rcS == TRADE_RETCODE_PLACED))
      {
         anySuccess = true;
         g_sellStopTicket = g_trade.ResultOrder();
      }
      else
         Print("ORB sell-stop failed retcode=", rcS);
   }

   if(anySuccess)
      g_ordersPlaced = true;

   PrintFormat("ORB %d orders: buy=%s sl=%s tp=%s | sell=%s sl=%s tp=%s (OR %s/%s, %d bars)",
               g_nyDate, DoubleToString(buyLevel), DoubleToString(buySL), DoubleToString(buyTP),
               DoubleToString(sellLevel), DoubleToString(sellSL), DoubleToString(sellTP),
               DoubleToString(g_rangeHigh), DoubleToString(g_rangeLow), g_orBars);
}

//+------------------------------------------------------------------+
//| New M5 bar state machine                                          |
//+------------------------------------------------------------------+
void OnNewM5Bar(const datetime t0)
{
   int nyDate = NyDateOf(t0);
   if(nyDate != g_nyDate)
   {
      g_nyDate = nyDate;
      StartNewDay(nyDate);
      PrintFormat("ORB new day %d (offset %+d h)", nyDate, NyOffsetHours(t0));
   }
   int di = DayIndex(nyDate);
   if(di < 0)
      return;
   if(IsBlockedSessionDate(nyDate) && !g_skipped)
   {
      g_skipped = true;
      g_skipReason = "nonstandard_session";
      PrintFormat("ORB %d SKIP: fixed nonstandard-session safety block", nyDate);
   }

   MqlDateTime ndt;
   TimeToStruct((datetime)((long)t0 + (long)NyOffsetHours(t0) * 3600), ndt);
   int nyMin = ndt.hour * 60 + ndt.min;
   int sessOpenMin = InpSessionOpenHour * 60 + InpSessionOpenMin;
   int rangeEndMin = InpRangeEndHour * 60 + InpRangeEndMin;
   int cancelMin   = InpOrderCancelHour * 60 + InpOrderCancelMin;

   if(!g_gapChecked && nyMin >= sessOpenMin && nyMin < rangeEndMin)
   {
      g_gapChecked = true;
      g_sessionOpen = iOpen(_Symbol, PERIOD_M5, 0);
      double pc = 0.0;
      int pd = 0;
      if(GetPriorSessionClose(nyDate, pc, pd))
      {
         g_prevClose = pc;
         if(pc > 0)
            g_gapPct = MathAbs(g_sessionOpen / pc - 1.0) * 100.0;
         if(g_gapPct > InpGapFilterPct)
         {
            g_skipped = true;
            g_skipReason = "gap_filter";
         }
      }
      else
      {
         g_skipped = true;
         g_skipReason = "no_prior_close";
      }
      PrintFormat("ORB %d gap check: session_open=%s prev_close=%s gap=%.4f%% -> %s",
                  nyDate, DoubleToString(g_sessionOpen), DoubleToString(g_prevClose),
                  g_gapPct, g_skipped ? "SKIP (" + g_skipReason + ")" : "OK");
   }

   datetime cTime = iTime(_Symbol, PERIOD_M5, 1);
   if(cTime > 0)
   {
      int cDate = NyDateOf(cTime);
      MqlDateTime cdt;
      TimeToStruct((datetime)((long)cTime + (long)NyOffsetHours(cTime) * 3600), cdt);
      int cMin = cdt.hour * 60 + cdt.min;
      if(cDate == nyDate && cMin >= sessOpenMin && cMin < rangeEndMin)
      {
         double h = iHigh(_Symbol, PERIOD_M5, 1);
         double l = iLow(_Symbol, PERIOD_M5, 1);
         if(g_orBars == 0)
         {
            g_rangeHigh = h;
            g_rangeLow  = l;
         }
         else
         {
            if(h > g_rangeHigh) g_rangeHigh = h;
            if(l < g_rangeLow)  g_rangeLow  = l;
         }
         if(cMin == sessOpenMin)
            g_orHasOpenBar = true;
         g_orBars++;
      }
   }

   if(!g_skipped && !g_ordersPlaced && !g_entryDone && nyMin >= rangeEndMin && nyMin < cancelMin
      && OurPositionTicket() == 0)
   {
      if(!g_orHasOpenBar)
      {
         g_skipped = true;
         g_skipReason = "no_session_open";
         PrintFormat("ORB %d SKIP: no 09:30 bar in opening range", nyDate);
      }
      else if(g_orBars < 5)
      {
         g_skipped = true;
         g_skipReason = "incomplete_range";
         PrintFormat("ORB %d SKIP: only %d opening-range bars", nyDate, g_orBars);
      }
      else
         PlaceOrders();
   }

   if(!g_cancelDone && g_ordersPlaced && !g_entryDone && nyMin >= cancelMin)
   {
      g_cancelDone = true;
      CancelPendings("10:30 no trigger");
      PrintFormat("ORB %d 10:30 cancel: no trigger", nyDate);
   }

   SyncDayRecord();
}

//+------------------------------------------------------------------+
//| Expert tick                                                       |
//+------------------------------------------------------------------+
void OnTick()
{
   ManageForceClose();

   datetime t0 = iTime(_Symbol, PERIOD_M5, 0);
   if(t0 == 0)
      return;
   if(t0 == g_lastM5Bar)
      return;
   g_lastM5Bar = t0;
   OnNewM5Bar(t0);
}
//+------------------------------------------------------------------+
//| Exit-reason fallback for deals whose transaction was not caught   |
//+------------------------------------------------------------------+
string FallbackReason(const double closePrice, const datetime closeTime, const double sl, const double tp)
{
   int nd = NyDateOf(closeTime);
   datetime closeUtc = NyToUtcOfDate(nd, InpSessionCloseHour, InpSessionCloseMin);
   long d = (long)closeTime - (long)closeUtc;
   if(d < 0) d = -d;
   if(d <= 600)
      return "session_end";
   double tol = MathMax(10.0 * _Point, 0.05);
   if(tp > 0 && MathAbs(closePrice - tp) <= tol)
      return "take_profit";
   if(sl > 0 && MathAbs(closePrice - sl) <= tol)
      return "stop_loss";
   return "other";
}

//+------------------------------------------------------------------+
//| Trade transactions: OCO + exit-reason capture                     |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
{
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD)
      return;
   if(!HistoryDealSelect(trans.deal))
      return;
   if((long)HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != InpMagicNumber)
      return;

   ENUM_DEAL_ENTRY de = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
   long posId = (long)HistoryDealGetInteger(trans.deal, DEAL_POSITION_ID);

   if(de == DEAL_ENTRY_IN)
   {
      g_entryDone = true;
      g_entryDir = ((ENUM_DEAL_TYPE)HistoryDealGetInteger(trans.deal, DEAL_TYPE) == DEAL_TYPE_BUY) ? "BUY" : "SELL";
      g_entryTime = (datetime)HistoryDealGetInteger(trans.deal, DEAL_TIME);
      g_posSL = HistoryDealGetDouble(trans.deal, DEAL_SL);
      g_posTP = HistoryDealGetDouble(trans.deal, DEAL_TP);
      PrintFormat("ORB entry: pos=%I64d %s @ %s sl=%s tp=%s",
                  posId, g_entryDir, DoubleToString(HistoryDealGetDouble(trans.deal, DEAL_PRICE)),
                  DoubleToString(g_posSL), DoubleToString(g_posTP));
      CancelPendings("OCO");
      SyncDayRecord();
   }
   else if(de == DEAL_ENTRY_OUT || de == DEAL_ENTRY_OUT_BY || de == DEAL_ENTRY_INOUT)
   {
      double closePrice = HistoryDealGetDouble(trans.deal, DEAL_PRICE);
      string reason = "other";
      long dealReason = HistoryDealGetInteger(trans.deal, DEAL_REASON);
      if(dealReason == DEAL_REASON_SL)
         reason = "stop_loss";
      else if(dealReason == DEAL_REASON_TP)
         reason = "take_profit";
      else if(g_forceClose)
         reason = "session_end";
      else
      {
         double tol = MathMax(10.0 * _Point, 0.05);
         if(g_posTP > 0 && MathAbs(closePrice - g_posTP) <= tol)
            reason = "take_profit";
         else if(g_posSL > 0 && MathAbs(closePrice - g_posSL) <= tol)
            reason = "stop_loss";
      }
      g_exitTime = (datetime)HistoryDealGetInteger(trans.deal, DEAL_TIME);
      g_exitReasonDay = reason;
      AppendExitReason(posId, reason);
      PrintFormat("EXEC_SAFE_EXIT strategy=ORB pos=%I64d deal_reason=%d reason=%s force_active=%s",
                  posId, dealReason, reason, g_forceClose ? "true" : "false");
      if(OurPositionTicket() == 0)
         g_forceClose = false;
      PrintFormat("ORB exit: pos=%I64d reason=%s @ %s", posId, reason, DoubleToString(closePrice));
      SyncDayRecord();
   }
}

//+------------------------------------------------------------------+
//| trades.csv — same schema family as the RR result collector        |
//| (semicolon-delimited, direction from the ENTRY deal — see         |
//|  PROGRESS.md 3.11/3.12)                                           |
//+------------------------------------------------------------------+
void WriteTradesCsv()
{
   int handle = FileOpen("trades.csv", FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(handle == INVALID_HANDLE)
   {
      Print("WARNING: could not open trades.csv for writing. Error=", GetLastError());
      return;
   }
   string header = "ticket;open_time;close_time;symbol;type;lot;open_price;stop_loss;take_profit;close_price;exit_reason;swap;commission;profit\r\n";
   FileWriteString(handle, header, StringLen(header));

   if(!HistorySelect(0, TimeCurrent()))
   {
      Print("HistorySelect failed");
      FileClose(handle);
      return;
   }

   int total = HistoryDealsTotal();
   int written = 0;

   for(int i = 0; i < total; i++)
   {
      ulong outTicket = HistoryDealGetTicket(i);
      if(outTicket == 0)
         continue;
      if((long)HistoryDealGetInteger(outTicket, DEAL_MAGIC) != InpMagicNumber)
         continue;
      ENUM_DEAL_ENTRY outEntry = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(outTicket, DEAL_ENTRY);
      if(outEntry != DEAL_ENTRY_OUT && outEntry != DEAL_ENTRY_OUT_BY && outEntry != DEAL_ENTRY_INOUT)
         continue;

      long posId = (long)HistoryDealGetInteger(outTicket, DEAL_POSITION_ID);

      ulong inTicket = 0;
      double openPrice = 0.0;
      datetime openTime = 0;
      long inDealType = -1;
      double inSL = 0.0, inTP = 0.0, inComm = 0.0;

      for(int j = 0; j < total; j++)
      {
         ulong dt = HistoryDealGetTicket(j);
         if(dt == 0 || dt == outTicket)
            continue;
         if((long)HistoryDealGetInteger(dt, DEAL_MAGIC) != InpMagicNumber)
            continue;
         if((ENUM_DEAL_ENTRY)HistoryDealGetInteger(dt, DEAL_ENTRY) != DEAL_ENTRY_IN)
            continue;
         if((long)HistoryDealGetInteger(dt, DEAL_POSITION_ID) != posId)
            continue;
         inTicket = dt;
         openPrice = HistoryDealGetDouble(dt, DEAL_PRICE);
         openTime = (datetime)HistoryDealGetInteger(dt, DEAL_TIME);
         inDealType = HistoryDealGetInteger(dt, DEAL_TYPE);
         inSL = HistoryDealGetDouble(dt, DEAL_SL);
         inTP = HistoryDealGetDouble(dt, DEAL_TP);
         inComm = HistoryDealGetDouble(dt, DEAL_COMMISSION);
         break;
      }
      if(inTicket == 0)
         continue;

      datetime closeTime = (datetime)HistoryDealGetInteger(outTicket, DEAL_TIME);
      double closePrice = HistoryDealGetDouble(outTicket, DEAL_PRICE);
      double volume = HistoryDealGetDouble(inTicket, DEAL_VOLUME);
      double swapVal = HistoryDealGetDouble(outTicket, DEAL_SWAP);
      double commVal = inComm + HistoryDealGetDouble(outTicket, DEAL_COMMISSION);
      double profitVal = HistoryDealGetDouble(outTicket, DEAL_PROFIT);
      string dir = (inDealType == DEAL_TYPE_BUY) ? "BUY" : "SELL";
      string dealSymbol = HistoryDealGetString(outTicket, DEAL_SYMBOL);

      string reason = FindExitReason(posId);
      if(reason == "")
         reason = FallbackReason(closePrice, closeTime, inSL, inTP);

      string line = IntegerToString((long)inTicket) + ";" +
                    TsStr(openTime) + ";" +
                    TsStr(closeTime) + ";" +
                    dealSymbol + ";" +
                    dir + ";" +
                    DoubleToString(volume, 2) + ";" +
                    DoubleToString(openPrice) + ";" +
                    DoubleToString(inSL) + ";" +
                    DoubleToString(inTP) + ";" +
                    DoubleToString(closePrice) + ";" +
                    reason + ";" +
                    DoubleToString(swapVal, 2) + ";" +
                    DoubleToString(commVal, 2) + ";" +
                    DoubleToString(profitVal, 2) + "\r\n";
      FileWriteString(handle, line, StringLen(line));
      written++;
   }

   FileClose(handle);
   Print("Trade CSV written: trades.csv (", written, " trades)");
}

//+------------------------------------------------------------------+
//| orb_days.csv — per-day state for Python-vs-MT5 reconciliation     |
//+------------------------------------------------------------------+
void WriteDaysCsv()
{
   int handle = FileOpen("orb_days.csv", FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(handle == INVALID_HANDLE)
   {
      Print("WARNING: could not open orb_days.csv for writing. Error=", GetLastError());
      return;
   }
   string header = "ny_date;prev_close;session_open;gap_pct;skipped;skip_reason;range_high;range_low;or_bars;buy_level;sell_level;orders_placed;entry_direction;entry_time;exit_time;exit_reason\r\n";
   FileWriteString(handle, header, StringLen(header));

   int n = ArraySize(g_days);
   for(int i = 0; i < n; i++)
   {
      string line = IntegerToString(g_days[i].nyDate) + ";" +
                    DoubleToString(g_days[i].prevClose, 4) + ";" +
                    DoubleToString(g_days[i].sessionOpen, 4) + ";" +
                    DoubleToString(g_days[i].gapPct, 4) + ";" +
                    (g_days[i].skipped ? "1" : "0") + ";" +
                    g_days[i].skipReason + ";" +
                    DoubleToString(g_days[i].rangeHigh, 4) + ";" +
                    DoubleToString(g_days[i].rangeLow, 4) + ";" +
                    IntegerToString(g_days[i].orBars) + ";" +
                    DoubleToString(g_days[i].buyLevel, 4) + ";" +
                    DoubleToString(g_days[i].sellLevel, 4) + ";" +
                    (g_days[i].ordersPlaced ? "1" : "0") + ";" +
                    g_days[i].entryDirection + ";" +
                    (g_days[i].entryTime > 0 ? TsStr(g_days[i].entryTime) : "") + ";" +
                    (g_days[i].exitTime > 0 ? TsStr(g_days[i].exitTime) : "") + ";" +
                    g_days[i].exitReason + "\r\n";
      FileWriteString(handle, line, StringLen(line));
   }

   FileClose(handle);
   Print("Day CSV written: orb_days.csv (", n, " days)");
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   PrintFormat("EXEC_SAFE_DEINIT strategy=ORB flat=%d owned_positions=%d reason=%d",
               OurPositionTicket() == 0 ? 1 : 0, OurPositionTicket() == 0 ? 0 : 1, reason);
   WriteTradesCsv();
   WriteDaysCsv();
}
//+------------------------------------------------------------------+
