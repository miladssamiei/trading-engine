//+------------------------------------------------------------------+
//|                                               ORB_Strategy.mq5   |
//|      Opening Range Breakout — native, self-contained EA          |
//|      No DLLs, no sockets, no Python bridge, no ML.               |
//+------------------------------------------------------------------+
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


// ── Inputs (defaults ARE the strategy spec) ─────────────────────────
input double ORB_InpRangeOffsetPct   = 0.15;     // Entry offset above/below opening range (%)
input double ORB_InpTakeProfitPct    = 0.8;      // Take-profit distance from entry (%)
input double ORB_InpGapFilterPct     = 1.0;      // Overnight gap skip threshold (%)
input double ORB_InpLotSize          = 1.0;      // Fixed lot size (clamped to symbol min/step)
input int    ORB_InpSessionOpenHour  = 9;        // NY session open hour
input int    ORB_InpSessionOpenMin   = 30;       // NY session open minute
input int    ORB_InpRangeEndHour     = 10;       // NY opening-range end hour
input int    ORB_InpRangeEndMin      = 0;        // NY opening-range end minute
input int    ORB_InpOrderCancelHour  = 10;       // NY pending-order cancel hour
input int    ORB_InpOrderCancelMin   = 30;       // NY pending-order cancel minute
input int    ORB_InpSessionCloseHour = 16;       // NY forced-close hour
input int    ORB_InpSessionCloseMin  = 0;        // NY forced-close minute
input long   ORB_InpMagicNumber      = 26082201; // Magic number

// ── Globals ─────────────────────────────────────────────────────────
CTrade   ORB_g_trade;
double   ORB_g_lot = 1.0;
datetime ORB_g_lastM5Bar = 0;

int      ORB_g_nyDate = 0;            // current NY date, YYYYMMDD
double   ORB_g_rangeHigh = 0.0, g_rangeLow = 0.0;
int      ORB_g_orBars = 0;
bool     ORB_g_orHasOpenBar = false;  // the 09:30 bar is part of the OR
bool     ORB_g_gapChecked = false, g_skipped = false;
string   ORB_g_skipReason = "";
bool     ORB_g_ordersPlaced = false, g_entryDone = false, g_cancelDone = false;
ulong    ORB_g_buyStopTicket = 0, g_sellStopTicket = 0;
double   ORB_g_buyLevel = 0.0, g_sellLevel = 0.0;
double   ORB_g_prevClose = 0.0, g_sessionOpen = 0.0, g_gapPct = 0.0;
string   ORB_g_entryDir = "";
datetime ORB_g_entryTime = 0, g_exitTime = 0;
string   ORB_g_exitReasonDay = "";
double   ORB_g_posSL = 0.0, g_posTP = 0.0;
bool     ORB_g_forceClose = false;    // set right before the 16:00 market close

struct ORB_DayRecord
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
ORB_DayRecord ORB_g_days[];

long     ORB_g_exitPosIds[];
string   ORB_g_exitReasons[];

//+------------------------------------------------------------------+
//| Time / DST helpers (server time == UTC on US500-tick)             |
//+------------------------------------------------------------------+
datetime ORB_NthSundayUtc(const int year, const int month, const int nth, const int hourUtc)
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
int ORB_NyOffsetHours(const datetime utc)
{
   MqlDateTime dt;
   TimeToStruct(utc, dt);
   datetime dstStart = ORB_NthSundayUtc(dt.year, 3, 2, 7);    // 2nd Sunday March 07:00 UTC
   datetime dstEnd   = ORB_NthSundayUtc(dt.year, 11, 1, 6);   // 1st Sunday Nov   06:00 UTC
   if(utc >= dstStart && utc < dstEnd)
      return -4;
   return -5;
}

// NY wall clock -> UTC (DST transitions never fall inside trading hours)
datetime ORB_NyToUtc(const int y, const int mo, const int d, const int h, const int mi)
{
   MqlDateTime dt;
   dt.year = y; dt.mon = mo; dt.day = d; dt.hour = h; dt.min = mi; dt.sec = 0;
   datetime naive = StructToTime(dt);
   return (datetime)((long)naive - (long)ORB_NyOffsetHours(naive) * 3600);
}

// NY date (YYYYMMDD) of a UTC timestamp
int ORB_NyDateOf(const datetime utc)
{
   MqlDateTime dt;
   TimeToStruct((datetime)((long)utc + (long)ORB_NyOffsetHours(utc) * 3600), dt);
   return dt.year * 10000 + dt.mon * 100 + dt.day;
}

datetime ORB_NyToUtcOfDate(const int nyDate, const int h, const int mi)
{
   return ORB_NyToUtc(nyDate / 10000, (nyDate / 100) % 100, nyDate % 100, h, mi);
}

string ORB_TsStr(const datetime t)
{
   return TimeToString(t, TIME_DATE | TIME_MINUTES);
}

double ORB_NormPrice(const double p)
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
int ORB_DayIndex(const int nyDate)
{
   for(int i = ArraySize(ORB_g_days) - 1; i >= 0; i--)
      if(ORB_g_days[i].nyDate == nyDate)
         return i;
   return -1;
}

void ORB_StartNewDay(const int nyDate)
{
   int n = ArraySize(ORB_g_days);
   ArrayResize(ORB_g_days, n + 1);
   ORB_g_days[n].nyDate       = nyDate;
   ORB_g_days[n].prevClose    = 0.0;
   ORB_g_days[n].sessionOpen  = 0.0;
   ORB_g_days[n].gapPct       = 0.0;
   ORB_g_days[n].skipped      = false;
   ORB_g_days[n].skipReason   = "";
   ORB_g_days[n].rangeHigh    = 0.0;
   ORB_g_days[n].rangeLow     = 0.0;
   ORB_g_days[n].orBars       = 0;
   ORB_g_days[n].buyLevel     = 0.0;
   ORB_g_days[n].sellLevel    = 0.0;
   ORB_g_days[n].ordersPlaced = false;
   ORB_g_days[n].entryDirection = "";
   ORB_g_days[n].entryTime    = 0;
   ORB_g_days[n].exitTime     = 0;
   ORB_g_days[n].exitReason   = "";

   ORB_g_rangeHigh = 0.0;    g_rangeLow = 0.0;
   ORB_g_orBars = 0;         ORB_g_orHasOpenBar = false;
   ORB_g_gapChecked = false; g_skipped = false; ORB_g_skipReason = "";
   ORB_g_ordersPlaced = false; g_entryDone = false; g_cancelDone = false;
   ORB_g_buyStopTicket = 0;  g_sellStopTicket = 0;
   ORB_g_buyLevel = 0.0;     g_sellLevel = 0.0;
   ORB_g_prevClose = 0.0;    g_sessionOpen = 0.0; g_gapPct = 0.0;
   ORB_g_entryDir = "";      ORB_g_entryTime = 0; g_exitTime = 0; ORB_g_exitReasonDay = "";
   ORB_g_posSL = 0.0;        g_posTP = 0.0; ORB_g_forceClose = false;
}

void ORB_SyncDayRecord()
{
   int di = ORB_DayIndex(ORB_g_nyDate);
   if(di < 0)
      return;
   ORB_g_days[di].prevClose      = ORB_g_prevClose;
   ORB_g_days[di].sessionOpen    = g_sessionOpen;
   ORB_g_days[di].gapPct         = g_gapPct;
   ORB_g_days[di].skipped        = g_skipped;
   ORB_g_days[di].skipReason     = ORB_g_skipReason;
   ORB_g_days[di].rangeHigh      = ORB_g_rangeHigh;
   ORB_g_days[di].rangeLow       = g_rangeLow;
   ORB_g_days[di].orBars         = ORB_g_orBars;
   ORB_g_days[di].buyLevel       = ORB_g_buyLevel;
   ORB_g_days[di].sellLevel      = g_sellLevel;
   ORB_g_days[di].ordersPlaced   = ORB_g_ordersPlaced;
   ORB_g_days[di].entryDirection = ORB_g_entryDir;
   ORB_g_days[di].entryTime      = ORB_g_entryTime;
   ORB_g_days[di].exitTime       = g_exitTime;
   ORB_g_days[di].exitReason     = ORB_g_exitReasonDay;
}

void ORB_AppendExitReason(const long posId, const string reason)
{
   int n = ArraySize(ORB_g_exitPosIds);
   ArrayResize(ORB_g_exitPosIds, n + 1);
   ArrayResize(ORB_g_exitReasons, n + 1);
   ORB_g_exitPosIds[n]   = posId;
   ORB_g_exitReasons[n]  = reason;
}

string ORB_FindExitReason(const long posId)
{
   for(int i = ArraySize(ORB_g_exitPosIds) - 1; i >= 0; i--)
      if(ORB_g_exitPosIds[i] == posId)
         return ORB_g_exitReasons[i];
   return "";
}

//+------------------------------------------------------------------+
//| Prior session close: M5 bar opening exactly 15:55 NY, most recent |
//| prior date that has one (search back up to 10 calendar days)      |
//+------------------------------------------------------------------+
bool ORB_GetPriorSessionClose(const int nyDate, double &closeOut, int &priorNyDateOut)
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

      datetime t1555 = ORB_NyToUtc(dt.year, dt.mon, dt.day, 15, 55);
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
ulong ORB_OurPositionTicket()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0)
         continue;
      if((long)PositionGetInteger(POSITION_MAGIC) == ORB_InpMagicNumber)
         return tk;
   }
   return 0;
}

bool ORB_PendingAlive(const ulong ticket)
{
   if(ticket == 0)
      return false;
   if(!OrderSelect(ticket))
      return false;   // filled / expired / deleted
   return ((ENUM_ORDER_STATE)OrderGetInteger(ORDER_STATE) == ORDER_STATE_PLACED);
}

void ORB_CancelPendings(const string why)
{
   if(ORB_PendingAlive(ORB_g_buyStopTicket))
   {
      if(!ORB_g_trade.OrderDelete(ORB_g_buyStopTicket))
         Print("ORB cancel buy-stop failed (", why, ") retcode=", ORB_g_trade.ResultRetcode());
      else
         Print("ORB cancelled buy-stop (", why, ")");
   }
   if(ORB_PendingAlive(g_sellStopTicket))
   {
      if(!ORB_g_trade.OrderDelete(g_sellStopTicket))
         Print("ORB cancel sell-stop failed (", why, ") retcode=", ORB_g_trade.ResultRetcode());
      else
         Print("ORB cancelled sell-stop (", why, ")");
   }
   ORB_g_buyStopTicket = 0;
   g_sellStopTicket = 0;
}
//+------------------------------------------------------------------+
//| Expert initialization                                             |
//+------------------------------------------------------------------+
int ORB_OnInit()
{
   ORB_g_trade.SetExpertMagicNumber((ulong)ORB_InpMagicNumber);
   ORB_g_trade.SetDeviationInPoints(50);

   double vmin  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double vstep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double lot   = ORB_InpLotSize;
   if(vstep > 0)
      lot = MathFloor(lot / vstep + 0.5) * vstep;
   if(vmin > 0 && lot < vmin)
      lot = vmin;
   ORB_g_lot = lot;

   Print("=== ORB_Strategy v1.0 init ===");
   PrintFormat("ORB inputs: offset=%.4f%% tp=%.4f%% gap=%.4f%% lot=%.2f session=%02d:%02d-%02d:%02d NY range_end=%02d:%02d cancel=%02d:%02d magic=%I64d",
               ORB_InpRangeOffsetPct, ORB_InpTakeProfitPct, ORB_InpGapFilterPct, ORB_InpLotSize,
               ORB_InpSessionOpenHour, ORB_InpSessionOpenMin, ORB_InpSessionCloseHour, ORB_InpSessionCloseMin,
               ORB_InpRangeEndHour, ORB_InpRangeEndMin, ORB_InpOrderCancelHour, ORB_InpOrderCancelMin, ORB_InpMagicNumber);

   int off = ORB_NyOffsetHours(TimeCurrent());
   PrintFormat("ORB NY offset now: %+d h -> server %s = NY %s",
               off, ORB_TsStr(TimeCurrent()), ORB_TsStr((datetime)((long)TimeCurrent() + (long)off * 3600)));

   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   PrintFormat("ORB DST boundaries %d: start=%s end=%s (UTC)",
               dt.year, ORB_TsStr(ORB_NthSundayUtc(dt.year, 3, 2, 7)), ORB_TsStr(ORB_NthSundayUtc(dt.year, 11, 1, 6)));

   int nd = ORB_NyDateOf(TimeCurrent());
   PrintFormat("ORB session map today: open=%s range_end=%s cancel=%s close=%s (UTC)",
               ORB_TsStr(ORB_NyToUtcOfDate(nd, ORB_InpSessionOpenHour, ORB_InpSessionOpenMin)),
               ORB_TsStr(ORB_NyToUtcOfDate(nd, ORB_InpRangeEndHour, ORB_InpRangeEndMin)),
               ORB_TsStr(ORB_NyToUtcOfDate(nd, ORB_InpOrderCancelHour, ORB_InpOrderCancelMin)),
               ORB_TsStr(ORB_NyToUtcOfDate(nd, ORB_InpSessionCloseHour, ORB_InpSessionCloseMin)));
   PrintFormat("ORB lot: input=%.2f -> trading=%.2f (min=%.2f step=%.2f digits=%d point=%s)",
               ORB_InpLotSize, ORB_g_lot, vmin, vstep,
               (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS), DoubleToString(_Point, 8));

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| 16:00 NY forced close — checked on every tick                     |
//+------------------------------------------------------------------+
void ORB_ManageForceClose()
{
   ulong ticket = ORB_OurPositionTicket();
   if(ticket == 0)
      return;
   datetime now = TimeCurrent();
   if(ORB_NyDateOf(now) != ORB_g_nyDate)
   {
      ORB_g_forceClose = true;
      if(!ORB_g_trade.PositionClose(ticket))
         Print("ORB force-close failed (past midnight) retcode=", ORB_g_trade.ResultRetcode());
      return;
   }
   datetime closeUtc = ORB_NyToUtcOfDate(ORB_g_nyDate, ORB_InpSessionCloseHour, ORB_InpSessionCloseMin);
   if(now >= closeUtc)
   {
      ORB_g_forceClose = true;
      if(!ORB_g_trade.PositionClose(ticket))
         Print("ORB force-close failed retcode=", ORB_g_trade.ResultRetcode());
   }
}

//+------------------------------------------------------------------+
//| Order placement at the first M5 bar >= 10:00 NY                   |
//+------------------------------------------------------------------+
void ORB_PlaceOrders()
{
   double buyLevel  = ORB_NormPrice(ORB_g_rangeHigh * (1.0 + ORB_InpRangeOffsetPct / 100.0));
   double sellLevel = ORB_NormPrice(g_rangeLow  * (1.0 - ORB_InpRangeOffsetPct / 100.0));
   double buySL     = ORB_NormPrice(g_rangeLow);
   double sellSL    = ORB_NormPrice(ORB_g_rangeHigh);
   double buyTP     = ORB_NormPrice(buyLevel  * (1.0 + ORB_InpTakeProfitPct / 100.0));
   double sellTP    = ORB_NormPrice(sellLevel * (1.0 - ORB_InpTakeProfitPct / 100.0));
   ORB_g_buyLevel  = buyLevel;
   g_sellLevel = sellLevel;

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double stopsDist = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
   datetime expiry = ORB_NyToUtcOfDate(ORB_g_nyDate, ORB_InpOrderCancelHour, ORB_InpOrderCancelMin);

   bool buyBeyond  = (ask >= buyLevel  - stopsDist);
   bool sellBeyond = (bid <= sellLevel + stopsDist);
   bool anySuccess = false;

   if(buyBeyond && sellBeyond)
      Print("ORB WARNING: both sides beyond level at placement (ask=", DoubleToString(ask),
            " bid=", DoubleToString(bid), ") — taking BUY side");

   if(buyBeyond)
   {
      if(ORB_g_trade.Buy(ORB_g_lot, _Symbol, 0.0, buySL, buyTP, "ORB market buy"))
      {
         anySuccess = true;
         PrintFormat("ORB %d BUY at market @ %s sl=%s tp=%s",
                     ORB_g_nyDate, DoubleToString(ORB_g_trade.ResultPrice()),
                     DoubleToString(buySL), DoubleToString(buyTP));
      }
      else
      {
         Print("ORB market buy failed retcode=", ORB_g_trade.ResultRetcode());
         if(ORB_g_trade.Buy(ORB_g_lot, _Symbol, 0.0, buySL, buyTP, "ORB market buy retry"))
            anySuccess = true;
         else
            Print("ORB market buy retry failed retcode=", ORB_g_trade.ResultRetcode());
      }
   }
   else if(sellBeyond)
   {
      if(ORB_g_trade.Sell(ORB_g_lot, _Symbol, 0.0, sellSL, sellTP, "ORB market sell"))
      {
         anySuccess = true;
         PrintFormat("ORB %d SELL at market @ %s sl=%s tp=%s",
                     ORB_g_nyDate, DoubleToString(ORB_g_trade.ResultPrice()),
                     DoubleToString(sellSL), DoubleToString(sellTP));
      }
      else
      {
         Print("ORB market sell failed retcode=", ORB_g_trade.ResultRetcode());
         if(ORB_g_trade.Sell(ORB_g_lot, _Symbol, 0.0, sellSL, sellTP, "ORB market sell retry"))
            anySuccess = true;
         else
            Print("ORB market sell retry failed retcode=", ORB_g_trade.ResultRetcode());
      }
   }
   else
   {
      bool okB = ORB_g_trade.BuyStop(ORB_g_lot, buyLevel, _Symbol, buySL, buyTP,
                                 ORDER_TIME_SPECIFIED, expiry, "ORB buy stop");
      uint rcB = ORB_g_trade.ResultRetcode();
      if((!okB || (rcB != TRADE_RETCODE_DONE && rcB != TRADE_RETCODE_PLACED))
         && rcB == TRADE_RETCODE_INVALID_EXPIRATION)
      {
         okB = ORB_g_trade.BuyStop(ORB_g_lot, buyLevel, _Symbol, buySL, buyTP,
                               ORDER_TIME_GTC, 0, "ORB buy stop");
         rcB = ORB_g_trade.ResultRetcode();
      }
      if(okB && (rcB == TRADE_RETCODE_DONE || rcB == TRADE_RETCODE_PLACED))
      {
         anySuccess = true;
         ORB_g_buyStopTicket = ORB_g_trade.ResultOrder();
      }
      else
         Print("ORB buy-stop failed retcode=", rcB);

      bool okS = ORB_g_trade.SellStop(ORB_g_lot, sellLevel, _Symbol, sellSL, sellTP,
                                  ORDER_TIME_SPECIFIED, expiry, "ORB sell stop");
      uint rcS = ORB_g_trade.ResultRetcode();
      if((!okS || (rcS != TRADE_RETCODE_DONE && rcS != TRADE_RETCODE_PLACED))
         && rcS == TRADE_RETCODE_INVALID_EXPIRATION)
      {
         okS = ORB_g_trade.SellStop(ORB_g_lot, sellLevel, _Symbol, sellSL, sellTP,
                                ORDER_TIME_GTC, 0, "ORB sell stop");
         rcS = ORB_g_trade.ResultRetcode();
      }
      if(okS && (rcS == TRADE_RETCODE_DONE || rcS == TRADE_RETCODE_PLACED))
      {
         anySuccess = true;
         g_sellStopTicket = ORB_g_trade.ResultOrder();
      }
      else
         Print("ORB sell-stop failed retcode=", rcS);
   }

   if(anySuccess)
      ORB_g_ordersPlaced = true;

   PrintFormat("ORB %d orders: buy=%s sl=%s tp=%s | sell=%s sl=%s tp=%s (OR %s/%s, %d bars)",
               ORB_g_nyDate, DoubleToString(buyLevel), DoubleToString(buySL), DoubleToString(buyTP),
               DoubleToString(sellLevel), DoubleToString(sellSL), DoubleToString(sellTP),
               DoubleToString(ORB_g_rangeHigh), DoubleToString(g_rangeLow), ORB_g_orBars);
}

//+------------------------------------------------------------------+
//| New M5 bar state machine                                          |
//+------------------------------------------------------------------+
void ORB_OnNewM5Bar(const datetime t0)
{
   int nyDate = ORB_NyDateOf(t0);
   if(nyDate != ORB_g_nyDate)
   {
      ORB_g_nyDate = nyDate;
      ORB_StartNewDay(nyDate);
      PrintFormat("ORB new day %d (offset %+d h)", nyDate, ORB_NyOffsetHours(t0));
   }
   int di = ORB_DayIndex(nyDate);
   if(di < 0)
      return;

   MqlDateTime ndt;
   TimeToStruct((datetime)((long)t0 + (long)ORB_NyOffsetHours(t0) * 3600), ndt);
   int nyMin = ndt.hour * 60 + ndt.min;
   int sessOpenMin = ORB_InpSessionOpenHour * 60 + ORB_InpSessionOpenMin;
   int rangeEndMin = ORB_InpRangeEndHour * 60 + ORB_InpRangeEndMin;
   int cancelMin   = ORB_InpOrderCancelHour * 60 + ORB_InpOrderCancelMin;

   if(!ORB_g_gapChecked && nyMin >= sessOpenMin && nyMin < rangeEndMin)
   {
      ORB_g_gapChecked = true;
      g_sessionOpen = iOpen(_Symbol, PERIOD_M5, 0);
      double pc = 0.0;
      int pd = 0;
      if(ORB_GetPriorSessionClose(nyDate, pc, pd))
      {
         ORB_g_prevClose = pc;
         if(pc > 0)
            g_gapPct = MathAbs(g_sessionOpen / pc - 1.0) * 100.0;
         if(g_gapPct > ORB_InpGapFilterPct)
         {
            g_skipped = true;
            ORB_g_skipReason = "gap_filter";
         }
      }
      else
      {
         g_skipped = true;
         ORB_g_skipReason = "no_prior_close";
      }
      PrintFormat("ORB %d gap check: session_open=%s prev_close=%s gap=%.4f%% -> %s",
                  nyDate, DoubleToString(g_sessionOpen), DoubleToString(ORB_g_prevClose),
                  g_gapPct, g_skipped ? "SKIP (" + ORB_g_skipReason + ")" : "OK");
   }

   datetime cTime = iTime(_Symbol, PERIOD_M5, 1);
   if(cTime > 0)
   {
      int cDate = ORB_NyDateOf(cTime);
      MqlDateTime cdt;
      TimeToStruct((datetime)((long)cTime + (long)ORB_NyOffsetHours(cTime) * 3600), cdt);
      int cMin = cdt.hour * 60 + cdt.min;
      if(cDate == nyDate && cMin >= sessOpenMin && cMin < rangeEndMin)
      {
         double h = iHigh(_Symbol, PERIOD_M5, 1);
         double l = iLow(_Symbol, PERIOD_M5, 1);
         if(ORB_g_orBars == 0)
         {
            ORB_g_rangeHigh = h;
            g_rangeLow  = l;
         }
         else
         {
            if(h > ORB_g_rangeHigh) ORB_g_rangeHigh = h;
            if(l < g_rangeLow)  g_rangeLow  = l;
         }
         if(cMin == sessOpenMin)
            ORB_g_orHasOpenBar = true;
         ORB_g_orBars++;
      }
   }

   if(!g_skipped && !ORB_g_ordersPlaced && !g_entryDone && nyMin >= rangeEndMin && nyMin < cancelMin
      && ORB_OurPositionTicket() == 0)
   {
      if(!ORB_g_orHasOpenBar)
      {
         g_skipped = true;
         ORB_g_skipReason = "no_session_open";
         PrintFormat("ORB %d SKIP: no 09:30 bar in opening range", nyDate);
      }
      else if(ORB_g_orBars < 5)
      {
         g_skipped = true;
         ORB_g_skipReason = "incomplete_range";
         PrintFormat("ORB %d SKIP: only %d opening-range bars", nyDate, ORB_g_orBars);
      }
      else if(RegimeAllowORB())
         ORB_PlaceOrders();
   }

   if(!g_cancelDone && ORB_g_ordersPlaced && !g_entryDone && nyMin >= cancelMin)
   {
      g_cancelDone = true;
      ORB_CancelPendings("10:30 no trigger");
      PrintFormat("ORB %d 10:30 cancel: no trigger", nyDate);
   }

   ORB_SyncDayRecord();
}

//+------------------------------------------------------------------+
//| Expert tick                                                       |
//+------------------------------------------------------------------+
void ORB_OnTick()
{
   ORB_ManageForceClose();

   datetime t0 = iTime(_Symbol, PERIOD_M5, 0);
   if(t0 == 0)
      return;
   if(t0 == ORB_g_lastM5Bar)
      return;
   ORB_g_lastM5Bar = t0;
   ORB_OnNewM5Bar(t0);
}
//+------------------------------------------------------------------+
//| Exit-reason fallback for deals whose transaction was not caught   |
//+------------------------------------------------------------------+
string ORB_FallbackReason(const double closePrice, const datetime closeTime, const double sl, const double tp)
{
   int nd = ORB_NyDateOf(closeTime);
   datetime closeUtc = ORB_NyToUtcOfDate(nd, ORB_InpSessionCloseHour, ORB_InpSessionCloseMin);
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
void ORB_OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
{
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD)
      return;
   if(!HistoryDealSelect(trans.deal))
      return;
   if((long)HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != ORB_InpMagicNumber)
      return;

   ENUM_DEAL_ENTRY de = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
   long posId = (long)HistoryDealGetInteger(trans.deal, DEAL_POSITION_ID);

   if(de == DEAL_ENTRY_IN)
   {
      g_entryDone = true;
      ORB_g_entryDir = ((ENUM_DEAL_TYPE)HistoryDealGetInteger(trans.deal, DEAL_TYPE) == DEAL_TYPE_BUY) ? "BUY" : "SELL";
      ORB_g_entryTime = (datetime)HistoryDealGetInteger(trans.deal, DEAL_TIME);
      ORB_g_posSL = HistoryDealGetDouble(trans.deal, DEAL_SL);
      g_posTP = HistoryDealGetDouble(trans.deal, DEAL_TP);
      PrintFormat("ORB entry: pos=%I64d %s @ %s sl=%s tp=%s",
                  posId, ORB_g_entryDir, DoubleToString(HistoryDealGetDouble(trans.deal, DEAL_PRICE)),
                  DoubleToString(ORB_g_posSL), DoubleToString(g_posTP));
      ORB_CancelPendings("OCO");
      ORB_SyncDayRecord();
   }
   else if(de == DEAL_ENTRY_OUT || de == DEAL_ENTRY_OUT_BY || de == DEAL_ENTRY_INOUT)
   {
      double closePrice = HistoryDealGetDouble(trans.deal, DEAL_PRICE);
      string reason = "other";
      if(ORB_g_forceClose)
         reason = "session_end";
      else
      {
         double tol = MathMax(10.0 * _Point, 0.05);
         if(g_posTP > 0 && MathAbs(closePrice - g_posTP) <= tol)
            reason = "take_profit";
         else if(ORB_g_posSL > 0 && MathAbs(closePrice - ORB_g_posSL) <= tol)
            reason = "stop_loss";
      }
      g_exitTime = (datetime)HistoryDealGetInteger(trans.deal, DEAL_TIME);
      ORB_g_exitReasonDay = reason;
      ORB_AppendExitReason(posId, reason);
      ORB_g_forceClose = false;
      PrintFormat("ORB exit: pos=%I64d reason=%s @ %s", posId, reason, DoubleToString(closePrice));
      ORB_SyncDayRecord();
   }
}

//+------------------------------------------------------------------+
//| trades.csv — same schema family as the RR result collector        |
//| (semicolon-delimited, direction from the ENTRY deal — see         |
//|  PROGRESS.md 3.11/3.12)                                           |
//+------------------------------------------------------------------+
void ORB_WriteTradesCsv()
{
   int handle = FileOpen("orb_trades.csv", FILE_WRITE | FILE_TXT | FILE_ANSI);
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
      if((long)HistoryDealGetInteger(outTicket, DEAL_MAGIC) != ORB_InpMagicNumber)
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
         if((long)HistoryDealGetInteger(dt, DEAL_MAGIC) != ORB_InpMagicNumber)
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

      string reason = ORB_FindExitReason(posId);
      if(reason == "")
         reason = ORB_FallbackReason(closePrice, closeTime, inSL, inTP);

      string line = IntegerToString((long)inTicket) + ";" +
                    ORB_TsStr(openTime) + ";" +
                    ORB_TsStr(closeTime) + ";" +
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
void ORB_WriteDaysCsv()
{
   int handle = FileOpen("orb_days.csv", FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(handle == INVALID_HANDLE)
   {
      Print("WARNING: could not open orb_days.csv for writing. Error=", GetLastError());
      return;
   }
   string header = "ny_date;prev_close;session_open;gap_pct;skipped;skip_reason;range_high;range_low;or_bars;buy_level;sell_level;orders_placed;entry_direction;entry_time;exit_time;exit_reason\r\n";
   FileWriteString(handle, header, StringLen(header));

   int n = ArraySize(ORB_g_days);
   for(int i = 0; i < n; i++)
   {
      string line = IntegerToString(ORB_g_days[i].nyDate) + ";" +
                    DoubleToString(ORB_g_days[i].prevClose, 4) + ";" +
                    DoubleToString(ORB_g_days[i].sessionOpen, 4) + ";" +
                    DoubleToString(ORB_g_days[i].gapPct, 4) + ";" +
                    (ORB_g_days[i].skipped ? "1" : "0") + ";" +
                    ORB_g_days[i].skipReason + ";" +
                    DoubleToString(ORB_g_days[i].rangeHigh, 4) + ";" +
                    DoubleToString(ORB_g_days[i].rangeLow, 4) + ";" +
                    IntegerToString(ORB_g_days[i].orBars) + ";" +
                    DoubleToString(ORB_g_days[i].buyLevel, 4) + ";" +
                    DoubleToString(ORB_g_days[i].sellLevel, 4) + ";" +
                    (ORB_g_days[i].ordersPlaced ? "1" : "0") + ";" +
                    ORB_g_days[i].entryDirection + ";" +
                    (ORB_g_days[i].entryTime > 0 ? ORB_TsStr(ORB_g_days[i].entryTime) : "") + ";" +
                    (ORB_g_days[i].exitTime > 0 ? ORB_TsStr(ORB_g_days[i].exitTime) : "") + ";" +
                    ORB_g_days[i].exitReason + "\r\n";
      FileWriteString(handle, line, StringLen(line));
   }

   FileClose(handle);
   Print("Day CSV written: orb_days.csv (", n, " days)");
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void ORB_OnDeinit(const int reason)
{
   Print("=== ORB_Strategy deinit (reason=", reason, ") ===");
   ORB_WriteTradesCsv();
   ORB_WriteDaysCsv();
}
//+------------------------------------------------------------------+