//+------------------------------------------------------------------+
//|                                          VWAP_MeanReversion.mq5  |
//|   Session-VWAP mean reversion — native, self-contained EA        |
//|   No DLLs, no sockets, no Python bridge, no ML.                  |
//+------------------------------------------------------------------+
#property copyright "RR project"
#property version   "1.00"
#property description "VWAP mean reversion on US500-tick (server time = UTC), M1"
//
// STRATEGY (published rule set, exact — interpretations documented):
//   Session: 09:30-16:00 New York time (RTH only). All signals and
//   exits live inside the session; forced exit 15:55 NY.
//
//   Session VWAP: cumulative over completed M1 bars of the session:
//     VWAP = sum(typical * tick_volume) / sum(tick_volume),
//     typical = (H+L+C)/3 of the completed bar. Resets every new
//     NY session. NO look-ahead: only fully completed bars enter the
//     accumulation; the VWAP used during a forming bar is the value
//     through the last completed bar.
//
//   Range-day condition (rolling, evaluated at every entry signal —
//   the published "by mid-session" wording is operationalized as a
//   continuous rolling check, since the strategy must know range-day
//   status as the session progresses, not in hindsight):
//     |close / session_open - 1|          <= 0.6 %   (net move)
//     (session_high - session_low) / open <= 1.0 %   (intraday range)
//   where session_high/low accumulate over completed session bars.
//   The H-L term is monotone: once broken it stays broken for the day.
//
//   Long entry:  signal-bar close trades >= 0.25% BELOW VWAP AND the
//   30-minute VWAP slope is flat-to-slightly-positive.
//   Short entry: close >= 0.25% ABOVE VWAP AND 30-min slope
//   flat-to-slightly-negative.
//     slope_pct = (vwap_now / vwap_{t - 30 min} - 1) * 100, referenced
//     by exact bar timestamp 1800 s earlier IN THE SAME SESSION
//     (missing reference bar => no signal). "Flat-to-slightly-X" is
//     implemented as slope >= 0 (long) / slope <= 0 (short); there is
//     no quantified upper bound in the published spec.
//
//   Entry execution: the signal fires on a COMPLETED bar; the market
//   order is sent on the first tick of the NEXT bar (entry bar). This
//   is the no-look-ahead, reproducible-in-both-engines convention.
//   Entries stop after 15:30 NY (documented operational cutoff so a
//   new position always has >= 25 min before the 15:55 force exit).
//
//   Stop loss: 0.2% beyond the entry price, away from VWAP
//     long:  SL = entry * (1 - 0.002);  short: SL = entry * (1 + 0.002).
//   Server-side stop, rounded to the symbol trade tick grid.
//
//   Take profit: touch of VWAP — a MOVING target. No server-side TP
//   order exists; the EA monitors every tick and closes when price
//   touches the current (last-completed-bar) VWAP:
//     long:  bid >= VWAP -> close;  short: ask <= VWAP -> close.
//   The trades.csv take_profit column records the VWAP snapshot at
//   entry (the initial TP level), for audit only.
//
//   Forced exit: 15:55 NY (five minutes before the ORB 16:00 close
//   convention), on the first tick at/after 15:55.
//
//   Max-move filter (published as "no trades when the underlying has
//   moved more than 30 points from the session open" — an ES-point
//   rule): implemented as a PERCENTAGE by default (consistency with
//   the other percentage rules of this strategy), default 0.42% =
//     30 points / ~7200 index level (April 2026 US500-tick ~ 7200).
//   DOCUMENTED SIMPLIFICATION: fixed-point research rule translated
//   proportionally at the CURRENT index level; at 2021 levels (~3800)
//   the same 0.42% equals only ~16 points. A fixed-points mode is
//   available via InpMaxMoveMode=1 (InpMaxMovePoints=30.0).
//
//   One position at a time; multiple trades per day allowed (the
//   published rule set states no per-day trade cap; signals are only
//   evaluated while flat).
//
// KNOWN SIMPLIFICATION: fixed lot sizing (same as ORB base run).
//
// TIMEBASE: M1 (probe 2026-08-24: full-range M1 served cleanly —
//   1,791,504 bars, 64/64 months, 0 multi-day gaps, date set identical
//   to the verified M5 export; tick_volume > 0 on every bar).
//
// TIMEZONE MODEL (identical to ORB_Strategy, verified): US500-tick
//   bar timestamps are RAW UTC; TimeCurrent() is UTC in backtests.
//   NY = UTC-5 (EST) / UTC-4 (EDT); US DST begins 2nd Sunday of March
//   07:00 UTC, ends 1st Sunday of November 06:00 UTC. The offset is
//   computed generically (no hardcoded years) and logged at startup.
//
// NOTE: MT5 batch /config .ini cannot override double input parameters
//   (PROGRESS.md 3.14) — the input defaults below ARE the strategy
//   spec. Int inputs can be overridden via the [Tester] ini section.
//
// OUTPUT FILES (tester agent MQL5\Files sandbox):
//   trades.csv  ticket;open_time;close_time;symbol;type;lot;open_price;
//               stop_loss;take_profit;close_price;exit_reason;swap;
//               commission;profit;dev_pct;slope_pct
//               (last two columns = signal diagnostics at entry)
//   vwap_days.csv ny_date;session_open;session_high;session_low;
//               net_pct_final;hl_pct_final;range_ok_final;skipped;
//               skip_reason;slope_ok_bars;entry_signals;entries;
//               exits_tp;exits_sl;exits_time;total_pnl
//+------------------------------------------------------------------+

#include <Trade\Trade.mqh>

// ── Inputs (defaults ARE the strategy spec) ─────────────────────────
input double InpDeviationPct   = 0.25;     // Entry deviation from VWAP (%)
input double InpStopPct        = 0.2;      // Stop distance beyond entry (%)
input double InpRangeNetPct    = 0.6;      // Range-day max net move from open (%)
input double InpRangeHLPct     = 1.0;      // Range-day max high-low (%)
input int    InpMaxMoveMode    = 0;        // Max-move filter mode: 0=percent, 1=points
input double InpMaxMovePct     = 0.42;     // Max-move filter, percent mode (%)
input double InpMaxMovePoints  = 30.0;     // Max-move filter, points mode (index pts)
input int    InpSlopeWindowMin = 30;       // VWAP slope window (minutes)
input double InpSlopeMinPct    = 0.0;      // Long needs slope >= +x%, short <= -x%
input double InpLotSize        = 1.0;      // Fixed lot size (clamped to symbol min/step)
input int    InpSessionOpenHour  = 9;      // NY session open hour
input int    InpSessionOpenMin   = 30;     // NY session open minute
input int    InpSessionCloseHour = 16;     // NY session end hour (bookkeeping)
input int    InpSessionCloseMin  = 0;      // NY session end minute
input int    InpForceExitHour  = 15;       // NY forced-exit hour
input int    InpForceExitMin   = 55;       // NY forced-exit minute
input int    InpLastEntryHour  = 15;       // NY last-entry hour (entry bar open)
input int    InpLastEntryMin   = 30;       // NY last-entry minute
input long   InpMagicNumber    = 26082401; // Magic number

// ── Globals ─────────────────────────────────────────────────────────
CTrade   g_trade;
double   g_lot = 1.0;
datetime g_lastM1Bar = 0;

int      g_nyDate = 0;                  // current NY date, YYYYMMDD
double   g_sessionOpen = 0.0;
bool     g_sessionOpenSet = false;
double   g_sessHigh = 0.0, g_sessLow = 0.0;
double   g_cumPV = 0.0, g_cumV = 0.0;   // session VWAP accumulators
double   g_vwap = 0.0;                  // VWAP through last completed session bar
datetime g_vwTimes[];                   // session bar-open times
double   g_vwVals[];                    // VWAP snapshot after each session bar
bool     g_skipped = false;
string   g_skipReason = "";
double   g_lastNetPct = 0.0, g_lastHLPct = 0.0;
bool     g_lastRangeOk = false;
int      g_slopeOkBars = 0;             // signal-eval bars where the slope ref existed

bool     g_forceClose = false;          // set right before a forced time exit
bool     g_vwapTouch  = false;          // set right before a VWAP-touch close

// entry context (set at order send, consumed at the IN deal)
double   g_entryDev = 0.0, g_entrySlope = 0.0;

struct DayRecord
{
   int      nyDate;
   double   sessionOpen, sessionHigh, sessionLow;
   double   netPctFinal, hlPctFinal;
   bool     rangeOkFinal;
   bool     skipped;
   string   skipReason;
   int      slopeOkBars;
   int      signals, entries;
   int      exitsTp, exitsSl, exitsTime;
   double   totalPnl;
};
DayRecord g_days[];

// per-position context registry keyed by position id (ORB pattern)
long     g_ctxPosId[];
double   g_ctxDev[], g_ctxSlope[], g_ctxTp[], g_ctxSL[];
long     g_exitPosIds[];
string   g_exitReasons[];

//+------------------------------------------------------------------+
//| Time / DST helpers (identical to ORB_Strategy.mq5)                |
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

datetime NyToUtc(const int y, const int mo, const int d, const int h, const int mi)
{
   MqlDateTime dt;
   dt.year = y; dt.mon = mo; dt.day = d; dt.hour = h; dt.min = mi; dt.sec = 0;
   datetime naive = StructToTime(dt);
   return (datetime)((long)naive - (long)NyOffsetHours(naive) * 3600);
}

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

// NY wall-clock minute-of-day of a UTC timestamp
int NyWallMin(const datetime utc)
{
   MqlDateTime dt;
   TimeToStruct((datetime)((long)utc + (long)NyOffsetHours(utc) * 3600), dt);
   return dt.hour * 60 + dt.min;
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
   g_days[n].sessionOpen  = 0.0;
   g_days[n].sessionHigh  = 0.0;
   g_days[n].sessionLow   = 0.0;
   g_days[n].netPctFinal  = 0.0;
   g_days[n].hlPctFinal   = 0.0;
   g_days[n].rangeOkFinal = false;
   g_days[n].skipped      = false;
   g_days[n].skipReason   = "";
   g_days[n].slopeOkBars  = 0;
   g_days[n].signals      = 0;
   g_days[n].entries      = 0;
   g_days[n].exitsTp      = 0;
   g_days[n].exitsSl      = 0;
   g_days[n].exitsTime    = 0;
   g_days[n].totalPnl     = 0.0;

   g_sessionOpen = 0.0;  g_sessionOpenSet = false;
   g_sessHigh = 0.0;     g_sessLow = 0.0;
   g_cumPV = 0.0;        g_cumV = 0.0;   g_vwap = 0.0;
   ArrayResize(g_vwTimes, 0);
   ArrayResize(g_vwVals, 0);
   g_skipped = false;    g_skipReason = "";
   g_lastNetPct = 0.0;   g_lastHLPct = 0.0;  g_lastRangeOk = false;
   g_slopeOkBars = 0;
}

void SyncDayRecord()
{
   int di = DayIndex(g_nyDate);
   if(di < 0)
      return;
   g_days[di].sessionOpen  = g_sessionOpen;
   g_days[di].sessionHigh  = g_sessHigh;
   g_days[di].sessionLow   = g_sessLow;
   g_days[di].netPctFinal  = g_lastNetPct;
   g_days[di].hlPctFinal   = g_lastHLPct;
   g_days[di].rangeOkFinal = g_lastRangeOk;
   g_days[di].skipped      = g_skipped;
   g_days[di].skipReason   = g_skipReason;
   g_days[di].slopeOkBars  = g_slopeOkBars;
}

//+------------------------------------------------------------------+
//| Position / context helpers                                        |
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

int CtxIndexOf(const long posId)
{
   for(int i = ArraySize(g_ctxPosId) - 1; i >= 0; i--)
      if(g_ctxPosId[i] == posId)
         return i;
   return -1;
}

double CtxVal(const long posId, const double &vals[], const double dflt = 0.0)
{
   int i = CtxIndexOf(posId);
   return (i >= 0) ? vals[i] : dflt;
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

void CountExit(const string reason)
{
   int di = DayIndex(g_nyDate);
   if(di < 0)
      return;
   if(reason == "vwap_touch")      g_days[di].exitsTp++;
   else if(reason == "stop_loss")  g_days[di].exitsSl++;
   else if(reason == "time_exit")  g_days[di].exitsTime++;
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

   Print("=== VWAP_MeanReversion v1.0 init ===");
   PrintFormat("VWAP inputs: dev=%.4f%% sl=%.4f%% range_net=%.4f%% range_hl=%.4f%% "
               "maxmove_mode=%d pct=%.4f%% pts=%.2f slope_win=%dmin slope_min=%.4f%% "
               "lot=%.2f session=%02d:%02d-%02d:%02d force_exit=%02d:%02d last_entry=%02d:%02d magic=%I64d",
               InpDeviationPct, InpStopPct, InpRangeNetPct, InpRangeHLPct,
               InpMaxMoveMode, InpMaxMovePct, InpMaxMovePoints,
               InpSlopeWindowMin, InpSlopeMinPct, InpLotSize,
               InpSessionOpenHour, InpSessionOpenMin,
               InpSessionCloseHour, InpSessionCloseMin,
               InpForceExitHour, InpForceExitMin,
               InpLastEntryHour, InpLastEntryMin, InpMagicNumber);

   int off = NyOffsetHours(TimeCurrent());
   PrintFormat("VWAP NY offset now: %+d h -> server %s = NY %s",
               off, TsStr(TimeCurrent()), TsStr((datetime)((long)TimeCurrent() + (long)off * 3600)));

   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   PrintFormat("VWAP DST boundaries %d: start=%s end=%s (UTC)",
               dt.year, TsStr(NthSundayUtc(dt.year, 3, 2, 7)), TsStr(NthSundayUtc(dt.year, 11, 1, 6)));

   int nd = NyDateOf(TimeCurrent());
   PrintFormat("VWAP session map today: open=%s force_exit=%s close=%s last_entry=%s (UTC)",
               TsStr(NyToUtcOfDate(nd, InpSessionOpenHour, InpSessionOpenMin)),
               TsStr(NyToUtcOfDate(nd, InpForceExitHour, InpForceExitMin)),
               TsStr(NyToUtcOfDate(nd, InpSessionCloseHour, InpSessionCloseMin)),
               TsStr(NyToUtcOfDate(nd, InpLastEntryHour, InpLastEntryMin)));
   PrintFormat("VWAP lot: input=%.2f -> trading=%.2f (min=%.2f step=%.2f digits=%d point=%s)",
               InpLotSize, g_lot, vmin, vstep,
               (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS), DoubleToString(_Point, 8));

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Grid-safe market close (CTrade::PositionClose sends the RAW bid/  |
//| ask and is rejected 10015 on this custom symbol — same root cause |
//| as ORB's force-close retries; closed via explicit rounded price). |
//+------------------------------------------------------------------+
bool ClosePositionGrid(const ulong ticket)
{
   if(!PositionSelectByTicket(ticket))
      return false;
   long ptype = PositionGetInteger(POSITION_TYPE);
   MqlTradeRequest req;
   MqlTradeResult  res;
   ZeroMemory(req);
   ZeroMemory(res);
   req.action    = TRADE_ACTION_DEAL;
   req.symbol    = _Symbol;
   req.volume    = PositionGetDouble(POSITION_VOLUME);
   req.type      = (ptype == POSITION_TYPE_BUY) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   req.position  = ticket;
   req.price     = (req.type == ORDER_TYPE_SELL)
                   ? NormPrice(SymbolInfoDouble(_Symbol, SYMBOL_BID))
                   : NormPrice(SymbolInfoDouble(_Symbol, SYMBOL_ASK));
   req.deviation = 50;
   req.magic     = (ulong)InpMagicNumber;
   req.comment   = "VWAP MR close";
   if(!OrderSend(req, res))
      return false;
   return (res.retcode == TRADE_RETCODE_DONE || res.retcode == TRADE_RETCODE_PLACED
           || res.retcode == TRADE_RETCODE_DONE_PARTIAL);
}

//+------------------------------------------------------------------+
//| 15:55 NY forced exit — checked on every tick                      |
//+------------------------------------------------------------------+
void ManageForceClose()
{
   ulong ticket = OurPositionTicket();
   if(ticket == 0)
      return;
   datetime now = TimeCurrent();
   if(NyDateOf(now) != g_nyDate)
   {
      g_forceClose = true;
      if(!ClosePositionGrid(ticket))
         Print("VWAP force-close failed (past midnight) retcode=", g_trade.ResultRetcode());
      return;
   }
   datetime exitUtc = NyToUtcOfDate(g_nyDate, InpForceExitHour, InpForceExitMin);
   if(now >= exitUtc)
   {
      g_forceClose = true;
      if(!ClosePositionGrid(ticket))
         Print("VWAP force-close failed retcode=", g_trade.ResultRetcode());
   }
}

//+------------------------------------------------------------------+
//| VWAP-touch take profit — tick-monitored dynamic target            |
//+------------------------------------------------------------------+
void ManageVwapTouchExit()
{
   if(g_vwap <= 0.0)
      return;
   ulong ticket = OurPositionTicket();
   if(ticket == 0)
      return;
   if(!PositionSelectByTicket(ticket))
      return;
   long ptype = PositionGetInteger(POSITION_TYPE);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   bool touch = false;
   if(ptype == POSITION_TYPE_BUY && bid >= g_vwap)
      touch = true;
   if(ptype == POSITION_TYPE_SELL && ask <= g_vwap)
      touch = true;
   if(touch)
   {
      g_vwapTouch = true;
      if(!ClosePositionGrid(ticket))
      {
         g_vwapTouch = false;   // retry on a later tick
         Print("VWAP touch-close failed retcode=", g_trade.ResultRetcode());
      }
   }
}

//+------------------------------------------------------------------+
//| VWAP snapshot at exact session bar time (for the slope)           |
//+------------------------------------------------------------------+
bool VwapAt(const datetime t, double &vOut)
{
   for(int i = ArraySize(g_vwTimes) - 1; i >= 0; i--)
   {
      if(g_vwTimes[i] == t)
      {
         vOut = g_vwVals[i];
         return true;
      }
      if(g_vwTimes[i] < t)
         break;               // times are ascending; t absent (data gap)
   }
   return false;
}

//+------------------------------------------------------------------+
//| Market entry                                                      |
//+------------------------------------------------------------------+
void PlaceMarketEntry(const int dir, const double devPct, const double slopePct)
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double refPx = (dir > 0) ? ask : bid;   // expected fill side
   double px = NormPrice(refPx);           // off-grid market price => 10015
   double sl = (dir > 0) ? NormPrice(refPx * (1.0 - InpStopPct / 100.0))
                         : NormPrice(refPx * (1.0 + InpStopPct / 100.0));

   g_entryDev   = devPct;
   g_entrySlope = slopePct;
   g_forceClose = false;      // clear stale exit flags for the new position
   g_vwapTouch  = false;

   bool ok = false;
   if(dir > 0)
   {
      ok = g_trade.Buy(g_lot, _Symbol, px, sl, 0.0, "VWAP MR long");
      if(!ok)
         ok = g_trade.Buy(g_lot, _Symbol, px, sl, 0.0, "VWAP MR long retry");
   }
   else
   {
      ok = g_trade.Sell(g_lot, _Symbol, px, sl, 0.0, "VWAP MR short");
      if(!ok)
         ok = g_trade.Sell(g_lot, _Symbol, px, sl, 0.0, "VWAP MR short retry");
   }
   if(!ok)
   {
      PrintFormat("VWAP %d %s entry FAILED retcode=%d", g_nyDate,
                  (dir > 0 ? "BUY" : "SELL"), g_trade.ResultRetcode());
      return;
   }

   int di = DayIndex(g_nyDate);
   if(di >= 0)
      g_days[di].entries++;
   PrintFormat("VWAP %d ENTRY %s @ %s sl=%s vwap=%s dev=%.4f%% slope=%.4f%%",
               g_nyDate, (dir > 0 ? "BUY" : "SELL"), DoubleToString(g_trade.ResultPrice()),
               DoubleToString(sl), DoubleToString(g_vwap), devPct, slopePct);
}

//+------------------------------------------------------------------+
//| New M1 bar state machine                                          |
//+------------------------------------------------------------------+
void OnNewM1Bar(const datetime t0)
{
   int nyDate = NyDateOf(t0);
   if(nyDate != g_nyDate)
   {
      g_nyDate = nyDate;
      StartNewDay(nyDate);
      PrintFormat("VWAP new day %d (offset %+d h)", nyDate, NyOffsetHours(t0));
   }
   int di = DayIndex(nyDate);
   if(di < 0)
      return;

   // completed bar = shift 1
   datetime cTime = iTime(_Symbol, PERIOD_M1, 1);
   if(cTime <= 0)
      return;
   if(NyDateOf(cTime) != nyDate)
      return;                              // completed bar belongs to a prior date
   int cWall = NyWallMin(cTime);
   int sessOpenMin  = InpSessionOpenHour * 60 + InpSessionOpenMin;       // 570
   int sessLastMin  = InpSessionCloseHour * 60 + InpSessionCloseMin - 1; // 959
   if(cWall < sessOpenMin || cWall > sessLastMin)
      return;                              // outside RTH: not part of the session

   double h = iHigh(_Symbol, PERIOD_M1, 1);
   double l = iLow(_Symbol, PERIOD_M1, 1);
   double c = iClose(_Symbol, PERIOD_M1, 1);
   long   v = iVolume(_Symbol, PERIOD_M1, 1);

   // session open bookkeeping
   if(!g_sessionOpenSet)
   {
      if(cWall == sessOpenMin)
      {
         g_sessionOpen    = iOpen(_Symbol, PERIOD_M1, 1);
         g_sessionOpenSet = true;
         PrintFormat("VWAP %d session start: open=%s vol=%I64d",
                     nyDate, DoubleToString(g_sessionOpen), v);
      }
      else
      {
         g_skipped = true;
         g_skipReason = "no_session_open";
         PrintFormat("VWAP %d SKIP: first session bar wall=%d (no 09:30 bar)", nyDate, cWall);
      }
   }

   // accumulate session VWAP / high / low (completed bars only — no look-ahead)
   double typ = (h + l + c) / 3.0;
   if(v > 0)
   {
      g_cumPV += typ * (double)v;
      g_cumV  += (double)v;
   }
   if(g_cumV > 0.0)
      g_vwap = g_cumPV / g_cumV;
   if(g_sessHigh == 0.0 || h > g_sessHigh) g_sessHigh = h;
   if(g_sessLow  == 0.0 || l < g_sessLow)  g_sessLow  = l;
   int nv = ArraySize(g_vwTimes);
   ArrayResize(g_vwTimes, nv + 1);
   ArrayResize(g_vwVals, nv + 1);
   g_vwTimes[nv] = cTime;
   g_vwVals[nv]  = g_vwap;

   // rolling range-day status after this bar
   double netPct = 0.0, hlPct = 0.0;
   if(g_sessionOpen > 0.0)
   {
      netPct = MathAbs(c / g_sessionOpen - 1.0) * 100.0;
      hlPct  = (g_sessHigh - g_sessLow) / g_sessionOpen * 100.0;
   }
   g_lastNetPct  = netPct;
   g_lastHLPct   = hlPct;
   g_lastRangeOk = (netPct <= InpRangeNetPct && hlPct <= InpRangeHLPct);

   // ---- entry evaluation on the completed signal bar ----------------
   int lastEntryMin = InpLastEntryHour * 60 + InpLastEntryMin;            // 930
   int t0Wall = NyWallMin(t0);
   if(!g_skipped && g_sessionOpenSet && g_vwap > 0.0
      && OurPositionTicket() == 0
      && t0Wall >= sessOpenMin && t0Wall <= lastEntryMin                 // entry-bar cutoff
      && NyDateOf(t0) == nyDate)
   {
      // slope over the exact window, same session, exact timestamp match
      datetime refT = cTime - (datetime)(InpSlopeWindowMin * 60);
      double vRef = 0.0;
      bool slopeOk = false;
      double slopePct = 0.0;
      if(VwapAt(refT, vRef) && vRef > 0.0)
      {
         slopePct = (g_vwap / vRef - 1.0) * 100.0;
         slopeOk = true;
         g_slopeOkBars++;
      }

      if(slopeOk)
      {
         double devPct   = (c / g_vwap - 1.0) * 100.0;
         double moveAbs  = MathAbs(c - g_sessionOpen);
         bool maxMoveOk  = (InpMaxMoveMode == 1)
                           ? (moveAbs <= InpMaxMovePoints)
                           : (moveAbs <= g_sessionOpen * InpMaxMovePct / 100.0);

         int dir = 0;
         if(g_lastRangeOk && maxMoveOk)
         {
            if(devPct <= -InpDeviationPct && slopePct >= InpSlopeMinPct)
               dir = +1;
            else if(devPct >= InpDeviationPct && slopePct <= -InpSlopeMinPct)
               dir = -1;
         }
         if(dir != 0)
         {
            g_days[di].signals++;
            PrintFormat("VWAP %d signal %s bar=%s close=%s vwap=%s dev=%.4f%% slope=%.4f%% "
                        "net=%.4f%% hl=%.4f%% maxmove=%s",
                        nyDate, (dir > 0 ? "LONG" : "SHORT"), TsStr(cTime),
                        DoubleToString(c), DoubleToString(g_vwap), devPct, slopePct,
                        netPct, hlPct, maxMoveOk ? "ok" : "BLOCKED");
            PlaceMarketEntry(dir, devPct, slopePct);
         }
      }
   }

   SyncDayRecord();
}

//+------------------------------------------------------------------+
//| Expert tick                                                       |
//+------------------------------------------------------------------+
void OnTick()
{
   ManageForceClose();
   ManageVwapTouchExit();

   datetime t0 = iTime(_Symbol, PERIOD_M1, 0);
   if(t0 == 0)
      return;
   if(t0 == g_lastM1Bar)
      return;
   g_lastM1Bar = t0;
   OnNewM1Bar(t0);
}
//+------------------------------------------------------------------+
//| Exit-reason fallback (flags missed the transaction)               |
//| SL tolerance 0.5 pt: tester stop fills can slip a few tenths past |
//| the level (observed 0.09 pt on 2026-01-02).                       |
//+------------------------------------------------------------------+
string FallbackReason(const double closePrice, const datetime closeTime,
                      const double sl, const double vwapTp)
{
   double tol = MathMax(0.5, 10.0 * _Point);
   if(sl > 0 && MathAbs(closePrice - sl) <= tol)
      return "stop_loss";
   int nd = NyDateOf(closeTime);
   datetime exitUtc = NyToUtcOfDate(nd, InpForceExitHour, InpForceExitMin);
   long d = (long)closeTime - (long)exitUtc;
   if(d >= -600 && d < 3600)          // within the forced-exit window
      return "time_exit";
   if(vwapTp > 0)                      // only remaining exit path is the VWAP touch
      return "vwap_touch";
   return "other";
}

//+------------------------------------------------------------------+
//| Trade transactions: exit-reason + context capture                 |
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
      // register per-position context: signal diagnostics + TP snapshot + SL
      int n = ArraySize(g_ctxPosId);
      ArrayResize(g_ctxPosId, n + 1);
      ArrayResize(g_ctxDev, n + 1);
      ArrayResize(g_ctxSlope, n + 1);
      ArrayResize(g_ctxTp, n + 1);
      ArrayResize(g_ctxSL, n + 1);
      g_ctxPosId[n] = posId;
      g_ctxDev[n]   = g_entryDev;
      g_ctxSlope[n] = g_entrySlope;
      g_ctxTp[n]    = g_vwap;                                   // VWAP at entry
      g_ctxSL[n]    = HistoryDealGetDouble(trans.deal, DEAL_SL); // order SL
      PrintFormat("VWAP entry deal: pos=%I64d @ %s (ctx dev=%.4f slope=%.4f tp0=%s)",
                  posId, DoubleToString(HistoryDealGetDouble(trans.deal, DEAL_PRICE)),
                  g_entryDev, g_entrySlope, DoubleToString(g_ctxTp[n]));
      SyncDayRecord();
   }
   else if(de == DEAL_ENTRY_OUT || de == DEAL_ENTRY_OUT_BY || de == DEAL_ENTRY_INOUT)
   {
      double closePrice = HistoryDealGetDouble(trans.deal, DEAL_PRICE);
      datetime closeTime = (datetime)HistoryDealGetInteger(trans.deal, DEAL_TIME);

      string reason;
      long dreason = HistoryDealGetInteger(trans.deal, DEAL_REASON);
      if(g_forceClose)
         reason = "time_exit";
      else if(g_vwapTouch)
         reason = "vwap_touch";
      else if(dreason == DEAL_REASON_SL)
         reason = "stop_loss";           // server-side SL fill (no flags set)
      else
      {
         double slCtx = CtxVal(posId, g_ctxSL, 0.0);
         double tpCtx = CtxVal(posId, g_ctxTp, 0.0);
         reason = FallbackReason(closePrice, closeTime, slCtx, tpCtx);
      }

      AppendExitReason(posId, reason);
      CountExit(reason);
      int di = DayIndex(g_nyDate);
      if(di >= 0)
         g_days[di].totalPnl += HistoryDealGetDouble(trans.deal, DEAL_PROFIT)
                                + HistoryDealGetDouble(trans.deal, DEAL_SWAP)
                                + HistoryDealGetDouble(trans.deal, DEAL_COMMISSION);
      g_forceClose = false;
      g_vwapTouch  = false;
      PrintFormat("VWAP exit: pos=%I64d reason=%s @ %s", posId, reason,
                  DoubleToString(closePrice));
      SyncDayRecord();
   }
}

//+------------------------------------------------------------------+
//| trades.csv — collector-compatible schema + signal diagnostics     |
//+------------------------------------------------------------------+
void WriteTradesCsv()
{
   int handle = FileOpen("trades.csv", FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(handle == INVALID_HANDLE)
   {
      Print("WARNING: could not open trades.csv for writing. Error=", GetLastError());
      return;
   }
   string header = "ticket;open_time;close_time;symbol;type;lot;open_price;stop_loss;"
                   "take_profit;close_price;exit_reason;swap;commission;profit;"
                   "dev_pct;slope_pct\r\n";
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

      // TP column: the target is the moving VWAP -> record the entry snapshot
      double tpSnap = CtxVal(posId, g_ctxTp, 0.0);
      if(tpSnap <= 0.0)
         tpSnap = inTP;
      double devCtx = CtxVal(posId, g_ctxDev, 0.0);
      double slopeCtx = CtxVal(posId, g_ctxSlope, 0.0);

      string reason = FindExitReason(posId);
      if(reason == "")
         reason = FallbackReason(closePrice, closeTime, inSL, tpSnap);

      string line = IntegerToString((long)inTicket) + ";" +
                    TsStr(openTime) + ";" +
                    TsStr(closeTime) + ";" +
                    dealSymbol + ";" +
                    dir + ";" +
                    DoubleToString(volume, 2) + ";" +
                    DoubleToString(openPrice) + ";" +
                    DoubleToString(inSL) + ";" +
                    DoubleToString(tpSnap) + ";" +
                    DoubleToString(closePrice) + ";" +
                    reason + ";" +
                    DoubleToString(swapVal, 2) + ";" +
                    DoubleToString(commVal, 2) + ";" +
                    DoubleToString(profitVal, 2) + ";" +
                    DoubleToString(devCtx, 4) + ";" +
                    DoubleToString(slopeCtx, 4) + "\r\n";
      FileWriteString(handle, line, StringLen(line));
      written++;
   }

   FileClose(handle);
   Print("Trade CSV written: trades.csv (", written, " trades)");
}

//+------------------------------------------------------------------+
//| vwap_days.csv — per-day state for Python-vs-MT5 reconciliation    |
//+------------------------------------------------------------------+
void WriteDaysCsv()
{
   int handle = FileOpen("vwap_days.csv", FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(handle == INVALID_HANDLE)
   {
      Print("WARNING: could not open vwap_days.csv for writing. Error=", GetLastError());
      return;
   }
   string header = "ny_date;session_open;session_high;session_low;net_pct_final;"
                   "hl_pct_final;range_ok_final;skipped;skip_reason;slope_ok_bars;"
                   "entry_signals;entries;exits_tp;exits_sl;exits_time;total_pnl\r\n";
   FileWriteString(handle, header, StringLen(header));

   int n = ArraySize(g_days);
   for(int i = 0; i < n; i++)
   {
      string line = IntegerToString(g_days[i].nyDate) + ";" +
                    DoubleToString(g_days[i].sessionOpen, 4) + ";" +
                    DoubleToString(g_days[i].sessionHigh, 4) + ";" +
                    DoubleToString(g_days[i].sessionLow, 4) + ";" +
                    DoubleToString(g_days[i].netPctFinal, 4) + ";" +
                    DoubleToString(g_days[i].hlPctFinal, 4) + ";" +
                    (g_days[i].rangeOkFinal ? "1" : "0") + ";" +
                    (g_days[i].skipped ? "1" : "0") + ";" +
                    g_days[i].skipReason + ";" +
                    IntegerToString(g_days[i].slopeOkBars) + ";" +
                    IntegerToString(g_days[i].signals) + ";" +
                    IntegerToString(g_days[i].entries) + ";" +
                    IntegerToString(g_days[i].exitsTp) + ";" +
                    IntegerToString(g_days[i].exitsSl) + ";" +
                    IntegerToString(g_days[i].exitsTime) + ";" +
                    DoubleToString(g_days[i].totalPnl, 2) + "\r\n";
      FileWriteString(handle, line, StringLen(line));
   }

   FileClose(handle);
   Print("Day CSV written: vwap_days.csv (", n, " days)");
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("=== VWAP_MeanReversion deinit (reason=", reason, ") ===");
   WriteTradesCsv();
   WriteDaysCsv();
}
//+------------------------------------------------------------------+
