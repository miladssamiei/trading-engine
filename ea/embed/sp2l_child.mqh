//+------------------------------------------------------------------+
//|                                                SP2L_SpikeGap.mq5  |
//|   SP2L spike/gap/continuation — native, self-contained EA        |
//|   No DLLs, no sockets, no Python bridge, no ML.                  |
//+------------------------------------------------------------------+
//
// WALK-FORWARD-CONFIRMED CONFIGURATION ONLY (mt5_rebuild_gate rule):
//   NY_E = NY session entry window, EMA60 filter ON; trend-structure,
//   ADX, and second-entry ablations were NOT confirmed and are
//   deliberately NOT implemented here. Parameters below are the
//   confirmed set from data/results/sp2l_ablation/ (SPEC.md, GATE.md,
//   step3/): TP_R = 2.0 selected by trailing expectancy in ALL 9
//   walk-forward folds (2022H1..2026H1).
//
// STRATEGY (full spec: data/results/sp2l_ablation/SPEC.md):
//   Setup on CLOSED M1 bars, indices relative to trigger bar t at its
//   close (shift1=t, shift2=t-1 post-spike, shift3=t-2 spike,
//   shift4=t-3 pre-spike):
//     BUY: low(t)<low(t-1); closes/opens rising across t-3..t-1; all
//          three candles bullish; gap low(t-1) > high(t-3)*(1+0.04%);
//          signed body(t-2) > 1.5 x bodies of t-3, t-1, t.
//     SELL: exact mirror. BUY+SELL same bar => neither.
//   Pending: created at t (NOT an entry). SL = low(t-3) [BUY] /
//     high(t-3) [SELL]. Multi-bar lifetime, same NY date only:
//     on each later completed bar u (wall <= 15:30 else expire):
//       break  : low(u) < low(u-1) [BUY] / high(u) > high(u-1) [SELL]
//       risk   : entry-side extreme minus SL; risk <= 0 => keep waiting
//       kill   : risk > 0.40% of price => pending INVALID (source rule)
//       EMA60  : close(u) strictly above [BUY] / below [SELL] EMA60(u)
//       session: wall(u) in [09:30, 15:30] NY else keep waiting
//     All pass => ENTRY: market order at the FIRST TICK of the next
//     bar (u+1) = E-NEXT convention (signal on completed bar, fill at
//     next bar open) — the no-look-ahead project convention, and the
//     convention whose walk-forward gates this configuration passed.
//     TP = entry +/- TP_R(=2.0) x risk, SL/TP on the 0.01 trade grid.
//   Occupancy: one position at a time; no setup detection while a
//     position is open; after an exit during bar eb, setups with
//     trigger bar <= eb are skipped (matches the Python reference).
//   Forced exit: 15:55 NY bar open, tick-monitored.
//
// EMA60: incremental over ALL M1 bars (overnight included), seeded at
//   the first bar of the test (same recursion as pandas ewm span=60
//   adjust=False). Yearly batches re-seed at each year's first bar;
//   the transient decays as (1-2/61)^n and entries cannot occur before
//   09:30 of the following NY day (>600 bars later), so the difference
//   vs a full-range seed is negligible — documented recon note.
//
// TIMEZONE MODEL (identical to ORB/VWAP EAs, verified): US500-tick bar
//   timestamps are RAW UTC; TimeCurrent() is UTC in backtests. NY =
//   UTC-5/UTC-4 with generic US DST rules (2nd Sunday March 07:00 UTC,
//   1st Sunday November 06:00 UTC).
//
// OUTPUT FILES (tester agent MQL5\Files sandbox):
//   trades.csv    ticket;open_time;close_time;symbol;type;lot;open_price;
//                 stop_loss;take_profit;close_price;exit_reason;swap;
//                 commission;profit;setup_time
//   sp2l_days.csv ny_date;setups;entered;killed;expired;entries;
//                 exits_tp;exits_sl;exits_time;total_pnl
//+------------------------------------------------------------------+


// ── Inputs (defaults ARE the confirmed spec) ─────────────────────────
input double SP2L_InpSpikeRatio     = 1.5;      // Spike body ratio vs neighbors
input double SP2L_InpGapPct         = 0.04;     // Gap threshold (% of price)
input double SP2L_InpMaxSlPct       = 0.40;     // Max SL distance (% of price)
input int    SP2L_InpEmaPeriod      = 60;       // EMA filter period (bars)
input double SP2L_InpTpR            = 2.0;      // Take-profit multiple of risk (R)
input double SP2L_InpLotSize        = 1.0;      // Fixed lot size
input int    SP2L_InpSessionOpenHour  = 9;      // NY entry-window open hour
input int    SP2L_InpSessionOpenMin   = 30;     // NY entry-window open minute
input int    SP2L_InpLastEntryHour    = 15;     // NY last signal-bar hour
input int    SP2L_InpLastEntryMin     = 30;     // NY last signal-bar minute
input int    SP2L_InpForceExitHour  = 15;       // NY forced-exit hour
input int    SP2L_InpForceExitMin   = 55;       // NY forced-exit minute
input long   SP2L_InpMagicNumber    = 26082701; // Magic number

// ── Globals ─────────────────────────────────────────────────────────
CTrade   SP2L_g_trade;
double   SP2L_g_lot = 1.0;
datetime SP2L_g_lastM1Bar = 0;
double   SP2L_g_ema = 0.0;
bool     SP2L_g_emaSeeded = false;
double   SP2L_g_emaK = 2.0 / 61.0;

int      SP2L_g_nyDate = 0;

// pending state (one at a time)
bool     SP2L_g_pendActive = false;
int      SP2L_g_pendDir = 0;                    // +1 BUY / -1 SELL
double   SP2L_g_pendSl = 0.0;                   // pre-spike extreme
datetime SP2L_g_pendTime = 0;                   // trigger bar time (setup bar t)

// occupancy: suppress setups with trigger bar <= this time
datetime SP2L_g_resumeAfter = 0;

bool     SP2L_g_forceClose = false;

struct SP2L_DayRecord
{
   int    nyDate;
   int    setups;
   int    entered;
   int    killed;
   int    expired;
   int    entries;
   int    exitsTp;
   int    exitsSl;
   int    exitsTime;
   double totalPnl;
};
SP2L_DayRecord SP2L_g_days[];

// per-position context registry keyed by position id (ORB pattern)
long     SP2L_g_ctxPosId[];
double   SP2L_g_ctxTp[], g_ctxSL[];
datetime SP2L_g_ctxSetupTime[];
long     SP2L_g_exitPosIds[];
string   SP2L_g_exitReasons[];

// entry context (set at order send, consumed at the IN deal)
datetime SP2L_g_entrySetupTime = 0;

//+------------------------------------------------------------------+
//| Time / DST helpers (identical to ORB/VWAP EAs)                    |
//+------------------------------------------------------------------+
datetime SP2L_NthSundayUtc(const int year, const int month, const int nth, const int hourUtc)
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

int SP2L_NyOffsetHours(const datetime utc)
{
   MqlDateTime dt;
   TimeToStruct(utc, dt);
   datetime dstStart = SP2L_NthSundayUtc(dt.year, 3, 2, 7);    // 2nd Sunday March 07:00 UTC
   datetime dstEnd   = SP2L_NthSundayUtc(dt.year, 11, 1, 6);   // 1st Sunday Nov   06:00 UTC
   if(utc >= dstStart && utc < dstEnd)
      return -4;
   return -5;
}

int SP2L_NyDateOf(const datetime utc)
{
   MqlDateTime dt;
   TimeToStruct((datetime)((long)utc + (long)SP2L_NyOffsetHours(utc) * 3600), dt);
   return dt.year * 10000 + dt.mon * 100 + dt.day;
}

int SP2L_NyWallMin(const datetime utc)
{
   MqlDateTime dt;
   TimeToStruct((datetime)((long)utc + (long)SP2L_NyOffsetHours(utc) * 3600), dt);
   return dt.hour * 60 + dt.min;
}

datetime SP2L_NyToUtcOfDate(const int nyDate, const int h, const int mi)
{
   MqlDateTime dt;
   dt.year = nyDate / 10000; dt.mon = (nyDate / 100) % 100; dt.day = nyDate % 100;
   dt.hour = h; dt.min = mi; dt.sec = 0;
   datetime naive = StructToTime(dt);
   return (datetime)((long)naive - (long)SP2L_NyOffsetHours(naive) * 3600);
}

string SP2L_TsStr(const datetime t)
{
   return TimeToString(t, TIME_DATE | TIME_MINUTES);
}

double SP2L_NormPrice(const double p)
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
//| Day bookkeeping                                                    |
//+------------------------------------------------------------------+
int SP2L_DayIndex(const int nyDate)
{
   for(int i = ArraySize(SP2L_g_days) - 1; i >= 0; i--)
      if(SP2L_g_days[i].nyDate == nyDate)
         return i;
   return -1;
}

void SP2L_StartNewDay(const int nyDate)
{
   // a pending still alive at date rollover expires on its own date
   if(SP2L_g_pendActive)
   {
      int old = SP2L_DayIndex(SP2L_g_nyDate);
      if(old >= 0)
         SP2L_g_days[old].expired++;
      SP2L_g_pendActive = false;
   }

   int n = ArraySize(SP2L_g_days);
   ArrayResize(SP2L_g_days, n + 1);
   SP2L_g_days[n].nyDate   = nyDate;
   SP2L_g_days[n].setups   = 0;
   SP2L_g_days[n].entered  = 0;
   SP2L_g_days[n].killed   = 0;
   SP2L_g_days[n].expired  = 0;
   SP2L_g_days[n].entries  = 0;
   SP2L_g_days[n].exitsTp  = 0;
   SP2L_g_days[n].exitsSl  = 0;
   SP2L_g_days[n].exitsTime = 0;
   SP2L_g_days[n].totalPnl = 0.0;

   SP2L_g_nyDate     = nyDate;
   SP2L_g_pendActive = false;      // pending expires at NY date end
   SP2L_g_pendDir    = 0;
   SP2L_g_pendSl     = 0.0;
   SP2L_g_pendTime   = 0;
}

//+------------------------------------------------------------------+
//| Position helpers                                                   |
//+------------------------------------------------------------------+
ulong SP2L_OurPositionTicket()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0)
         continue;
      if((long)PositionGetInteger(POSITION_MAGIC) == SP2L_InpMagicNumber)
         return tk;
   }
   return 0;
}

void SP2L_AppendExitReason(const long posId, const string reason)
{
   int n = ArraySize(SP2L_g_exitPosIds);
   ArrayResize(SP2L_g_exitPosIds, n + 1);
   ArrayResize(SP2L_g_exitReasons, n + 1);
   SP2L_g_exitPosIds[n]   = posId;
   SP2L_g_exitReasons[n]  = reason;
}

string SP2L_FindExitReason(const long posId)
{
   for(int i = ArraySize(SP2L_g_exitPosIds) - 1; i >= 0; i--)
      if(SP2L_g_exitPosIds[i] == posId)
         return SP2L_g_exitReasons[i];
   return "";
}

int SP2L_CtxIndexOf(const long posId)
{
   for(int i = ArraySize(SP2L_g_ctxPosId) - 1; i >= 0; i--)
      if(SP2L_g_ctxPosId[i] == posId)
         return i;
   return -1;
}

double SP2L_CtxVal(const long posId, const double &vals[], const double dflt = 0.0)
{
   int i = SP2L_CtxIndexOf(posId);
   return (i >= 0) ? vals[i] : dflt;
}

void SP2L_CountExit(const string reason)
{
   int di = SP2L_DayIndex(SP2L_g_nyDate);
   if(di < 0)
      return;
   if(reason == "target")         SP2L_g_days[di].exitsTp++;
   else if(reason == "stop_loss") SP2L_g_days[di].exitsSl++;
   else if(reason == "time_exit") SP2L_g_days[di].exitsTime++;
}

//+------------------------------------------------------------------+
//| Expert initialization                                             |
//+------------------------------------------------------------------+
int SP2L_OnInit()
{
   SP2L_g_trade.SetExpertMagicNumber((ulong)SP2L_InpMagicNumber);
   SP2L_g_trade.SetDeviationInPoints(50);

   double vmin  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double vstep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double lot   = SP2L_InpLotSize;
   if(vstep > 0)
      lot = MathFloor(lot / vstep + 0.5) * vstep;
   if(vmin > 0 && lot < vmin)
      lot = vmin;
   SP2L_g_lot = lot;
   SP2L_g_emaK = 2.0 / (SP2L_InpEmaPeriod + 1.0);

   Print("=== SP2L_SpikeGap v1.0 init (NY_E, WF-confirmed) ===");
   PrintFormat("SP2L inputs: spike=%.2f gap=%.4f%% maxsl=%.4f%% ema=%d tp_r=%.2f "
               "lot=%.2f window=%02d:%02d-%02d:%02d force_exit=%02d:%02d magic=%I64d",
               SP2L_InpSpikeRatio, SP2L_InpGapPct, SP2L_InpMaxSlPct, SP2L_InpEmaPeriod, SP2L_InpTpR,
               SP2L_g_lot, SP2L_InpSessionOpenHour, SP2L_InpSessionOpenMin,
               SP2L_InpLastEntryHour, SP2L_InpLastEntryMin,
               SP2L_InpForceExitHour, SP2L_InpForceExitMin, SP2L_InpMagicNumber);

   int off = SP2L_NyOffsetHours(TimeCurrent());
   PrintFormat("SP2L NY offset now: %+d h -> server %s = NY %s",
               off, SP2L_TsStr(TimeCurrent()),
               SP2L_TsStr((datetime)((long)TimeCurrent() + (long)off * 3600)));
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   PrintFormat("SP2L DST boundaries %d: start=%s end=%s (UTC)",
               dt.year, SP2L_TsStr(SP2L_NthSundayUtc(dt.year, 3, 2, 7)),
               SP2L_TsStr(SP2L_NthSundayUtc(dt.year, 11, 1, 6)));

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Grid-safe market close (CTrade::PositionClose sends the RAW bid/  |
//| ask and is rejected 10015 on this custom symbol — VWAP EA fix).   |
//+------------------------------------------------------------------+
bool SP2L_ClosePositionGrid(const ulong ticket)
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
                   ? SP2L_NormPrice(SymbolInfoDouble(_Symbol, SYMBOL_BID))
                   : SP2L_NormPrice(SymbolInfoDouble(_Symbol, SYMBOL_ASK));
   req.deviation = 50;
   req.magic     = (ulong)SP2L_InpMagicNumber;
   req.comment   = "SP2L close";
   if(!OrderSend(req, res))
      return false;
   return (res.retcode == TRADE_RETCODE_DONE || res.retcode == TRADE_RETCODE_PLACED
           || res.retcode == TRADE_RETCODE_DONE_PARTIAL);
}

//+------------------------------------------------------------------+
//| 15:55 NY forced exit — checked on every tick                      |
//+------------------------------------------------------------------+
void SP2L_ManageForceClose()
{
   ulong ticket = SP2L_OurPositionTicket();
   if(ticket == 0)
      return;
   datetime now = TimeCurrent();
   int today = SP2L_NyDateOf(now);
   datetime exitUtc = SP2L_NyToUtcOfDate(today, SP2L_InpForceExitHour, SP2L_InpForceExitMin);
   if(now >= exitUtc)
   {
      SP2L_g_forceClose = true;
      if(!SP2L_ClosePositionGrid(ticket))
         Print("SP2L force-close failed retcode=", SP2L_g_trade.ResultRetcode());
   }
}

//+------------------------------------------------------------------+
//| Market entry (E-NEXT: first tick of the bar after signal bar u)   |
//+------------------------------------------------------------------+
void SP2L_PlaceMarketEntry(const int dir, const datetime setupTime)
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double refPx = (dir > 0) ? ask : bid;   // expected fill side
   double px = SP2L_NormPrice(refPx);           // off-grid market price => 10015
   double sl = SP2L_NormPrice(SP2L_g_pendSl);
   double risk = (dir > 0) ? (px - sl) : (sl - px);
   if(risk <= 0)
   {
      Print("SP2L entry aborted: grid risk <= 0");
      return;
   }
   double tp = SP2L_NormPrice(px + dir * SP2L_InpTpR * risk);

   SP2L_g_entrySetupTime = setupTime;
   SP2L_g_forceClose = false;

   bool ok = false;
   if(dir > 0)
   {
      ok = SP2L_g_trade.Buy(SP2L_g_lot, _Symbol, px, sl, tp, "SP2L long");
      if(!ok)
         ok = SP2L_g_trade.Buy(SP2L_g_lot, _Symbol, px, sl, tp, "SP2L long retry");
   }
   else
   {
      ok = SP2L_g_trade.Sell(SP2L_g_lot, _Symbol, px, sl, tp, "SP2L short");
      if(!ok)
         ok = SP2L_g_trade.Sell(SP2L_g_lot, _Symbol, px, sl, tp, "SP2L short retry");
   }
   if(!ok)
   {
      PrintFormat("SP2L %d %s entry FAILED retcode=%d", SP2L_g_nyDate,
                  (dir > 0 ? "BUY" : "SELL"), SP2L_g_trade.ResultRetcode());
      return;
   }

   int di = SP2L_DayIndex(SP2L_g_nyDate);
   if(di >= 0)
      SP2L_g_days[di].entries++;
   PrintFormat("SP2L %d ENTRY %s @ %s sl=%s tp=%s (setup %s)",
               SP2L_g_nyDate, (dir > 0 ? "BUY" : "SELL"),
               DoubleToString(SP2L_g_trade.ResultPrice()),
               DoubleToString(sl), DoubleToString(tp), SP2L_TsStr(setupTime));
}

//+------------------------------------------------------------------+
//| Setup detection on completed trigger bar t (shift 1)              |
//| Returns +1 BUY / -1 SELL / 0 none; sets SP2L_g_pendSl accordingly.     |
//+------------------------------------------------------------------+
double SP2L_g_newPendSl = 0.0;

int SP2L_DetectSetup(const datetime tTime)
{
   if(tTime <= SP2L_g_resumeAfter)
      return 0;                             // occupancy shadow (t <= eb)
   if(iTime(_Symbol, PERIOD_M1, 4) <= 0)
      return 0;                             // not enough history

   // trigger t = shift1, post-spike t-1 = shift2, spike t-2 = shift3,
   // pre-spike t-3 = shift4
   double l1 = iLow(_Symbol,  PERIOD_M1, 1),  h1 = iHigh(_Symbol, PERIOD_M1, 1);
   double o1 = iOpen(_Symbol, PERIOD_M1, 1),  c1 = iClose(_Symbol, PERIOD_M1, 1);
   double o2 = iOpen(_Symbol, PERIOD_M1, 2),  c2 = iClose(_Symbol, PERIOD_M1, 2);
   double o3 = iOpen(_Symbol, PERIOD_M1, 3),  c3 = iClose(_Symbol, PERIOD_M1, 3);
   double o4 = iOpen(_Symbol, PERIOD_M1, 4),  c4 = iClose(_Symbol, PERIOD_M1, 4);
   double h4 = iHigh(_Symbol, PERIOD_M1, 4),  l4 = iLow(_Symbol,  PERIOD_M1, 4);

   double b2 = c2 - o2, b3 = c3 - o3, b4 = c4 - o4, bt = c1 - o1;
   double gapF = 1.0 + SP2L_InpGapPct / 100.0;

   bool buy = (l1 < iLow(_Symbol, PERIOD_M1, 2))
              && (c2 > c3) && (o2 > o3)
              && (c3 > c4) && (o3 > o4)
              && (c2 > o2) && (c3 > o3) && (c4 > o4)
              && (iLow(_Symbol, PERIOD_M1, 2) > h4 * gapF)
              && (b3 > SP2L_InpSpikeRatio * b2)
              && (b3 > SP2L_InpSpikeRatio * b4)
              && (b3 > SP2L_InpSpikeRatio * bt);

   bool sell = (h1 > iHigh(_Symbol, PERIOD_M1, 2))
               && (c2 < c3) && (o2 < o3)
               && (c3 < c4) && (o3 < o4)
               && (c2 < o2) && (c3 < o3) && (c4 < o4)
               && (iHigh(_Symbol, PERIOD_M1, 2) < l4 * (2.0 - gapF))
               && (-b3 > SP2L_InpSpikeRatio * -b2)
               && (-b3 > SP2L_InpSpikeRatio * -b4)
               && (-b3 > SP2L_InpSpikeRatio * -bt);

   if(buy && sell)
      return 0;                             // source: buy and not sell
   if(buy)
   {
      SP2L_g_newPendSl = l4;                     // SL = low of pre-spike candle
      return 1;
   }
   if(sell)
   {
      SP2L_g_newPendSl = h4;                     // SL = high of pre-spike candle
      return -1;
   }
   return 0;
}

//+------------------------------------------------------------------+
//| Pending evaluation on completed bar u (shift 1)                   |
//| Returns +1 entry / 0 keep waiting / -1 killed / -2 expired        |
//+------------------------------------------------------------------+
int SP2L_CheckPending()
{
   double lu = iLow(_Symbol,  PERIOD_M1, 1), hu = iHigh(_Symbol, PERIOD_M1, 1);
   double cu = iClose(_Symbol, PERIOD_M1, 1);
   datetime uTime = iTime(_Symbol, PERIOD_M1, 1);
   int w = SP2L_NyWallMin(uTime);
   int lastEntryMin = SP2L_InpLastEntryHour * 60 + SP2L_InpLastEntryMin;

   if(w > lastEntryMin)
      return -2;                            // window over: expire at date end

   double risk;
   if(SP2L_g_pendDir > 0)
   {
      if(lu >= iLow(_Symbol, PERIOD_M1, 2))
         return 0;                          // no break
      risk = lu - SP2L_g_pendSl;
      if(risk <= 0)
         return 0;                          // pullback not deep enough
      if(risk > SP2L_InpMaxSlPct / 100.0 * lu)
         return -1;                         // INVALID: kill
      if(!(cu > SP2L_g_ema))
         return 0;                          // EMA filter
   }
   else
   {
      if(hu <= iHigh(_Symbol, PERIOD_M1, 2))
         return 0;
      risk = SP2L_g_pendSl - hu;
      if(risk <= 0)
         return 0;
      if(risk > SP2L_InpMaxSlPct / 100.0 * hu)
         return -1;
      if(!(cu < SP2L_g_ema))
         return 0;
   }

   if(w < SP2L_InpSessionOpenHour * 60 + SP2L_InpSessionOpenMin)
      return 0;                             // before session: keep waiting

   return 1;                                // ENTRY at next bar's first tick
}

//+------------------------------------------------------------------+
//| New M1 bar state machine                                          |
//+------------------------------------------------------------------+
void SP2L_OnNewM1Bar(const datetime t0)
{
   int nyDate = SP2L_NyDateOf(t0);
   if(nyDate != SP2L_g_nyDate)
   {
      SP2L_StartNewDay(nyDate);
      PrintFormat("SP2L new day %d (offset %+d h)", nyDate, SP2L_NyOffsetHours(t0));
   }
   int di = SP2L_DayIndex(nyDate);
   if(di < 0)
      return;

   datetime uTime = iTime(_Symbol, PERIOD_M1, 1);
   if(uTime <= 0)
      return;
   if(SP2L_NyDateOf(uTime) != nyDate)
   {
      // completed bar belongs to the prior NY date: only EMA advances
      double cPrev = iClose(_Symbol, PERIOD_M1, 1);
      if(!SP2L_g_emaSeeded) { SP2L_g_ema = cPrev; SP2L_g_emaSeeded = true; }
      else               SP2L_g_ema += SP2L_g_emaK * (cPrev - SP2L_g_ema);
      return;
   }

   // EMA over every completed bar (overnight included)
   double cu = iClose(_Symbol, PERIOD_M1, 1);
   if(!SP2L_g_emaSeeded) { SP2L_g_ema = cu; SP2L_g_emaSeeded = true; }
   else               SP2L_g_ema += SP2L_g_emaK * (cu - SP2L_g_ema);

   bool holding = (SP2L_OurPositionTicket() != 0);

   // ---- pending first (one machine at a time) ------------------------
   if(SP2L_g_pendActive && !holding)
   {
      int r = SP2L_CheckPending();
      if(r == 1)
      {
         SP2L_g_days[di].entered++;
         datetime setupT = SP2L_g_pendTime;
         int dir = SP2L_g_pendDir;
         SP2L_g_pendActive = false;              // consumed by the entry
         if(RegimeAllowSP2L())
            SP2L_PlaceMarketEntry(dir, setupT);
         holding = true;                    // block detection this event
      }
      else if(r == -1 || r == -2)
      {
         SP2L_g_pendActive = false;
         if(r == -2)
            SP2L_g_days[di].expired++;           // window over
         else
            SP2L_g_days[di].killed++;            // risk cap (INVALID)
         SP2L_g_resumeAfter = uTime;             // skip setups t <= outcome bar
      }
      // r == 0: keep waiting
   }

   // ---- setup detection (only while flat and pending-free) -----------
   if(!holding && !SP2L_g_pendActive)
   {
      int d = (RegimeAllowSP2L() ? SP2L_DetectSetup(uTime) : 0);
      if(d != 0)
      {
         SP2L_g_pendActive = true;
         SP2L_g_pendDir    = d;
         SP2L_g_pendSl     = SP2L_g_newPendSl;
         SP2L_g_pendTime   = uTime;
         SP2L_g_days[di].setups++;
         PrintFormat("SP2L %d %s pending setup (bar %s, sl=%s)",
                     nyDate, (d > 0 ? "BUY" : "SELL"), SP2L_TsStr(uTime),
                     DoubleToString(SP2L_g_pendSl));
      }
   }
}

//+------------------------------------------------------------------+
//| Expert tick                                                       |
//+------------------------------------------------------------------+
void SP2L_OnTick()
{
   SP2L_ManageForceClose();

   datetime t0 = iTime(_Symbol, PERIOD_M1, 0);
   if(t0 == 0)
      return;
   if(t0 == SP2L_g_lastM1Bar)
      return;
   SP2L_g_lastM1Bar = t0;
   SP2L_OnNewM1Bar(t0);
}

//+------------------------------------------------------------------+
//| Exit-reason fallback (SL tolerance like VWAP EA: 0.5 pt)          |
//+------------------------------------------------------------------+
string SP2L_FallbackReason(const double closePrice, const datetime closeTime,
                      const double sl, const double tp)
{
   double tol = MathMax(0.5, 10.0 * _Point);
   if(sl > 0 && MathAbs(closePrice - sl) <= tol)
      return "stop_loss";
   if(tp > 0 && MathAbs(closePrice - tp) <= tol)
      return "target";
   int nd = SP2L_NyDateOf(closeTime);
   datetime exitUtc = SP2L_NyToUtcOfDate(nd, SP2L_InpForceExitHour, SP2L_InpForceExitMin);
   long d = (long)closeTime - (long)exitUtc;
   if(d >= -600 && d < 3600)
      return "time_exit";
   return "other";
}

//+------------------------------------------------------------------+
//| Trade transactions: exit-reason + context capture                 |
//+------------------------------------------------------------------+
void SP2L_OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
{
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD)
      return;
   if(!HistoryDealSelect(trans.deal))
      return;
   if((long)HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != SP2L_InpMagicNumber)
      return;

   ENUM_DEAL_ENTRY de = (ENUM_DEAL_ENTRY)HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
   long posId = (long)HistoryDealGetInteger(trans.deal, DEAL_POSITION_ID);

   if(de == DEAL_ENTRY_IN)
   {
      int n = ArraySize(SP2L_g_ctxPosId);
      ArrayResize(SP2L_g_ctxPosId, n + 1);
      ArrayResize(SP2L_g_ctxTp, n + 1);
      ArrayResize(g_ctxSL, n + 1);
      ArrayResize(SP2L_g_ctxSetupTime, n + 1);
      SP2L_g_ctxPosId[n] = posId;
      SP2L_g_ctxTp[n]    = HistoryDealGetDouble(trans.deal, DEAL_TP);
      g_ctxSL[n]    = HistoryDealGetDouble(trans.deal, DEAL_SL);
      SP2L_g_ctxSetupTime[n] = SP2L_g_entrySetupTime;
      PrintFormat("SP2L entry deal: pos=%I64d @ %s (setup %s)",
                  posId,
                  DoubleToString(HistoryDealGetDouble(trans.deal, DEAL_PRICE)),
                  SP2L_TsStr(SP2L_g_entrySetupTime));
   }
   else if(de == DEAL_ENTRY_OUT || de == DEAL_ENTRY_OUT_BY || de == DEAL_ENTRY_INOUT)
   {
      double closePrice = HistoryDealGetDouble(trans.deal, DEAL_PRICE);
      datetime closeTime = (datetime)HistoryDealGetInteger(trans.deal, DEAL_TIME);

      string reason;
      long dreason = HistoryDealGetInteger(trans.deal, DEAL_REASON);
      if(SP2L_g_forceClose)
         reason = "time_exit";
      else if(dreason == DEAL_REASON_SL)
         reason = "stop_loss";
      else if(dreason == DEAL_REASON_TP)
         reason = "target";
      else
      {
         double slCtx = SP2L_CtxVal(posId, g_ctxSL, 0.0);
         double tpCtx = SP2L_CtxVal(posId, SP2L_g_ctxTp, 0.0);
         reason = SP2L_FallbackReason(closePrice, closeTime, slCtx, tpCtx);
      }

      SP2L_AppendExitReason(posId, reason);
      SP2L_CountExit(reason);
      int di = SP2L_DayIndex(SP2L_g_nyDate);
      if(di >= 0)
         SP2L_g_days[di].totalPnl += HistoryDealGetDouble(trans.deal, DEAL_PROFIT)
                                + HistoryDealGetDouble(trans.deal, DEAL_SWAP)
                                + HistoryDealGetDouble(trans.deal, DEAL_COMMISSION);

      // occupancy: setups with trigger bar <= exit bar are skipped;
      // exit happens during the CURRENT forming bar
      SP2L_g_resumeAfter = iTime(_Symbol, PERIOD_M1, 0);
      if(SP2L_g_resumeAfter == 0)
         SP2L_g_resumeAfter = closeTime;
      SP2L_g_forceClose = false;
      PrintFormat("SP2L exit: pos=%I64d reason=%s @ %s", posId, reason,
                  DoubleToString(closePrice));
   }
}

//+------------------------------------------------------------------+
//| trades.csv — collector-compatible schema                          |
//+------------------------------------------------------------------+
void SP2L_WriteTradesCsv()
{
   int handle = FileOpen("sp2l_trades.csv", FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(handle == INVALID_HANDLE)
   {
      Print("WARNING: could not open trades.csv for writing. Error=", GetLastError());
      return;
   }
   string header = "ticket;open_time;close_time;symbol;type;lot;open_price;stop_loss;"
                   "take_profit;close_price;exit_reason;swap;commission;profit;"
                   "setup_time\r\n";
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
      if((long)HistoryDealGetInteger(outTicket, DEAL_MAGIC) != SP2L_InpMagicNumber)
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
         ulong dtk = HistoryDealGetTicket(j);
         if(dtk == 0 || dtk == outTicket)
            continue;
         if((long)HistoryDealGetInteger(dtk, DEAL_MAGIC) != SP2L_InpMagicNumber)
            continue;
         if((ENUM_DEAL_ENTRY)HistoryDealGetInteger(dtk, DEAL_ENTRY) != DEAL_ENTRY_IN)
            continue;
         if((long)HistoryDealGetInteger(dtk, DEAL_POSITION_ID) != posId)
            continue;
         inTicket = dtk;
         openPrice = HistoryDealGetDouble(dtk, DEAL_PRICE);
         openTime = (datetime)HistoryDealGetInteger(dtk, DEAL_TIME);
         inDealType = HistoryDealGetInteger(dtk, DEAL_TYPE);
         inSL = HistoryDealGetDouble(dtk, DEAL_SL);
         inTP = HistoryDealGetDouble(dtk, DEAL_TP);
         inComm = HistoryDealGetDouble(dtk, DEAL_COMMISSION);
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

      double tpSnap = SP2L_CtxVal(posId, SP2L_g_ctxTp, 0.0);
      if(tpSnap <= 0.0)
         tpSnap = inTP;
      double slSnap = SP2L_CtxVal(posId, g_ctxSL, 0.0);
      if(slSnap <= 0.0)
         slSnap = inSL;

      string reason = SP2L_FindExitReason(posId);
      if(reason == "")
         reason = SP2L_FallbackReason(closePrice, closeTime, slSnap, tpSnap);

      int ci = SP2L_CtxIndexOf(posId);
      datetime setupTs = (ci >= 0) ? SP2L_g_ctxSetupTime[ci] : openTime;

      string line = IntegerToString((long)inTicket) + ";" +
                    SP2L_TsStr(openTime) + ";" +
                    SP2L_TsStr(closeTime) + ";" +
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
                    SP2L_TsStr(setupTs) + "\r\n";
      FileWriteString(handle, line, StringLen(line));
      written++;
   }

   FileClose(handle);
   Print("Trade CSV written: trades.csv (", written, " trades)");
}

//+------------------------------------------------------------------+
//| sp2l_days.csv — per-day state for reconciliation                  |
//+------------------------------------------------------------------+
void SP2L_WriteDaysCsv()
{
   int handle = FileOpen("sp2l_days.csv", FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(handle == INVALID_HANDLE)
   {
      Print("WARNING: could not open sp2l_days.csv for writing. Error=", GetLastError());
      return;
   }
   string header = "ny_date;setups;entered;killed;expired;entries;"
                   "exits_tp;exits_sl;exits_time;total_pnl\r\n";
   FileWriteString(handle, header, StringLen(header));

   int n = ArraySize(SP2L_g_days);
   for(int i = 0; i < n; i++)
   {
      string line = IntegerToString(SP2L_g_days[i].nyDate) + ";" +
                    IntegerToString(SP2L_g_days[i].setups) + ";" +
                    IntegerToString(SP2L_g_days[i].entered) + ";" +
                    IntegerToString(SP2L_g_days[i].killed) + ";" +
                    IntegerToString(SP2L_g_days[i].expired) + ";" +
                    IntegerToString(SP2L_g_days[i].entries) + ";" +
                    IntegerToString(SP2L_g_days[i].exitsTp) + ";" +
                    IntegerToString(SP2L_g_days[i].exitsSl) + ";" +
                    IntegerToString(SP2L_g_days[i].exitsTime) + ";" +
                    DoubleToString(SP2L_g_days[i].totalPnl, 2) + "\r\n";
      FileWriteString(handle, line, StringLen(line));
   }

   FileClose(handle);
   Print("Day CSV written: sp2l_days.csv (", n, " days)");
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void SP2L_OnDeinit(const int reason)
{
   Print("=== SP2L_SpikeGap deinit (reason=", reason, ") ===");
   SP2L_WriteTradesCsv();
   SP2L_WriteDaysCsv();
}
//+------------------------------------------------------------------+