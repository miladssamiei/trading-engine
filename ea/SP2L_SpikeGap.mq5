//+------------------------------------------------------------------+
//|                                                SP2L_SpikeGap.mq5  |
//|   SP2L spike/gap/continuation — native, self-contained EA        |
//|   No DLLs, no sockets, no Python bridge, no ML.                  |
//+------------------------------------------------------------------+
#property copyright "RR project"
#property version   "1.00"
#property description "SP2L spike/gap strategy on US500-tick (server time = UTC), M1"
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

#include <Trade\Trade.mqh>

// ── Inputs (defaults ARE the confirmed spec) ─────────────────────────
input double InpSpikeRatio     = 1.5;      // Spike body ratio vs neighbors
input double InpGapPct         = 0.04;     // Gap threshold (% of price)
input double InpMaxSlPct       = 0.40;     // Max SL distance (% of price)
input int    InpEmaPeriod      = 60;       // EMA filter period (bars)
input double InpTpR            = 2.0;      // Take-profit multiple of risk (R)
input double InpLotSize        = 1.0;      // Fixed lot size
input int    InpSessionOpenHour  = 9;      // NY entry-window open hour
input int    InpSessionOpenMin   = 30;     // NY entry-window open minute
input int    InpLastEntryHour    = 15;     // NY last signal-bar hour
input int    InpLastEntryMin     = 30;     // NY last signal-bar minute
input int    InpForceExitHour  = 15;       // NY forced-exit hour
input int    InpForceExitMin   = 55;       // NY forced-exit minute
input long   InpMagicNumber    = 26082701; // Magic number

// ── Globals ─────────────────────────────────────────────────────────
CTrade   g_trade;
double   g_lot = 1.0;
datetime g_lastM1Bar = 0;
double   g_ema = 0.0;
bool     g_emaSeeded = false;
double   g_emaK = 2.0 / 61.0;

int      g_nyDate = 0;

// pending state (one at a time)
bool     g_pendActive = false;
int      g_pendDir = 0;                    // +1 BUY / -1 SELL
double   g_pendSl = 0.0;                   // pre-spike extreme
datetime g_pendTime = 0;                   // trigger bar time (setup bar t)

// occupancy: suppress setups with trigger bar <= this time
datetime g_resumeAfter = 0;

bool     g_forceClose = false;

struct DayRecord
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
DayRecord g_days[];

// per-position context registry keyed by position id (ORB pattern)
long     g_ctxPosId[];
double   g_ctxTp[], g_ctxSL[];
datetime g_ctxSetupTime[];
long     g_exitPosIds[];
string   g_exitReasons[];

// entry context (set at order send, consumed at the IN deal)
datetime g_entrySetupTime = 0;

//+------------------------------------------------------------------+
//| Time / DST helpers (identical to ORB/VWAP EAs)                    |
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

int NyDateOf(const datetime utc)
{
   MqlDateTime dt;
   TimeToStruct((datetime)((long)utc + (long)NyOffsetHours(utc) * 3600), dt);
   return dt.year * 10000 + dt.mon * 100 + dt.day;
}

int NyWallMin(const datetime utc)
{
   MqlDateTime dt;
   TimeToStruct((datetime)((long)utc + (long)NyOffsetHours(utc) * 3600), dt);
   return dt.hour * 60 + dt.min;
}

datetime NyToUtcOfDate(const int nyDate, const int h, const int mi)
{
   MqlDateTime dt;
   dt.year = nyDate / 10000; dt.mon = (nyDate / 100) % 100; dt.day = nyDate % 100;
   dt.hour = h; dt.min = mi; dt.sec = 0;
   datetime naive = StructToTime(dt);
   return (datetime)((long)naive - (long)NyOffsetHours(naive) * 3600);
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
//| Day bookkeeping                                                    |
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
   // a pending still alive at date rollover expires on its own date
   if(g_pendActive)
   {
      int old = DayIndex(g_nyDate);
      if(old >= 0)
         g_days[old].expired++;
      g_pendActive = false;
   }

   int n = ArraySize(g_days);
   ArrayResize(g_days, n + 1);
   g_days[n].nyDate   = nyDate;
   g_days[n].setups   = 0;
   g_days[n].entered  = 0;
   g_days[n].killed   = 0;
   g_days[n].expired  = 0;
   g_days[n].entries  = 0;
   g_days[n].exitsTp  = 0;
   g_days[n].exitsSl  = 0;
   g_days[n].exitsTime = 0;
   g_days[n].totalPnl = 0.0;

   g_nyDate     = nyDate;
   g_pendActive = false;      // pending expires at NY date end
   g_pendDir    = 0;
   g_pendSl     = 0.0;
   g_pendTime   = 0;
}

//+------------------------------------------------------------------+
//| Position helpers                                                   |
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

void CountExit(const string reason)
{
   int di = DayIndex(g_nyDate);
   if(di < 0)
      return;
   if(reason == "target")         g_days[di].exitsTp++;
   else if(reason == "stop_loss") g_days[di].exitsSl++;
   else if(reason == "time_exit") g_days[di].exitsTime++;
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
   g_emaK = 2.0 / (InpEmaPeriod + 1.0);

   Print("=== SP2L_SpikeGap v1.0 init (NY_E, WF-confirmed) ===");
   PrintFormat("SP2L inputs: spike=%.2f gap=%.4f%% maxsl=%.4f%% ema=%d tp_r=%.2f "
               "lot=%.2f window=%02d:%02d-%02d:%02d force_exit=%02d:%02d magic=%I64d",
               InpSpikeRatio, InpGapPct, InpMaxSlPct, InpEmaPeriod, InpTpR,
               g_lot, InpSessionOpenHour, InpSessionOpenMin,
               InpLastEntryHour, InpLastEntryMin,
               InpForceExitHour, InpForceExitMin, InpMagicNumber);

   int off = NyOffsetHours(TimeCurrent());
   PrintFormat("SP2L NY offset now: %+d h -> server %s = NY %s",
               off, TsStr(TimeCurrent()),
               TsStr((datetime)((long)TimeCurrent() + (long)off * 3600)));
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   PrintFormat("SP2L DST boundaries %d: start=%s end=%s (UTC)",
               dt.year, TsStr(NthSundayUtc(dt.year, 3, 2, 7)),
               TsStr(NthSundayUtc(dt.year, 11, 1, 6)));

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Grid-safe market close (CTrade::PositionClose sends the RAW bid/  |
//| ask and is rejected 10015 on this custom symbol — VWAP EA fix).   |
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
   req.comment   = "SP2L close";
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
   int today = NyDateOf(now);
   datetime exitUtc = NyToUtcOfDate(today, InpForceExitHour, InpForceExitMin);
   if(now >= exitUtc)
   {
      g_forceClose = true;
      if(!ClosePositionGrid(ticket))
         Print("SP2L force-close failed retcode=", g_trade.ResultRetcode());
   }
}

//+------------------------------------------------------------------+
//| Market entry (E-NEXT: first tick of the bar after signal bar u)   |
//+------------------------------------------------------------------+
void PlaceMarketEntry(const int dir, const datetime setupTime)
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double refPx = (dir > 0) ? ask : bid;   // expected fill side
   double px = NormPrice(refPx);           // off-grid market price => 10015
   double sl = NormPrice(g_pendSl);
   double risk = (dir > 0) ? (px - sl) : (sl - px);
   if(risk <= 0)
   {
      Print("SP2L entry aborted: grid risk <= 0");
      return;
   }
   double tp = NormPrice(px + dir * InpTpR * risk);

   g_entrySetupTime = setupTime;
   g_forceClose = false;

   bool ok = false;
   if(dir > 0)
   {
      ok = g_trade.Buy(g_lot, _Symbol, px, sl, tp, "SP2L long");
      if(!ok)
         ok = g_trade.Buy(g_lot, _Symbol, px, sl, tp, "SP2L long retry");
   }
   else
   {
      ok = g_trade.Sell(g_lot, _Symbol, px, sl, tp, "SP2L short");
      if(!ok)
         ok = g_trade.Sell(g_lot, _Symbol, px, sl, tp, "SP2L short retry");
   }
   if(!ok)
   {
      PrintFormat("SP2L %d %s entry FAILED retcode=%d", g_nyDate,
                  (dir > 0 ? "BUY" : "SELL"), g_trade.ResultRetcode());
      return;
   }

   int di = DayIndex(g_nyDate);
   if(di >= 0)
      g_days[di].entries++;
   PrintFormat("SP2L %d ENTRY %s @ %s sl=%s tp=%s (setup %s)",
               g_nyDate, (dir > 0 ? "BUY" : "SELL"),
               DoubleToString(g_trade.ResultPrice()),
               DoubleToString(sl), DoubleToString(tp), TsStr(setupTime));
}

//+------------------------------------------------------------------+
//| Setup detection on completed trigger bar t (shift 1)              |
//| Returns +1 BUY / -1 SELL / 0 none; sets g_pendSl accordingly.     |
//+------------------------------------------------------------------+
double g_newPendSl = 0.0;

int DetectSetup(const datetime tTime)
{
   if(tTime <= g_resumeAfter)
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
   double gapF = 1.0 + InpGapPct / 100.0;

   bool buy = (l1 < iLow(_Symbol, PERIOD_M1, 2))
              && (c2 > c3) && (o2 > o3)
              && (c3 > c4) && (o3 > o4)
              && (c2 > o2) && (c3 > o3) && (c4 > o4)
              && (iLow(_Symbol, PERIOD_M1, 2) > h4 * gapF)
              && (b3 > InpSpikeRatio * b2)
              && (b3 > InpSpikeRatio * b4)
              && (b3 > InpSpikeRatio * bt);

   bool sell = (h1 > iHigh(_Symbol, PERIOD_M1, 2))
               && (c2 < c3) && (o2 < o3)
               && (c3 < c4) && (o3 < o4)
               && (c2 < o2) && (c3 < o3) && (c4 < o4)
               && (iHigh(_Symbol, PERIOD_M1, 2) < l4 * (2.0 - gapF))
               && (-b3 > InpSpikeRatio * -b2)
               && (-b3 > InpSpikeRatio * -b4)
               && (-b3 > InpSpikeRatio * -bt);

   if(buy && sell)
      return 0;                             // source: buy and not sell
   if(buy)
   {
      g_newPendSl = l4;                     // SL = low of pre-spike candle
      return 1;
   }
   if(sell)
   {
      g_newPendSl = h4;                     // SL = high of pre-spike candle
      return -1;
   }
   return 0;
}

//+------------------------------------------------------------------+
//| Pending evaluation on completed bar u (shift 1)                   |
//| Returns +1 entry / 0 keep waiting / -1 killed / -2 expired        |
//+------------------------------------------------------------------+
int CheckPending()
{
   double lu = iLow(_Symbol,  PERIOD_M1, 1), hu = iHigh(_Symbol, PERIOD_M1, 1);
   double cu = iClose(_Symbol, PERIOD_M1, 1);
   datetime uTime = iTime(_Symbol, PERIOD_M1, 1);
   int w = NyWallMin(uTime);
   int lastEntryMin = InpLastEntryHour * 60 + InpLastEntryMin;

   if(w > lastEntryMin)
      return -2;                            // window over: expire at date end

   double risk;
   if(g_pendDir > 0)
   {
      if(lu >= iLow(_Symbol, PERIOD_M1, 2))
         return 0;                          // no break
      risk = lu - g_pendSl;
      if(risk <= 0)
         return 0;                          // pullback not deep enough
      if(risk > InpMaxSlPct / 100.0 * lu)
         return -1;                         // INVALID: kill
      if(!(cu > g_ema))
         return 0;                          // EMA filter
   }
   else
   {
      if(hu <= iHigh(_Symbol, PERIOD_M1, 2))
         return 0;
      risk = g_pendSl - hu;
      if(risk <= 0)
         return 0;
      if(risk > InpMaxSlPct / 100.0 * hu)
         return -1;
      if(!(cu < g_ema))
         return 0;
   }

   if(w < InpSessionOpenHour * 60 + InpSessionOpenMin)
      return 0;                             // before session: keep waiting

   return 1;                                // ENTRY at next bar's first tick
}

//+------------------------------------------------------------------+
//| New M1 bar state machine                                          |
//+------------------------------------------------------------------+
void OnNewM1Bar(const datetime t0)
{
   int nyDate = NyDateOf(t0);
   if(nyDate != g_nyDate)
   {
      StartNewDay(nyDate);
      PrintFormat("SP2L new day %d (offset %+d h)", nyDate, NyOffsetHours(t0));
   }
   int di = DayIndex(nyDate);
   if(di < 0)
      return;

   datetime uTime = iTime(_Symbol, PERIOD_M1, 1);
   if(uTime <= 0)
      return;
   if(NyDateOf(uTime) != nyDate)
   {
      // completed bar belongs to the prior NY date: only EMA advances
      double cPrev = iClose(_Symbol, PERIOD_M1, 1);
      if(!g_emaSeeded) { g_ema = cPrev; g_emaSeeded = true; }
      else               g_ema += g_emaK * (cPrev - g_ema);
      return;
   }

   // EMA over every completed bar (overnight included)
   double cu = iClose(_Symbol, PERIOD_M1, 1);
   if(!g_emaSeeded) { g_ema = cu; g_emaSeeded = true; }
   else               g_ema += g_emaK * (cu - g_ema);

   bool holding = (OurPositionTicket() != 0);

   // ---- pending first (one machine at a time) ------------------------
   if(g_pendActive && !holding)
   {
      int r = CheckPending();
      if(r == 1)
      {
         g_days[di].entered++;
         datetime setupT = g_pendTime;
         int dir = g_pendDir;
         g_pendActive = false;              // consumed by the entry
         PlaceMarketEntry(dir, setupT);
         holding = true;                    // block detection this event
      }
      else if(r == -1 || r == -2)
      {
         g_pendActive = false;
         if(r == -2)
            g_days[di].expired++;           // window over
         else
            g_days[di].killed++;            // risk cap (INVALID)
         g_resumeAfter = uTime;             // skip setups t <= outcome bar
      }
      // r == 0: keep waiting
   }

   // ---- setup detection (only while flat and pending-free) -----------
   if(!holding && !g_pendActive)
   {
      int d = DetectSetup(uTime);
      if(d != 0)
      {
         g_pendActive = true;
         g_pendDir    = d;
         g_pendSl     = g_newPendSl;
         g_pendTime   = uTime;
         g_days[di].setups++;
         PrintFormat("SP2L %d %s pending setup (bar %s, sl=%s)",
                     nyDate, (d > 0 ? "BUY" : "SELL"), TsStr(uTime),
                     DoubleToString(g_pendSl));
      }
   }
}

//+------------------------------------------------------------------+
//| Expert tick                                                       |
//+------------------------------------------------------------------+
void OnTick()
{
   ManageForceClose();

   datetime t0 = iTime(_Symbol, PERIOD_M1, 0);
   if(t0 == 0)
      return;
   if(t0 == g_lastM1Bar)
      return;
   g_lastM1Bar = t0;
   OnNewM1Bar(t0);
}

//+------------------------------------------------------------------+
//| Exit-reason fallback (SL tolerance like VWAP EA: 0.5 pt)          |
//+------------------------------------------------------------------+
string FallbackReason(const double closePrice, const datetime closeTime,
                      const double sl, const double tp)
{
   double tol = MathMax(0.5, 10.0 * _Point);
   if(sl > 0 && MathAbs(closePrice - sl) <= tol)
      return "stop_loss";
   if(tp > 0 && MathAbs(closePrice - tp) <= tol)
      return "target";
   int nd = NyDateOf(closeTime);
   datetime exitUtc = NyToUtcOfDate(nd, InpForceExitHour, InpForceExitMin);
   long d = (long)closeTime - (long)exitUtc;
   if(d >= -600 && d < 3600)
      return "time_exit";
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
      int n = ArraySize(g_ctxPosId);
      ArrayResize(g_ctxPosId, n + 1);
      ArrayResize(g_ctxTp, n + 1);
      ArrayResize(g_ctxSL, n + 1);
      ArrayResize(g_ctxSetupTime, n + 1);
      g_ctxPosId[n] = posId;
      g_ctxTp[n]    = HistoryDealGetDouble(trans.deal, DEAL_TP);
      g_ctxSL[n]    = HistoryDealGetDouble(trans.deal, DEAL_SL);
      g_ctxSetupTime[n] = g_entrySetupTime;
      PrintFormat("SP2L entry deal: pos=%I64d @ %s (setup %s)",
                  posId,
                  DoubleToString(HistoryDealGetDouble(trans.deal, DEAL_PRICE)),
                  TsStr(g_entrySetupTime));
   }
   else if(de == DEAL_ENTRY_OUT || de == DEAL_ENTRY_OUT_BY || de == DEAL_ENTRY_INOUT)
   {
      double closePrice = HistoryDealGetDouble(trans.deal, DEAL_PRICE);
      datetime closeTime = (datetime)HistoryDealGetInteger(trans.deal, DEAL_TIME);

      string reason;
      long dreason = HistoryDealGetInteger(trans.deal, DEAL_REASON);
      if(g_forceClose)
         reason = "time_exit";
      else if(dreason == DEAL_REASON_SL)
         reason = "stop_loss";
      else if(dreason == DEAL_REASON_TP)
         reason = "target";
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

      // occupancy: setups with trigger bar <= exit bar are skipped;
      // exit happens during the CURRENT forming bar
      g_resumeAfter = iTime(_Symbol, PERIOD_M1, 0);
      if(g_resumeAfter == 0)
         g_resumeAfter = closeTime;
      g_forceClose = false;
      PrintFormat("SP2L exit: pos=%I64d reason=%s @ %s", posId, reason,
                  DoubleToString(closePrice));
   }
}

//+------------------------------------------------------------------+
//| trades.csv — collector-compatible schema                          |
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
         ulong dtk = HistoryDealGetTicket(j);
         if(dtk == 0 || dtk == outTicket)
            continue;
         if((long)HistoryDealGetInteger(dtk, DEAL_MAGIC) != InpMagicNumber)
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

      double tpSnap = CtxVal(posId, g_ctxTp, 0.0);
      if(tpSnap <= 0.0)
         tpSnap = inTP;
      double slSnap = CtxVal(posId, g_ctxSL, 0.0);
      if(slSnap <= 0.0)
         slSnap = inSL;

      string reason = FindExitReason(posId);
      if(reason == "")
         reason = FallbackReason(closePrice, closeTime, slSnap, tpSnap);

      int ci = CtxIndexOf(posId);
      datetime setupTs = (ci >= 0) ? g_ctxSetupTime[ci] : openTime;

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
                    TsStr(setupTs) + "\r\n";
      FileWriteString(handle, line, StringLen(line));
      written++;
   }

   FileClose(handle);
   Print("Trade CSV written: trades.csv (", written, " trades)");
}

//+------------------------------------------------------------------+
//| sp2l_days.csv — per-day state for reconciliation                  |
//+------------------------------------------------------------------+
void WriteDaysCsv()
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

   int n = ArraySize(g_days);
   for(int i = 0; i < n; i++)
   {
      string line = IntegerToString(g_days[i].nyDate) + ";" +
                    IntegerToString(g_days[i].setups) + ";" +
                    IntegerToString(g_days[i].entered) + ";" +
                    IntegerToString(g_days[i].killed) + ";" +
                    IntegerToString(g_days[i].expired) + ";" +
                    IntegerToString(g_days[i].entries) + ";" +
                    IntegerToString(g_days[i].exitsTp) + ";" +
                    IntegerToString(g_days[i].exitsSl) + ";" +
                    IntegerToString(g_days[i].exitsTime) + ";" +
                    DoubleToString(g_days[i].totalPnl, 2) + "\r\n";
      FileWriteString(handle, line, StringLen(line));
   }

   FileClose(handle);
   Print("Day CSV written: sp2l_days.csv (", n, " days)");
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("=== SP2L_SpikeGap deinit (reason=", reason, ") ===");
   WriteTradesCsv();
   WriteDaysCsv();
}
//+------------------------------------------------------------------+
