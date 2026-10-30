//+------------------------------------------------------------------+
//|                                                 Pairs_FTMO.mq5   |
//| Dual-Symbol Signal-Reader EA: replays the FROZEN pairs_us500_     |
//| usa30 OOS spread trades (18 pairs) under simulated FTMO $10k      |
//| Challenge rules. US500.c leg = REAL tester orders; Dow leg =      |
//| VIRTUAL, priced from the verified Dukascopy USA30 M5 tape (EET    |
//| wall clock, $1/pt/lot, 2.28-pt RT spread). See SPEC, ADDENDUM 1   |
//| and ADDENDUM 3 at data/results/ftmo_pairs_us500_usa30/.           |
//| Inputs staged via FILE_COMMON (build 6182 agent sandbox rule).    |
//+------------------------------------------------------------------+
#property copyright "RR project"
#property version   "2.00"
#property strict

#include "..\Include\PropFirmGuard.mqh"

#define PF_TAG                 "PairsFTMO"
#define PF_CSV_IN              "pairs_trades.csv"
#define PF_CSV_OUT             "ftmo_pairs_trades.csv"
#define PF_CSV_STATUS          "ftmo_pairs_status.csv"
#define PF_CSV_DOW             "usa30_m5_eet.csv"
#define FTMO_DAILY_LOSS_PCT    5.0
#define FTMO_MAX_LOSS_PCT      10.0
#define PF_EXEC_DAY_OFFSET     86400
#define PF_DOW_HALF_SPREAD     1.14
#define PF_DOW_PT_VALUE        1.0

input double FixedLots          = 1.0;
input double InpProfitTargetUSD = 10800.0;
input long   InpMagicNumber     = 26091301;

struct PfSignal
{
   datetime entry_exec;
   datetime exit_exec;
   int      dir;
   double   ratio30;
};

struct PfTrade
{
   ulong    ticket500;
   int      dir;
   double   lots500;
   double   lots30;
   double   req30;
   double   open500;
   double   open30;
   datetime open_time;
   double   close500;
   double   close30;
   datetime close_time;
   datetime exit_exec;
   double   profit500;
   double   profit30;
   double   swap500;
   double   swap30;
   double   comm500;
   double   comm30;
   string   reason;
};

PfSignal g_sig[];
int      g_nSig      = 0;
int      g_next      = 0;
PfTrade  g_open;
bool     g_haveOpen  = false;
PfTrade  g_done[];
int      g_nDone     = 0;
double   g_initEquity = 0.0;
bool     g_targetReached = false;
string   g_breachReason  = "";
double   g_virtualCash   = 0.0;

datetime g_dowT[];
double   g_dowC[];
int      g_nDow = 0;

//+------------------------------------------------------------------+
bool LoadDowTape()
{
   int h = FileOpen(PF_CSV_DOW, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE)
   {
      PrintFormat("PF FATAL: cannot open %s (error %d)", PF_CSV_DOW, GetLastError());
      return false;
   }
   ArrayResize(g_dowT, 0, 65536);
   ArrayResize(g_dowC, 0, 65536);
   while(!FileIsEnding(h))
   {
      string line = FileReadString(h);
      if(StringLen(line) < 10)
         continue;
      string parts[];
      if(StringSplit(line, ',', parts) < 5)
         continue;
      if(StringFind(parts[0], "time") == 0)
         continue;
      long tv = (long)StringToInteger(parts[0]);
      if(tv <= 0)
         continue;
      int k = ArraySize(g_dowT);
      ArrayResize(g_dowT, k + 1, 65536);
      ArrayResize(g_dowC, k + 1, 65536);
      g_dowT[k] = (datetime)tv;
      g_dowC[k] = StringToDouble(parts[4]);
   }
   FileClose(h);
   g_nDow = ArraySize(g_dowT);
   for(int i = 1; i < g_nDow; i++)
   {
      if(g_dowT[i] < g_dowT[i - 1])
      {
         Print("PF FATAL: Dow tape not sorted");
         return false;
      }
   }
   PrintFormat("PF loaded %d Dow M5 bars (%s .. %s)", g_nDow,
               TimeToString(g_dowT[0], TIME_DATE), TimeToString(g_dowT[g_nDow - 1], TIME_DATE));
   return g_nDow > 100000;
}

//+------------------------------------------------------------------+
double DowCloseAt(datetime t)
{
   if(g_nDow == 0)
      return 0.0;
   int lo = 0;
   int hi = g_nDow - 1;
   if(t < g_dowT[0])
      return 0.0;
   while(lo < hi)
   {
      int mid = (lo + hi + 1) / 2;
      if(g_dowT[mid] <= t)
         lo = mid;
      else
         hi = mid - 1;
   }
   return g_dowC[lo];
}

//+------------------------------------------------------------------+
double NormalizeVolume500(double want)
{
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minv = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxv = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   if(step <= 0.0)
      step = 0.01;
   double vol = MathFloor(want / step + 1e-9) * step;
   if(minv > 0.0 && vol < minv)
      vol = minv;
   if(maxv > 0.0 && vol > maxv)
      vol = maxv;
   return NormalizeDouble(vol, 2);
}

//+------------------------------------------------------------------+
datetime ParseYmd(const string s)
{
   if(StringLen(s) < 8)
      return 0;
   string ymd = StringSubstr(s, 0, 4) + "." + StringSubstr(s, 4, 2) + "." + StringSubstr(s, 6, 2);
   return StringToTime(ymd);
}

//+------------------------------------------------------------------+
int LoadSignals()
{
   int h = FileOpen(PF_CSV_IN, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE)
   {
      PrintFormat("PF FATAL: cannot open %s (error %d)", PF_CSV_IN, GetLastError());
      return -1;
   }
   int n = 0;
   int skipped = 0;
   while(!FileIsEnding(h))
   {
      string line = FileReadString(h);
      if(StringLen(line) < 10)
         continue;
      string parts[];
      if(StringSplit(line, ',', parts) < 7)
         continue;
      if(StringFind(parts[0], "entry_date") == 0)
         continue;
      datetime ev = ParseYmd(parts[0]);
      datetime xv = ParseYmd(parts[1]);
      if(ev <= 0 || xv <= 0 || xv <= ev)
      {
         skipped++;
         continue;
      }
      int dir = 0;
      if(StringFind(parts[2], "LONG_SPREAD") == 0)
         dir = 1;
      else if(StringFind(parts[2], "SHORT_SPREAD") == 0)
         dir = -1;
      if(dir == 0)
      {
         skipped++;
         continue;
      }
      double ratio = StringToDouble(parts[6]);
      if(ratio <= 0.0)
      {
         skipped++;
         continue;
      }
      if(n >= ArraySize(g_sig))
         ArrayResize(g_sig, n + 32);
      g_sig[n].entry_exec = ev + PF_EXEC_DAY_OFFSET;
      g_sig[n].exit_exec  = xv + PF_EXEC_DAY_OFFSET;
      g_sig[n].dir        = dir;
      g_sig[n].ratio30    = ratio;
      n++;
   }
   FileClose(h);
   for(int i = 1; i < n; i++)
   {
      if(g_sig[i].entry_exec < g_sig[i - 1].entry_exec)
      {
         Print("PF FATAL: signals not sorted by entry");
         return -1;
      }
   }
   PrintFormat("PF loaded %d pair signals (%d rows skipped) from %s", n, skipped, PF_CSV_IN);
   return n;
}

//+------------------------------------------------------------------+
double VirtualFloating()
{
   if(!g_haveOpen)
      return 0.0;
   double px = DowCloseAt(TimeCurrent());
   if(px <= 0.0)
      px = g_open.open30;
   int d30 = -g_open.dir;
   return d30 * (px - g_open.open30) * PF_DOW_PT_VALUE * g_open.lots30;
}

double TotalEquity()
{
   return AccountInfoDouble(ACCOUNT_EQUITY) + g_virtualCash + VirtualFloating();
}

double TotalBalance()
{
   return AccountInfoDouble(ACCOUNT_BALANCE) + g_virtualCash;
}

//+------------------------------------------------------------------+
bool PfGuardTick()
{
   if(PG_IsBreached())
      return false;

   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   double today = dt.year * 10000.0 + dt.mon * 100.0 + dt.day;
   if(GlobalVariableGet(PG_GV_EOD_DATE) != today)
   {
      GlobalVariableSet(PG_GV_EOD_DATE, today);
      GlobalVariableSet(PG_GV_EOD_BAL, TotalBalance());
   }

   double equity = TotalEquity();
   double eod = GlobalVariableGet(PG_GV_EOD_BAL);
   if(eod > 0.0 && (eod - equity) / eod * 100.0 >= FTMO_DAILY_LOSS_PCT)
   {
      g_breachReason = "FTMO_DAILY_DRAWDOWN";
      PrintFormat("PF guard: daily breach eod=%.2f equity=%.2f", eod, equity);
      PG_Breach("FTMO_DAILY_DRAWDOWN");
      return false;
   }
   double floorUsd = g_initEquity * (1.0 - FTMO_MAX_LOSS_PCT / 100.0);
   if(equity <= floorUsd)
   {
      g_breachReason = "FTMO_MAX_LOSS";
      PrintFormat("PF guard: max-loss breach equity=%.2f floor=%.2f", equity, floorUsd);
      PG_Breach("FTMO_MAX_LOSS");
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
ulong OpenLeg500(int dir, double vol)
{
   double price = (dir > 0) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                            : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   price = NormalizeDouble(price, 2);
   if(price <= 0.0)
   {
      PrintFormat("PF leg US500: no price (dir=%d)", dir);
      return 0;
   }
   MqlTradeRequest req;
   MqlTradeResult  res;
   ZeroMemory(req);
   ZeroMemory(res);
   req.action    = TRADE_ACTION_DEAL;
   req.symbol    = _Symbol;
   req.volume    = vol;
   req.type      = (dir > 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   req.price     = price;
   req.deviation = 20;
   req.magic     = (ulong)InpMagicNumber;
   req.comment   = "PF leg";
   if(!OrderSend(req, res))
   {
      PrintFormat("PF leg US500 OrderSend failed ret=%u", res.retcode);
      return 0;
   }
   if(res.retcode != TRADE_RETCODE_DONE &&
      res.retcode != TRADE_RETCODE_DONE_PARTIAL &&
      res.retcode != TRADE_RETCODE_PLACED)
   {
      PrintFormat("PF leg US500 rejected ret=%u", res.retcode);
      return 0;
   }
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
         PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
         return tk;
   }
   Print("PF leg US500 FATAL: filled but no position found");
   return 0;
}

//+------------------------------------------------------------------+
bool CloseLeg500(ulong ticket, int dir, double vol)
{
   double price = (dir > 0) ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                            : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   price = NormalizeDouble(price, 2);
   MqlTradeRequest req;
   MqlTradeResult  res;
   ZeroMemory(req);
   ZeroMemory(res);
   req.action    = TRADE_ACTION_DEAL;
   req.symbol    = _Symbol;
   req.volume    = vol;
   req.type      = (dir > 0) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   req.position  = ticket;
   req.price     = price;
   req.deviation = 20;
   req.magic     = (ulong)InpMagicNumber;
   req.comment   = "PF close";
   if(!OrderSend(req, res))
   {
      PrintFormat("PF close US500 OrderSend failed ret=%u", res.retcode);
      return false;
   }
   if(res.retcode != TRADE_RETCODE_DONE &&
      res.retcode != TRADE_RETCODE_DONE_PARTIAL &&
      res.retcode != TRADE_RETCODE_PLACED)
   {
      PrintFormat("PF close US500 rejected ret=%u", res.retcode);
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
void CollectLeg500(ulong ticket, double &profit, double &swap, double &comm,
                   double &openPrice, datetime &openTime,
                   double &closePrice, datetime &closeTime)
{
   profit = 0.0; swap = 0.0; comm = 0.0;
   openPrice = 0.0; closePrice = 0.0; openTime = 0; closeTime = 0;
   if(!HistorySelectByPosition(ticket))
      return;
   int n = HistoryDealsTotal();
   for(int i = 0; i < n; i++)
   {
      ulong d = HistoryDealGetTicket(i);
      if(d == 0)
         continue;
      profit += HistoryDealGetDouble(d, DEAL_PROFIT);
      swap   += HistoryDealGetDouble(d, DEAL_SWAP);
      comm   += HistoryDealGetDouble(d, DEAL_COMMISSION);
      long entryType = HistoryDealGetInteger(d, DEAL_ENTRY);
      if(entryType == DEAL_ENTRY_IN)
      {
         openPrice = HistoryDealGetDouble(d, DEAL_PRICE);
         openTime  = (datetime)HistoryDealGetInteger(d, DEAL_TIME);
      }
      else if(entryType == DEAL_ENTRY_OUT)
      {
         closePrice = HistoryDealGetDouble(d, DEAL_PRICE);
         closeTime  = (datetime)HistoryDealGetInteger(d, DEAL_TIME);
      }
   }
}

//+------------------------------------------------------------------+
void FinalizeOpen(const string reason)
{
   double p500, s500, c500, o500, c500p;
   datetime ot500, ct500;
   CollectLeg500(g_open.ticket500, p500, s500, c500, o500, ot500, c500p, ct500);
   int d30 = -g_open.dir;
   double px = DowCloseAt(TimeCurrent());
   if(px <= 0.0)
      px = g_open.open30;
   double close30 = px - d30 * PF_DOW_HALF_SPREAD;
   double pnl30 = d30 * (close30 - g_open.open30) * PF_DOW_PT_VALUE * g_open.lots30;
   g_virtualCash += pnl30;
   g_open.profit500 = p500;
   g_open.profit30  = pnl30;
   g_open.swap500   = s500;
   g_open.swap30    = 0.0;
   g_open.comm500   = c500;
   g_open.comm30    = 0.0;
   g_open.open500   = o500;
   g_open.close500  = c500p;
   g_open.close30   = close30;
   g_open.open_time = ot500;
   g_open.close_time = ct500;
   g_open.reason    = reason;
   if(g_nDone >= ArraySize(g_done))
      ArrayResize(g_done, g_nDone + 16);
   g_done[g_nDone] = g_open;
   g_nDone++;
   PrintFormat("PF CLOSE pair t500=%I64u reason=%s profit=%.2f (500:%.2f / 30:%.2f) swap=%.2f comm=%.2f",
               g_open.ticket500, reason, p500 + pnl30, p500, pnl30, s500, c500);
   g_haveOpen = false;
}

//+------------------------------------------------------------------+
bool OpenPair(int dir, double ratio30)
{
   double vol500 = NormalizeVolume500(1.0 * FixedLots);
   if(vol500 <= 0.0)
   {
      Print("PF: US500 volume normalization produced zero; signal skipped");
      return false;
   }
   datetime now = TimeCurrent();
   double dow = DowCloseAt(now);
   if(dow <= 0.0)
   {
      PrintFormat("PF: no Dow tape price at %s; signal skipped", TimeToString(now));
      return false;
   }
   ulong t500 = OpenLeg500(dir, vol500);
   if(t500 == 0)
      return false;
   int d30 = -dir;
   ZeroMemory(g_open);
   g_open.ticket500 = t500;
   g_open.dir       = dir;
   g_open.lots500   = vol500;
   g_open.lots30    = ratio30 * FixedLots;
   g_open.req30     = ratio30 * FixedLots;
   g_open.open500   = SymbolInfoDouble(_Symbol, dir > 0 ? SYMBOL_ASK : SYMBOL_BID);
   g_open.open30    = dow + d30 * PF_DOW_HALF_SPREAD;
   g_open.open_time = now;
   g_haveOpen       = true;
   PrintFormat("PF OPEN dir=%d vol500=%.2f lots30=%.4f p500=%.2f p30=%.2f t500=%I64u",
               dir, vol500, g_open.lots30, g_open.open500, g_open.open30, t500);
   return true;
}

//+------------------------------------------------------------------+
string TStr(datetime t)
{
   return TimeToString(t, TIME_DATE | TIME_MINUTES);
}

//+------------------------------------------------------------------+
int OnInit()
{
   g_initEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   PG_Init(PF_TAG, InpMagicNumber);
   GlobalVariableSet(PG_GV_HWM, g_initEquity);

   g_next = 0;
   g_nDone = 0;
   g_haveOpen = false;
   g_targetReached = false;
   g_breachReason = "";
   g_virtualCash = 0.0;

   if(!LoadDowTape())
      return INIT_FAILED;
   int n = LoadSignals();
   if(n < 0)
      return INIT_FAILED;
   g_nSig = n;
   PrintFormat("PF init: equity=%.2f target=%.2f mult=%.2f signals=%d dow_bars=%d leg30=virtual_dukascopy",
               g_initEquity, InpProfitTargetUSD, FixedLots, g_nSig, g_nDow);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnTick()
{
   if(!PfGuardTick())
   {
      if(g_haveOpen && !PositionSelectByTicket(g_open.ticket500))
         FinalizeOpen("guard_flatten");
      return;
   }

   datetime now = TimeCurrent();

   if(!g_targetReached && TotalBalance() >= InpProfitTargetUSD)
   {
      g_targetReached = true;
      PrintFormat("### PF TARGET_REACHED total_balance=%.2f (account=%.2f virtual=%.2f) at %s",
                  TotalBalance(), AccountInfoDouble(ACCOUNT_BALANCE), g_virtualCash, TStr(now));
   }

   if(g_haveOpen && now >= g_open.exit_exec)
   {
      if(CloseLeg500(g_open.ticket500, g_open.dir, g_open.lots500))
         FinalizeOpen("signal");
   }

   if(!g_haveOpen && !g_targetReached)
   {
      while(g_next < g_nSig)
      {
         if(now < g_sig[g_next].entry_exec)
            break;
         if(now >= g_sig[g_next].exit_exec)
         {
            g_next++;
            continue;
         }
         if(OpenPair(g_sig[g_next].dir, g_sig[g_next].ratio30))
         {
            g_open.exit_exec = g_sig[g_next].exit_exec;
            g_next++;
         }
         else
         {
            PrintFormat("PF signal at %s skipped after failed pair open",
                        TStr(g_sig[g_next].entry_exec));
            g_next++;
         }
         break;
      }
   }
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(g_haveOpen)
   {
      if(!PositionSelectByTicket(g_open.ticket500))
         FinalizeOpen("guard_flatten");
      else
      {
         CloseLeg500(g_open.ticket500, g_open.dir, g_open.lots500);
         FinalizeOpen("open_at_end");
      }
   }

   int h = FileOpen(PF_CSV_OUT, FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
   if(h != INVALID_HANDLE)
   {
      FileWrite(h, "ticket500", "ticket30", "entry_time", "exit_time", "direction",
                "lots500", "lots30", "req30", "open500", "close500", "open30", "close30",
                "profit500", "profit30", "profit_total", "commission_total", "swap_total",
                "tick_value500", "tick_size500", "point500",
                "tick_value30", "tick_size30", "point30", "leg30_mode", "reason");
      double tv500 = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      double ts500 = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      double pt500 = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      for(int i = 0; i < g_nDone; i++)
      {
         FileWrite(h,
                   (long)g_done[i].ticket500,
                   0,
                   TStr(g_done[i].open_time),
                   TStr(g_done[i].close_time),
                   g_done[i].dir > 0 ? "LONG_SPREAD" : "SHORT_SPREAD",
                   DoubleToString(g_done[i].lots500, 2),
                   DoubleToString(g_done[i].lots30, 4),
                   DoubleToString(g_done[i].req30, 5),
                   DoubleToString(g_done[i].open500, 2),
                   DoubleToString(g_done[i].close500, 2),
                   DoubleToString(g_done[i].open30, 2),
                   DoubleToString(g_done[i].close30, 2),
                   DoubleToString(g_done[i].profit500, 2),
                   DoubleToString(g_done[i].profit30, 2),
                   DoubleToString(g_done[i].profit500 + g_done[i].profit30, 2),
                   DoubleToString(g_done[i].comm500 + g_done[i].comm30, 2),
                   DoubleToString(g_done[i].swap500 + g_done[i].swap30, 2),
                   DoubleToString(tv500, 4),
                   DoubleToString(ts500, 4),
                   DoubleToString(pt500, 4),
                   "0.0100",
                   "0.0100",
                   "0.0100",
                   "virtual_dukascopy",
                   g_done[i].reason);
      }
      FileClose(h);
      PrintFormat("PF wrote %d pair trades to %s", g_nDone, PF_CSV_OUT);
   }
   else
      PrintFormat("PF FATAL: cannot write %s (error %d)", PF_CSV_OUT, GetLastError());

   int s = FileOpen(PF_CSV_STATUS, FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
   if(s != INVALID_HANDLE)
   {
      FileWrite(s, "initial_equity", "final_balance", "final_equity",
                "account_balance", "virtual_cash",
                "target_reached", "breached", "breach_reason",
                "signals_total", "signals_next", "trades_executed", "last_time");
      FileWrite(s,
                DoubleToString(g_initEquity, 2),
                DoubleToString(TotalBalance(), 2),
                DoubleToString(TotalEquity(), 2),
                DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2),
                DoubleToString(g_virtualCash, 2),
                (g_targetReached ? 1 : 0),
                (PG_IsBreached() ? 1 : 0),
                g_breachReason,
                g_nSig,
                g_next,
                g_nDone,
                TStr(TimeCurrent()));
      FileClose(s);
   }
}
//+------------------------------------------------------------------+
