//+------------------------------------------------------------------+
//|                                              HybridLong_FTMO.mq5 |
//| Single-Symbol Signal-Reader EA: replays the FROZEN youtube_hybrid |
//| scalping US500/Long (ATR band-touch) trade list under simulated   |
//| FTMO $10k Challenge Swap-Free rules. Real tester market orders on |
//| US500.c, Model=0. See SPEC/GATE at data/results/ftmo_hybrid_long/ |
//| Inputs and outputs staged via FILE_COMMON (build 6182 sandbox).   |
//+------------------------------------------------------------------+
#property copyright "RR project"
#property version   "1.00"
#property strict

#include "..\Include\PropFirmGuard.mqh"

#define HL_TAG                 "HybridLongFTMO"
#define HL_CSV_IN              "hybrid_scalp_trades.csv"
#define HL_CSV_OUT             "ftmo_hybrid_long_trades.csv"
#define HL_CSV_STATUS          "ftmo_hybrid_long_status.csv"
#define FTMO_DAILY_LOSS_PCT    5.0
#define FTMO_MAX_LOSS_PCT      10.0
#define HL_EXIT_ARM_S          300

input double FixedLots          = 1.0;
input double InpProfitTargetUSD = 10800.0;
input long   InpMagicNumber     = 26091601;

struct HlSignal
{
   datetime entry_exec;
   datetime exit_exec;
};

struct HlTrade
{
   ulong    ticket;
   double   lots;
   double   open_price;
   double   close_price;
   datetime open_time;
   datetime close_time;
   datetime exit_exec;
   double   profit;
   double   swap;
   double   comm;
   double   mae;
   double   mfe;
   string   reason;
};

HlSignal g_sig[];
int      g_nSig      = 0;
int      g_next      = 0;
HlTrade  g_open;
bool     g_haveOpen  = false;
HlTrade  g_done[];
int      g_nDone     = 0;
double   g_initEquity = 0.0;
bool     g_targetReached = false;
string   g_breachReason  = "";

//+------------------------------------------------------------------+
double NormalizeVolume(double want)
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
int LoadSignals()
{
   int h = FileOpen(HL_CSV_IN, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE)
   {
      PrintFormat("HL FATAL: cannot open %s (error %d)", HL_CSV_IN, GetLastError());
      return -1;
   }
   int n = 0;
   int skipped = 0;
   int keptOthers = 0;
   while(!FileIsEnding(h))
   {
      string line = FileReadString(h);
      if(StringLen(line) < 10)
         continue;
      string parts[];
      if(StringSplit(line, ',', parts) < 5)
         continue;
      if(StringFind(parts[0], "instrument") == 0)
         continue;
      if(parts[0] != "US500" || parts[1] != "Long")
      {
         keptOthers++;
         continue;
      }
      long ev = (long)StringToInteger(parts[3]);
      long xv = (long)StringToInteger(parts[4]);
      if(ev <= 0 || xv < ev)
      {
         skipped++;
         continue;
      }
      if(n >= ArraySize(g_sig))
         ArrayResize(g_sig, n + 64);
      g_sig[n].entry_exec = (datetime)ev;
      g_sig[n].exit_exec  = (datetime)(xv + HL_EXIT_ARM_S);
      n++;
   }
   FileClose(h);
   for(int i = 1; i < n; i++)
   {
      if(g_sig[i].entry_exec < g_sig[i - 1].entry_exec)
      {
         Print("HL FATAL: signals not sorted by entry");
         return -1;
      }
   }
   PrintFormat("HL loaded %d US500/Long signals (%d off-filter rows, %d malformed skipped) from %s",
               n, keptOthers, skipped, HL_CSV_IN);
   return n;
}

//+------------------------------------------------------------------+
bool HlGuardTick()
{
   if(PG_IsBreached())
      return false;

   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   double today = dt.year * 10000.0 + dt.mon * 100.0 + dt.day;
   if(GlobalVariableGet(PG_GV_EOD_DATE) != today)
   {
      GlobalVariableSet(PG_GV_EOD_DATE, today);
      GlobalVariableSet(PG_GV_EOD_BAL, AccountInfoDouble(ACCOUNT_BALANCE));
   }

   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double eod = GlobalVariableGet(PG_GV_EOD_BAL);
   if(eod > 0.0 && (eod - equity) / eod * 100.0 >= FTMO_DAILY_LOSS_PCT)
   {
      g_breachReason = "FTMO_DAILY_DRAWDOWN";
      PrintFormat("HL guard: daily breach eod=%.2f equity=%.2f", eod, equity);
      PG_Breach("FTMO_DAILY_DRAWDOWN");
      return false;
   }
   double floorUsd = g_initEquity * (1.0 - FTMO_MAX_LOSS_PCT / 100.0);
   if(equity <= floorUsd)
   {
      g_breachReason = "FTMO_MAX_LOSS";
      PrintFormat("HL guard: max-loss breach equity=%.2f floor=%.2f", equity, floorUsd);
      PG_Breach("FTMO_MAX_LOSS");
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
ulong OpenLong(double vol)
{
   double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   price = NormalizeDouble(price, 2);
   if(price <= 0.0)
   {
      Print("HL open: no ask price");
      return 0;
   }
   MqlTradeRequest req;
   MqlTradeResult  res;
   ZeroMemory(req);
   ZeroMemory(res);
   req.action    = TRADE_ACTION_DEAL;
   req.symbol    = _Symbol;
   req.volume    = vol;
   req.type      = ORDER_TYPE_BUY;
   req.price     = price;
   req.deviation = 20;
   req.magic     = (ulong)InpMagicNumber;
   req.comment   = "HL long entry";
   if(!OrderSend(req, res))
   {
      PrintFormat("HL open OrderSend failed ret=%u", res.retcode);
      return 0;
   }
   if(res.retcode != TRADE_RETCODE_DONE &&
      res.retcode != TRADE_RETCODE_DONE_PARTIAL &&
      res.retcode != TRADE_RETCODE_PLACED)
   {
      PrintFormat("HL open rejected ret=%u", res.retcode);
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
   Print("HL open FATAL: filled but no position found");
   return 0;
}

//+------------------------------------------------------------------+
bool CloseLong(ulong ticket, double vol)
{
   double price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   price = NormalizeDouble(price, 2);
   MqlTradeRequest req;
   MqlTradeResult  res;
   ZeroMemory(req);
   ZeroMemory(res);
   req.action    = TRADE_ACTION_DEAL;
   req.symbol    = _Symbol;
   req.volume    = vol;
   req.type      = ORDER_TYPE_SELL;
   req.position  = ticket;
   req.price     = price;
   req.deviation = 20;
   req.magic     = (ulong)InpMagicNumber;
   req.comment   = "HL close";
   if(!OrderSend(req, res))
   {
      PrintFormat("HL close OrderSend failed ret=%u", res.retcode);
      return false;
   }
   if(res.retcode != TRADE_RETCODE_DONE &&
      res.retcode != TRADE_RETCODE_DONE_PARTIAL &&
      res.retcode != TRADE_RETCODE_PLACED)
   {
      PrintFormat("HL close rejected ret=%u", res.retcode);
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
void FinalizeOpen(const string reason)
{
   if(!HistorySelectByPosition(g_open.ticket))
      PrintFormat("HL finalize: HistorySelectByPosition(%I64u) empty", g_open.ticket);
   int n = HistoryDealsTotal();
   for(int i = 0; i < n; i++)
   {
      ulong d = HistoryDealGetTicket(i);
      if(d == 0)
         continue;
      g_open.profit += HistoryDealGetDouble(d, DEAL_PROFIT);
      g_open.swap   += HistoryDealGetDouble(d, DEAL_SWAP);
      g_open.comm   += HistoryDealGetDouble(d, DEAL_COMMISSION);
      long entryType = HistoryDealGetInteger(d, DEAL_ENTRY);
      if(entryType == DEAL_ENTRY_IN)
      {
         g_open.open_price = HistoryDealGetDouble(d, DEAL_PRICE);
         g_open.open_time  = (datetime)HistoryDealGetInteger(d, DEAL_TIME);
      }
      else if(entryType == DEAL_ENTRY_OUT)
      {
         g_open.close_price = HistoryDealGetDouble(d, DEAL_PRICE);
         g_open.close_time  = (datetime)HistoryDealGetInteger(d, DEAL_TIME);
      }
   }
   g_open.reason = reason;
   if(g_nDone >= ArraySize(g_done))
      ArrayResize(g_done, g_nDone + 64);
   g_done[g_nDone] = g_open;
   g_nDone++;
   PrintFormat("HL CLOSE t=%I64u reason=%s profit=%.2f swap=%.2f comm=%.2f mae=%.2f mfe=%.2f",
               g_open.ticket, reason, g_open.profit, g_open.swap, g_open.comm,
               g_open.mae, g_open.mfe);
   g_haveOpen = false;
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
   PG_Init(HL_TAG, InpMagicNumber);
   GlobalVariableSet(PG_GV_HWM, g_initEquity);

   g_next = 0;
   g_nDone = 0;
   g_haveOpen = false;
   g_targetReached = false;
   g_breachReason = "";
   ZeroMemory(g_open);
   g_open.reason = "";

   int n = LoadSignals();
   if(n < 0)
      return INIT_FAILED;
   g_nSig = n;
   PrintFormat("HL init: equity=%.2f target=%.2f lots=%.2f signals=%d exit_arm=+%ds",
               g_initEquity, InpProfitTargetUSD, FixedLots, g_nSig, HL_EXIT_ARM_S);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnTick()
{
   if(!HlGuardTick())
   {
      if(g_haveOpen && !PositionSelectByTicket(g_open.ticket))
         FinalizeOpen("guard_flatten");
      return;
   }

   datetime now = TimeCurrent();

   if(!g_targetReached && AccountInfoDouble(ACCOUNT_BALANCE) >= InpProfitTargetUSD)
   {
      g_targetReached = true;
      PrintFormat("### HL TARGET_REACHED balance=%.2f at %s",
                  AccountInfoDouble(ACCOUNT_BALANCE), TStr(now));
   }

   if(g_haveOpen)
   {
      if(PositionSelectByTicket(g_open.ticket))
      {
         double fl = PositionGetDouble(POSITION_PROFIT)
                     + PositionGetDouble(POSITION_SWAP);
         if(fl < g_open.mae)
            g_open.mae = fl;
         if(fl > g_open.mfe)
            g_open.mfe = fl;
         if(now >= g_open.exit_exec)
         {
            if(CloseLong(g_open.ticket, g_open.lots))
               FinalizeOpen("signal");
         }
      }
      else
         FinalizeOpen("guard_flatten");
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
         double vol = NormalizeVolume(1.0 * FixedLots);
         ulong tk = OpenLong(vol);
         if(tk > 0)
         {
            g_open.ticket    = tk;
            g_open.lots      = vol;
            g_open.exit_exec = g_sig[g_next].exit_exec;
            g_open.profit    = 0.0;
            g_open.swap      = 0.0;
            g_open.comm      = 0.0;
            g_open.mae       = 0.0;
            g_open.mfe       = 0.0;
            g_open.open_price = 0.0;
            g_open.close_price = 0.0;
            g_open.open_time  = 0;
            g_open.close_time = 0;
            g_open.reason     = "";
            g_haveOpen = true;
            g_next++;
         }
         else
         {
            PrintFormat("HL signal at %s consumed after failed open", TStr(g_sig[g_next].entry_exec));
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
      if(!PositionSelectByTicket(g_open.ticket))
         FinalizeOpen("guard_flatten");
      else
      {
         CloseLong(g_open.ticket, g_open.lots);
         FinalizeOpen("open_at_end");
      }
   }

   int h = FileOpen(HL_CSV_OUT, FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
   if(h != INVALID_HANDLE)
   {
      FileWrite(h, "ticket", "entry_time", "exit_time", "direction", "lots",
                "open_price", "close_price", "profit", "swap", "commission",
                "mae_usd", "mfe_usd", "reason",
                "tick_value", "tick_size", "point");
      double tv = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      double ts = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      double pt = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      for(int i = 0; i < g_nDone; i++)
      {
         FileWrite(h,
                   (long)g_done[i].ticket,
                   TStr(g_done[i].open_time),
                   TStr(g_done[i].close_time),
                   "Long",
                   DoubleToString(g_done[i].lots, 2),
                   DoubleToString(g_done[i].open_price, 2),
                   DoubleToString(g_done[i].close_price, 2),
                   DoubleToString(g_done[i].profit, 2),
                   DoubleToString(g_done[i].swap, 2),
                   DoubleToString(g_done[i].comm, 2),
                   DoubleToString(g_done[i].mae, 2),
                   DoubleToString(g_done[i].mfe, 2),
                   g_done[i].reason,
                   DoubleToString(tv, 4),
                   DoubleToString(ts, 4),
                   DoubleToString(pt, 4));
      }
      FileClose(h);
      PrintFormat("HL wrote %d trades to %s", g_nDone, HL_CSV_OUT);
   }
   else
      PrintFormat("HL FATAL: cannot write %s (error %d)", HL_CSV_OUT, GetLastError());

   int s = FileOpen(HL_CSV_STATUS, FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
   if(s != INVALID_HANDLE)
   {
      FileWrite(s, "initial_equity", "final_balance", "final_equity",
                "target_reached", "breached", "breach_reason",
                "signals_total", "signals_next", "trades_executed", "last_time");
      FileWrite(s,
                DoubleToString(g_initEquity, 2),
                DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2),
                DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2),
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
