//+------------------------------------------------------------------+
//|                                        MacroReversal_FTMO.mq5    |
//| Signal-Reader EA: replays the FROZEN short_term_reversal g2 OOS  |
//| trade list (h=3) on US500.c under simulated FTMO $10k Challenge  |
//| rules. SPEC: data/results/ftmo_macro_reversal/                   |
//| SPEC_FTMO_MACRO_REVERSAL.txt (frozen 2026-09-12).                |
//| Trade list source: data/results/short_term_reversal/g2_trades.csv |
//| (copied by the runner into the tester agent sandbox MQL5\Files). |
//+------------------------------------------------------------------+
#property copyright "RR project"
#property version   "1.00"
#property strict

#include "..\Include\PropFirmGuard.mqh"

#define MRFTMO_TAG              "MacroReversalFTMO"
#define MRFTMO_CSV_IN           "g2_trades.csv"
#define MRFTMO_CSV_OUT          "ftmo_reversal_trades.csv"
#define MRFTMO_CSV_STATUS       "ftmo_reversal_status.csv"
#define FTMO_DAILY_LOSS_PCT     5.0
#define FTMO_MAX_LOSS_PCT       10.0
#define FTMO_EXEC_DAY_OFFSET    86400

input double FixedLots          = 5.0;
input double InpProfitTargetUSD = 10800.0;
input long   InpMagicNumber     = 26091201;

struct MrSignal
{
   datetime entry_exec;
   datetime exit_exec;
   int      dir;
};

struct MrTrade
{
   ulong    ticket;
   int      dir;
   double   lots;
   double   open_price;
   datetime open_time;
   double   close_price;
   datetime close_time;
   datetime exit_exec;
   double   profit;
   double   swap;
   double   commission;
   string   reason;
};

MrSignal g_sig[];
int      g_nSig      = 0;
int      g_next      = 0;
MrTrade  g_open;
bool     g_haveOpen  = false;
MrTrade  g_done[];
int      g_nDone     = 0;
double   g_initEquity = 0.0;
bool     g_targetReached = false;
string   g_breachReason  = "";

//+------------------------------------------------------------------+
int NormalizeVolume(double want)
{
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minv = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxv = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   if(step <= 0.0)
      step = 0.01;
   double vol = MathFloor(want / step + 1e-9) * step;
   if(minv > 0.0 && vol < minv)
      return 0;
   if(maxv > 0.0 && vol > maxv)
      vol = maxv;
   return (int)MathRound(vol / step);
}

//+------------------------------------------------------------------+
int LoadSignals()
{
   int h = FileOpen(MRFTMO_CSV_IN, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE)
   {
      PrintFormat("MR-FTMO FATAL: cannot open %s (error %d)", MRFTMO_CSV_IN, GetLastError());
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
      if(StringSplit(line, ',', parts) < 3)
         continue;
      if(StringFind(parts[0], "entry") == 0)
         continue;
      string es = parts[0];
      string xs = parts[1];
      StringReplace(es, "-", ".");
      StringReplace(xs, "-", ".");
      datetime ev = StringToTime(es);
      datetime xv = StringToTime(xs);
      if(ev <= 0 || xv <= 0 || xv <= ev)
      {
         skipped++;
         continue;
      }
      int dir = (int)StringToInteger(parts[2]);
      if(dir != 1 && dir != -1)
      {
         skipped++;
         continue;
      }
      if(n >= ArraySize(g_sig))
         ArrayResize(g_sig, n + 64);
      g_sig[n].entry_exec = ev + FTMO_EXEC_DAY_OFFSET;
      g_sig[n].exit_exec  = xv + FTMO_EXEC_DAY_OFFSET;
      g_sig[n].dir        = dir;
      n++;
   }
   FileClose(h);
   for(int i = 1; i < n; i++)
   {
      if(g_sig[i].entry_exec < g_sig[i - 1].entry_exec)
      {
         Print("MR-FTMO FATAL: signals not sorted by entry");
         return -1;
      }
   }
   PrintFormat("MR-FTMO loaded %d signals (%d rows skipped) from %s",
               n, skipped, MRFTMO_CSV_IN);
   return n;
}

//+------------------------------------------------------------------+
bool MrGuardTick()
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
      PrintFormat("MR-FTMO guard: daily breach eod=%.2f equity=%.2f", eod, equity);
      PG_Breach("FTMO_DAILY_DRAWDOWN");
      return false;
   }
   double floorUsd = g_initEquity * (1.0 - FTMO_MAX_LOSS_PCT / 100.0);
   if(equity <= floorUsd)
   {
      g_breachReason = "FTMO_MAX_LOSS";
      PrintFormat("MR-FTMO guard: max-loss breach equity=%.2f floor=%.2f", equity, floorUsd);
      PG_Breach("FTMO_MAX_LOSS");
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
void CollectFromHistory(MrTrade &t)
{
   if(!HistorySelectByPosition(t.ticket))
      return;
   int n = HistoryDealsTotal();
   for(int i = 0; i < n; i++)
   {
      ulong d = HistoryDealGetTicket(i);
      if(d == 0)
         continue;
      t.profit     += HistoryDealGetDouble(d, DEAL_PROFIT);
      t.swap       += HistoryDealGetDouble(d, DEAL_SWAP);
      t.commission += HistoryDealGetDouble(d, DEAL_COMMISSION);
      long entryType = HistoryDealGetInteger(d, DEAL_ENTRY);
      if(entryType == DEAL_ENTRY_IN)
      {
         t.open_price = HistoryDealGetDouble(d, DEAL_PRICE);
         t.open_time  = (datetime)HistoryDealGetInteger(d, DEAL_TIME);
      }
      else if(entryType == DEAL_ENTRY_OUT)
      {
         t.close_price = HistoryDealGetDouble(d, DEAL_PRICE);
         t.close_time  = (datetime)HistoryDealGetInteger(d, DEAL_TIME);
      }
   }
}

//+------------------------------------------------------------------+
bool OpenTrade(int dir)
{
   int volSteps = NormalizeVolume(FixedLots);
   if(volSteps <= 0)
   {
      Print("MR-FTMO: volume normalized to zero, entry skipped");
      return false;
   }
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(step <= 0.0)
      step = 0.01;
   double vol = volSteps * step;

   double price = (dir > 0) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                            : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   price = NormalizeDouble(price, 2);

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
   req.comment   = "MR-FTMO";

   if(!OrderSend(req, res))
   {
      PrintFormat("MR-FTMO entry OrderSend failed ret=%u %s", res.retcode, res.comment);
      return false;
   }
   if(res.retcode != TRADE_RETCODE_DONE &&
      res.retcode != TRADE_RETCODE_DONE_PARTIAL &&
      res.retcode != TRADE_RETCODE_PLACED)
   {
      PrintFormat("MR-FTMO entry rejected ret=%u", res.retcode);
      return false;
   }

   ulong posTicket = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0)
         continue;
      if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
         PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
      {
         posTicket = tk;
         break;
      }
   }
   if(posTicket == 0)
   {
      Print("MR-FTMO FATAL: filled but no position found");
      return false;
   }

   ZeroMemory(g_open);
   g_open.ticket     = posTicket;
   g_open.dir        = dir;
   g_open.lots       = vol;
   g_open.open_price = (dir > 0) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK)
                                 : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   g_open.open_time  = TimeCurrent();
   g_open.reason     = "";
   g_haveOpen        = true;
   PrintFormat("MR-FTMO OPEN dir=%d vol=%.2f price=%.2f ticket=%I64u",
               dir, vol, g_open.open_price, posTicket);
   return true;
}

//+------------------------------------------------------------------+
bool CloseTrade()
{
   double price = (g_open.dir > 0) ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
                                   : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   price = NormalizeDouble(price, 2);
   MqlTradeRequest req;
   MqlTradeResult  res;
   ZeroMemory(req);
   ZeroMemory(res);
   req.action    = TRADE_ACTION_DEAL;
   req.symbol    = _Symbol;
   req.volume    = g_open.lots;
   req.type      = (g_open.dir > 0) ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
   req.position  = g_open.ticket;
   req.price     = price;
   req.deviation = 20;
   req.magic     = (ulong)InpMagicNumber;
   req.comment   = "MR-FTMO close";

   if(!OrderSend(req, res))
   {
      PrintFormat("MR-FTMO close OrderSend failed ret=%u", res.retcode);
      return false;
   }
   if(res.retcode != TRADE_RETCODE_DONE &&
      res.retcode != TRADE_RETCODE_DONE_PARTIAL &&
      res.retcode != TRADE_RETCODE_PLACED)
   {
      PrintFormat("MR-FTMO close rejected ret=%u", res.retcode);
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
void FinalizeOpen(const string reason)
{
   CollectFromHistory(g_open);
   g_open.reason = reason;
   if(g_nDone >= ArraySize(g_done))
      ArrayResize(g_done, g_nDone + 64);
   g_done[g_nDone] = g_open;
   g_nDone++;
   PrintFormat("MR-FTMO CLOSE ticket=%I64u reason=%s profit=%.2f swap=%.2f comm=%.2f",
               g_open.ticket, reason, g_open.profit, g_open.swap, g_open.commission);
   g_haveOpen = false;
}

//+------------------------------------------------------------------+
string TimeToStr(datetime t)
{
   return TimeToString(t, TIME_DATE | TIME_MINUTES);
}

//+------------------------------------------------------------------+
int OnInit()
{
   g_initEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   PG_Init(MRFTMO_TAG, InpMagicNumber);
   GlobalVariableSet(PG_GV_HWM, g_initEquity);

   g_next = 0;
   g_nDone = 0;
   g_haveOpen = false;
   g_targetReached = false;
   g_breachReason = "";

   int n = LoadSignals();
   if(n < 0)
      return INIT_FAILED;
   g_nSig = n;
   PrintFormat("MR-FTMO init: equity=%.2f target=%.2f lots=%.2f signals=%d",
               g_initEquity, InpProfitTargetUSD, FixedLots, g_nSig);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnTick()
{
   if(!MrGuardTick())
   {
      if(g_haveOpen && !PositionSelectByTicket(g_open.ticket))
         FinalizeOpen("guard_flatten");
      return;
   }

   datetime now = TimeCurrent();

   if(!g_targetReached && AccountInfoDouble(ACCOUNT_BALANCE) >= InpProfitTargetUSD)
   {
      g_targetReached = true;
      PrintFormat("### MR-FTMO TARGET_REACHED balance=%.2f at %s",
                  AccountInfoDouble(ACCOUNT_BALANCE), TimeToStr(now));
   }

   if(g_haveOpen && now >= g_open.exit_exec)
   {
      if(CloseTrade())
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
         if(OpenTrade(g_sig[g_next].dir))
         {
            g_open.exit_exec = g_sig[g_next].exit_exec;
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
         CollectFromHistory(g_open);
         g_open.reason = "open_at_end";
         if(g_nDone >= ArraySize(g_done))
            ArrayResize(g_done, g_nDone + 8);
         g_done[g_nDone] = g_open;
         g_nDone++;
         g_haveOpen = false;
      }
   }

   int h = FileOpen(MRFTMO_CSV_OUT, FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
   if(h != INVALID_HANDLE)
   {
      FileWrite(h, "ticket", "entry_time", "exit_time", "type", "lot",
                "open_price", "close_price", "commission", "profit", "swap",
                "tick_value", "tick_size", "point", "reason");
      double tv = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
      double ts = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
      double pt = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      for(int i = 0; i < g_nDone; i++)
      {
         FileWrite(h,
                   (long)g_done[i].ticket,
                   TimeToStr(g_done[i].open_time),
                   TimeToStr(g_done[i].close_time),
                   g_done[i].dir,
                   DoubleToString(g_done[i].lots, 2),
                   DoubleToString(g_done[i].open_price, 2),
                   DoubleToString(g_done[i].close_price, 2),
                   DoubleToString(g_done[i].commission, 2),
                   DoubleToString(g_done[i].profit, 2),
                   DoubleToString(g_done[i].swap, 2),
                   DoubleToString(tv, 4),
                   DoubleToString(ts, 4),
                   DoubleToString(pt, 4),
                   g_done[i].reason);
      }
      FileClose(h);
      PrintFormat("MR-FTMO wrote %d trades to %s", g_nDone, MRFTMO_CSV_OUT);
   }
   else
      PrintFormat("MR-FTMO FATAL: cannot write %s (error %d)", MRFTMO_CSV_OUT, GetLastError());

   int s = FileOpen(MRFTMO_CSV_STATUS, FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ',');
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
                TimeToStr(TimeCurrent()));
      FileClose(s);
   }
}
//+------------------------------------------------------------------+
