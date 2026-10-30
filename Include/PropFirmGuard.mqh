//+------------------------------------------------------------------+
//|                                              PropFirmGuard.mqh   |
//|  Account-level prop-firm risk limits, PANIC_FLATTEN protocol,     |
//|  microstructure filters. SPEC: deploy/infrastructure/             |
//|  SPEC_EXECUTION_INFRASTRUCTURE.txt (frozen 2026-09-08).           |
//|                                                                   |
//|  RESET PROCEDURE (human only): after a breach has been reviewed,  |
//|  delete the terminal global variable "PROP_GUARD_BREACHED"        |
//|  (Tools > Global Variables / F3) and restart the EAs. HWM and     |
//|  EOD anchors persist deliberately so a restart cannot launder a   |
//|  drawdown.                                                        |
//+------------------------------------------------------------------+
#property copyright "RR project"
#property strict

#define PG_MAX_DAILY_LOSS_PCT      3.5
#define PG_MAX_TRAILING_DD_PCT     8.0
#define PG_MAX_SPREAD_US500        0.60
#define PG_MAX_SPREAD_XAUUSD       0.45
#define PG_MAX_SLIPPAGE_INDEX_PT   0.20
#define PG_RISK_PER_TRADE_PCT      0.5
#define PG_MAX_CONCURRENT_RISK_PCT 1.0

#define PG_MAGIC_ORB               26082201
#define PG_MAGIC_SP2L              26082701

#define PG_GV_BREACHED   "PROP_GUARD_BREACHED"
#define PG_GV_HWM        "PROP_GUARD_HWM"
#define PG_GV_EOD_BAL    "PROP_GUARD_EOD_BAL"
#define PG_GV_EOD_DATE   "PROP_GUARD_EOD_DATE"

string g_pgTag = "";
long   g_pgMagic = 0;
bool   g_pgFlattenDone = false;

//+------------------------------------------------------------------+
//| Symbol-aware thresholds (all SPEC limits are in PRICE units)      |
//+------------------------------------------------------------------+
double PG_SpreadLimit(const string symbol)
{
   string s = symbol;
   StringToUpper(s);
   if(StringFind(s, "XAUUSD") >= 0)
      return PG_MAX_SPREAD_XAUUSD;
   return PG_MAX_SPREAD_US500;
}

int PG_DeviationPoints(const string symbol)
{
   double pt = SymbolInfoDouble(symbol, SYMBOL_POINT);
   if(pt <= 0.0)
      return 20;
   int dev = (int)MathRound(PG_MAX_SLIPPAGE_INDEX_PT / pt);
   return (dev < 1 ? 1 : dev);
}

bool PG_IsBreached()
{
   return GlobalVariableCheck(PG_GV_BREACHED)
          && GlobalVariableGet(PG_GV_BREACHED) > 0.0;
}

//+------------------------------------------------------------------+
//| Registration (each _Live EA calls once in OnInit)                 |
//+------------------------------------------------------------------+
void PG_Init(const string strategyTag, const long magicNumber)
{
   g_pgTag   = strategyTag;
   g_pgMagic = magicNumber;
   g_pgFlattenDone = false;

   double equity  = AccountInfoDouble(ACCOUNT_EQUITY);
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   if(!GlobalVariableCheck(PG_GV_HWM))
      GlobalVariableSet(PG_GV_HWM, MathMax(equity, balance));
   if(!GlobalVariableCheck(PG_GV_EOD_BAL))
      GlobalVariableSet(PG_GV_EOD_BAL, balance);

   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   double today = dt.year * 10000.0 + dt.mon * 100.0 + dt.day;
   if(!GlobalVariableCheck(PG_GV_EOD_DATE))
      GlobalVariableSet(PG_GV_EOD_DATE, today);

   PrintFormat("PG init: tag=%s magic=%I64d equity=%.2f balance=%.2f "
               "HWM=%.2f EOD_BAL=%.2f breached=%s",
               g_pgTag, g_pgMagic, equity, balance,
               GlobalVariableGet(PG_GV_HWM),
               GlobalVariableGet(PG_GV_EOD_BAL),
               PG_IsBreached() ? "YES" : "no");
}

//+------------------------------------------------------------------+
//| PANIC_FLATTEN: close ALL positions, delete ALL pendings, latch    |
//+------------------------------------------------------------------+
void PG_CloseEverything()
{
   for(int attempt = 0; attempt < 3; attempt++)
   {
      bool remaining = false;

      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         ulong tk = PositionGetTicket(i);
         if(tk == 0)
            continue;
         if(!PositionSelectByTicket(tk))
            continue;
         string sym = PositionGetString(POSITION_SYMBOL);
         long   ptype = PositionGetInteger(POSITION_TYPE);
         MqlTradeRequest req;
         MqlTradeResult  res;
         ZeroMemory(req);
         ZeroMemory(res);
         req.action    = TRADE_ACTION_DEAL;
         req.symbol    = sym;
         req.volume    = PositionGetDouble(POSITION_VOLUME);
         req.type      = (ptype == POSITION_TYPE_BUY)
                         ? ORDER_TYPE_SELL : ORDER_TYPE_BUY;
         req.position  = tk;
         req.price     = (req.type == ORDER_TYPE_SELL)
                         ? SymbolInfoDouble(sym, SYMBOL_BID)
                         : SymbolInfoDouble(sym, SYMBOL_ASK);
         req.deviation = PG_DeviationPoints(sym);
         req.magic     = (ulong)g_pgMagic;
         req.comment   = "PG panic flatten";
         if(!OrderSend(req, res))
            remaining = true;
         else if(res.retcode != TRADE_RETCODE_DONE
                 && res.retcode != TRADE_RETCODE_DONE_PARTIAL
                 && res.retcode != TRADE_RETCODE_PLACED)
            remaining = true;
      }

      for(int i = OrdersTotal() - 1; i >= 0; i--)
      {
         ulong tk = OrderGetTicket(i);
         if(tk == 0)
            continue;
         MqlTradeRequest req;
         MqlTradeResult  res;
         ZeroMemory(req);
         ZeroMemory(res);
         req.action = TRADE_ACTION_REMOVE;
         req.order  = tk;
         if(!OrderSend(req, res))
            remaining = true;
      }

      if(!remaining)
         break;
   }
}

void PG_Breach(const string reason)
{
   if(!PG_IsBreached())
      GlobalVariableSet(PG_GV_BREACHED, 1.0);
   Print("### PROPFIRMGUARD BREACH: ", reason,
         " — PANIC_FLATTEN (all positions closed, pendings deleted). "
         "Trading halted until human reset of ", PG_GV_BREACHED, ".");
   if(!g_pgFlattenDone)
   {
      PG_CloseEverything();
      g_pgFlattenDone = true;
   }
}

//+------------------------------------------------------------------+
//| Heartbeat: call first in OnTick. false = breached, return now     |
//+------------------------------------------------------------------+
bool PG_OnTick()
{
   if(PG_IsBreached())
   {
      if(!g_pgFlattenDone)
      {
         PG_CloseEverything();
         g_pgFlattenDone = true;
      }
      return false;
   }

   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   double today = dt.year * 10000.0 + dt.mon * 100.0 + dt.day;
   if(GlobalVariableGet(PG_GV_EOD_DATE) != today)
   {
      GlobalVariableSet(PG_GV_EOD_DATE, today);
      GlobalVariableSet(PG_GV_EOD_BAL,
                        AccountInfoDouble(ACCOUNT_BALANCE));
   }

   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(equity > GlobalVariableGet(PG_GV_HWM))
      GlobalVariableSet(PG_GV_HWM, equity);

   double eod = GlobalVariableGet(PG_GV_EOD_BAL);
   if(eod > 0.0 && (eod - equity) / eod * 100.0 >= PG_MAX_DAILY_LOSS_PCT)
   {
      PG_Breach("DAILY_DRAWDOWN");
      return false;
   }

   double hwm = GlobalVariableGet(PG_GV_HWM);
   if(hwm > 0.0 && (hwm - equity) / hwm * 100.0 >= PG_MAX_TRAILING_DD_PCT)
   {
      PG_Breach("TRAILING_DRAWDOWN");
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| Spread filter (entry-time, price units)                           |
//+------------------------------------------------------------------+
bool PG_SpreadOK(const string symbol)
{
   double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
   if(ask <= 0.0 || bid <= 0.0)
      return false;
   return (ask - bid) <= PG_SpreadLimit(symbol) + 1e-9;
}

//+------------------------------------------------------------------+
//| Concurrent-risk controller: 0.5% per strategy slot, 1.0% cap      |
//+------------------------------------------------------------------+
int PG_StrategyPositions()
{
   int n = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong tk = PositionGetTicket(i);
      if(tk == 0)
         continue;
      long magic = (long)PositionGetInteger(POSITION_MAGIC);
      if(magic == PG_MAGIC_ORB || magic == PG_MAGIC_SP2L)
         n++;
   }
   return n;
}

bool PG_ConcurrentRiskOK()
{
   double openRiskPct = PG_StrategyPositions() * PG_RISK_PER_TRADE_PCT;
   return (openRiskPct + PG_RISK_PER_TRADE_PCT)
          <= PG_MAX_CONCURRENT_RISK_PCT + 1e-9;
}

//+------------------------------------------------------------------+
//| Entry gate: breach + spread + concurrency (signal skipped)        |
//+------------------------------------------------------------------+
bool PG_EntryAllowed(const string symbol, const string strategyTag)
{
   if(PG_IsBreached())
   {
      PrintFormat("PG entry blocked (%s): breach latched", strategyTag);
      return false;
   }
   if(!PG_SpreadOK(symbol))
   {
      double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
      double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
      PrintFormat("PG entry blocked (%s): spread %.3f > limit %.2f "
                  "(signal skipped, not queued)",
                  strategyTag, ask - bid, PG_SpreadLimit(symbol));
      return false;
   }
   if(!PG_ConcurrentRiskOK())
   {
      PrintFormat("PG entry blocked (%s): concurrent risk %.1f%% + "
                  "%.1f%% would exceed %.1f%% cap",
                  strategyTag,
                  PG_StrategyPositions() * PG_RISK_PER_TRADE_PCT,
                  PG_RISK_PER_TRADE_PCT, PG_MAX_CONCURRENT_RISK_PCT);
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
//| OrderSend wrapper: deviation enforcement; entries die on breach   |
//+------------------------------------------------------------------+
bool PG_OrderSend(MqlTradeRequest &request, MqlTradeResult &result)
{
   bool isEntry = (request.action == TRADE_ACTION_DEAL
                   && request.position == 0 && request.volume > 0.0);
   if(isEntry && PG_IsBreached())
   {
      Print("PG OrderSend refused: breach latched (entry blocked)");
      return false;
   }
   if(request.action == TRADE_ACTION_DEAL && request.symbol != "")
      request.deviation = PG_DeviationPoints(request.symbol);
   return OrderSend(request, result);
}

//+------------------------------------------------------------------+
//| Test hooks (scripts/TestPropGuard.mq5 only)                       |
//+------------------------------------------------------------------+
void PG_TestReset()
{
   GlobalVariableDel(PG_GV_BREACHED);
   GlobalVariableDel(PG_GV_HWM);
   GlobalVariableDel(PG_GV_EOD_BAL);
   GlobalVariableDel(PG_GV_EOD_DATE);
   g_pgFlattenDone = false;
}

void PG_TestForceDailyBreach()
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   GlobalVariableSet(PG_GV_BREACHED, 0.0);
   GlobalVariableDel(PG_GV_BREACHED);
   GlobalVariableSet(PG_GV_EOD_BAL, equity / (1.0 - 0.036));
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   GlobalVariableSet(PG_GV_EOD_DATE,
                     dt.year * 10000.0 + dt.mon * 100.0 + dt.day);
   GlobalVariableSet(PG_GV_HWM, equity / (1.0 - 0.036));
   g_pgFlattenDone = false;
}

void PG_TestForceTrailingBreach()
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   GlobalVariableDel(PG_GV_BREACHED);
   GlobalVariableSet(PG_GV_HWM, equity / (1.0 - 0.081));
   GlobalVariableSet(PG_GV_EOD_BAL, equity);
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   GlobalVariableSet(PG_GV_EOD_DATE,
                     dt.year * 10000.0 + dt.mon * 100.0 + dt.day);
   g_pgFlattenDone = false;
}
//+------------------------------------------------------------------+
