//+------------------------------------------------------------------+
//| MacroTrendEnsemble_FTMO.mq5                                      |
//| Native H1/H4 Engine A or B with account-wide safety latches.     |
//| Chart instances: one symbol/engine/magic per chart.              |
//| Historical proxy results do not validate this broker execution. |
//+------------------------------------------------------------------+
#property copyright "RR project"
#property version   "1.02"
#property strict

#include <Trade/Trade.mqh>
#include <Trade/PositionInfo.mqh>

enum ENUM_ENGINE_TYPE
{
   ENGINE_A_KELTNER_4H = 0,
   ENGINE_B_DONCHIAN_1H = 1
};

input ENUM_ENGINE_TYPE InpEngineType = ENGINE_A_KELTNER_4H;
input ulong  InpMagicNumber = 20261001;
input string InpTradeComment = "MTE_FTMO";
input double InpRiskPercent = 0.5;
input bool   InpUseBrokerMinimumLot = true; // Permit a minimum-lot entry above target risk.
input double InpMaxMinimumLotRiskPct = 3.5; // Hard risk ceiling below the 4% daily guard.
input double InpMaxSpreadPriceUnits = 0.0;
input int    InpMaxPortfolioPositions = 6;
input double InpMaxDailyLossPct = 4.0;
input double InpMaxTotalLossPct = 9.0;
input int    InpPragueResetHourUTC = 0;
input double InpChallengeStartBalance = 10000.0;
input int    InpGuardGroupId = 1; // Increment for a new challenge on the same login.

input int    InpEMA50_D1_Period = 50;
input int    InpEMA29_H4_Period = 29;
input int    InpATR29_H4_Period = 29;
input double InpKeltnerMult = 2.77;
input double InpInitialStopATRMult = 1.5;

input int    InpDonchianPeriod = 20;
input int    InpADXPeriod = 14;
input double InpADXMin = 20.0;
input int    InpATR14_H1_Period = 14;
input double InpTrailATRMult = 4.0;
input int    InpSupertrendPeriod = 10;
input double InpSupertrendMult = 3.0;

// FTMO-style broker clock: UTC+2 winter / UTC+3 European-DST summer.
// The H4/D1 anchors below are reconstructed in America/New_York wall time.
input int    InpServerWinterUTCOffsetHours = 2;
input int    InpServerSummerUTCOffsetHours = 3;
input int    InpMaxEntryDelaySeconds = 120;
input int    InpHistoryH1Bars = 20000;

struct GridBar
{
   datetime start;
   double open;
   double high;
   double low;
   double close;
   bool real;
};

CTrade        g_trade;
CPositionInfo g_position;
string        g_prefix = "";
string        g_day_key_name = "";
string        g_day_balance_name = "";
string        g_daily_latch_name = "";
string        g_total_latch_name = "";
string        g_initial_balance_name = "";
string        g_reset_hour_name = "";
string        g_cap_name = "";
datetime      g_last_h1 = 0;
datetime      g_last_h4_bucket = 0;
bool          g_guard_fault = false;
string        g_guard_fault_reason = "";

bool GoodNumber(const double value)
{
   return MathIsValidNumber(value) && value != EMPTY_VALUE;
}

double TickSize(const string symbol)
{
   return SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
}

double NormPrice(const string symbol, const double price)
{
   double tick = TickSize(symbol);
   if(tick <= 0.0 || !GoodNumber(price))
      return 0.0;
   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   return NormalizeDouble(MathRound(price / tick) * tick, digits);
}

datetime LastSunday(const int year, const int month)
{
   MqlDateTime parts;
   ZeroMemory(parts);
   parts.year = year;
   parts.mon = month;
   parts.day = 31;
   datetime last = StructToTime(parts);
   TimeToStruct(last, parts);
   return last - parts.day_of_week * 86400;
}

int ServerUTCOffsetHours(const datetime utc_time)
{
   MqlDateTime parts;
   TimeToStruct(utc_time, parts);
   datetime summer_start = LastSunday(parts.year, 3) + 3600;
   datetime summer_end = LastSunday(parts.year, 10) + 3600;
   if(utc_time >= summer_start && utc_time < summer_end)
      return InpServerSummerUTCOffsetHours;
   return InpServerWinterUTCOffsetHours;
}

datetime FirstSunday(const int year, const int month)
{
   MqlDateTime parts;
   ZeroMemory(parts);
   parts.year = year;
   parts.mon = month;
   parts.day = 1;
   datetime first = StructToTime(parts);
   TimeToStruct(first, parts);
   return first + ((7 - parts.day_of_week) % 7) * 86400;
}

datetime ServerToUTC(const datetime server_time)
{
   // Both candidate UTC times are checked against the configured European
   // DST calendar. The repeated autumn server hour is necessarily ambiguous
   // without per-bar UTC offsets; choosing summer affects one transition
   // hour but not the adjacent NY 4H or NY-17 daily bucket boundary.
   datetime summer = server_time - InpServerSummerUTCOffsetHours * 3600;
   if(ServerUTCOffsetHours(summer) == InpServerSummerUTCOffsetHours)
      return summer;
   return server_time - InpServerWinterUTCOffsetHours * 3600;
}

int NewYorkUTCOffsetHours(const datetime utc_time)
{
   MqlDateTime parts;
   TimeToStruct(utc_time, parts);
   datetime start = FirstSunday(parts.year, 3) + 7 * 86400 + 7 * 3600;
   datetime end = FirstSunday(parts.year, 11) + 6 * 3600;
   return (utc_time >= start && utc_time < end ? -4 : -5);
}

datetime NewYorkWallHour(const datetime server_hour)
{
   datetime utc = ServerToUTC(server_hour);
   return utc + NewYorkUTCOffsetHours(utc) * 3600;
}

datetime GuardUTCNow()
{
   if((bool)MQLInfoInteger(MQL_TESTER))
   {
      // MT5 tester makes TimeGMT equal server time. Reconstruct UTC from
      // the configured broker offsets, including European DST.
      return ServerToUTC(TimeCurrent());
   }
   return TimeGMT();
}

long GuardDay(const datetime utc_now)
{
   return (long)((utc_now - InpPragueResetHourUTC * 3600) / 86400);
}

datetime H4Bucket(const datetime server_hour)
{
   datetime wall = NewYorkWallHour(server_hour);
   return (datetime)((long)wall / 14400 * 14400);
}

datetime DailyBucket(const datetime server_hour)
{
   datetime wall = NewYorkWallHour(server_hour);
   return (datetime)(((long)wall - 17 * 3600) / 86400 * 86400 + 17 * 3600);
}

bool ValidateInputs()
{
   if(_Period != PERIOD_H1 && _Period != PERIOD_H4)
   {
      Print("MTE: attach to an H1 or H4 chart");
      return false;
   }
   if(StringFind(_Symbol, "XAUUSD") != 0 && StringFind(_Symbol, "BTCUSD") != 0
      && StringFind(_Symbol, "ETHUSD") != 0)
   {
      PrintFormat("MTE: unsupported chart symbol %s", _Symbol);
      return false;
   }
   if(InpMagicNumber == 0 || StringLen(InpTradeComment) == 0 ||
      InpRiskPercent <= 0.0 || InpRiskPercent > 5.0 ||
      InpMaxMinimumLotRiskPct <= 0.0 || InpMaxMinimumLotRiskPct > 5.0 ||
      InpMaxSpreadPriceUnits < 0.0 ||
      InpMaxPortfolioPositions < 1 || InpMaxPortfolioPositions > 6 ||
      InpMaxDailyLossPct <= 0.0 || InpMaxDailyLossPct >= 5.0 ||
      InpMaxTotalLossPct <= 0.0 || InpMaxTotalLossPct >= 10.0 ||
      InpChallengeStartBalance <= 0.0 ||
      InpGuardGroupId < 1 ||
      InpPragueResetHourUTC < 0 || InpPragueResetHourUTC > 23 ||
      InpServerWinterUTCOffsetHours < -12 || InpServerWinterUTCOffsetHours > 14 ||
      InpServerSummerUTCOffsetHours < -12 || InpServerSummerUTCOffsetHours > 14 ||
      InpMaxEntryDelaySeconds < 0 || InpMaxEntryDelaySeconds > 3600 ||
      InpHistoryH1Bars < 1000 || InpHistoryH1Bars > 50000)
   {
      Print("MTE: invalid risk, guard or clock input");
      return false;
   }
   if(InpEMA50_D1_Period < 2 || InpEMA29_H4_Period < 2 || InpATR29_H4_Period < 2 ||
      InpKeltnerMult <= 0 || InpInitialStopATRMult <= 0 ||
      InpDonchianPeriod < 2 || InpADXPeriod < 2 || InpADXMin < 0 ||
      InpATR14_H1_Period < 2 || InpTrailATRMult <= 0 ||
      InpSupertrendPeriod < 2 || InpSupertrendMult <= 0)
   {
      Print("MTE: invalid indicator parameter");
      return false;
   }
   if(TickSize(_Symbol) <= 0.0 || SymbolInfoDouble(_Symbol, SYMBOL_POINT) <= 0.0 ||
      SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP) <= 0.0)
   {
      Print("MTE: symbol tick or volume specification is unavailable");
      return false;
   }
   return true;
}

bool BuildGrids(const datetime current_h1, GridBar &hourly[], GridBar &h4[],
                GridBar &daily[])
{
   MqlRates raw[];
   ArraySetAsSeries(raw, false);
   int got = CopyRates(_Symbol, PERIOD_H1, 0, InpHistoryH1Bars, raw);
   if(got < 1000)
   {
      PrintFormat("MTE: H1 history incomplete: %d bars, error %d", got, GetLastError());
      return false;
   }
   int first = 0;
   while(first < got && raw[first].time >= current_h1)
      first++;
   if(first >= got)
      return false;
   long needed = ((long)current_h1 - (long)raw[first].time) / 3600;
   if(needed < 1000 || needed > 70000)
   {
      PrintFormat("MTE: H1 grid length invalid: %I64d", needed);
      return false;
   }
   ArrayResize(hourly, (int)needed);
   int real_index = first;
   GridBar previous;
   ZeroMemory(previous);
   int n1 = 0;
   for(datetime clock = raw[first].time; clock < current_h1; clock += 3600)
   {
      GridBar bar;
      ZeroMemory(bar);
      bar.start = clock;
      if(real_index < got && raw[real_index].time == clock)
      {
         bar.open = raw[real_index].open;
         bar.high = raw[real_index].high;
         bar.low = raw[real_index].low;
         bar.close = raw[real_index].close;
         bar.real = true;
         real_index++;
      }
      else
      {
         // Fieldwise forward fill matches the research grid, including
         // separate carried open/high/low/close values.
         if(n1 == 0)
            return false;
         bar.open = previous.open;
         bar.high = previous.high;
         bar.low = previous.low;
         bar.close = previous.close;
         bar.real = false;
      }
      hourly[n1++] = bar;
      previous = bar;
   }
   ArrayResize(hourly, n1);
   ArrayResize(h4, n1 / 3 + 2);
   ArrayResize(daily, n1 / 20 + 2);
   int n4 = 0;
   int nd = 0;
   for(int i = 0; i < n1; i++)
   {
      datetime key4 = H4Bucket(hourly[i].start);
      if(n4 == 0 || h4[n4 - 1].start != key4)
      {
         h4[n4] = hourly[i];
         h4[n4].start = key4;
         n4++;
      }
      else
      {
         h4[n4 - 1].high = MathMax(h4[n4 - 1].high, hourly[i].high);
         h4[n4 - 1].low = MathMin(h4[n4 - 1].low, hourly[i].low);
         h4[n4 - 1].close = hourly[i].close;
         h4[n4 - 1].real = h4[n4 - 1].real || hourly[i].real;
      }
      datetime keyd = DailyBucket(hourly[i].start);
      if(nd == 0 || daily[nd - 1].start != keyd)
      {
         daily[nd] = hourly[i];
         daily[nd].start = keyd;
         nd++;
      }
      else
      {
         daily[nd - 1].high = MathMax(daily[nd - 1].high, hourly[i].high);
         daily[nd - 1].low = MathMin(daily[nd - 1].low, hourly[i].low);
         daily[nd - 1].close = hourly[i].close;
         daily[nd - 1].real = daily[nd - 1].real || hourly[i].real;
      }
   }
   ArrayResize(h4, n4);
   ArrayResize(daily, nd);
   return n4 >= 60 && nd >= 60;
}

void EMAFromBars(GridBar &bars[], const int period, double &out[])
{
   int n = ArraySize(bars);
   ArrayResize(out, n);
   if(n <= 0)
      return;
   double alpha = 2.0 / (period + 1.0);
   out[0] = bars[0].close;
   for(int i = 1; i < n; i++)
      out[i] = alpha * bars[i].close + (1.0 - alpha) * out[i - 1];
}

void WilderATR(GridBar &bars[], const int period, const bool sma_seed,
               double &out[])
{
   int n = ArraySize(bars);
   ArrayResize(out, n);
   double sum = 0.0;
   for(int i = 0; i < n; i++)
   {
      double tr = bars[i].high - bars[i].low;
      if(i > 0)
      {
         tr = MathMax(tr, MathAbs(bars[i].high - bars[i - 1].close));
         tr = MathMax(tr, MathAbs(bars[i].low - bars[i - 1].close));
      }
      if(sma_seed)
      {
         if(i < period)
            sum += tr;
         if(i < period - 1)
            out[i] = EMPTY_VALUE;
         else if(i == period - 1)
            out[i] = sum / period;
         else
            out[i] = (out[i - 1] * (period - 1) + tr) / period;
      }
      else
         out[i] = (i == 0 ? tr : (out[i - 1] * (period - 1) + tr) / period);
   }
}

void WilderADX(GridBar &bars[], const int period, double &out[])
{
   int n = ArraySize(bars);
   ArrayResize(out, n);
   double atr = 0.0, plus_sm = 0.0, minus_sm = 0.0, adx = 0.0;
   bool seeded = false;
   for(int i = 0; i < n; i++)
   {
      double tr = bars[i].high - bars[i].low;
      double plus_dm = 0.0, minus_dm = 0.0;
      if(i > 0)
      {
         tr = MathMax(tr, MathAbs(bars[i].high - bars[i - 1].close));
         tr = MathMax(tr, MathAbs(bars[i].low - bars[i - 1].close));
         double up = bars[i].high - bars[i - 1].high;
         double down = bars[i - 1].low - bars[i].low;
         if(up > down && up > 0.0)
            plus_dm = up;
         if(down > up && down > 0.0)
            minus_dm = down;
      }
      if(i == 0)
      {
         atr = tr;
         plus_sm = plus_dm;
         minus_sm = minus_dm;
      }
      else
      {
         atr = (atr * (period - 1) + tr) / period;
         plus_sm = (plus_sm * (period - 1) + plus_dm) / period;
         minus_sm = (minus_sm * (period - 1) + minus_dm) / period;
      }
      double den = plus_sm + minus_sm;
      if(atr <= 0.0 || den <= 0.0)
      {
         out[i] = EMPTY_VALUE;
         continue;
      }
      double dx = 100.0 * MathAbs(plus_sm - minus_sm) / den;
      adx = (seeded ? (adx * (period - 1) + dx) / period : dx);
      seeded = true;
      out[i] = adx;
   }
}

void SupertrendDirection(GridBar &bars[], const int period, const double mult,
                         int &direction[])
{
   int n = ArraySize(bars);
   ArrayResize(direction, n);
   double atr[];
   WilderATR(bars, period, false, atr);
   double upper = 0.0, lower = 0.0;
   for(int i = 0; i < n; i++)
   {
      double mid = (bars[i].high + bars[i].low) / 2.0;
      double basic_up = mid + mult * atr[i];
      double basic_low = mid - mult * atr[i];
      if(i == 0)
      {
         upper = basic_up;
         lower = basic_low;
         direction[i] = -1;
         continue;
      }
      double prev_upper = upper, prev_lower = lower;
      upper = (basic_up < prev_upper || bars[i - 1].close > prev_upper)
              ? basic_up : prev_upper;
      lower = (basic_low > prev_lower || bars[i - 1].close < prev_lower)
              ? basic_low : prev_lower;
      if(direction[i - 1] == -1)
         direction[i] = (bars[i].close > upper ? 1 : -1);
      else
         direction[i] = (bars[i].close < lower ? -1 : 1);
   }
}

int LastCompletedH4(GridBar &bars[], const datetime current_h1)
{
   datetime key = H4Bucket(current_h1);
   for(int i = ArraySize(bars) - 1; i >= 0; i--)
      if(bars[i].start < key)
         return i;
   return -1;
}

int LastCompletedDay(GridBar &bars[], const datetime current_h1)
{
   datetime key = DailyBucket(current_h1);
   for(int i = ArraySize(bars) - 1; i >= 0; i--)
      if(bars[i].start < key)
         return i;
   return -1;
}

string MagicKey(const ulong magic)
{
   return g_prefix + ".M." + StringFormat("%I64u", magic);
}

bool RegisterGuard()
{
   g_prefix = "MTE." + IntegerToString((long)AccountInfoInteger(ACCOUNT_LOGIN))
              + "." + IntegerToString(InpGuardGroupId);
   g_day_key_name = g_prefix + ".DAY";
   g_day_balance_name = g_prefix + ".BAL";
   g_daily_latch_name = g_prefix + ".DL";
   g_total_latch_name = g_prefix + ".TL";
   g_initial_balance_name = g_prefix + ".INIT";
   g_reset_hour_name = g_prefix + ".UTC";
   g_cap_name = g_prefix + ".CAP";
   double stored = 0.0;
   if(GlobalVariableGet(g_initial_balance_name, stored))
   {
      if(MathAbs(stored - InpChallengeStartBalance) > 0.01)
      {
         PrintFormat("MTE: challenge start balance mismatch: stored %.2f, input %.2f",
                     stored, InpChallengeStartBalance);
         return false;
      }
   }
   else if(GlobalVariableSet(g_initial_balance_name, InpChallengeStartBalance) == 0)
      return false;
   if(GlobalVariableGet(g_reset_hour_name, stored))
   {
      if((int)stored != InpPragueResetHourUTC)
      {
         Print("MTE: UTC reset hour differs from another chart in this guard group");
         return false;
      }
   }
   else if(GlobalVariableSet(g_reset_hour_name, InpPragueResetHourUTC) == 0)
      return false;
   if(GlobalVariableGet(g_cap_name, stored))
   {
      if((int)stored != InpMaxPortfolioPositions)
      {
         Print("MTE: portfolio cap differs from another chart in this guard group");
         return false;
      }
   }
   else if(GlobalVariableSet(g_cap_name, InpMaxPortfolioPositions) == 0)
      return false;
   if(GlobalVariableSet(MagicKey(InpMagicNumber), 1.0) == 0)
      return false;
   GlobalVariablesFlush();
   return true;
}

void GuardFault(const string reason)
{
   if(!g_guard_fault)
      PrintFormat("MTE: GUARD FAULT: %s; entries disabled", reason);
   g_guard_fault = true;
   g_guard_fault_reason = reason;
}

bool RefreshDailyBaseline(const datetime utc_now, const long day_key)
{
   double stored_day = 0.0, baseline = 0.0;
   if(GlobalVariableGet(g_day_key_name, stored_day) &&
      (long)stored_day == day_key &&
      GlobalVariableGet(g_day_balance_name, baseline) && baseline > 0.0)
      return true;

   datetime reset_utc = (datetime)(day_key * 86400 + InpPragueResetHourUTC * 3600);
   datetime reset_server = reset_utc + ServerUTCOffsetHours(reset_utc) * 3600;
   datetime server_now = TimeTradeServer();
   if(server_now <= 0)
      server_now = TimeCurrent();
   if(reset_utc > utc_now || reset_server > server_now + 120)
   {
      GuardFault("clock conversion invalid at daily reset");
      return false;
   }
   if(!HistorySelect(reset_server, server_now + 60))
   {
      GuardFault("cannot reconstruct balance at UTC reset from deal history");
      return false;
   }
   double realized_change = 0.0;
   int deals = HistoryDealsTotal();
   for(int i = 0; i < deals; i++)
   {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0 || (datetime)HistoryDealGetInteger(ticket, DEAL_TIME) < reset_server)
         continue;
      ENUM_DEAL_TYPE deal_type = (ENUM_DEAL_TYPE)HistoryDealGetInteger(ticket, DEAL_TYPE);
      // Initial tester funding is a BALANCE deal. It is not trading PnL;
      // subtracting it would produce a zero day-start balance and latch
      // the guard before the first signal. The same applies to deposits,
      // credits and bonuses on a live account.
      if(deal_type == DEAL_TYPE_BALANCE || deal_type == DEAL_TYPE_CREDIT ||
         deal_type == DEAL_TYPE_BONUS)
         continue;
      realized_change += HistoryDealGetDouble(ticket, DEAL_PROFIT);
      realized_change += HistoryDealGetDouble(ticket, DEAL_SWAP);
      realized_change += HistoryDealGetDouble(ticket, DEAL_COMMISSION);
      realized_change += HistoryDealGetDouble(ticket, DEAL_FEE);
   }
   baseline = AccountInfoDouble(ACCOUNT_BALANCE) - realized_change;
   if(!GoodNumber(baseline) || baseline <= 0.0)
   {
      GuardFault("reconstructed daily starting balance is invalid");
      return false;
   }
   // Write baseline first and date second; the date marks completion.
   if(GlobalVariableSet(g_day_balance_name, baseline) == 0 ||
      GlobalVariableSet(g_day_key_name, (double)day_key) == 0)
   {
      GuardFault("cannot persist daily balance baseline");
      return false;
   }
   GlobalVariablesFlush();
   PrintFormat("MTE: daily balance baseline %.2f at UTC day %I64d", baseline, day_key);
   return true;
}

bool ManagedPositionSelected()
{
   ulong magic = (ulong)PositionGetInteger(POSITION_MAGIC);
   if(GlobalVariableCheck(MagicKey(magic)))
      return true;
   string group_comment = InpTradeComment + "#" + IntegerToString(InpGuardGroupId);
   return StringFind(PositionGetString(POSITION_COMMENT), group_comment) == 0;
}

ENUM_ORDER_TYPE_FILLING FillingFor(const string symbol)
{
   long flags = SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE);
   if((flags & SYMBOL_FILLING_IOC) != 0)
      return ORDER_FILLING_IOC;
   if((flags & SYMBOL_FILLING_FOK) != 0)
      return ORDER_FILLING_FOK;
   return ORDER_FILLING_RETURN;
}

bool CloseTicket(const ulong ticket, const string reason)
{
   if(!PositionSelectByTicket(ticket))
      return true;
   string symbol = PositionGetString(POSITION_SYMBOL);
   MqlTick quote;
   if(!SymbolInfoTick(symbol, quote) || quote.bid <= 0.0 || quote.ask <= 0.0)
   {
      PrintFormat("MTE: cannot close %I64u: no valid quote", ticket);
      return false;
   }
   MqlTradeRequest request;
   MqlTradeResult result;
   ZeroMemory(request);
   ZeroMemory(result);
   request.action = TRADE_ACTION_DEAL;
   request.position = ticket;
   request.symbol = symbol;
   request.volume = PositionGetDouble(POSITION_VOLUME);
   request.magic = (ulong)PositionGetInteger(POSITION_MAGIC);
   request.deviation = 20;
   request.type_filling = FillingFor(symbol);
   if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY)
   {
      request.type = ORDER_TYPE_SELL;
      request.price = NormPrice(symbol, quote.bid);
   }
   else
   {
      request.type = ORDER_TYPE_BUY;
      request.price = NormPrice(symbol, quote.ask);
   }
   if(request.price <= 0.0 || request.volume <= 0.0)
      return false;
   bool sent = OrderSend(request, result);
   bool done = sent && (result.retcode == TRADE_RETCODE_DONE ||
                        result.retcode == TRADE_RETCODE_DONE_PARTIAL);
   PrintFormat("MTE: close ticket=%I64u reason=%s sent=%s retcode=%u volume=%.4f price=%.8f %s",
               ticket, reason, (sent ? "true" : "false"), result.retcode,
               request.volume, request.price, result.comment);
   return done;
}

void PanicFlatten(const string reason)
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !ManagedPositionSelected())
         continue;
      CloseTicket(ticket, reason);
   }
}

double CurrentDailyDrawdownPct()
{
   double baseline = 0.0;
   if(!GlobalVariableGet(g_day_balance_name, baseline) || baseline <= 0.0)
      return 0.0;
   return MathMax(0.0, (baseline - AccountInfoDouble(ACCOUNT_EQUITY)) / baseline * 100.0);
}

string GuardStatus()
{
   if(g_guard_fault)
      return "FAULT: " + g_guard_fault_reason;
   if(GlobalVariableCheck(g_total_latch_name) &&
      GlobalVariableGet(g_total_latch_name) > 0.0)
      return "TOTAL LOSS LATCH (manual reset only)";
   long today = GuardDay(GuardUTCNow());
   if(GlobalVariableCheck(g_daily_latch_name) &&
      (long)GlobalVariableGet(g_daily_latch_name) == today)
      return "DAILY LOSS LATCH";
   return "ARMED";
}

bool GuardCheck()
{
   if(g_guard_fault)
   {
      PanicFlatten("GUARD_FAULT");
      return false;
   }
   double registered = 0.0;
   if(!GlobalVariableGet(MagicKey(InpMagicNumber), registered) &&
      GlobalVariableSet(MagicKey(InpMagicNumber), 1.0) == 0)
   {
      GuardFault("cannot refresh managed-magic registry");
      PanicFlatten("GUARD_FAULT");
      return false;
   }
   datetime utc_now = GuardUTCNow();
   if(utc_now <= 0)
   {
      GuardFault("UTC clock unavailable");
      PanicFlatten("GUARD_FAULT");
      return false;
   }
   long today = GuardDay(utc_now);
   if(!RefreshDailyBaseline(utc_now, today))
   {
      PanicFlatten("GUARD_FAULT");
      return false;
   }
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double baseline = GlobalVariableGet(g_day_balance_name);
   double initial = GlobalVariableGet(g_initial_balance_name);
   if(equity <= 0.0 || baseline <= 0.0 || initial <= 0.0)
   {
      GuardFault("account equity or guard baseline unavailable");
      PanicFlatten("GUARD_FAULT");
      return false;
   }
   double daily_dd = (baseline - equity) / baseline * 100.0;
   double total_dd = (initial - equity) / initial * 100.0;
   if(total_dd >= InpMaxTotalLossPct)
   {
      bool first_breach = !GlobalVariableCheck(g_total_latch_name);
      if(GlobalVariableSet(g_total_latch_name, 1.0) == 0)
         GuardFault("cannot persist total-loss latch");
      GlobalVariablesFlush();
      if(first_breach)
         PrintFormat("MTE: PANIC_FLATTEN total equity drawdown %.3f%%", total_dd);
   }
   if(daily_dd >= InpMaxDailyLossPct)
   {
      bool first_breach = (!GlobalVariableCheck(g_daily_latch_name) ||
                           (long)GlobalVariableGet(g_daily_latch_name) != today);
      if(GlobalVariableSet(g_daily_latch_name, (double)today) == 0)
         GuardFault("cannot persist daily-loss latch");
      GlobalVariablesFlush();
      if(first_breach)
         PrintFormat("MTE: PANIC_FLATTEN daily equity drawdown %.3f%%", daily_dd);
   }
   if(GuardStatus() != "ARMED")
   {
      PanicFlatten("PROP_GUARD");
      return false;
   }
   return true;
}

int ManagedPositionCount()
{
   int count = 0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(PositionGetTicket(i) != 0 && ManagedPositionSelected())
         count++;
   }
   return count;
}

ulong OwnPositionTicket()
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(g_position.SelectByIndex(i) && g_position.Symbol() == _Symbol &&
         (ulong)g_position.Magic() == InpMagicNumber)
         return g_position.Ticket();
   }
   return 0;
}

bool SymbolOccupiedByOtherOnNetting()
{
   if((ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE) ==
      ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)
      return false;
   for(int i = 0; i < PositionsTotal(); i++)
      if(PositionGetTicket(i) != 0 && PositionGetString(POSITION_SYMBOL) == _Symbol)
         return true;
   return false;
}

double CalculateLots(const int direction, const double entry, const double stop,
                     bool &minimum_override, double &planned_risk_cash)
{
   minimum_override = false;
   planned_risk_cash = 0.0;
   double tick_size = TickSize(_Symbol);
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_loss = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE_LOSS);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double distance = MathAbs(entry - stop);
   if(tick_size <= 0.0 || point <= 0.0 || step <= 0.0 ||
      min_lot <= 0.0 || max_lot < min_lot || distance < tick_size)
      return 0.0;
   double per_lot_tick = MathMax(tick_value, tick_loss);
   if(per_lot_tick <= 0.0)
      return 0.0;
   double loss_per_lot = distance / point * (point / tick_size) * per_lot_tick;
   double calc_profit = 0.0;
   ENUM_ORDER_TYPE type = (direction == 1 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
   if(OrderCalcProfit(type, _Symbol, 1.0, entry, stop, calc_profit) && calc_profit < 0.0)
      loss_per_lot = MathMax(loss_per_lot, -calc_profit);
   double risk_cash = AccountInfoDouble(ACCOUNT_BALANCE) * InpRiskPercent / 100.0;
   if(loss_per_lot <= 0.0 || risk_cash <= 0.0)
      return 0.0;
   double raw = risk_cash / loss_per_lot;
   if(raw + 1e-12 < min_lot)
   {
      double minimum_risk = min_lot * loss_per_lot;
      double minimum_risk_pct = minimum_risk / AccountInfoDouble(ACCOUNT_BALANCE) * 100.0;
      if(!InpUseBrokerMinimumLot || minimum_risk_pct > InpMaxMinimumLotRiskPct + 1e-9)
      {
         PrintFormat("MTE: minimum lot blocked: requested %.5f, broker minimum %.5f, "
                     "minimum-stop risk %.2f (%.3f%%), override=%s, cap=%.3f%%",
                     raw, min_lot, minimum_risk, minimum_risk_pct,
                     (InpUseBrokerMinimumLot ? "on" : "off"), InpMaxMinimumLotRiskPct);
         return 0.0;
      }
      minimum_override = true;
      planned_risk_cash = minimum_risk;
      PrintFormat("MTE: MIN_LOT_OVERRIDE requested %.5f -> %.5f lots; "
                  "initial-stop risk %.2f (%.3f%% of balance; target %.3f%%)",
                  raw, min_lot, minimum_risk, minimum_risk_pct, InpRiskPercent);
      return min_lot;
   }
   double lots = MathFloor(MathMin(raw, max_lot) / step + 1e-9) * step;
   int volume_digits = 0;
   double s = step;
   while(volume_digits < 8 && MathAbs(s - MathRound(s)) > 1e-9)
   {
      s *= 10.0;
      volume_digits++;
   }
   lots = NormalizeDouble(lots, volume_digits);
   if(lots < min_lot || lots > max_lot || lots * loss_per_lot > risk_cash + 0.01)
      return 0.0;
   planned_risk_cash = lots * loss_per_lot;
   return lots;
}

bool StopIsBrokerValid(const int direction, const double stop, const MqlTick &quote)
{
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int stop_points = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double gap = (stop_points + 1) * point;
   if(direction == 1)
      return stop < quote.bid - gap;
   return stop > quote.ask + gap;
}

void RatchetStop(const ulong ticket, const int direction, const double candidate_raw)
{
   if(ticket == 0 || !PositionSelectByTicket(ticket))
      return;
   double candidate = NormPrice(_Symbol, candidate_raw);
   double old_stop = PositionGetDouble(POSITION_SL);
   double tick_size = TickSize(_Symbol);
   if(candidate <= 0.0 || old_stop <= 0.0)
   {
      PrintFormat("MTE: position %I64u has missing/invalid stop; closing", ticket);
      CloseTicket(ticket, "MISSING_SL");
      return;
   }
   if(direction == 1 && candidate <= old_stop + tick_size / 2.0)
      return;
   if(direction == -1 && candidate >= old_stop - tick_size / 2.0)
      return;
   MqlTick quote;
   if(!SymbolInfoTick(_Symbol, quote))
      return;
   // A crossed trailing level is a market exit, not an invalid SL request.
   if((direction == 1 && candidate >= quote.bid) ||
      (direction == -1 && candidate <= quote.ask))
   {
      CloseTicket(ticket, "TRAIL_CROSSED");
      return;
   }
   int stops = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   int freeze = (int)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL);
   double gap = (MathMax(stops, freeze) + 1) * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if((direction == 1 && quote.bid - candidate <= gap) ||
      (direction == -1 && candidate - quote.ask <= gap))
      return;
   double tp = PositionGetDouble(POSITION_TP);
   bool ok = g_trade.PositionModify(ticket, candidate, tp);
   uint code = g_trade.ResultRetcode();
   if(!ok || code != TRADE_RETCODE_DONE)
      PrintFormat("MTE: trail modify failed ticket=%I64u SL=%.8f retcode=%u %s",
                  ticket, candidate, code, g_trade.ResultRetcodeDescription());
   else
      PrintFormat("MTE: ratcheted ticket=%I64u SL %.8f -> %.8f", ticket, old_stop, candidate);
}

bool PlaceEntry(const int direction, const double anchor_ema, const double atr)
{
   if(!GuardCheck() || atr <= 0.0 || !GoodNumber(atr))
      return false;
   if(OwnPositionTicket() != 0 || SymbolOccupiedByOtherOnNetting() ||
      ManagedPositionCount() >= InpMaxPortfolioPositions)
      return false;
   MqlTick quote;
   if(!SymbolInfoTick(_Symbol, quote) || quote.bid <= 0.0 || quote.ask <= 0.0)
      return false;
   double spread = quote.ask - quote.bid;
   if(spread < 0.0 || (InpMaxSpreadPriceUnits > 0.0 && spread > InpMaxSpreadPriceUnits))
   {
      PrintFormat("MTE: entry blocked by spread %.8f (limit %.8f)",
                  spread, InpMaxSpreadPriceUnits);
      return false;
   }
   double entry = NormPrice(_Symbol, direction == 1 ? quote.ask : quote.bid);
   if(entry <= 0.0)
      return false;
   double distance = (InpEngineType == ENGINE_A_KELTNER_4H
                      ? MathMax(MathAbs(entry - anchor_ema), InpInitialStopATRMult * atr)
                      : InpTrailATRMult * atr);
   double stop = NormPrice(_Symbol, entry - direction * distance);
   if(stop <= 0.0 || !StopIsBrokerValid(direction, stop, quote))
   {
      PrintFormat("MTE: stop %.8f is invalid for current quote %.8f/%.8f",
                  stop, quote.bid, quote.ask);
      return false;
   }
   bool minimum_override = false;
   double planned_risk_cash = 0.0;
   double lots = CalculateLots(direction, entry, stop, minimum_override, planned_risk_cash);
   if(lots <= 0.0)
      return false;
   string comment = InpTradeComment + "#" + IntegerToString(InpGuardGroupId);
   bool sent = (direction == 1
                ? g_trade.Buy(lots, _Symbol, entry, stop, 0.0, comment)
                : g_trade.Sell(lots, _Symbol, entry, stop, 0.0, comment));
   uint code = g_trade.ResultRetcode();
   bool done = sent && (code == TRADE_RETCODE_DONE || code == TRADE_RETCODE_DONE_PARTIAL);
   PrintFormat("MTE: %s %s engine=%s lots=%.4f entry=%.8f SL=%.8f spread=%.8f "
               "planned_stop_risk=%.2f min_lot_override=%s retcode=%u %s",
               (done ? "OPEN" : "REJECT"), (direction == 1 ? "BUY" : "SELL"),
               (InpEngineType == ENGINE_A_KELTNER_4H ? "A" : "B"),
               lots, entry, stop, spread, planned_risk_cash,
               (minimum_override ? "yes" : "no"), code, g_trade.ResultRetcodeDescription());
   if(done)
   {
      ulong ticket = OwnPositionTicket();
      if(ticket != 0 && PositionSelectByTicket(ticket))
      {
         double actual_open = PositionGetDouble(POSITION_PRICE_OPEN);
         double actual_sl = PositionGetDouble(POSITION_SL);
         double actual_lots = PositionGetDouble(POSITION_VOLUME);
         double loss = 0.0;
         ENUM_ORDER_TYPE side = (direction == 1 ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
         double allowed_risk_pct = (minimum_override ? InpMaxMinimumLotRiskPct : InpRiskPercent);
         double allowed_risk_cash = AccountInfoDouble(ACCOUNT_BALANCE) * allowed_risk_pct / 100.0;
         bool calculated = (actual_sl > 0.0 &&
                            OrderCalcProfit(side, _Symbol, actual_lots, actual_open, actual_sl, loss));
         if(!calculated || -loss > allowed_risk_cash + 0.01)
         {
            PrintFormat("MTE: filled position exceeds %.3f%% risk cap, lacks stop, or "
                        "risk cannot be calculated; closing ticket %I64u (loss %.2f)",
                        allowed_risk_pct, ticket, -loss);
            CloseTicket(ticket, "POST_FILL_RISK");
         }
      }
   }
   return done;
}

void ProcessA(const datetime current_h1, GridBar &h4[], GridBar &daily[])
{
   // Called once when a new NY-wall H4 bucket first has a real quote.
   // The prior H4 and D1 bars are both complete.
   int h4_index = LastCompletedH4(h4, current_h1);
   int day_index = LastCompletedDay(daily, current_h1);
   if(h4_index < InpEMA29_H4_Period - 1 ||
      h4_index < InpATR29_H4_Period - 1 ||
      day_index < InpEMA50_D1_Period - 1 ||
      !h4[h4_index].real)
      return;
   double ema4[], atr4[], ema_day[];
   EMAFromBars(h4, InpEMA29_H4_Period, ema4);
   WilderATR(h4, InpATR29_H4_Period, true, atr4);
   EMAFromBars(daily, InpEMA50_D1_Period, ema_day);
   double baseline = ema4[h4_index];
   double atr = atr4[h4_index];
   if(!GoodNumber(atr) || atr <= 0.0)
      return;
   ulong ticket = OwnPositionTicket();
   if(ticket != 0 && PositionSelectByTicket(ticket))
   {
      int direction = ((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) ==
                       POSITION_TYPE_BUY ? 1 : -1);
      RatchetStop(ticket, direction, baseline);
      return;
   }
   if(TimeCurrent() - current_h1 > InpMaxEntryDelaySeconds)
      return;
   double close4 = h4[h4_index].close;
   double day_close = daily[day_index].close;
   int signal = 0;
   if(close4 > baseline + InpKeltnerMult * atr && day_close > ema_day[day_index])
      signal = 1;
   else if(close4 < baseline - InpKeltnerMult * atr && day_close < ema_day[day_index])
      signal = -1;
   if(signal != 0)
      PlaceEntry(signal, baseline, atr);
}

void ProcessB(const datetime current_h1, GridBar &hourly[], GridBar &h4[])
{
   int last = ArraySize(hourly) - 1;
   if(last < InpDonchianPeriod + 2 || !hourly[last].real)
      return;
   double atr1[], adx1[];
   WilderATR(hourly, InpATR14_H1_Period, false, atr1);
   WilderADX(hourly, InpADXPeriod, adx1);
   double atr = atr1[last];
   if(!GoodNumber(atr) || atr <= 0.0)
      return;
   ulong ticket = OwnPositionTicket();
   if(ticket != 0 && PositionSelectByTicket(ticket))
   {
      int direction = ((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) ==
                       POSITION_TYPE_BUY ? 1 : -1);
      RatchetStop(ticket, direction,
                  hourly[last].close - direction * InpTrailATRMult * atr);
      return;
   }
   if(TimeCurrent() - current_h1 > InpMaxEntryDelaySeconds ||
      !GoodNumber(adx1[last]) || adx1[last] < InpADXMin)
      return;
   int h4_index = LastCompletedH4(h4, current_h1);
   if(h4_index < InpSupertrendPeriod + 5)
      return;
   int direction4[];
   SupertrendDirection(h4, InpSupertrendPeriod, InpSupertrendMult, direction4);
   double upper = -DBL_MAX, lower = DBL_MAX;
   // Donchian20 ending at t-2, excluding the trigger bar t-1.
   int end = last - 1;
   int begin = end - InpDonchianPeriod + 1;
   if(begin < 0)
      return;
   for(int i = begin; i <= end; i++)
   {
      upper = MathMax(upper, hourly[i].high);
      lower = MathMin(lower, hourly[i].low);
   }
   int signal = 0;
   if(hourly[last].close > upper && direction4[h4_index] == 1)
      signal = 1;
   else if(hourly[last].close < lower && direction4[h4_index] == -1)
      signal = -1;
   if(signal != 0)
      PlaceEntry(signal, 0.0, atr);
}

void CheckOwnStops()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || PositionGetString(POSITION_SYMBOL) != _Symbol ||
         (ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;
      if(PositionGetDouble(POSITION_SL) <= 0.0)
         CloseTicket(ticket, "MISSING_SL");
   }
}

void Dashboard()
{
   double open_pnl = 0.0, stop = 0.0, open_stop_risk = 0.0;
   int n = 0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(!g_position.SelectByIndex(i) || g_position.Symbol() != _Symbol ||
         (ulong)g_position.Magic() != InpMagicNumber)
         continue;
      open_pnl += g_position.Profit() + g_position.Swap();
      stop = g_position.StopLoss();
      double stop_profit = 0.0;
      ENUM_ORDER_TYPE side = (g_position.PositionType() == POSITION_TYPE_BUY
                              ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
      if(stop > 0.0 && OrderCalcProfit(side, _Symbol, g_position.Volume(),
                                       g_position.PriceOpen(), stop, stop_profit))
         open_stop_risk += MathMax(0.0, -stop_profit);
      n++;
   }
   string engine = (InpEngineType == ENGINE_A_KELTNER_4H
                    ? "A: H4 Keltner / D1 EMA" : "B: H1 Donchian / H4 Supertrend");
   Comment("MacroTrendEnsemble FTMO\n", engine, "  ", _Symbol,
           "\nOpen positions: ", n,
           "  Open PnL: ", DoubleToString(open_pnl, 2),
           "\nTrailing SL: ", DoubleToString(stop, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS)),
           "\nOpen stop risk: ", DoubleToString(open_stop_risk, 2),
           " (", DoubleToString(100.0 * open_stop_risk /
                                  MathMax(1.0, AccountInfoDouble(ACCOUNT_BALANCE)), 2), "%)",
           "\nMinimum lot: ", (InpUseBrokerMinimumLot ? "ON" : "OFF"),
           "  cap ", DoubleToString(InpMaxMinimumLotRiskPct, 2), "%",
           "\nDaily DD: ", DoubleToString(CurrentDailyDrawdownPct(), 2), "%",
           "\nGuard: ", GuardStatus());
}

int OnInit()
{
   if(!ValidateInputs() || !RegisterGuard())
      return INIT_FAILED;
   g_trade.SetExpertMagicNumber(InpMagicNumber);
   g_trade.SetDeviationInPoints(20);
   g_trade.SetTypeFillingBySymbol(_Symbol);
   g_trade.SetAsyncMode(false);
   datetime current_h1 = iTime(_Symbol, PERIOD_H1, 0);
   if(current_h1 > 0)
      g_last_h4_bucket = H4Bucket(current_h1);
   if(!GuardCheck())
      PrintFormat("MTE: initialized with guard status %s", GuardStatus());
   EventSetTimer(5);
   PrintFormat("MTE v1.02: initialized %s %s magic=%I64u guard=%s; check broker NY grid offsets before live use",
               _Symbol, (InpEngineType == ENGINE_A_KELTNER_4H ? "A" : "B"),
               InpMagicNumber, GuardStatus());
   Dashboard();
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   Comment("");
   PrintFormat("MTE: deinitialized reason=%d", reason);
}

void OnTimer()
{
   GuardCheck();
   CheckOwnStops();
   Dashboard();
}

void OnTick()
{
   bool guard_ok = GuardCheck();
   CheckOwnStops();
   if(!guard_ok)
   {
      // Do not replay a bucket that began while the account was latched.
      datetime blocked_h1 = iTime(_Symbol, PERIOD_H1, 0);
      if(blocked_h1 > 0)
      {
         g_last_h1 = blocked_h1;
         g_last_h4_bucket = H4Bucket(blocked_h1);
      }
      Dashboard();
      return;
   }
   datetime current_h1 = iTime(_Symbol, PERIOD_H1, 0);
   if(current_h1 <= 0 || current_h1 == g_last_h1)
      return;
   g_last_h1 = current_h1;
   GridBar hourly[], h4[], daily[];
   if(!BuildGrids(current_h1, hourly, h4, daily))
   {
      Print("MTE: signal grid unavailable; no order on this H1 bar");
      Dashboard();
      return;
   }
   if(InpEngineType == ENGINE_A_KELTNER_4H)
   {
      datetime bucket = H4Bucket(current_h1);
      if(g_last_h4_bucket != 0 && bucket != g_last_h4_bucket)
         ProcessA(current_h1, h4, daily);
      g_last_h4_bucket = bucket;
   }
   else
      ProcessB(current_h1, hourly, h4);
   Dashboard();
}
