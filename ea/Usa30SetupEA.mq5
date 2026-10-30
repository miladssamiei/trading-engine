//+------------------------------------------------------------------+
//|                                               UsA30SetupEA.mq5   |
//| One-shot terminal-side setup EA: builds the USA30Duk custom       |
//| symbol (Dukascopy USA30 M5 tape, EET wall clock, $1/pt/lot) at    |
//| TERMINAL level so Strategy Tester agents can synchronize it.      |
//| Attached to a chart via the usa30setup.tpl start config because   |
//| custom symbols cannot be created inside tester agents (error      |
//| 4014) and services do not auto-start on first deployment.         |
//| SPEC: data/results/ftmo_pairs_us500_usa30/SPEC_ADDENDUM_2_        |
//| FTMO_PAIRS.txt (frozen 2026-09-15).                               |
//+------------------------------------------------------------------+
#property copyright "RR project"
#property version   "1.00"
#property strict

#define SU_SYM      "USA30Duk"
#define SU_CSV      "usa30_m5_eet.csv"
#define SU_CMD      "usa30duk_cmd.txt"
#define SU_STATUS   "usa30duk_status.txt"

//+------------------------------------------------------------------+
bool ImportUsa30M5()
{
   int h = FileOpen(SU_CSV, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE)
   {
      PrintFormat("SU FATAL: cannot open %s (error %d)", SU_CSV, GetLastError());
      return false;
   }
   MqlRates rates[];
   ArrayResize(rates, 0, 65536);
   int total = 0;
   int chunks = 0;
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
      int k = ArraySize(rates);
      ArrayResize(rates, k + 1, 65536);
      rates[k].time = (datetime)tv;
      rates[k].open = StringToDouble(parts[1]);
      rates[k].high = StringToDouble(parts[2]);
      rates[k].low = StringToDouble(parts[3]);
      rates[k].close = StringToDouble(parts[4]);
      rates[k].tick_volume = 0;
      rates[k].spread = 0;
      rates[k].real_volume = 0;
      total++;
      if(ArraySize(rates) >= 65536)
      {
         datetime from = rates[0].time;
         datetime to = rates[ArraySize(rates) - 1].time;
         if(CustomRatesReplace(SU_SYM, from, to, rates) < 0)
         {
            PrintFormat("SU FATAL: CustomRatesReplace failed (error %d)", GetLastError());
            FileClose(h);
            return false;
         }
         ArrayResize(rates, 0, 65536);
         chunks++;
      }
   }
   FileClose(h);
   if(ArraySize(rates) > 0)
   {
      datetime from2 = rates[0].time;
      datetime to2 = rates[ArraySize(rates) - 1].time;
      if(CustomRatesReplace(SU_SYM, from2, to2, rates) < 0)
      {
         PrintFormat("SU FATAL: final CustomRatesReplace failed (error %d)", GetLastError());
         return false;
      }
      chunks++;
   }
   PrintFormat("SU imported %d M5 bars into %s (%d chunks)", total, SU_SYM, chunks);
   return true;
}

//+------------------------------------------------------------------+
bool EnsureUsa30Duk()
{
   bool exists = SymbolSelect(SU_SYM, true);
   if(!exists)
   {
      if(!CustomSymbolCreate(SU_SYM, NULL))
      {
         PrintFormat("SU FATAL: CustomSymbolCreate(%s) failed (error %d)", SU_SYM, GetLastError());
         return false;
      }
      CustomSymbolSetInteger(SU_SYM, SYMBOL_DIGITS, 2);
      CustomSymbolSetDouble(SU_SYM, SYMBOL_POINT, 0.01);
      CustomSymbolSetDouble(SU_SYM, SYMBOL_TRADE_TICK_SIZE, 0.01);
      CustomSymbolSetDouble(SU_SYM, SYMBOL_TRADE_TICK_VALUE, 0.01);
      CustomSymbolSetInteger(SU_SYM, SYMBOL_SPREAD, 228);
      CustomSymbolSetDouble(SU_SYM, SYMBOL_VOLUME_MIN, 0.01);
      CustomSymbolSetDouble(SU_SYM, SYMBOL_VOLUME_STEP, 0.01);
      CustomSymbolSetDouble(SU_SYM, SYMBOL_VOLUME_MAX, 100.0);
      CustomSymbolSetDouble(SU_SYM, SYMBOL_TRADE_CONTRACT_SIZE, 1.0);
      CustomSymbolSetString(SU_SYM, SYMBOL_CURRENCY_BASE, "USD");
      CustomSymbolSetString(SU_SYM, SYMBOL_CURRENCY_PROFIT, "USD");
      CustomSymbolSetString(SU_SYM, SYMBOL_DESCRIPTION, "Dukascopy USA30 M5, $1/pt, EET wall clock");
      CustomSymbolSetInteger(SU_SYM, SYMBOL_TRADE_MODE, SYMBOL_TRADE_MODE_FULL);
      PrintFormat("SU created custom symbol %s", SU_SYM);
   }
   MqlRates chk[];
   int have = CopyRates(SU_SYM, PERIOD_M5,
                        StringToTime("2021.01.01"), StringToTime("2026.07.01"), chk);
   if(have < 400000)
   {
      PrintFormat("SU %s coverage %d M5 bars < 400000 -> importing", SU_SYM, have);
      if(!ImportUsa30M5())
         return false;
   }
   else
      PrintFormat("SU %s already has %d M5 bars in range", SU_SYM, have);
   if(!SymbolSelect(SU_SYM, true))
   {
      PrintFormat("SU FATAL: cannot select %s after build (error %d)", SU_SYM, GetLastError());
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
void WriteStatus(const string text)
{
   int s = FileOpen(SU_STATUS, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(s != INVALID_HANDLE)
   {
      FileWrite(s, text);
      FileClose(s);
   }
}

//+------------------------------------------------------------------+
bool OneshotRequested()
{
   int h = FileOpen(SU_CMD, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE)
      return false;
   string cmd = FileReadString(h);
   FileClose(h);
   return (StringFind(cmd, "ONESHOT") >= 0);
}

//+------------------------------------------------------------------+
int OnInit()
{
   bool ok = EnsureUsa30Duk();
   WriteStatus(ok ? "DONE" : "FAILED");
   PrintFormat("SU %s %s", SU_SYM, ok ? "ready" : "FAILED");
   if(OneshotRequested())
   {
      Print("SU oneshot: closing terminal");
      TerminalClose(0);
   }
   return INIT_SUCCEEDED;
}

void OnTick() {}
void OnDeinit(const int reason) {}
//+------------------------------------------------------------------+
