//+------------------------------------------------------------------+
//|                                            UsA30DukService.mq5   |
//| Terminal SERVICE: one-shot builder of the USA30Duk custom symbol |
//| (Dukascopy USA30 M5 tape, EET wall clock, $1/pt/lot) at TERMINAL |
//| level so Strategy Tester agents can synchronize it. Services run |
//| automatically at terminal start; custom-symbol functions are NOT |
//| allowed inside tester agents (error 4014).                       |
//| SPEC: data/results/ftmo_pairs_us500_usa30/SPEC_ADDENDUM_         |
//| FTMO_PAIRS.txt (frozen 2026-09-13).                               |
//+------------------------------------------------------------------+
#property copyright "RR project"
#property version   "1.00"
#property service
#property strict

#define SVC_SYM      "USA30Duk"
#define SVC_CSV      "usa30_m5_eet.csv"
#define SVC_CMD      "usa30duk_cmd.txt"
#define SVC_STATUS   "usa30duk_status.txt"

//+------------------------------------------------------------------+
bool ImportUsa30M5()
{
   int h = FileOpen(SVC_CSV, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE)
   {
      PrintFormat("SVC FATAL: cannot open %s (error %d)", SVC_CSV, GetLastError());
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
         if(CustomRatesReplace(SVC_SYM, from, to, rates) < 0)
         {
            PrintFormat("SVC FATAL: CustomRatesReplace failed (error %d)", GetLastError());
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
      if(CustomRatesReplace(SVC_SYM, from2, to2, rates) < 0)
      {
         PrintFormat("SVC FATAL: final CustomRatesReplace failed (error %d)", GetLastError());
         return false;
      }
      chunks++;
   }
   PrintFormat("SVC imported %d M5 bars into %s (%d chunks)", total, SVC_SYM, chunks);
   return true;
}

//+------------------------------------------------------------------+
bool EnsureUsa30Duk()
{
   bool exists = SymbolSelect(SVC_SYM, true);
   if(!exists)
   {
      if(!CustomSymbolCreate(SVC_SYM, NULL))
      {
         PrintFormat("SVC FATAL: CustomSymbolCreate(%s) failed (error %d)", SVC_SYM, GetLastError());
         return false;
      }
      CustomSymbolSetInteger(SVC_SYM, SYMBOL_DIGITS, 2);
      CustomSymbolSetDouble(SVC_SYM, SYMBOL_POINT, 0.01);
      CustomSymbolSetDouble(SVC_SYM, SYMBOL_TRADE_TICK_SIZE, 0.01);
      CustomSymbolSetDouble(SVC_SYM, SYMBOL_TRADE_TICK_VALUE, 0.01);
      CustomSymbolSetInteger(SVC_SYM, SYMBOL_SPREAD, 228);
      CustomSymbolSetDouble(SVC_SYM, SYMBOL_VOLUME_MIN, 0.01);
      CustomSymbolSetDouble(SVC_SYM, SYMBOL_VOLUME_STEP, 0.01);
      CustomSymbolSetDouble(SVC_SYM, SYMBOL_VOLUME_MAX, 100.0);
      CustomSymbolSetDouble(SVC_SYM, SYMBOL_TRADE_CONTRACT_SIZE, 1.0);
      CustomSymbolSetString(SVC_SYM, SYMBOL_CURRENCY_BASE, "USD");
      CustomSymbolSetString(SVC_SYM, SYMBOL_CURRENCY_PROFIT, "USD");
      CustomSymbolSetString(SVC_SYM, SYMBOL_DESCRIPTION, "Dukascopy USA30 M5, $1/pt, EET wall clock");
      CustomSymbolSetInteger(SVC_SYM, SYMBOL_TRADE_MODE, SYMBOL_TRADE_MODE_FULL);
      PrintFormat("SVC created custom symbol %s", SVC_SYM);
   }
   MqlRates chk[];
   int have = CopyRates(SVC_SYM, PERIOD_M5,
                        StringToTime("2021.01.01"), StringToTime("2026.07.01"), chk);
   if(have < 400000)
   {
      PrintFormat("SVC %s coverage %d M5 bars < 400000 -> importing", SVC_SYM, have);
      if(!ImportUsa30M5())
         return false;
   }
   else
      PrintFormat("SVC %s already has %d M5 bars in range", SVC_SYM, have);
   if(!SymbolSelect(SVC_SYM, true))
   {
      PrintFormat("SVC FATAL: cannot select %s after build (error %d)", SVC_SYM, GetLastError());
      return false;
   }
   return true;
}

//+------------------------------------------------------------------+
void WriteStatus(const string text)
{
   int s = FileOpen(SVC_STATUS, FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(s != INVALID_HANDLE)
   {
      FileWrite(s, text);
      FileClose(s);
   }
}

//+------------------------------------------------------------------+
bool OneshotRequested()
{
   int h = FileOpen(SVC_CMD, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(h == INVALID_HANDLE)
      return false;
   string cmd = FileReadString(h);
   FileClose(h);
   return (StringFind(cmd, "ONESHOT") >= 0);
}

//+------------------------------------------------------------------+
void OnStart()
{
   bool ok = EnsureUsa30Duk();
   WriteStatus(ok ? "DONE" : "FAILED");
   PrintFormat("SVC %s %s", SVC_SYM, ok ? "ready" : "FAILED");
   bool oneshot = OneshotRequested();
   if(oneshot)
   {
      Print("SVC oneshot: closing terminal");
      TerminalClose(0);
   }
}
//+------------------------------------------------------------------+
