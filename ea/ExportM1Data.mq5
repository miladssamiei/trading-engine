//+------------------------------------------------------------------+
//|                                                  ExportM1Data.mq5 |
//| Exports M1 OHLC + tick volume + spread for a symbol/date range    |
//| to CSV for import into the ohlc_1min PostgreSQL table.            |
//|                                                                   |
//| Usage: drag onto a chart of the symbol, set FromDate/ToDate,      |
//| run once. Output: MQL5/Files/{symbol}_m1_{from}_{to}.csv          |
//| Columns: timestamp_iso, open, high, low, close, avg_spread,       |
//|          tick_count                                               |
//+------------------------------------------------------------------+
#property strict
#property script_show_inputs

input datetime InpFromDate = D'2024.08.01 00:00';
input datetime InpToDate   = D'2026.08.01 00:00';
input string   InpFileName = "";   // empty = auto-generated

//+------------------------------------------------------------------+
void OnStart()
  {
   string sym       = _Symbol;
   string base      = sym + "_m1";
   string fname;
   if(InpFileName != "")
      fname = InpFileName;
   else
      fname = base + "_" +
              TimeToString(InpFromDate, TIME_DATE) + "_" +
              TimeToString(InpToDate,   TIME_DATE) + ".csv";

   int    chunk     = 20000;
   int    total     = 0;
   string path      = fname;
   int    handle    = FileOpen(path, FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
   if(handle == INVALID_HANDLE)
     {
      Print("ExportM1Data: cannot open ", path, " error=", GetLastError());
      return;
     }
   FileWrite(handle, "timestamp_iso", "open", "high", "low", "close",
             "avg_spread", "tick_count");

   datetime t = InpFromDate;
   while(t < InpToDate)
     {
      MqlRates rates[];
      long     vols[];
      int      spreads[];
      int got = CopyRates(sym, PERIOD_M1, t, chunk, rates);
      if(got <= 0)
        {
         Print("ExportM1Data: CopyRates error=", GetLastError(), " at ", t);
         break;
        }
      ArraySetAsSeries(rates, false);
      CopyTickVolume(sym, PERIOD_M1, t, got, vols);
      ArraySetAsSeries(vols, false);
      CopySpread(sym, PERIOD_M1, t, got, spreads);
      ArraySetAsSeries(spreads, false);

      for(int i = 0; i < got; i++)
        {
         if(rates[i].time >= InpToDate)
           {
            t = InpToDate;
            break;
           }
         datetime tt   = rates[i].time;
         string   iso  = TimeToString(tt, TIME_DATE|TIME_SECONDS);
         string   iso2 = StringSubstr(iso, 0, 4) + "-" + StringSubstr(iso, 5, 2) + "-" +
                         StringSubstr(iso, 8, 2) + " " + StringSubstr(iso, 11, 8);
         double spread = (i < ArraySize(spreads)) ? spreads[i] : 0;
         double pips   = sym_spread_to_pips(sym, spread);
         long   tvol   = (i < ArraySize(vols)) ? vols[i] : 0;
         FileWrite(handle, iso2,
                   DoubleToString(rates[i].open,  _digits(sym)),
                   DoubleToString(rates[i].high,  _digits(sym)),
                   DoubleToString(rates[i].low,   _digits(sym)),
                   DoubleToString(rates[i].close, _digits(sym)),
                   DoubleToString(pips, 8),
                   LongToString(tvol));
         total++;
        }
      t = rates[got - 1].time + 60;
      if(got < chunk)
         break;
     }

   FileClose(handle);
   Print("ExportM1Data: wrote ", total, " M1 bars to ", path);
  }

//+------------------------------------------------------------------+
double sym_spread_to_pips(string sym, int spread_points)
  {
   double pips;
   if(StringFind(sym, "XAU", 0) >= 0)
      pips = spread_points * 0.01;
   else if(StringFind(sym, "US500", 0) >= 0 || StringFind(sym, "US30", 0) >= 0 ||
           StringFind(sym, "JPY", 0) >= 0)
      pips = spread_points * 0.01;
   else
      pips = spread_points * 0.00001;
   return pips;
  }

//+------------------------------------------------------------------+
int _digits(string sym)
  {
   if(StringFind(sym, "XAU", 0) >= 0)
      return 2;
   if(StringFind(sym, "US500", 0) >= 0 || StringFind(sym, "US30", 0) >= 0)
      return 2;
   if(StringFind(sym, "JPY", 0) >= 0)
      return 3;
   return 5;
  }
//+------------------------------------------------------------------+
