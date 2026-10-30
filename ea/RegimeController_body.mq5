//+------------------------------------------------------------------+
//| RegimeController_body.mq5 (generated body; see wrappers)         |
//| Phase-2 master EA for SPEC_REGIME_ROTATION (hypothesis #13).     |
//| Embeds faithful, prefixed copies of the frozen ORB and SP2L      |
//| children; adds causal monthly D1 regime labels (US500.c) and the |
//| pre-declared Arm P gating. REGIME_UNGATED=1 forces all children  |
//| enabled (reproduction mode; must reproduce child trade sets).    |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>
#include "embed/orb_child.mqh"
#include "embed/sp2l_child.mqh"

input string InpD1Symbol = "US500.c";   // D1 label source (SPEC section 3)

// ---- label parameters (SPEC_REGIME_ROTATION, fixed) ----
#define RC_EMA_PERIOD   200
#define RC_VOL_WIN      20
#define RC_MED_WIN      252
#define RC_BAND_PCT     1.0

string   g_label = "";
int      g_labelMonth = 0;
double   g_trendRel = 0.0, g_vol20 = 0.0, g_volMed = 0.0;
int      g_lastNyDate = 0;
bool     g_d1ProbeLogged = false;

bool RegimeAllowORB()   { return (REGIME_UNGATED == 1) || (g_label == "UP-CALM"   || g_label == "DOWN-CALM"); }
bool RegimeAllowSP2L()  { return (REGIME_UNGATED == 1) || (g_label == "UP-VOL"    || g_label == "DOWN-VOL");  }

//+------------------------------------------------------------------+
//| Causal monthly label from US500.c D1 (SPEC section 3)            |
//| Cutoff = UTC midnight of the month's FIRST CALENDAR DAY: bars    |
//| strictly before it only (the month's own first trading day's bar |
//| is EXCLUDED - the corrected cutoff; see Phase-1 report).         |
//+------------------------------------------------------------------+
bool ComputeMonthLabel(const datetime nowUtc)
{
   MqlDateTime dt;
   TimeToStruct(nowUtc, dt);
   dt.day = 1; dt.hour = 0; dt.min = 0; dt.sec = 0;
   datetime cutoff = StructToTime(dt);          // UTC midnight, 1st of month

   int need = RC_MED_WIN + RC_VOL_WIN + 40;
   MqlRates r[];
   int n = CopyRates(InpD1Symbol, PERIOD_D1,
                     cutoff - (datetime)(need * 2 * 86400), cutoff - 1, r);
   if(!g_d1ProbeLogged)
   {
      PrintFormat("RC D1 probe: symbol=%s bars=%d (need %d) last=%s",
                  InpD1Symbol, n, RC_MED_WIN + RC_VOL_WIN + 2,
                  n > 0 ? TimeToString(r[n-1].time, TIME_DATE) : "n/a");
      g_d1ProbeLogged = true;
   }
   if(n < RC_MED_WIN + RC_VOL_WIN + 2)
   {
      PrintFormat("RC LABEL FAIL: only %d D1 bars before %s", n,
                  TimeToString(cutoff, TIME_DATE));
      return false;
   }
   // EMA over all available closes (ewm adjust=False equivalent;
   // 6.5y history in the terminal DB, converged for 2021+)
   double ema = r[0].close;
   double k = 2.0 / (RC_EMA_PERIOD + 1.0);
   for(int i = 1; i < n; i++)
      ema += k * (r[i].close - ema);
   // vol20 series (annualized %, ddof=1)
   int last = n - 1;
   double v[];
   ArrayResize(v, n);
   ArrayInitialize(v, 0.0);
   for(int i = RC_VOL_WIN; i < n; i++)
   {
      double s1 = 0.0, s2 = 0.0;
      for(int j = i - RC_VOL_WIN + 1; j <= i; j++)
      {
         double lr = MathLog(r[j].close / r[j-1].close);
         s1 += lr; s2 += lr * lr;
      }
      double m = s1 / RC_VOL_WIN;
      v[i] = MathSqrt(MathMax(0.0, (s2 - RC_VOL_WIN * m * m)
                              / (RC_VOL_WIN - 1)) * 252.0) * 100.0;
   }
   g_vol20 = v[last];
   // trailing median of vol20 over the MED_WIN values strictly before last
   double win[];
   ArrayResize(win, RC_MED_WIN);
   for(int i = 0; i < RC_MED_WIN; i++)
      win[i] = v[last - RC_MED_WIN + i];
   ArraySort(win);
   g_volMed = 0.5 * (win[RC_MED_WIN/2 - 1] + win[RC_MED_WIN/2]);
   g_trendRel = (r[last].close / ema - 1.0) * 100.0;

   if(MathAbs(g_trendRel) < RC_BAND_PCT)      g_label = "RANGE";
   else if(g_trendRel >= RC_BAND_PCT)         g_label = (g_vol20 <= g_volMed ? "UP-CALM"   : "UP-VOL");
   else                                       g_label = (g_vol20 <= g_volMed ? "DOWN-CALM" : "DOWN-VOL");
   MqlDateTime ld;
   TimeToStruct(nowUtc, ld);
   g_labelMonth = ld.year * 100 + ld.mon;
   return true;
}

void UpdateRegime(const datetime nowUtc)
{
   MqlDateTime dt;
   TimeToStruct(nowUtc, dt);
   int ym = dt.year * 100 + dt.mon;
   if(ym != g_labelMonth)
      if(!ComputeMonthLabel(nowUtc))
      {
         g_label = "";            // label unavailable -> fail safe: no trading
         g_labelMonth = ym;
      }
}

void LogRegimeDay(const int nyDate)
{
   int h = FileOpen("regime_days.csv",
                    FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(h == INVALID_HANDLE)
      return;
   FileSeek(h, 0, SEEK_END);
   FileWrite(h, StringFormat("%d;%s;%.3f;%.2f;%.2f;%d;%d",
            nyDate, g_label, g_trendRel, g_vol20, g_volMed,
            RegimeAllowORB() ? 1 : 0, RegimeAllowSP2L() ? 1 : 0));
   FileClose(h);
}

//+------------------------------------------------------------------+
int OnInit()
{
   if(ORB_OnInit() != INIT_SUCCEEDED) return INIT_FAILED;
   if(SP2L_OnInit() != INIT_SUCCEEDED) return INIT_FAILED;
   int h = FileOpen("regime_days.csv", FILE_WRITE | FILE_TXT | FILE_ANSI);
   if(h != INVALID_HANDLE)
   {
      FileWrite(h, "ny_date;label;trend_rel;vol20;vol_med;orb_on;sp2l_on");
      FileClose(h);
   }
   PrintFormat("RC init: gating=%s d1=%s magicORB=%I64d magicSP2L=%I64d",
               REGIME_UNGATED == 1 ? "OFF (reproduction)" : "ARM P",
               InpD1Symbol, ORB_InpMagicNumber, SP2L_InpMagicNumber);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   ORB_OnDeinit(reason);
   SP2L_OnDeinit(reason);
}

void OnTick()
{
   datetime now = TimeCurrent();          // UTC on US500-tick (verified)
   UpdateRegime(now);
   int nd = ORB_NyDateOf(now);
   if(nd != g_lastNyDate && nd > 0)
   {
      g_lastNyDate = nd;
      LogRegimeDay(nd);
   }
   ORB_OnTick();
   SP2L_OnTick();
}

void OnTradeTransaction(const MqlTradeTransaction &trans,
                        const MqlTradeRequest &request,
                        const MqlTradeResult &result)
{
   ORB_OnTradeTransaction(trans, request, result);
   SP2L_OnTradeTransaction(trans, request, result);
}
//+------------------------------------------------------------------+
