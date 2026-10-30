//+------------------------------------------------------------------+
//|                                       CryptoKC_Breakout_EA.mq5   |
//|                    Hypothesis #55: Crypto Portfolio KC Breakout   |
//|                                  Strict 4H / D1 Multi-Timeframe  |
//+------------------------------------------------------------------+
#property copyright "Milad Samieifar / Quantitative Research Pipeline"
#property link      ""
#property version   "1.00"
#property description "Dual-timeframe 4H Keltner Breakout with D1 Anchor and EMA-Riding Trail"

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include <Trade\PositionInfo.mqh>

//--- Input Parameters
input group "=== Strategy Parameters ==="
input int      InpEMA_Period       = 29;       // 4H EMA Baseline Period
input double   InpKC_Multiplier    = 2.77;     // 4H Keltner Channel ATR Multiplier
input int      InpATR_Period       = 29;       // 4H ATR Period
input double   InpMinStopATRMult   = 1.5;      // Minimum Initial Stop Multiplier (ATR)
input int      InpD1_EMA_Period    = 50;       // D1 Anchor Trend EMA Period

input group "=== Risk Management ==="
input double   InpRiskPercent      = 1.0;      // Risk Per Trade (% of Equity)
input ulong    InpMagicNumber      = 20261003; // Expert Advisor Magic Number
input ulong    InpSlippage         = 30;       // Max Allowed Slippage (Points)

//--- Objects & Handles
CTrade         m_trade;
CSymbolInfo    m_symbol;
CPositionInfo  m_position;

int            h_ema_4h;
int            h_atr_4h;
int            h_ema_d1;
datetime       last_4h_bar_time;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Initialize symbol
   if(!m_symbol.Name(_Symbol))
   {
      PrintFormat("Error: Failed to initialize CSymbolInfo for %s", _Symbol);
      return INIT_FAILED;
   }
   m_symbol.Refresh();

   // Initialize CTrade parameters
   m_trade.SetExpertMagicNumber(InpMagicNumber);
   m_trade.SetDeviationInPoints(InpSlippage);
   m_trade.SetTypeFillingBySymbol(_Symbol);

   // Initialize Indicator Handles
   h_ema_4h = iMA(_Symbol, PERIOD_H4, InpEMA_Period, 0, MODE_EMA, PRICE_CLOSE);
   h_atr_4h = iATR(_Symbol, PERIOD_H4, InpATR_Period);
   h_ema_d1 = iMA(_Symbol, PERIOD_D1, InpD1_EMA_Period, 0, MODE_EMA, PRICE_CLOSE);

   if(h_ema_4h == INVALID_HANDLE || h_atr_4h == INVALID_HANDLE || h_ema_d1 == INVALID_HANDLE)
   {
      Print("Error: Failed to create indicator handles.");
      return INIT_FAILED;
   }

   last_4h_bar_time = 0;
   PrintFormat("CryptoKC_Breakout_EA Initialized successfully on %s (4H/D1).", _Symbol);
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   IndicatorRelease(h_ema_4h);
   IndicatorRelease(h_atr_4h);
   IndicatorRelease(h_ema_d1);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // 1. Synchronize execution strictly to the open of a new 4H candle
   datetime current_4h_time = iTime(_Symbol, PERIOD_H4, 0);
   if(current_4h_time == 0 || current_4h_time == last_4h_bar_time)
      return;

   // Bar index 1 is now strictly completed
   last_4h_bar_time = current_4h_time;

   m_symbol.Refresh();
   m_symbol.RefreshRates();

   // 2. Fetch causal completed indicator values (Shift 1 = Completed Bar)
   double ema4h[2], atr4h[2], close4h[2];
   double emad1[1], closed1[1];

   if(CopyBuffer(h_ema_4h, 0, 1, 2, ema4h) < 2 ||
      CopyBuffer(h_atr_4h, 0, 1, 2, atr4h) < 2 ||
      CopyClose(_Symbol, PERIOD_H4, 1, 2, close4h) < 2 ||
      CopyBuffer(h_ema_d1, 0, 1, 1, emad1) < 1 ||
      CopyClose(_Symbol, PERIOD_D1, 1, 1, closed1) < 1)
   {
      Print("Warning: Insufficient historical buffer data. Waiting for next candle.");
      return;
   }

   // Most recently completed 4H bar: index 1 in retrieved array (shift 1)
   double completed_ema4h   = ema4h[1];
   double completed_atr4h   = atr4h[1];
   double completed_close4h = close4h[1];

   // Most recently completed D1 bar
   double completed_emad1   = emad1[0];
   double completed_closed1 = closed1[0];

   // Calculate Keltner Channel boundaries
   double kc_upper = completed_ema4h + (InpKC_Multiplier * completed_atr4h);
   double kc_lower = completed_ema4h - (InpKC_Multiplier * completed_atr4h);

   bool d1_bullish = completed_closed1 > completed_emad1;
   bool d1_bearish = completed_closed1 < completed_emad1;

   // 3. Manage Open Positions (EMA-Baseline Riding Trail)
   bool has_open_position = false;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(m_position.SelectByIndex(i))
      {
         if(m_position.Symbol() == _Symbol && m_position.Magic() == InpMagicNumber)
         {
            has_open_position = true;
            ManageTrailingStop(completed_ema4h);
            break;
         }
      }
   }

   // 4. Signal Generation & Order Execution (Max 1 Open Trade per Instrument)
   if(!has_open_position)
   {
      // Check Long Breakout: 4H Close > KC Upper AND D1 Anchor Bullish
      if(completed_close4h > kc_upper && d1_bullish)
      {
         ExecuteEntry(ORDER_TYPE_BUY, completed_ema4h, completed_atr4h);
      }
      // Check Short Breakout: 4H Close < KC Lower AND D1 Anchor Bearish
      else if(completed_close4h < kc_lower && d1_bearish)
      {
         ExecuteEntry(ORDER_TYPE_SELL, completed_ema4h, completed_atr4h);
      }
   }
}

//+------------------------------------------------------------------+
//| Execute Market Entry with Calculated Risk & Stop Loss            |
//+------------------------------------------------------------------+
void ExecuteEntry(ENUM_ORDER_TYPE order_type, double ema4h, double atr4h)
{
   double entry_price = (order_type == ORDER_TYPE_BUY) ? m_symbol.Ask() : m_symbol.Bid();

   // Structural Risk Distance: max(|Entry - EMA29|, 1.5 * ATR29)
   double risk_distance = MathMax(MathAbs(entry_price - ema4h), InpMinStopATRMult * atr4h);

   double sl_price = 0.0;
   if(order_type == ORDER_TYPE_BUY)
      sl_price = entry_price - risk_distance;
   else
      sl_price = entry_price + risk_distance;

   sl_price = m_symbol.NormalizePrice(sl_price);

   // Calculate Position Size off Account Equity
   double lot_size = CalculateLotSize(entry_price, sl_price);
   if(lot_size <= 0.0)
   {
      PrintFormat("Execution Rejected: Invalid calculated lot size: %.4f", lot_size);
      return;
   }

   // Send Market Order
   if(order_type == ORDER_TYPE_BUY)
   {
      if(!m_trade.Buy(lot_size, _Symbol, entry_price, sl_price, 0.0, "KC Breakout Long"))
         PrintFormat("Error executing Buy order: %s", m_trade.ResultComment());
   }
   else
   {
      if(!m_trade.Sell(lot_size, _Symbol, entry_price, sl_price, 0.0, "KC Breakout Short"))
         PrintFormat("Error executing Sell order: %s", m_trade.ResultComment());
   }
}

//+------------------------------------------------------------------+
//| Ratchet Trailing Stop Loss to the EMA Baseline                   |
//+------------------------------------------------------------------+
void ManageTrailingStop(double current_ema4h)
{
   double current_sl   = m_position.StopLoss();
   double current_tp   = m_position.TakeProfit();
   ulong  ticket       = m_position.Ticket();
   double normalized_ema = m_symbol.NormalizePrice(current_ema4h);

   double stops_level = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;
   double min_stop_dist = MathMax(stops_level, m_symbol.Spread() * _Point);

   if(m_position.PositionType() == POSITION_TYPE_BUY)
   {
      // Long Trail: Only ratchet upwards toward the rising EMA baseline
      if(normalized_ema > current_sl)
      {
         // Verify stop level distance from current market bid
         if((m_symbol.Bid() - normalized_ema) > min_stop_dist)
         {
            if(!m_trade.PositionModify(ticket, normalized_ema, current_tp))
               PrintFormat("Error modifying Long SL to %.5f: %s", normalized_ema, m_trade.ResultComment());
         }
      }
   }
   else if(m_position.PositionType() == POSITION_TYPE_SELL)
   {
      // Short Trail: Only ratchet downwards toward the falling EMA baseline
      if(normalized_ema < current_sl || current_sl == 0.0)
      {
         // Verify stop level distance from current market ask
         if((normalized_ema - m_symbol.Ask()) > min_stop_dist)
         {
            if(!m_trade.PositionModify(ticket, normalized_ema, current_tp))
               PrintFormat("Error modifying Short SL to %.5f: %s", normalized_ema, m_trade.ResultComment());
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Calculate Lot Size based on Account Equity & Distance to SL      |
//+------------------------------------------------------------------+
double CalculateLotSize(double entry_price, double sl_price)
{
   double risk_points = MathAbs(entry_price - sl_price);
   if(risk_points <= 0.0)
      return 0.0;

   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double risk_capital = equity * (InpRiskPercent / 100.0);

   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tick_size  = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

   if(tick_value <= 0.0 || tick_size <= 0.0)
   {
      Print("Warning: Could not retrieve valid tick size/value from broker.");
      return 0.0;
   }

   // Value of 1 lot moving risk_points
   double loss_per_lot = (risk_points / tick_size) * tick_value;
   if(loss_per_lot <= 0.0)
      return 0.0;

   double raw_lots = risk_capital / loss_per_lot;

   // Clamp lot size to broker specifications
   double volume_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double volume_min  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double volume_max  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

   double lots = MathFloor(raw_lots / volume_step) * volume_step;
   lots = MathMax(volume_min, MathMin(volume_max, lots));

   return lots;
}
