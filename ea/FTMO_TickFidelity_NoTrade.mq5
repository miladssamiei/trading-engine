#property strict
#property version "1.000"

int OnInit()
{
   PrintFormat("TickFidelity probe symbol=%s digits=%d point=%.8f contract=%.4f",
               _Symbol, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS),
               SymbolInfoDouble(_Symbol, SYMBOL_POINT),
               SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE));
   return INIT_SUCCEEDED;
}

void OnTick() {}
