"""
check_v2_setup.py
=================
Verificacion rapida del setup V2 antes de arrancar el bot.

Uso:
  poetry run python check_v2_setup.py
"""
import sys


def check(label, ok, details=""):
    mark = "[OK] " if ok else "[FAIL]"
    print(f"  {mark} {label}{(' - ' + details) if details else ''}")
    return ok


print("=" * 60)
print("V2 paper-trade setup check")
print("=" * 60)

all_ok = True

# 1. Settings
try:
    from config.settings import ML_V15_PAIRS, ML_V15_SIZING
    pairs_ok = (len(ML_V15_PAIRS) == 5 and
                set(ML_V15_PAIRS) == {'BTC/USDT', 'BNB/USDT', 'DOGE/USDT',
                                       'ETH/USDT', 'OP/USDT'})
    all_ok &= check("ML_V15_PAIRS (5 pares)", pairs_ok,
                    f"{len(ML_V15_PAIRS)} pares: {ML_V15_PAIRS}")
    sizing_ok = all(p in ML_V15_SIZING for p in ML_V15_PAIRS)
    all_ok &= check("ML_V15_SIZING completo", sizing_ok,
                    f"{ML_V15_SIZING}")
except Exception as e:
    all_ok &= check("config.settings import", False, str(e))

# 2. V2 engine
try:
    from src import v2_engine
    all_ok &= check("v2_engine module", True,
                    f"PARAMS_V2 con {len(v2_engine.PARAMS_V2)} keys")
    all_ok &= check("v2_engine.get_live_signal callable",
                    callable(v2_engine.get_live_signal))
    all_ok &= check("v2_engine.run_v2_backtest callable",
                    callable(v2_engine.run_v2_backtest))
except Exception as e:
    all_ok &= check("v2_engine import", False, str(e))

# 3. Strategy V15 con routing V2
try:
    from src.ml_strategy_v15 import MLStrategyV15, V2_AVAILABLE
    all_ok &= check("MLStrategyV15 import", True)
    all_ok &= check("V2_AVAILABLE flag", V2_AVAILABLE)
    s = MLStrategyV15()
    pairs_ok = len(s.pairs) == 5
    all_ok &= check("MLStrategyV15 lee 5 pares", pairs_ok,
                    f"{s.pairs}")
except Exception as e:
    all_ok &= check("MLStrategyV15 import", False, str(e))

# 4. Meta files V2 para cada par
import os
from pathlib import Path
root = Path(__file__).parent
for coin in ['btc', 'bnb', 'doge', 'eth', 'op']:
    p = root / 'strategies' / f'{coin}_v15' / 'models' / 'meta_v2_paper.json'
    all_ok &= check(f"meta_v2_paper.json para {coin.upper()}",
                    p.exists(), str(p))

print("=" * 60)
if all_ok:
    print(">>> SETUP OK — el bot esta listo para arrancar")
    sys.exit(0)
else:
    print(">>> SETUP TIENE PROBLEMAS — revisa los [FAIL] arriba")
    sys.exit(1)
