"""
reset_yield_state.py
====================
Resetea el state del yield manager por un bug previo de loop de sweeps
en modo SIMULATE (inflaba earn_balance > balance real).

Tras el fix en yield_manager.py:
  - En SIMULATE: futures_virtual = real_balance - earn_balance_virtual
  - No mas loops infinitos

Uso:
  poetry run python reset_yield_state.py
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

STATE_FILE = Path('data/yield_state.json')

if not STATE_FILE.exists():
    print(f"No existe {STATE_FILE} — nada que resetear.")
    exit(0)

# Backup del state previo
backup = STATE_FILE.with_suffix('.json.bak')
with open(STATE_FILE) as f:
    old = json.load(f)
with open(backup, 'w') as f:
    json.dump(old, f, indent=2)
print(f"Backup guardado en {backup}")
print(f"State previo: earn_balance=${old.get('earn_balance', 0):.2f}, "
      f"interest=${old.get('accumulated_interest', 0):.4f}, "
      f"n_sweeps={old.get('n_sweeps', 0)}")

# Reset preservando started_at (continuidad de tracking)
new_state = {
    'earn_balance': 0.0,
    'accumulated_interest': 0.0,
    'last_accrual_ts': 0.0,
    'last_rebalance_ts': 0.0,
    'total_swept': 0.0,
    'total_redeemed': 0.0,
    'n_sweeps': 0,
    'n_redeems': 0,
    'started_at': old.get('started_at') or datetime.now(timezone.utc).isoformat(),
}
with open(STATE_FILE, 'w') as f:
    json.dump(new_state, f, indent=2)
print(f"Reset listo. Al reiniciar el bot, yield manager arrancara limpio.")
print(f"started_at preservado: {new_state['started_at']}")
