# Ejecución en vivo — mediciones reproducibles (plan 2.2)

Deja en scripts las dos mediciones de `docs/AUDITORIA_2026-09.md` §2.1 y §2.2.
Ambos con `C:/Python/python.exe` desde la raíz del repo, sobre
`data/btcusdt_{4h_v15,1d_v15,1h}.parquet` (4h/1d hasta 2026-03-04, 1h desde
2022-01-01).

## 1. Granularidad del trail — `trail_granularity.py`

Trades de `run_v2_backtest` re-simulados con `_sim_long_trailing` sobre velas
1h (`max_bars × 4`): misma entrada (close de la vela de señal), mismo
`trail_dist`, mismo `max_bars`. Solo cambia cada cuánto sube el peak.

| granularidad del trail | n | suma PnL | WR | PF | media/trade |
|---|--:|--:|--:|--:|--:|
| vela 4h (backtest) | 81 | +88,7% | 49% | 2,11 | +1,10% |
| vela 1h (aprox. vivo) | 81 | +82,1% | 49% | 2,01 | +1,01% |

10 de 81 trades cambian de PnL; ninguno cambia de signo. Primer trade con
datos 1h: 2023-03-13 (2022 no tiene trades: régimen bajista, `bull_1d` apagado).

**Diferencia con la auditoría** (82 trades, PF 2,06 → 1,93, 13 cambian): el
script de la auditoría no se conservó. Lo más probable es otra alineación de
la vela de entrada o del corte final de los datos 1h (aquí se exige que el
trade entero quepa en el parquet 1h, lo que deja fuera un trade). La
conclusión no cambia: **refinar el trail intrabar siempre empeora** (PF −0,10;
−6,6 pp de PnL acumulado). A 30 s sería peor que a 1h.

## 2. Warm-up del régimen diario — `regime_warmup.py`

Para cada día, `bull_1d` (EMA50 > EMA200 diarias, `adjust=False`) calculado
solo con las N últimas velas frente al de la historia completa.

| velas en vivo | rango evaluado | días | desacuerdos | meses |
|--:|---|--:|--:|---|
| 300 | 2020-07-03 → 2026-03-04 | 2.071 | 29 | 2022-01, 2023-02, 2023-09, 2023-10, 2024-09, 2025-11 |
| 1000 | 2022-06-03 → 2026-03-04 | 1.371 | **0** | — |
| 300 (mismo rango que 1000) | 2022-06-03 → 2026-03-04 | 1.371 | 23 | 2023-02, 2023-09, 2023-10, 2024-09, 2025-11 |

Los 29 de la auditoría salen de evaluar desde el primer día con 300 velas;
los 23 de `tests/test_regime_warmup.py` de evaluar solo los días donde también
cabe la ventana de 1000 (comparación sobre los mismos días). Los desacuerdos
caen en los cruces de régimen, justo cuando hay señal.

## Estado en producción

- Régimen: desde `7c60de8` el bot baja **1000** velas diarias (plan 0.2) → 0
  desacuerdos.
- Trail: desde `50bbcb4` el bot mueve el trail **solo con velas 4h cerradas**
  (plan 2.3, opción a), igual que el backtest. La fila "vela 1h" de la tabla 1
  describe el comportamiento **anterior** (trail por tick de 30 s) y ya no el
  actual.
