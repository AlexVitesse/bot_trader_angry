# Analisis de Limpieza del Proyecto
## Bot de Trading V13 - Febrero 2026

---

## Resumen Ejecutivo

**Total archivos analizados:** ~215 archivos
**Archivos CORE (mantener):** ~25 archivos
**Archivos CANDIDATOS A ELIMINAR:** ~55 archivos
**Espacio recuperable estimado:** ~230 MB

---

## 1. ARCHIVOS CORE (NO ELIMINAR)

### Codigo de Produccion (/src/)
| Archivo | Tamano | Proposito |
|---------|--------|-----------|
| ml_bot.py | 43K | Bot principal ML |
| portfolio_manager.py | 41K | Gestor de posiciones |
| ml_strategy.py | 34K | Estrategia V13 |
| telegram_alerts.py | 7.8K | Alertas Telegram |
| shadow_portfolio_manager.py | 18K | Shadow (futuro uso) |

### Configuracion (/config/)
| Archivo | Mantener |
|---------|----------|
| settings.py | SI - Config V13 |
| __init__.py | SI |

### Modelos Activos (/models/)
| Archivo | Proposito |
|---------|-----------|
| v7_*.pkl (8 pares activos) | Modelos V7 produccion |
| v85_conviction_scorer.pkl | ConvictionScorer activo |
| v85_meta.json | Metadata V8.5 |
| v7_meta.json | Metadata V7 |

### Documentacion (MANTENER)
- PLAN_REENTRENAMIENTO_PARES.md
- DOCUMENTACION_V12_COMPLETA.txt
- METAS_FUTURAS_V12.txt
- PLAN_EXPERIMENTOS_V95.md

### Datos Esenciales (/data/)
- *_4h_history.parquet (8 pares activos)
- *_1d_history.parquet (8 pares activos)
- ml_bot.db
- bot_trades.db

---

## 2. CANDIDATOS A ELIMINAR - SCRIPTS

### Scripts de Backtest OBSOLETOS (31 archivos)
Estos scripts fueron usados para experimentos pero ya no son necesarios:

```
ELIMINAR:
- backtest_v10_1.py
- backtest_v11_pro_filters.py
- backtest_v12_1_specialized.py
- backtest_v12_2_combined.py
- backtest_v12_2b_tpsl_only.py
- backtest_v12_3_adaptive_filters.py
- backtest_v12_4_calibrated_detector.py
- backtest_v12_5_multitf.py
- backtest_v12_filtered_pairs.py
- backtest_v12_hybrid.py
- backtest_v12_optimized.py
- backtest_v12_sr.py
- backtest_v12_sr_fast.py
- backtest_v85_vs_v95atr.py
- backtest_v95.py
- backtest_v95_fast.py
- backtest_adaptive_v10.py
- backtest_specialized_bot.py
- backtest_specialized_fast.py
- backtest_specialized_v2.py
- backtest_experiments.py
- backtest_experiments_v2.py
- backtest_experiments_v3.py
```

**Justificacion:** Ya tenemos los resultados en JSON y la documentacion. V13 es la version final.

### Scripts de Analisis OBSOLETOS (10 archivos)
```
ELIMINAR:
- analyze_regime_distribution.py (resultados ya documentados)
- analyze_losing_trades.py (analisis completo)
- analyze_tpsl_optimization.py (optimizacion hecha)
- diagnose_model_bias.py (bug ya corregido)
- optimize_per_pair.py (optimizacion completa)
- optimize_per_regime.py (optimizacion completa)
- test_funding_atr_combos.py (experimento completo)
```

### Scripts de Entrenamiento LEGACY (5 archivos)
```
ELIMINAR:
- ml_train_v84.py (V8.4 superado por V8.5)
- ml_train_v95.py (V9.5 desactivado en V13)
- ml_test_v9_lossdetector.py (LossDetector desactivado)
- regime_detector.py (sustituido por regime_detector_v2)
```

### Scripts DUPLICADOS o Temporales
```
ELIMINAR:
- nul (archivo vacio)
- ml_backup.db (ya analizado, datos en ml_bot.db)
```

---

## 3. CANDIDATOS A ELIMINAR - MODELOS

### Modelos V9/V9.5 LossDetector (DESACTIVADOS en V13)
```
ELIMINAR:
- v9_loss_detector.pkl
- v9_meta.json
- v95_ld_*.pkl (11 archivos)
- v95_meta.json
```

**Justificacion:** V13 no usa LossDetector. Si en el futuro queremos reactivar, podemos re-entrenar.

### Modelos V7 de Pares EXCLUIDOS
```
ELIMINAR:
- v7_BTC_USDT.pkl (2.1M - par excluido)
- v7_SOL_USDT.pkl (689K - par excluido)
- v7_BNB_USDT.pkl (407K - par excluido)
```

**Justificacion:** Estos pares estan excluidos de V13. Cuando los re-entrenemos, crearemos nuevos.

### Configs de Backtest V12 (ya documentados)
```
ELIMINAR:
- v12_*_config.json (8 archivos)
- backtest_comparison*.json (3 archivos)
- experiment_results*.json (3 archivos)
- specialized_*_results.json (2 archivos)
```

**Justificacion:** Resultados ya estan en DOCUMENTACION_V12_COMPLETA.txt

---

## 4. CANDIDATOS A ELIMINAR - DATOS

### Datos de Pares EXCLUIDOS (~230MB)
```
ELIMINAR:
- BTC_USDT_1m_history.parquet (221M) - ENORME, innecesario
- BTC_USDT_1h_history.parquet (2.4M)
- BTC_USDT_4h_*.parquet
- BTC_USDT_1d_*.parquet
- SOL_USDT_*.parquet (todos)
- BNB_USDT_*.parquet (todos)
```

**Justificacion:** Estos pares estan excluidos. El archivo de 1m de BTC (221MB) es el 85% del espacio de datos.

### Datos de Pares NO USADOS
```
CONSIDERAR ELIMINAR:
- APT_USDT_*.parquet
- SUI_USDT_*.parquet
- OP_USDT_*.parquet
- ARB_USDT_*.parquet
```

**Justificacion:** Estos pares nunca fueron incluidos en ML_PAIRS.

### Datos de Funding (NO usados en V13)
```
CONSIDERAR ELIMINAR:
- *_funding_history.parquet (todos)
```

**Justificacion:** V13 no usa funding rate como feature.

---

## 5. REVISION TELEGRAM - OK

El codigo de `telegram_alerts.py` esta correcto:
- Usa emojis Unicode correctamente
- Mensajes claros y bien formateados
- Threading para no bloquear el bot
- Poller para comandos /status y /resume

**No se requieren cambios.**

---

## 6. ARCHIVOS A MANTENER DEFINITIVAMENTE

### Documentacion
- PLAN_REENTRENAMIENTO_PARES.md
- DOCUMENTACION_V12_COMPLETA.txt
- METAS_FUTURAS_V12.txt
- PLAN_EXPERIMENTOS_V95.md
- plan.txt
- resumen.txt

### Scripts Utiles para Futuro
- backtest_last_14_days.py (validacion rapida)
- backtest_daily_breakdown.py (analisis detallado)
- compare_v12_vs_production.py (comparaciones)
- analyze_market_conditions.py (analisis por regimen)
- ml_export_models.py (re-entrenamiento)
- ml_train_v7.py (base de entrenamiento)
- ml_train_v85.py (ConvictionScorer)
- fetch_history.py (descargar datos)
- macro_data.py (datos macro)
- compare_live.py (comparar produccion)

### Datos Esenciales (8 pares activos)
- XRP_USDT_4h_history.parquet
- NEAR_USDT_4h_history.parquet
- DOT_USDT_4h_history.parquet
- ETH_USDT_4h_history.parquet
- DOGE_USDT_4h_history.parquet
- AVAX_USDT_4h_history.parquet
- LINK_USDT_4h_history.parquet
- ADA_USDT_4h_history.parquet
(y sus versiones 1d)

---

## 7. PLAN DE EJECUCION

### Fase 1: Backup (ANTES de eliminar)
```bash
# Crear backup de modelos
mkdir backup_models_v13
cp models/v9* backup_models_v13/
cp models/v95* backup_models_v13/
cp models/v7_BTC* backup_models_v13/
cp models/v7_SOL* backup_models_v13/
cp models/v7_BNB* backup_models_v13/

# Crear backup de datos grandes
mkdir backup_data_v13
mv data/BTC_USDT_1m_history.parquet backup_data_v13/
```

### Fase 2: Limpiar Scripts (Bajo riesgo)
```bash
# Crear carpeta archive
mkdir archive_scripts
mv backtest_v10*.py archive_scripts/
mv backtest_v11*.py archive_scripts/
mv backtest_v12*.py archive_scripts/
mv backtest_v95*.py archive_scripts/
mv backtest_specialized*.py archive_scripts/
mv backtest_experiments*.py archive_scripts/
mv analyze_regime*.py archive_scripts/
mv diagnose_*.py archive_scripts/
mv optimize_*.py archive_scripts/
mv ml_train_v84.py archive_scripts/
mv ml_train_v95.py archive_scripts/
mv ml_test_v9*.py archive_scripts/
mv regime_detector.py archive_scripts/
```

### Fase 3: Limpiar Modelos
```bash
rm models/v9_*
rm models/v95_*
rm models/v12_*_config.json
rm models/*_results.json
rm models/backtest_*.json
rm models/v7_BTC_USDT.pkl
rm models/v7_SOL_USDT.pkl
rm models/v7_BNB_USDT.pkl
```

### Fase 4: Limpiar Datos (Mayor impacto)
```bash
rm data/BTC_USDT_*.parquet
rm data/SOL_USDT_*.parquet
rm data/BNB_USDT_*.parquet
rm data/APT_USDT_*.parquet
rm data/SUI_USDT_*.parquet
rm data/OP_USDT_*.parquet
rm data/ARB_USDT_*.parquet
rm data/*_funding_history.parquet
```

### Fase 5: Limpiar Misc
```bash
rm nul
rm ml_backup.db
rm -rf __pycache__
rm -rf .pytest_cache
```

---

## 8. ESPACIO RECUPERADO ESTIMADO

| Categoria | Espacio |
|-----------|---------|
| BTC_USDT_1m_history.parquet | 221 MB |
| Otros datos pares excluidos | ~15 MB |
| Modelos V9/V9.5 | ~1 MB |
| Scripts obsoletos | ~500 KB |
| **TOTAL** | **~237 MB** |

---

## 9. ESTRUCTURA FINAL LIMPIA

```
BOTDETRADINGAGRESIVO/
├── src/
│   ├── ml_bot.py
│   ├── portfolio_manager.py
│   ├── ml_strategy.py
│   ├── telegram_alerts.py
│   └── shadow_portfolio_manager.py
├── config/
│   └── settings.py
├── models/
│   ├── v7_XRP_USDT.pkl
│   ├── v7_NEAR_USDT.pkl
│   ├── v7_DOT_USDT.pkl
│   ├── v7_ETH_USDT.pkl
│   ├── v7_DOGE_USDT.pkl
│   ├── v7_AVAX_USDT.pkl
│   ├── v7_LINK_USDT.pkl
│   ├── v7_ADA_USDT.pkl
│   ├── v7_meta.json
│   ├── v85_conviction_scorer.pkl
│   └── v85_meta.json
├── data/
│   ├── *_4h_history.parquet (8 pares)
│   ├── *_1d_history.parquet (8 pares)
│   ├── ml_bot.db
│   └── bot_trades.db
├── docs/ (nuevo)
│   ├── DOCUMENTACION_V12_COMPLETA.txt
│   ├── METAS_FUTURAS_V12.txt
│   ├── PLAN_REENTRENAMIENTO_PARES.md
│   └── PLAN_EXPERIMENTOS_V95.md
├── scripts/ (utilidades)
│   ├── backtest_last_14_days.py
│   ├── backtest_daily_breakdown.py
│   ├── compare_v12_vs_production.py
│   ├── analyze_market_conditions.py
│   ├── ml_export_models.py
│   ├── ml_train_v7.py
│   ├── ml_train_v85.py
│   ├── fetch_history.py
│   ├── macro_data.py
│   └── compare_live.py
├── archive/ (opcional, para no perder nada)
│   └── (scripts y modelos viejos)
└── logs/
```

---

*Documento creado: 25 Feb 2026*
*Version: V13 CORE*
