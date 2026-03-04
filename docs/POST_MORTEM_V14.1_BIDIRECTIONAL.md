# Post-Mortem: feature/v14.1-bidirectional

**Fecha:** 2026-03-03
**Resultado:** Rama descartada. Solo se rescató CLAUDE.md.

---

## Qué se intentó

1. Conectar los modelos ML de ETH (`ethusdt_v14`) al pipeline de señales
2. Entrenar y activar modelos SHORT para DOGE/ADA/DOT/SOL
3. Hacer el bot bidireccional (LONG + SHORT simultáneo)

---

## Qué falló y por qué

### Error 1: Conectar ETH ML sin validar las métricas primero

**Lo que se hizo:** Se escribió código para cargar `ethusdt_v14` (regime_detector + ensemble context/momentum/volume) y generar señales.

**El error:** No se leyó `meta.json` antes de conectar. Al leerlo *después* de implementar, se descubrió:
- AUC ~0.5 en todos los sub-modelos → predicción aleatoria
- total_pnl: -996 en walk-forward
- status: NEEDS_REVIEW (el training script mismo lo marcó como fallido)
- 5/12 folds positivos (umbral mínimo: 7/12)

**Lección:** Verificar métricas del modelo ANTES de escribir una línea de código de integración. Un modelo con AUC ~0.5 no se arregla con código — se descarta o se reentrena.

**Adicionalmente:** ETH ha sido excluido en TODAS las versiones anteriores (V13, Low-Overfit experiments). Ese patrón histórico debía haber detenido la implementación de inmediato.

---

### Error 2: Entrenar SHORT sin cross-asset validation

**Lo que se hizo:** Se creó `train_ensemble_short.py` y se entrenaron modelos SHORT para DOGE/ADA/DOT/SOL.

**El error:** Se entrenaron y activaron los modelos antes de hacer la validación cruzada entre activos — que es un requisito obligatorio documentado en `METODOLOGIA_TESTING.md`.

**Resultados del walk-forward SHORT:**
| Par | Folds positivos | WR | PnL total | Break-even WR |
|-----|----------------|-----|-----------|---------------|
| DOGE | 4/12 | 43.3% | +108% | > 40% |
| ADA | 5/12 | 35.3% | -222% | > 40% |
| DOT | 4/12 | 35.8% | -52% | > 40% |
| SOL | 4/12 | 38.4% | -200% | > 40% |

- ADA y SOL están POR DEBAJO del break-even WR → pérdida garantizada sin importar los umbrales
- Ninguno pasa el mínimo de 7/12 folds positivos
- No se hizo validación cruzada (DOGE → SHIB/PEPE, ADA → DOT/SOL/ATOM)

**Lección:** Walk-forward fallido = rechazo inmediato. No se conecta al bot. No se "filtra mejor". El problema es el modelo, no el umbral.

---

### Error 3: Implementar antes de entender el objetivo

**Lo que se hizo:** Se procedió a implementar ETH ML + SHORT antes de haber leído toda la documentación del proyecto.

**El error:** El CLAUDE.md (que documenta el patrón histórico de overfitting y las exclusiones) no existía aún. Al crearlo, se evidenció que lo implementado contradecía directamente el historial del proyecto.

**Lección:** Leer TODA la documentación antes de implementar. En este proyecto existe evidencia documentada de por qué ciertos enfoques no funcionan. Ignorar esa historia = repetir los mismos errores.

---

### Error 4: Confundir "técnicamente correcto" con "útil"

**Lo que se hizo:** El código de integración ETH funcionaba sintácticamente. Los modelos SHORT se entrenaban y guardaban correctamente.

**El error:** Código que funciona pero que conecta modelos sin edge real no agrega valor — agrega ruido y riesgo. La calidad del código es irrelevante si el modelo subyacente no tiene edge.

**Lección:** En ML trading, la pregunta no es "¿funciona el código?" sino "¿tiene el modelo edge estadístico validado en out-of-sample data?"

---

## Qué se rescata

1. **`CLAUDE.md`** — documento de contexto del proyecto, filosofía, historial de errores, reglas para Claude. Mergeado a main.
2. **`train_ensemble_short.py`** — script de entrenamiento SHORT. Útil como referencia metodológica si algún día se intenta SHORT con validación correcta. Se deja en la rama, no en main.

---

## Checklist obligatorio antes de conectar cualquier modelo nuevo

- [ ] Leer `meta.json` / resultados de training antes de escribir código
- [ ] Walk-forward ≥ 7/12 folds positivos
- [ ] Cross-asset validation (todos los activos correlacionados positivos)
- [ ] WR > break-even (`SL / (TP + SL)`)
- [ ] Revisar historial del proyecto — ¿este activo/dirección fue excluido antes? ¿por qué?
- [ ] Documentar resultados antes de hacer merge a main

Si alguno de estos puntos falla → rechazar el modelo, no ajustar umbrales.
