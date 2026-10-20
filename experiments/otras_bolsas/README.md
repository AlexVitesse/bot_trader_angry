# E11: Hong Kong y Reino Unido como alerta de caídas de BTC

> **Pre-registro escrito el 2026-10-20, ANTES de descargar datos o correr nada.**
> Idea del usuario. Viene de `docs/PLAN_ESPECIALISTAS_2026-10.md`. Se hace porque
> E7 pasó, que era la condición del plan. **Un solo test, α = 0,05.**
>
> ⚠️ **Selección declarada.** La forma de la feature se copia de `oro_z`, la
> única feature macro que aportó en E7. Se eligió **después** de ver E7.

## Pregunta

¿Las bolsas de Hong Kong y Reino Unido añaden información sobre caídas de BTC
**además** de la base de BTC y del macro de E7?

## Datos

yfinance, diario desde 2018-01-01, columna `Close`:
- Hang Seng `^HSI`;
- FTSE 100 `^FTSE`.

Se guardan en `experiments/especialistas/data/`.

## Etiqueta, base, alineado y walk-forward

Exactamente los de E7 (`experiments/especialistas/README.md`):
- Etiqueta: caída de al menos 10% en 10 días.
- Valor del último día hábil **estrictamente anterior** a `d`.
- Logística con `C=1`, walk-forward anual 2021-2026 con purga de 10 días.

Hong Kong cierra hacia las 08:00 UTC y podría usarse el mismo día. **No se
hace:** se mantiene la convención conservadora de E7 para no cambiar dos cosas
a la vez.

## Features (2, fijas)

- `hsi_z`: z-score de 60 días naturales de `log(close_BTC / HSI)`.
- `ftse_z`: lo mismo con el FTSE.

## Test

- **H11:** `ΔAUC = AUC(base + macro E7 + hsi_z + ftse_z) − AUC(base + macro E7) > 0`.
- **p:** bootstrap pareado por bloques de 30 días de calendario, 10.000
  réplicas, `p = P(ΔAUC* ≤ 0)`. **Pasa si p < 0,05.**
- **Descriptivo:**
  - AUC por año;
  - `base + hsi_z` y `base + ftse_z` por separado;
  - null por rotación del bloque `[hsi_z, ftse_z]` (1.000 rotaciones de al
    menos 90 días).

## Riesgo conocido

`log(BTC/índice)` se mueve sobre todo por BTC, así que `hsi_z` y `ftse_z`
estarán muy correlacionadas con `oro_z` y `ret20`. Lo más probable es que no
añadan nada nuevo.

---

## Resultados (2026-10-20): `test_otras_bolsas.py`, salida en `salida.txt`

| modelo | AUC fuera de muestra |
|---|--:|
| base + macro E7 (referencia) | 0,568 |
| **referencia + hsi_z + ftse_z** | **0,524** |
| base + hsi_z | 0,585 |
| base + ftse_z | 0,555 |
| base | 0,475 |

**H11: NO pasa.** ΔAUC = −0,045, IC90 [−0,082; −0,012], **p = 0,99**, p_rot = 0,78.
Añadir Hong Kong y Reino Unido **empeora** el modelo.

### Hallazgo colateral que corrige la lectura de E7 (`diag_btc_z.py`, post-hoc)

`hsi_z` y `ftse_z` correlacionan **0,90 y 0,94** con `oro_z`. Si se divide BTC
entre cualquier cosa, sale casi la misma serie. Se probó la misma feature
**sin dividir por nada**: `btc_z`, el z-score de 60 días de `log(BTC)`.

| base + … | AUC |
|---|--:|
| `btc_z` (solo BTC) | **0,586** |
| `oro_z` (BTC / oro) | 0,588 |
| las dos | 0,591 |

`corr(btc_z, oro_z) = 0,945`.

**El efecto de E7 no es del oro.** Es del propio BTC: le faltaba a la base una
feature, el precio respecto a su media de 60 días. La señal (coef `btc_z` −0,59,
`ret20` +0,26) es: **BTC sube en 20 días pero sigue por debajo de su media de 60
→ rebote dentro de una bajada → más riesgo de caída.** El oro no añade nada
(0,588 contra 0,586). El veredicto de E7 se mantiene como se pre-registró, pero
**su interpretación cambia**: no hay evidencia de que el macro aporte.
