# Resumen de Re-Validación V15 — Motor Honesto

> Resultado de correr `revalidate_v15.py` sobre los 20 pares
> `rule_based_trailing` (todos los V15 excepto BTC ML y ETH reglas).
> Fecha: 2026-05-19 · Motor con dos fixes vs. el original:
> (1) una posición a la vez, (2) trailing sin look-ahead intrabar.

---

## Veredicto agregado

| Veredicto | Pares | # |
|-----------|-------|---|
| **KEEP** | (ninguno) | **0** |
| **PAPER-ONLY** | ETC, BCH, UNI (marginales) | 3 |
| **REJECT (provisional)** | ADA, SOL, DOGE, LINK, AVAX, DOT, NEAR, XRP, ATOM, INJ, ALGO, FIL, 1000SHIB, BNB, LTC, AAVE, OP | 17 |

> "Provisional" significa que la Capa A (forward-OOS post-cutoff) aún no se pudo
> ejecutar: los `.parquet` locales terminan en febrero/marzo 2026, antes o justo
> en los cutoffs. El veredicto definitivo llegará al refrescar datos.

---

## Tabla completa: declarado vs. medido honestamente

| Par | Veredicto | WF real | PF medio | p-value | WF declarado | PF L declar. | PF S declar. |
|-----|-----------|---------|----------|---------|--------------|--------------|--------------|
| ADA | REJECT | 5/12 | 1.05 | 0.326 | 10/12 | 2.86 | 13.51 |
| SOL | REJECT | 6/11 | 1.35 | 0.155 | 8/10 | 2.56 | 15.04 |
| DOGE | REJECT | 4/12 | 1.10 | 0.346 | 12/12 | 7.51 | 12.73 |
| LINK | REJECT | 3/12 | 0.75 | 0.958 | 12/12 | 7.40 | 15.02 |
| AVAX | REJECT | 4/11 | 0.83 | 0.713 | 10/11 | 8.33 | 15.43 |
| DOT | REJECT | 4/11 | 1.07 | 0.560 | 11/11 | 7.62 | 13.90 |
| NEAR | REJECT | 5/11 | 1.12 | 0.440 | 11/11 | 11.05 | 16.80 |
| XRP | REJECT | 4/12 | 1.13 | 0.623 | 12/12 | 4.36 | 18.80 |
| ATOM | REJECT | 4/12 | 0.98 | 0.548 | 12/12 | 7.85 | 14.81 |
| INJ | REJECT | 5/11 | 1.07 | 0.386 | 10/10 | 8.89 | 18.96 |
| ALGO | REJECT | 4/12 | 0.93 | 0.575 | 12/12 | 6.62 | 15.36 |
| FIL | REJECT | 5/11 | 1.21 | 0.435 | 9/10 | 9.88 | 18.43 |
| 1000SHIB | REJECT | 1/1 | 1.25 | 0.237 | 9/9 | 11.68 | 17.08 |
| BNB | REJECT | 4/12 | 0.97 | 0.831 | 11/12 | 2.85 | 8.64 |
| LTC | REJECT | 5/12 | 1.18 | 0.563 | 12/12 | 7.10 | 18.21 |
| **ETC** | **PAPER-ONLY** | **7/12** | **1.43** | **0.050** | 12/12 | 8.27 | 17.12 |
| **BCH** | **PAPER-ONLY** | **7/12** | **1.72** | **0.083** | 12/12 | 7.86 | 20.13 |
| **UNI** | **PAPER-ONLY** | **7/11** | **1.37** | **0.390** | 11/11 | 8.34 | 20.17 |
| AAVE | REJECT | 4/11 | 0.96 | 0.808 | 11/11 | 6.99 | 19.76 |
| OP | REJECT | 4/7 | 1.20 | 0.199 | 7/7 | 14.91 | 13.13 |

> "PF medio" = mediana del PF por fold del WF corregido (excluyendo `inf`).
> "p-value" = prob. de que el retorno total acumulado del WF pudiera salir por
> azar (bootstrap n=3000). Significativo si < 0.05.

---

## Lecturas

### 1. El colapso es sistemático y enorme
- PF SHORT declarado medio: **~16**. Real (PF medio fold): **~1.13**.
- WF declarado medio: **~11/12** (92%). Real: **~4.4/11** (40%).
- Caída en folds aprobados: **55-75%** en pares con PF declarado más alto.

### 2. Ningún par tiene edge estadísticamente significativo
- Solo ETC roza el corte de bootstrap (p=0.050) — marginal, casi azar.
- BCH y UNI, los otros "PAPER-ONLY", **fallan el bootstrap** (p=0.083 y 0.390).
- Los 17 rechazados tienen p-values entre 0.15 y 0.96 — ruido.

### 3. SOL fue el mejor de los "validados" originales
SOL: WF 6/11, PF 1.35, p=0.155. Cerca pero no llega. Su mejor semestre real
(2025-07: PF 2.8) vs. declarado (PF 2.56 LONG / 15.04 SHORT) muestra que el
LONG declarado estaba en el rango razonable; el SHORT era pura inflación.

### 4. El SHORT era el bug
Cada par declaraba PF SHORT entre 8 y 20. Después de los fixes, la mayoría de
pares pierden dinero o quedan neutros en BEAR. Confirma lo que CLAUDE.md ya
decía: **SHORT en altcoins no funciona** — el OOS positivo declarado era un
artefacto del trailing look-ahead intrabar (vela bajista volátil → "vendí en el
piso de cada vela + 0.8%").

### 5. La estrategia template no tiene edge
La plantilla `BTC-follower + BTC-breakdown + trailing tight` aplicada a 20
altcoins produce **PF cercano a 1.0 en todos** — la firma de "no edge, solo
ruido + costes". No es un problema de afinación: es la estrategia misma.

---

## Implicaciones

1. **No mover capital real a ningún par** — ni siquiera ETC/BCH/UNI hasta tener
   forward-OOS post-cutoff que confirme algún edge. Paper trading testnet OK.
2. **Reducir `ML_V15_PAIRS`** a los 4 con docs históricos (BTC, ETH, ADA, SOL)
   en cuanto se actualice el bot — o pausar el bot multi-par hasta tener un
   resultado positivo del forward-OOS con datos refrescados.
3. **BTC y ETH siguen pendientes** — usan motores diferentes (ML GBM / rule_based
   propio); este script no los cubre. Necesitan re-validación dedicada con los
   mismos dos fixes aplicados a sus simuladores.
4. **Replantear la estrategia.** El template trailing-tight + BTC-follower no
   tiene edge en altcoins. Opciones a explorar (no obvias):
   - Volver a la filosofía V7 (trailing más amplio, no tan tight).
   - Cambiar el universo (solo BTC/ETH como pares líquidos).
   - Adoptar metodología tipo Jesse (bootstrap + Monte Carlo) en producción.
5. **Refrescar datos** (`download_new_pairs.py`) y re-correr para activar
   Capa A — única manera de cerrar el veredicto definitivo.

---

## Fichas detalladas
Una por par en `docs/revalidation/{PAR}.md`. Cada una incluye la tabla
semestre-por-semestre y el bootstrap completo.
