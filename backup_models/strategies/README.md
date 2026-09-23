# Modelos y metas V14/V15 — archivados, NO usar

Movidos aquí en la limpieza de 2026-09 (`docs/PLAN_MEJORAS_2026-09.md` Fase 6.4).
El bot en vivo ya no los lee: el motor por par lo fija `ML_V15_ENGINE` en
`config/settings.py` y hoy solo existe V2 (`src/v2_engine.py`), sin modelos.

Las métricas de los `meta_v15.json` (PF 12-20, DD 1-4%, "WF 8/12") salieron de
un simulador con look-ahead intrabar y trades solapados
(`AUDITORIA_2026-09.md` §4.2, `docs/revalidation/PASO0_lookahead.md`). No son
evidencia de nada. Los `.pkl` son GBM/ensembles que ningún camino vivo carga.
