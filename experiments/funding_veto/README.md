# Veto de funding — medido y NO conectado

> Fecha: 2026-08-22 · `test_funding_veto.py` · BTC 2019-09 → 2026-03 (6,5 años)
> Cierra el punto 3 de "Parte 7 — Abierto" de `docs/SESION_2026-08-09.md`.

## El punto de partida

El motor soporta el veto (`a_funding_z_max=2,5`, `f_funding_z_max_long=2,0`) y
hay **6 años de funding en disco** (`data/btc_v15_funding.parquet`, 2020-01 →
2026-03). Pero en vivo la llamada es:

```python
sig = _v2_engine.get_live_signal(df_4h, df_1d=df_1d, df_funding=None)
```

El veto lleva **muerto desde siempre**. Parecía dinero tirado: mecanismo
escrito, datos descargados, un `None` en medio.

## La medida

```
funding_z: media +0,005  std 1,155  min −10,92  max +12,83
  velas con z > 2,5 (veta A):  272  (1,92%)
  velas con z > 2,0 (veta F):  418  (2,95%)
```

| config | n | WR | PF | anual | DD |
|---|--:|--:|--:|--:|--:|
| SIN funding (actual) | 131 | 45,0% | 1,83 | **+19,1%** | −14,7% |
| CON veto de funding | 127 | 45,7% | 1,87 | **+19,1%** | −14,3% |

**El retorno anual es idéntico.** PF y DD mejoran en el tercer decimal.

### Los trades bloqueados, uno a uno

**6 de 131 en 6,5 años = 0,9 al año.**

| fecha | tipo | z | pnl |
|---|---|--:|--:|
| 2021-01-03 | `A_LONG` | 4,05 | −5,36% |
| 2021-01-06 | `A_LONG` | 2,56 | −0,42% |
| 2021-01-08 | `A_LONG` | 3,35 | −4,45% |
| 2021-09-02 | `F_LONG` | 2,23 | −2,09% |
| 2024-02-26 | `F_LONG` | 2,90 | **+6,82%** |
| 2024-11-19 | `F_LONG` | 2,41 | **+2,89%** |

Suma **−2,60%** en 6,5 años. 4 perdedores, **2 ganadores**. Tres de los seis
son la misma semana de enero de 2021.

## Veredicto: no se conecta

**6 eventos no admiten significancia estadística.** Por el criterio recién
adoptado en `CLAUDE.md` — bootstrap p < 0,05 + tamaño de efecto — el veto no
califica. Un −2,60% acumulado en 6,5 años repartido entre 6 trades, con 2 de
ellos ganadores, es indistinguible de ruido.

Comparación que lo ancla: `f_enable_short` se desactivó con **44 trades** y
p=0,644. Aquí hay **6**. Si 44 no bastaron para afirmar un efecto, 6 no bastan
para adoptarlo.

Y hay un coste real al otro lado: conectarlo hace que el comportamiento en vivo
difiera del simulado en `portfolio_sim/`, que es la base de la expectativa
publicada (+12,6%/año). Cambiar producción por un mecanismo que dispara **una
vez al año** y cuyo efecto no se distingue de cero es exactamente el tipo de
movimiento que este proyecto ha aprendido a no hacer.

> Nota de implementación, por si alguna vez cambia el veredicto: el bot **ya**
> llama a `fapi/v1/fundingRate` cada vela para la línea de régimen
> (`_fetch_funding_zscore`, `limit=100`). Subir ese `limit` y pasar el
> histórico a `get_live_signal` no añadiría ni una llamada de red. El coste de
> conectarlo es bajo; lo que no hay es motivo.

## Qué queda

El `df_funding=None` **se queda como está**, ahora por decisión medida y no por
olvido. El punto 3 de "Parte 7" pasa de *pendiente* a *resuelto: no procede*.

Esto no contradice la conclusión de `presupuesto_informacion/`: el funding
sigue siendo la única fuente no-OHLCV disponible, y sigue siendo cierto que
información ortogonal es lo único que movería la aguja. Lo que dice este
experimento es que **este uso concreto** del funding — un veto de umbral sobre
señales ya generadas — no es esa palanca. Un uso distinto (funding como
feature de régimen, o como señal de posicionamiento extremo del mercado) es
otra pregunta, y no está medida.
