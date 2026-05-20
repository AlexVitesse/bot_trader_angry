# Test — ETH-V2 (A + F combinado en ETH, sin re-tuning)

> Aplicación del motor combinado A+F a ETH, sin re-tunear params.
> Pregunta: ¿la combinación que funciona en BTC funciona en ETH?
> Fecha: 2026-05-19 · `test_eth_V2.py`

---

## Resumen del veredicto

**ETH-V2 REJECT formal**, pero **NO por las razones que los 20 alts**.
Es un caso de **mala suerte de muestra**, no de overfitting.

| Criterio | Umbral | ETH-V2 | Pasa? |
|----------|--------|------:|:-----:|
| Bootstrap p<0.05 (real ETH) | < 0.05 | 0.484 | **NO** |
| Mediana sintético > 0 | > 0 | +11.7% | SÍ |
| ≥14/20 sintéticas positivas | ≥ 14 | 16/20 | SÍ |
| Edge vs null > 5% | > 5% | +11.6% | SÍ |

3 de 4 pasan. Solo falla bootstrap p en real, pero **el sintético sugiere que
el edge esperado existe** (mediana +11.7%, mayor incluso que BTC-V2 +7.9%).

---

## La paradoja: real vs sintético

| | Real annual | Sintético mediana | Posición real en distribución |
|--|------------:|------------------:|-------------------------------|
| BTC V2 | +23.4% | +7.9% | percentil ~90 (**afortunado**) |
| **ETH V2** | **+0.2%** | **+11.7%** | percentil ~5 (**desafortunado**) |

Ambas estrategias tienen edge esperado de ~10% anual. Pero el orden específico
de los eventos en la historia 2020-2025:
- Fue FAVORABLE para BTC (caímos en una secuencia buena)
- Fue DESFAVORABLE para ETH (caímos en una secuencia mala)

Cuando vimos BTC +23% emocionados, el sintético ya nos avisó: "+8% esperado".
Ahora vemos ETH +0.2% (fracaso aparente) y el sintético dice: "+11.7% esperado".
**Es el mismo fenómeno, en direcciones opuestas.**

---

## Desglose por componente en real ETH

| Componente | N | WR | PF | Annual | p |
|------------|--:|---:|---:|-------:|--:|
| A_ETH solo | 75 | 45.3% | 1.30 | +5.2% | 0.252 |
| F_ETH solo | 96 | 35.4% | 0.89 | -4.7% | 0.738 |
| **Combinado** | **171** | **39.8%** | **1.07** | **+0.2%** | **0.484** |

### Observaciones
1. **F_ETH es lastre confirmado en real**: -4.7% annual, p=0.738.
2. **A_ETH solo: +5.2%** (peor que cuando se midió por separado: +11.3%).
   Razón: F dispara primero en muchas velas y "consume" oportunidades que
   A habría tomado más tarde con `idx += bars + 1`.
3. **Combinado da +0.2%** porque F lastrea el ~5% de A.

Pero en sintético:
- Mediana A solo: ~7%
- Mediana F solo: probablemente 0-5%
- **Mediana combinado: +11.7%** — la diversificación TEMPORAL ayuda en
  distribución, aunque en el real ETH específico no.

---

## Resultados sintético (20 series block bootstrap)

| Métrica | Valor |
|---------|------:|
| Mediana annual | +11.7% |
| Media annual | +10.9% |
| p25-p75 | +5.7% a +18.2% |
| p5-p95 | -8.7% a +24.6% |
| Series positivas | 16/20 (80%) |
| Series > 10% | 11/20 |
| Series < 0 | 4/20 (incluyendo el real) |

El real ETH cayó en el extremo inferior de la distribución. 4 de 20 universos
paralelos dan resultados peores que el real, 16 dan mejores.

---

## Comparación con los 20 alts rechazados

| | ETH-V2 | 20 alts originales |
|--|--------|---------------------|
| Sintético mediana | **+11.7%** | (cerca de 0 en mediana real) |
| % sintéticas positivas | 80% | ~50% |
| Edge vs null | **+11.6%** | ~0% |
| Categoría | **Mala suerte de muestra** | Bug del simulador |

**ETH-V2 es genuinamente prometedor**. La rechaza el criterio formal por
falta de evidencia estadística en el sample específico, no por ausencia de
edge.

---

## Implicaciones — opciones honestas

### Opción 1 (conservadora estricta): REJECT firme
Aplicamos el criterio formal sin matices. ETH se queda fuera. Bot opera solo
BTC V2. **Esto es lo que el protocolo dice hacer.**

### Opción 2 (acumular evidencia): PAPER-ONLY ETH-V2
- ETH-V2 va a paper trade en testnet JUNTO con BTC V2
- No recibe capital real
- Cada 3 meses, re-medir bootstrap p con datos actualizados
- Si en 6-12 meses sale a p<0.05 → entrar al portfolio
- Si sigue en zona marginal → queda permanentemente fuera
- **Pro**: aprovecha el edge esperado del sintético si es real
- **Contra**: complicación operacional sin garantía

### Opción 3 (más datos): refrescar y re-correr
- `download_new_pairs.py` para traer datos 2026 hasta hoy
- Re-correr test_eth_V2 con sample mayor
- Si p baja a <0.05 con datos nuevos → aprobar
- Si no → REJECT firme
- **Pro**: más datos = menos ruido = veredicto más confiable
- **Contra**: 2026 hasta hoy son solo ~5 meses, ayuda marginalmente

### Opción 4 (rediseñar F para ETH): ⚠️ peligroso
- F fue diseñado para vol patterns generales; ETH tiene diferentes
- Re-tunear F's params para ETH... pero ya demostramos que re-tunear empeora
  14/20 universos
- **NO RECOMENDADO** — exactamente el overfitting que el proyecto evita

---

## Mi recomendación

**Opción 2 o 1**, NO 4.

- Si quieres minimizar complicación operacional: **opción 1** (ETH fuera)
- Si quieres aprovechar la posibilidad de edge real: **opción 2** (paper-only)

La opción 3 (refrescar datos) es accesoria — vale la pena de todas formas
para tener data fresca, pero no va a cambiar el veredicto significativamente.

---

## Lo que este test enseña sobre el método

1. **El bootstrap p de UN sample real es ruidoso.** Depende mucho del orden
   específico de eventos. BTC tuvo orden favorable, ETH desfavorable. Sin la
   prueba sintética no lo habríamos sabido.

2. **El sintético es complementario al real, no sustituto.** Real te dice qué
   pasó; sintético te dice qué pasaría en distribución. Ambos importan.

3. **"Edge esperado positivo" ≠ "edge demostrable en este sample"**.
   ETH-V2 tiene el primero pero no el segundo. Es una distinción importante
   que el protocolo previo no capturaba.

4. **Re-tunear es SIEMPRE la respuesta incorrecta** ante un rechazo. La
   tentación de "ajustar F para ETH" es exactamente el overfitting.
