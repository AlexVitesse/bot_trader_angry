# Ideas - Proyectos Futuros

Fecha: 2026-02-26

---

## 1. Macro LLM Analyst Bot (Prioridad: Media)

### Concepto
Bot que usa LLM para analizar datos macro y generar sesgo direccional semanal.
No ejecuta trades, solo da la direccion (BULL/BEAR/NEUTRAL).

### Arquitectura propuesta

```
CAPA 1: DATA INGESTION
├── FRED API (M2 global, tasas Fed, DXY)
├── Calendario economico (reuniones Fed/ECB/BOJ)
├── News API o RSS (headlines crypto)
└── On-chain basico (fear & greed, funding rates)
            │
            ▼
CAPA 2: LLM ANALYSIS (Claude/GPT)
├── Prompt estructurado con datos
├── Analisis semanal automatico
├── Output: BULL / BEAR / NEUTRAL + confianza %
└── Razonamiento explicado
            │
            ▼
CAPA 3: INTEGRACION
├── Webhook a Telegram con reporte semanal
├── API endpoint para que bot actual consulte sesgo
└── Override manual si el usuario quiere
```

### Datos necesarios

| Fuente | Costo | Lag | Prioridad |
|--------|-------|-----|-----------|
| FRED API | Gratis | 1-2 sem | Alta |
| Fear & Greed Index | Gratis | Diario | Alta |
| Funding Rates | Gratis (Binance) | Real-time | Alta |
| DXY (dolar index) | Gratis | Diario | Media |
| News headlines | $$$ o scraping | Real-time | Media |
| Glassnode on-chain | $$$ | Diario | Baja |

### Integracion con bot actual

```python
# En ml_strategy.py
def get_macro_bias():
    """Consulta sesgo del Macro LLM Bot."""
    # API call al servicio macro
    response = requests.get("http://macro-bot/api/bias")
    return response.json()  # {"bias": "BULL", "confidence": 0.75}

def generate_signals(...):
    macro = get_macro_bias()

    # Solo tomar trades alineados con macro
    if macro['bias'] == 'BULL' and macro['confidence'] > 0.6:
        # Solo LONGs
        signals = [s for s in signals if s['direction'] == 1]
    elif macro['bias'] == 'BEAR' and macro['confidence'] > 0.6:
        # Solo SHORTs
        signals = [s for s in signals if s['direction'] == -1]
    # NEUTRAL = ambos lados permitidos
```

### Validacion

Dificil de backtestear automaticamente. Opciones:
1. Paper trading 3 meses comparando con/sin filtro macro
2. Analisis historico manual: "que hubiera dicho el LLM en marzo 2024?"
3. Metricas de precision del sesgo vs movimiento real de BTC

### Estimacion de esfuerzo

| Fase | Tiempo estimado |
|------|-----------------|
| Research APIs y datos | 1-2 dias |
| Pipeline de datos | 2-3 dias |
| Prompt engineering LLM | 2-3 dias |
| Bot Telegram basico | 1 dia |
| Integracion con bot actual | 1 dia |
| Testing y ajustes | 1 semana |

### Prerequisitos
- [ ] Bot actual V13 validado en produccion (14 dias demo)
- [ ] Cuenta API Claude/OpenAI con credito
- [ ] Definir frecuencia de analisis (diario vs semanal)

### Notas
- Usuario no es analista, necesita automatizacion completa
- Empezar simple: solo M2 + Fear&Greed + funding rates
- LLM como "traductor" de datos a sesgo, no como predictor magico

---

## 2. [Placeholder para futuras ideas]

...

---

## Prioridades actuales

1. **V13 en produccion** - Validar 14 dias (hasta Mar 12, 2026)
2. **Monitorear metricas** - WR > 50%, PF > 1.5
3. **Luego explorar** - Macro LLM Bot como proyecto separado
