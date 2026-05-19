"""
Analisis de filtros optimos POR MODELO
Usando datos de all_models_trades.json
"""

import json
from collections import defaultdict

def load_trades():
    with open('analysis/all_models_trades.json', 'r') as f:
        return json.load(f)

def analyze_filter(trades, filter_fn):
    filtered = [t for t in trades if filter_fn(t)]
    if not filtered:
        return None
    wins = len([t for t in filtered if t['trade']['exit_reason'] == 'TP'])
    total = len(filtered)
    pnl = sum(t['trade']['pnl_pct'] for t in filtered)
    return {
        'trades': total,
        'wins': wins,
        'wr': round(wins / total * 100, 1) if total > 0 else 0,
        'pnl': round(pnl, 1)
    }

def get_feature(t, name, default=0):
    """Obtiene feature de forma segura"""
    return t.get('features', {}).get(name, default)

def main():
    trades = load_trades()

    # Agrupar por modelo
    models = defaultdict(list)
    for t in trades:
        models[t['model']].append(t)

    print("=" * 80)
    print("ANALISIS DE FILTROS OPTIMOS POR MODELO")
    print("=" * 80)

    # Filtros a probar (genericos)
    filters_generic = [
        ("Sin filtro", lambda t: True),
        ("ret_3 > -0.03", lambda t: get_feature(t, 'ret_3', 0) > -0.03),
        ("ret_3 > -0.02", lambda t: get_feature(t, 'ret_3', 0) > -0.02),
        ("ret_3 > -0.01", lambda t: get_feature(t, 'ret_3', 0) > -0.01),
        ("ret_3 > 0", lambda t: get_feature(t, 'ret_3', 0) > 0),
        ("vol_ratio > 1.3", lambda t: get_feature(t, 'vol_ratio', 0) > 1.3),
        ("vol_ratio > 1.5", lambda t: get_feature(t, 'vol_ratio', 0) > 1.5),
        ("vol_ratio > 1.7", lambda t: get_feature(t, 'vol_ratio', 0) > 1.7),
        ("vol_ratio > 2.0", lambda t: get_feature(t, 'vol_ratio', 0) > 2.0),
        ("trend == 1", lambda t: get_feature(t, 'trend', 0) == 1.0),
        ("rsi > 0.4", lambda t: get_feature(t, 'rsi', get_feature(t, 'rsi14', 50)/100) > 0.4),
        ("rsi < 0.6", lambda t: get_feature(t, 'rsi', get_feature(t, 'rsi14', 50)/100) < 0.6),
        ("bb_pct > 0.3", lambda t: get_feature(t, 'bb_pct', 0.5) > 0.3),
        ("bb_pct < 0.7", lambda t: get_feature(t, 'bb_pct', 0.5) < 0.7),
        # Combinados
        ("ret_3 > -0.02 AND vol > 1.5",
         lambda t: get_feature(t, 'ret_3', 0) > -0.02 and get_feature(t, 'vol_ratio', 0) > 1.5),
        ("ret_3 > 0 AND vol > 1.5",
         lambda t: get_feature(t, 'ret_3', 0) > 0 and get_feature(t, 'vol_ratio', 0) > 1.5),
        ("ret_3 > -0.01 AND trend == 1",
         lambda t: get_feature(t, 'ret_3', 0) > -0.01 and get_feature(t, 'trend', 0) == 1.0),
        ("vol > 1.5 AND trend == 1",
         lambda t: get_feature(t, 'vol_ratio', 0) > 1.5 and get_feature(t, 'trend', 0) == 1.0),
    ]

    # Filtros especificos para BTC (usa features diferentes)
    filters_btc = [
        ("Sin filtro", lambda t: True),
        ("ret_3 > -0.03", lambda t: get_feature(t, 'ret_3', 0) > -0.03),
        ("ret_3 > -0.02", lambda t: get_feature(t, 'ret_3', 0) > -0.02),
        ("ret_3 > -0.01", lambda t: get_feature(t, 'ret_3', 0) > -0.01),
        ("ret_3 > 0", lambda t: get_feature(t, 'ret_3', 0) > 0),
        ("vol_ratio > 1.3", lambda t: get_feature(t, 'vol_ratio', 0) > 1.3),
        ("vol_ratio > 1.5", lambda t: get_feature(t, 'vol_ratio', 0) > 1.5),
        ("rsi14 > 35", lambda t: get_feature(t, 'rsi14', 50) > 35),
        ("rsi14 < 65", lambda t: get_feature(t, 'rsi14', 50) < 65),
        ("rsi14 35-65", lambda t: 35 < get_feature(t, 'rsi14', 50) < 65),
        ("adx > 20", lambda t: get_feature(t, 'adx', 0) > 20),
        ("adx > 25", lambda t: get_feature(t, 'adx', 0) > 25),
        ("atr_pct < 3", lambda t: get_feature(t, 'atr_pct', 0) < 3),
        ("bb_pct 0.2-0.8", lambda t: 0.2 < get_feature(t, 'bb_pct', 0.5) < 0.8),
        # Por regimen
        ("TREND_UP only", lambda t: 'TREND_UP' in t.get('features', {}).get('regime', '')),
        ("TREND_DOWN only", lambda t: 'TREND_DOWN' in t.get('features', {}).get('regime', '')),
        ("RANGE only", lambda t: 'RANGE' in t.get('features', {}).get('regime', '')),
        # Combinados
        ("adx > 25 AND vol > 1.3",
         lambda t: get_feature(t, 'adx', 0) > 25 and get_feature(t, 'vol_ratio', 0) > 1.3),
        ("ret_3 > -0.02 AND adx > 20",
         lambda t: get_feature(t, 'ret_3', 0) > -0.02 and get_feature(t, 'adx', 0) > 20),
    ]

    best_filters = {}

    for model_name in sorted(models.keys()):
        model_trades = models[model_name]
        print(f"\n{'='*60}")
        print(f"MODELO: {model_name} ({len(model_trades)} trades)")
        print(f"{'='*60}")

        # Elegir filtros segun modelo
        filters_to_test = filters_btc if model_name == 'BTC' else filters_generic

        baseline = analyze_filter(model_trades, lambda t: True)
        print(f"Baseline: {baseline['trades']} trades, WR {baseline['wr']}%, PnL {baseline['pnl']}%")

        print(f"\n{'Filtro':<40} {'Trades':<8} {'WR%':<8} {'Cambio':<10} {'PnL':<10}")
        print("-" * 76)

        best_improvement = 0
        best_filter_name = "Sin filtro"

        for name, filter_fn in filters_to_test:
            stats = analyze_filter(model_trades, filter_fn)
            if stats and stats['trades'] >= max(5, len(model_trades) * 0.1):  # Min 10% de trades
                diff = stats['wr'] - baseline['wr']
                marker = ""
                if diff >= 5:
                    marker = " [+]"
                elif diff >= 10:
                    marker = " [++]"
                elif diff <= -5:
                    marker = " [-]"

                print(f"{name:<40} {stats['trades']:<8} {stats['wr']:<8} {diff:+.1f}%{marker:<6} {stats['pnl']}")

                # Mejor filtro: balance entre WR improvement y trades retenidos
                if diff > best_improvement and stats['trades'] >= len(model_trades) * 0.2:
                    best_improvement = diff
                    best_filter_name = name

        best_filters[model_name] = {
            'filter': best_filter_name,
            'improvement': best_improvement,
            'baseline_wr': baseline['wr']
        }

    # Resumen final
    print("\n" + "=" * 80)
    print("RESUMEN: MEJOR FILTRO POR MODELO")
    print("=" * 80)

    print(f"\n{'Modelo':<10} {'WR Base':<10} {'Mejor Filtro':<35} {'Mejora':<10}")
    print("-" * 65)

    for model, info in sorted(best_filters.items()):
        new_wr = info['baseline_wr'] + info['improvement']
        print(f"{model:<10} {info['baseline_wr']:.1f}%     {info['filter']:<35} +{info['improvement']:.1f}% -> {new_wr:.1f}%")

    # Recomendaciones
    print("\n" + "=" * 80)
    print("RECOMENDACIONES DE IMPLEMENTACION")
    print("=" * 80)

    for model, info in sorted(best_filters.items()):
        if info['improvement'] >= 3:
            print(f"\n{model}: APLICAR FILTRO '{info['filter']}'")
            print(f"  - Mejora esperada: +{info['improvement']:.1f}% WR")
        else:
            print(f"\n{model}: NO NECESITA FILTRO (ya funciona bien)")
            print(f"  - WR actual: {info['baseline_wr']:.1f}%")


if __name__ == '__main__':
    main()
