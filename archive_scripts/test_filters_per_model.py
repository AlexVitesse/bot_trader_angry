"""
Analisis de filtros POR MODELO
Para ver si cada modelo necesita filtros diferentes
"""

import json

def load_trades():
    with open('analysis/all_trades.json', 'r') as f:
        return json.load(f)

def analyze_filter(trades, filter_fn):
    filtered = [t for t in trades if filter_fn(t)]
    if not filtered:
        return None
    wins = len([t for t in filtered if t['trade']['exit_reason'] == 'TP'])
    total = len(filtered)
    return {
        'trades': total,
        'wr': round(wins / total * 100, 1) if total > 0 else 0,
    }

def main():
    trades = load_trades()

    # Agrupar por modelo
    models = {}
    for t in trades:
        model = t['model']
        if model not in models:
            models[model] = []
        models[model].append(t)

    print("=" * 70)
    print("ANALISIS DE FILTROS POR MODELO")
    print("=" * 70)

    filters = [
        ("Sin filtro", lambda t: True),
        ("ret_3 > -0.02", lambda t: t['features']['ret_3'] > -0.02),
        ("ret_3 > -0.01", lambda t: t['features']['ret_3'] > -0.01),
        ("ret_3 > 0", lambda t: t['features']['ret_3'] > 0),
        ("vol_ratio > 1.5", lambda t: t['features']['vol_ratio'] > 1.5),
        ("vol_ratio > 1.7", lambda t: t['features']['vol_ratio'] > 1.7),
        ("vol_ratio > 2.0", lambda t: t['features']['vol_ratio'] > 2.0),
        ("ret_3 > -0.01 AND vol > 1.5", lambda t: t['features']['ret_3'] > -0.01 and t['features']['vol_ratio'] > 1.5),
    ]

    for model_name, model_trades in sorted(models.items()):
        print(f"\n{'='*50}")
        print(f"MODELO: {model_name} ({len(model_trades)} trades)")
        print(f"{'='*50}")

        # Baseline
        baseline = analyze_filter(model_trades, lambda t: True)
        print(f"Baseline: {baseline['trades']} trades, WR {baseline['wr']}%")

        print(f"\n{'Filtro':<35} {'Trades':<10} {'WR%':<10} {'Cambio':<10}")
        print("-" * 65)

        for name, filter_fn in filters:
            stats = analyze_filter(model_trades, filter_fn)
            if stats and stats['trades'] >= 5:
                diff = stats['wr'] - baseline['wr']
                marker = ""
                if diff >= 5:
                    marker = " [MEJOR]"
                elif diff <= -5:
                    marker = " [PEOR]"
                print(f"{name:<35} {stats['trades']:<10} {stats['wr']:<10} {diff:+.1f}%{marker}")

    # Resumen: mejor filtro por modelo
    print("\n" + "=" * 70)
    print("RESUMEN: MEJOR FILTRO POR MODELO")
    print("=" * 70)

    for model_name, model_trades in sorted(models.items()):
        baseline = analyze_filter(model_trades, lambda t: True)
        best_name = "Sin filtro"
        best_wr = baseline['wr']
        best_trades = baseline['trades']

        for name, filter_fn in filters[1:]:  # Skip baseline
            stats = analyze_filter(model_trades, filter_fn)
            if stats and stats['trades'] >= 10 and stats['wr'] > best_wr:
                best_name = name
                best_wr = stats['wr']
                best_trades = stats['trades']

        improvement = best_wr - baseline['wr']
        print(f"\n{model_name}:")
        print(f"  Baseline: {baseline['wr']}% WR")
        print(f"  Mejor: {best_name} -> {best_wr}% WR (+{improvement:.1f}%), {best_trades} trades")

if __name__ == '__main__':
    main()
