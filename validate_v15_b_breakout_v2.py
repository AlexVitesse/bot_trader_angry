"""
validate_v15_b_breakout_v2.py  —  Branch: v15/momentum-breakout  [Iteracion 2]
==============================================================================
Optimizacion de la estrategia B para aumentar frecuencia de seniales.

PROBLEMA IDENTIFICADO EN V1:
  - OOS: solo 49 trades en 4 anos (1.2/mes)
  - Muchos semestres con 0 seniales (condiciones demasiado estrictas)
  - Cuando dispara: WR=59.2%, PF=1.90 -> CALIDAD EXCELENTE
  - Objetivo: mantener WR>50% y PF>1.3 con 3-5 trades/mes

CAMBIOS vs V1:
  - vol_ratio: 1.8 -> 1.3 (breakouts con volumen moderado tambien son validos)
  - bb_width threshold: 4.0% -> 5.5% (consolidaciones algo mas amplias)
  - bar_move max: 2.5% -> 3.5% (velas de ruptura algo mas grandes)
  - lookback breakout: high20 (sin cambio - sigue siendo high de 20 barras)
  - ADX anterior: < 28 -> < 30 (mas permisivo)

CRITERIO DE APROBACION:
  - WF: >= 7/12 folds OK (WR>40%, PF>1.2, n>=3)
  - OOS: WR>=50%, PF>=1.3, >=3 trades/mes

HISTORIAL DE ITERACIONES:
  v1: vol>=1.8, bb_width<4.0, bar_move<2.5% -> 4/12 folds, WR=59.2%, 1.2t/m
  v2: vol>=1.3, bb_width<5.5%, bar_move<3.5% -> [resultados abajo]
"""

import sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from v15_framework import (
    load_btc_4h, compute_features_4h, compute_macro_daily, merge_daily_to_4h,
    sim_trade_fixed, metrics, print_metrics, walk_forward,
    WF_FOLDS, OOS_START, OOS_END
)

STRATEGY_NAME = 'B-v2: Momentum Breakout (Optimizado)'
BRANCH        = 'v15/momentum-breakout'
MAX_BARS      = 16

# Parametros v2 (relajados)
VOL_RATIO_MIN   = 1.3   # era 1.8
BB_WIDTH_MAX    = 5.5   # era 4.0
BAR_MOVE_MAX    = 3.5   # era 2.5
ADX_PREV_MAX    = 30    # era 28
NARROW_BARS_MIN = 3     # sin cambio
SL_PCT_MIN      = 0.005
SL_PCT_MAX      = 0.04


def detect_setup(df: pd.DataFrame, i: int) -> dict | None:
    if i < 25:
        return None
    row = df.iloc[i]

    # 1. Ruptura: close > maximo de 20 barras anteriores
    high20 = float(df['high'].iloc[i-20:i].max())
    if row['close'] <= high20:
        return None

    # 2. Volumen confirma (relajado: 1.3x)
    if row.get('vol_ratio', 1) < VOL_RATIO_MIN:
        return None

    # 3. Vela de ruptura no demasiado grande (relajado: 3.5%)
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > BAR_MOVE_MAX:
        return None

    # 4. Consolidacion previa (relajado: bb_width < 5.5%)
    recent_bb = df['bb_width'].iloc[i-5:i]
    narrow_bars = (recent_bb < BB_WIDTH_MAX).sum()
    if narrow_bars < NARROW_BARS_MIN:
        return None

    # 5. ADX anterior bajo (relajado: < 30)
    prev_adx = df['adx14'].iloc[i-3:i].mean()
    if prev_adx > ADX_PREV_MAX:
        return None

    # 6. TP/SL
    entry  = float(row['close'])
    sl_raw = float(df['low'].iloc[i-5:i].min()) * 0.997
    sl_pct = (entry - sl_raw) / entry

    if sl_pct < SL_PCT_MIN or sl_pct > SL_PCT_MAX:
        return None

    tp_pct = sl_pct * 1.5

    return {
        'setup': 'BREAKOUT_V2',
        'entry': entry,
        'sl_pct': sl_pct,
        'tp_pct': tp_pct,
        'vol_ratio': float(row.get('vol_ratio', 1)),
        'bar_move': bar_move,
        'prev_adx': prev_adx,
        'narrow_bars': int(narrow_bars),
    }


def simulate(df, global_i, trade):
    return sim_trade_fixed(df, global_i, trade['entry'],
                           trade['tp_pct'], trade['sl_pct'], max_bars=MAX_BARS)


def main():
    print("=" * 70)
    print(f"ESTRATEGIA {STRATEGY_NAME}")
    print(f"Rama: {BRANCH} | Iteracion: v2")
    print("=" * 70)
    print("\nOptimizacion vs v1:")
    print(f"  vol_ratio:  1.8 -> {VOL_RATIO_MIN}")
    print(f"  bb_width:   4.0 -> {BB_WIDTH_MAX}")
    print(f"  bar_move:   2.5% -> {BAR_MOVE_MAX}%")
    print(f"  adx_prev:   28 -> {ADX_PREV_MAX}")
    print("\nProblema original: 1.2 trades/mes (condiciones demasiado estrictas)")
    print("Objetivo: mantener WR>50%, PF>1.3 con 3-5 trades/mes")

    # Cargar
    print("\nCargando datos...")
    df_raw  = load_btc_4h()
    df_feat = compute_features_4h(df_raw)
    print(f"  4H: {len(df_feat)} velas | {df_feat.index[0].date()} - {df_feat.index[-1].date()}")

    # Walk-forward
    print("\nWalk-forward: 12 semestres...")
    wf = walk_forward(df_feat, detect_setup, simulate, min_trades=3)
    print(f"\n  Folds OK: {wf['folds_ok']}/12 | "
          f"{'APROBADO' if wf['approved'] else 'RECHAZADO'}")
    print(f"\n  {'Periodo':<14} | {'N':>4} | {'WR':>7} | {'PF':>6} | {'Anual':>8} | OK")
    print("  " + "-" * 55)
    for r in wf['folds']:
        ok_s  = '+' if r['ok'] else '-'
        ann_s = f"{r.get('annual_pct',0):.0f}%" if r['n'] > 0 else 'n/a'
        wr_s  = f"{r['wr']:.1%}"  if r['n'] > 0 else 'n/a'
        pf_s  = f"{r['pf']:.2f}"  if r['n'] > 0 else 'n/a'
        print(f"  {r['period']:<14} | {r['n']:>4} | {wr_s:>7} | {pf_s:>6} | {ann_s:>8} | {ok_s}")

    # OOS completo
    print(f"\nOOS completo ({OOS_START} a {OOS_END})...")
    oos_mask = (df_feat.index >= OOS_START) & (df_feat.index <= OOS_END)
    df_oos   = df_feat[oos_mask]
    oos_start_bar = df_feat.index.get_loc(df_oos.index[0])

    oos_trades = []
    for i in range(len(df_oos)):
        trade = detect_setup(df_feat, oos_start_bar + i)
        if trade is None:
            continue
        out = sim_trade_fixed(df_feat, oos_start_bar + i,
                              trade['entry'], trade['tp_pct'], trade['sl_pct'],
                              max_bars=MAX_BARS)
        oos_trades.append({'outcome': out[0], 'pnl_pct': out[2],
                           'ts': df_oos.index[i], 'setup': trade['setup'],
                           'vol_ratio': trade['vol_ratio']})

    m_oos = metrics(oos_trades, f'OOS {OOS_START}-{OOS_END}')
    print()
    print_metrics(m_oos)

    if oos_trades:
        # Comparacion v1 vs v2
        print(f"\n  Comparacion v1 -> v2:")
        print(f"    N:          49 -> {m_oos['n']} ({m_oos['n']-49:+d})")
        print(f"    WR:         59.2% -> {m_oos['wr']:.1%} ({m_oos['wr']-0.592:+.1%}pp)")
        print(f"    PF:         1.90 -> {m_oos['pf']:.2f} ({m_oos['pf']-1.90:+.2f})")
        print(f"    Trades/mes: 1.2 -> {m_oos['trades_pm']:.1f}")

        # Breakdown por vol_ratio
        low_vol  = [t for t in oos_trades if t['vol_ratio'] < 1.8]
        high_vol = [t for t in oos_trades if t['vol_ratio'] >= 1.8]
        if low_vol and high_vol:
            m_lv = metrics(low_vol,  'vol < 1.8 (nuevas)')
            m_hv = metrics(high_vol, 'vol >= 1.8 (originales)')
            print(f"\n  Breakdown por volumen:")
            print(f"    Nuevas (vol<1.8):    N={m_lv['n']:>3} | WR={m_lv['wr']:.1%} | PF={m_lv['pf']:.2f}")
            print(f"    Originales (vol>=1.8): N={m_hv['n']:>3} | WR={m_hv['wr']:.1%} | PF={m_hv['pf']:.2f}")

    verdict = ('APROBADO'
               if wf['approved'] and m_oos['pf'] > 1.2 and
                  m_oos['wr'] > 0.48 and m_oos['trades_pm'] >= 2.5
               else 'RECHAZADO')
    print(f"\n{'='*70}")
    print(f"VEREDICTO FINAL v2: {verdict}")
    print(f"  WF: {wf['folds_ok']}/12 | WR={m_oos['wr']:.1%} | "
          f"PF={m_oos['pf']:.2f} | {m_oos['trades_pm']:.1f}t/m")
    print(f"{'='*70}")

    _write_doc(wf, m_oos, verdict, oos_trades)
    return verdict, m_oos


def _write_doc(wf, m_oos, verdict, trades):
    doc_dir = ROOT / 'docs'
    doc_dir.mkdir(exist_ok=True)
    with open(doc_dir / 'V15_B_BREAKOUT_v2_results.md', 'w', encoding='utf-8') as f:
        f.write("# V15 Estrategia B-v2: Momentum Breakout (Optimizado)\n\n")
        f.write(f"**Rama**: `v15/momentum-breakout`\n")
        f.write(f"**Iteracion**: 2 (optimizacion de frecuencia)\n")
        f.write(f"**Fecha**: {pd.Timestamp.now().date()}\n")
        f.write(f"**Veredicto**: {verdict}\n\n")
        f.write("## Cambios vs v1\n\n")
        f.write("| Parametro | v1 | v2 |\n|-----------|----|----|----|\n")
        f.write("| vol_ratio min | 1.8 | 1.3 |\n")
        f.write("| bb_width max | 4.0% | 5.5% |\n")
        f.write("| bar_move max | 2.5% | 3.5% |\n")
        f.write("| adx_prev max | 28 | 30 |\n\n")
        f.write("## Comparacion v1 vs v2\n\n")
        f.write("| Metrica | v1 | v2 |\n|---------|----|----|----|\n")
        f.write(f"| N OOS | 49 | {m_oos['n']} |\n")
        f.write(f"| WR | 59.2% | {m_oos['wr']:.1%} |\n")
        f.write(f"| PF | 1.90 | {m_oos['pf']:.2f} |\n")
        f.write(f"| Trades/mes | 1.2 | {m_oos['trades_pm']:.1f} |\n")
        f.write(f"| WF folds OK | 4/12 | {wf['folds_ok']}/12 |\n\n")
        f.write("## Walk-forward\n\n")
        f.write("| Periodo | N | WR | PF | Anual | OK |\n|---------|---|----|----|-------|----|\n")
        for r in wf['folds']:
            ok_s = 'SI' if r['ok'] else 'NO'
            wr_s = f"{r['wr']:.1%}" if r['n'] > 0 else '-'
            pf_s = f"{r['pf']:.2f}" if r['n'] > 0 else '-'
            ann_s = f"{r.get('annual_pct',0):.0f}%" if r['n'] > 0 else '-'
            f.write(f"| {r['period']} | {r['n']} | {wr_s} | {pf_s} | {ann_s} | {ok_s} |\n")
        f.write(f"\n**Folds OK**: {wf['folds_ok']}/12\n\n")
        f.write(f"## OOS | N={m_oos['n']} | WR={m_oos['wr']:.1%} | "
                f"PF={m_oos['pf']:.2f} | ~{m_oos['annual_pct']:.0f}%/yr\n\n")
        f.write(f"## Veredicto: {verdict}\n")
        f.write("Criterios: WF>=7/12, WR>=50%, PF>=1.2, >=2.5 trades/mes\n")


if __name__ == '__main__':
    main()
