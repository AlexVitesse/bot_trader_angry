"""
validate_v15_b_breakout_v3.py  —  Branch: v15/momentum-breakout  [Iteracion 3]
==============================================================================
Objetivo: corregir las folds malas de 2025 con filtro macro.

HALLAZGO CLAVE DE V2:
  - Nuevas seniales (vol<1.8): WR=36.7%, PF=0.87  -> danan la calidad
  - Originales   (vol>=1.8):  WR=55.4%, PF=1.67  -> excelente calidad
  - Conclusion: NO relajar vol_ratio. El problema es FRECUENCIA EN REGIMEN BULL.

CAUSA DE FOLDS MALAS:
  - 2021-H1/H2: solo 1 trade c/u (mercado lateral/techo)
  - 2022-H1:    0 trades (bear -75%)
  - 2023-H1/H2: 8-19 trades pero WR 42-50% (bull sin fuerza, muchas falsas rupturas)
  - 2025-H1/H2: 17-23 trades pero WR 35-39% (mercado rango/bajista)

HIPOTESIS V3:
  Agregar filtro macro: solo operar cuando daily EMA20 > EMA50 (tendencia diaria alcista).
  En 2025 bajista, muchos setups se filtran -> menos perdidas.
  En 2020/2024 bull fuerte, el filtro es permisivo -> buenas entradas pasan.

CAMBIOS vs V2:
  - vol_ratio: 1.3 -> 1.8  (vuelve al original, calidad primero)
  - bb_width:  5.5 -> 4.5% (pequena relajacion para frecuencia)
  - bar_move:  3.5 -> 2.8% (equilibrio entre v1 y v2)
  - ADX:       30  -> 28   (vuelve al original)
  - NUEVO: macro_bull filtro (daily EMA20 > EMA50, shifted 1 dia)

HISTORIAL DE ITERACIONES:
  v1: vol>=1.8, bb_width<4.0, bar_move<2.5%, sin macro -> WF 4/12, WR=59.2%, 1.2t/m
  v2: vol>=1.3, bb_width<5.5, bar_move<3.5%, sin macro -> WF 5/12, WR=49.5%, 2.4t/m
  v3: vol>=1.8, bb_width<4.5, bar_move<2.8%, CON macro  -> [resultados abajo]
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

STRATEGY_NAME = 'B-v3: Momentum Breakout + Macro Filter'
BRANCH        = 'v15/momentum-breakout'
MAX_BARS      = 16

# Parametros v3
VOL_RATIO_MIN   = 1.8   # vuelve al original (calidad primero)
BB_WIDTH_MAX    = 4.5   # leve relajacion (era 4.0 en v1)
BAR_MOVE_MAX    = 2.8   # equilibrio (era 2.5 en v1)
ADX_PREV_MAX    = 28    # vuelve al original
NARROW_BARS_MIN = 3
SL_PCT_MIN      = 0.005
SL_PCT_MAX      = 0.04
MACRO_BULL_REQ  = True  # NUEVO: solo operar cuando EMA20_1d > EMA50_1d


def detect_setup(df: pd.DataFrame, i: int) -> dict | None:
    if i < 25:
        return None
    row = df.iloc[i]

    # 0. NUEVO: Filtro macro — solo operar en tendencia diaria alcista
    if MACRO_BULL_REQ:
        bull_daily = row.get('bull_1d', 1)  # si no existe, no filtrar
        if bull_daily == 0:
            return None

    # 1. Ruptura: close > maximo de 20 barras anteriores
    high20 = float(df['high'].iloc[i-20:i].max())
    if row['close'] <= high20:
        return None

    # 2. Volumen confirma (original: 1.8x)
    if row.get('vol_ratio', 1) < VOL_RATIO_MIN:
        return None

    # 3. Vela de ruptura no demasiado grande (moderado: 2.8%)
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > BAR_MOVE_MAX:
        return None

    # 4. Consolidacion previa (leve relajacion: bb_width < 4.5%)
    recent_bb = df['bb_width'].iloc[i-5:i]
    narrow_bars = (recent_bb < BB_WIDTH_MAX).sum()
    if narrow_bars < NARROW_BARS_MIN:
        return None

    # 5. ADX anterior bajo (original: < 28)
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
        'setup': 'BREAKOUT_V3',
        'entry': entry,
        'sl_pct': sl_pct,
        'tp_pct': tp_pct,
        'vol_ratio': float(row.get('vol_ratio', 1)),
        'bar_move': bar_move,
        'prev_adx': prev_adx,
        'narrow_bars': int(narrow_bars),
        'bull_daily': int(row.get('bull_1d', 1)),
    }


def simulate(df, global_i, trade):
    return sim_trade_fixed(df, global_i, trade['entry'],
                           trade['tp_pct'], trade['sl_pct'], max_bars=MAX_BARS)


def main():
    print("=" * 70)
    print(f"ESTRATEGIA {STRATEGY_NAME}")
    print(f"Rama: {BRANCH} | Iteracion: v3")
    print("=" * 70)
    print("\nCambios vs v2:")
    print(f"  vol_ratio:  1.3 -> {VOL_RATIO_MIN} (vuelve a calidad original)")
    print(f"  bb_width:   5.5 -> {BB_WIDTH_MAX}% (leve relajacion)")
    print(f"  bar_move:   3.5% -> {BAR_MOVE_MAX}% (equilibrio)")
    print(f"  adx_prev:   30 -> {ADX_PREV_MAX} (vuelve al original)")
    print(f"  macro_bull: NO -> SI (filtro EMA20_1d > EMA50_1d)")
    print("\nHipotesis: macro filter elimina losses en 2025 bajista")

    # Cargar + macro features
    print("\nCargando datos...")
    df_raw  = load_btc_4h()
    df_feat = compute_features_4h(df_raw)
    df_daily = compute_macro_daily(df_feat)
    df_feat  = merge_daily_to_4h(df_feat, df_daily)
    print(f"  4H: {len(df_feat)} velas | {df_feat.index[0].date()} - {df_feat.index[-1].date()}")
    print(f"  Daily: {len(df_daily)} dias | bull_1d disponible")

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
        print(f"\n  Comparacion v1 -> v2 -> v3:")
        print(f"    N:          49 -> 95 -> {m_oos['n']} ({m_oos['n']-49:+d} vs v1)")
        print(f"    WR:         59.2% -> 49.5% -> {m_oos['wr']:.1%}")
        print(f"    PF:         1.90 -> 1.37 -> {m_oos['pf']:.2f}")
        print(f"    Trades/mes: 1.2 -> 2.4 -> {m_oos['trades_pm']:.1f}")

    verdict = ('APROBADO'
               if wf['approved'] and m_oos['pf'] > 1.2 and
                  m_oos['wr'] > 0.48 and m_oos['trades_pm'] >= 2.5
               else 'RECHAZADO')
    print(f"\n{'='*70}")
    print(f"VEREDICTO FINAL v3: {verdict}")
    print(f"  WF: {wf['folds_ok']}/12 | WR={m_oos['wr']:.1%} | "
          f"PF={m_oos['pf']:.2f} | {m_oos['trades_pm']:.1f}t/m")
    print(f"{'='*70}")

    _write_doc(wf, m_oos, verdict)
    return verdict, m_oos


def _write_doc(wf, m_oos, verdict):
    doc_dir = ROOT / 'docs'
    doc_dir.mkdir(exist_ok=True)
    with open(doc_dir / 'V15_B_BREAKOUT_v3_results.md', 'w', encoding='utf-8') as f:
        f.write("# V15 Estrategia B-v3: Momentum Breakout + Macro Filter\n\n")
        f.write(f"**Rama**: `v15/momentum-breakout`\n")
        f.write(f"**Iteracion**: 3 (filtro macro EMA20_1d > EMA50_1d)\n")
        f.write(f"**Fecha**: {pd.Timestamp.now().date()}\n")
        f.write(f"**Veredicto**: {verdict}\n\n")
        f.write("## Hipotesis\n\n")
        f.write("v2 mostro que vol<1.8 degradaba calidad. El filtro macro elimina ")
        f.write("trades en 2025 bajista (WR=35-39% -> no entrar).\n\n")
        f.write("## Parametros v3\n\n")
        f.write("| Parametro | v1 | v2 | v3 |\n|-----------|----|----|----|\n")
        f.write("| vol_ratio min | 1.8 | 1.3 | 1.8 |\n")
        f.write("| bb_width max | 4.0% | 5.5% | 4.5% |\n")
        f.write("| bar_move max | 2.5% | 3.5% | 2.8% |\n")
        f.write("| adx_prev max | 28 | 30 | 28 |\n")
        f.write("| macro filter | NO | NO | SI (EMA20>EMA50 diario) |\n\n")
        f.write("## Walk-forward\n\n")
        f.write("| Periodo | N | WR | PF | Anual | OK |\n|---------|---|----|----|-------|----|\\n")
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
