"""
validate_v15_b_breakout.py  —  Branch: v15/momentum-breakout
============================================================
Estrategia B: Momentum Breakout (RUPTURA DE CONSOLIDACION)

Lo nuevo vs V14:
- V14 tenia BREAKOUT_UP pero en regimen VOLATILE con condiciones casi imposibles
  (bb_pct>1.0 AND vol_ratio>1.5 AND consec_up>=3) -> casi nunca disparaba
- Esta estrategia detecta CONSOLIDACIONES REALES (BB estrecho por varios bars)
  y entra en la ruptura con confirmacion de volumen
- Es la tecnica mas usada por trend traders institucionales

Condiciones de entrada:
  1. Consolidacion: bb_width < 3.5% durante al menos 3 de las 5 barras anteriores
  2. ADX anterior < 25 (mercado en rango, no trending)
  3. Ruptura: close > high20 (maximos de 20 barras)
  4. Volumen: vol_ratio > 1.8 (volumen real detras de la ruptura)
  5. No tarde: precio no subio mas del 2% en la barra de ruptura (evitar chasers)

TP/SL:
  SL = minimo de las ultimas 5 barras - 0.3%
  TP = entry + (entry - SL) * 1.5 (1.5:1 RR, conservador por fallas)
  Max SL: 4%
"""

import sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pandas_ta as pta
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from v15_framework import (
    load_btc_4h, compute_features_4h, compute_macro_daily, merge_daily_to_4h,
    sim_trade_fixed, metrics, print_metrics, walk_forward,
    WF_FOLDS, OOS_START, OOS_END, COMMISSION
)

STRATEGY_NAME = 'B: Momentum Breakout'
BRANCH        = 'v15/momentum-breakout'
MAX_BARS      = 16   # ~4 dias


# ============================================================
# DETECCION DE SETUP
# ============================================================
def detect_setup(df: pd.DataFrame, i: int) -> dict | None:
    if i < 25:
        return None

    row = df.iloc[i]

    # 1. Ruptura: close > maximo de las 20 barras anteriores
    high20 = float(df['high'].iloc[i-20:i].max())
    if row['close'] <= high20:
        return None

    # 2. Volumen confirma la ruptura
    if row.get('vol_ratio', 1) < 1.8:
        return None

    # 3. La vela de ruptura no es demasiado grande (evitar entrar tarde)
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > 2.5:
        return None

    # 4. Consolidacion previa: bb_width estrecho en al menos 3 de 5 barras anteriores
    recent_bb_widths = df['bb_width'].iloc[i-5:i]
    narrow_bars = (recent_bb_widths < 4.0).sum()
    if narrow_bars < 3:
        return None

    # 5. ADX anterior bajo (mercado en rango antes de la ruptura)
    prev_adx = df['adx14'].iloc[i-3:i].mean()
    if prev_adx > 28:
        return None

    # 6. Calcular TP/SL
    entry  = float(row['close'])
    sl_raw = float(df['low'].iloc[i-5:i].min()) * 0.997
    sl_pct = (entry - sl_raw) / entry

    if sl_pct < 0.005 or sl_pct > 0.04:
        return None

    tp_pct = sl_pct * 1.5  # 1.5:1 RR

    return {
        'setup': 'BREAKOUT',
        'entry': entry,
        'sl_pct': sl_pct,
        'tp_pct': tp_pct,
        'vol_ratio': float(row.get('vol_ratio', 1)),
        'bar_move': bar_move,
        'prev_adx': prev_adx,
        'narrow_bars': int(narrow_bars),
    }


def simulate(df: pd.DataFrame, global_i: int, trade: dict) -> tuple:
    return sim_trade_fixed(
        df, global_i, trade['entry'],
        trade['tp_pct'], trade['sl_pct'],
        max_bars=MAX_BARS
    )


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print(f"ESTRATEGIA {STRATEGY_NAME}")
    print(f"Rama: {BRANCH}")
    print(f"OOS: {OOS_START} a {OOS_END}")
    print("=" * 70)
    print("\nNovedoso vs V14:")
    print("  - V14 BREAKOUT_UP era casi imposible (bb_pct>1.0 en VOLATILE)")
    print("  - Esta detecta CONSOLIDACION REAL (BB estrecho 3/5 barras)")
    print("  - Ruptura confirmada con volumen 1.8x + cierre sobre maximo 20 barras")
    print("  - SL bajo el minimo de la consolidacion (logica real de traders)")

    # ----- Cargar datos -----
    print("\nCargando datos...")
    df_raw  = load_btc_4h()
    df_feat = compute_features_4h(df_raw)
    print(f"  4H: {len(df_feat)} velas | {df_feat.index[0].date()} - {df_feat.index[-1].date()}")

    # ----- Walk-forward -----
    print("\nWalk-forward: 12 semestres (2020-H1 a 2025-H2)...")
    wf = walk_forward(df_feat, detect_setup, simulate, min_trades=2)
    print(f"\n  Folds OK: {wf['folds_ok']}/12 | "
          f"{'APROBADO' if wf['approved'] else 'RECHAZADO'}")
    print(f"\n  {'Periodo':<14} | {'N':>4} | {'WR':>7} | {'PF':>6} | {'Anual':>8} | OK")
    print("  " + "-" * 55)
    for r in wf['folds']:
        ok_s  = '+' if r['ok'] else '-'
        ann_s = f"{r.get('annual_pct', 0):.0f}%" if r['n'] > 0 else 'n/a'
        wr_s  = f"{r['wr']:.1%}" if r['n'] > 0 else 'n/a'
        pf_s  = f"{r['pf']:.2f}" if r['n'] > 0 else 'n/a'
        print(f"  {r['period']:<14} | {r['n']:>4} | {wr_s:>7} | {pf_s:>6} | {ann_s:>8} | {ok_s}")

    # ----- OOS Completo -----
    print(f"\nOOS completo ({OOS_START} a {OOS_END})...")
    oos_mask = (df_feat.index >= OOS_START) & (df_feat.index <= OOS_END)
    df_oos   = df_feat[oos_mask]
    oos_start_bar = df_feat.index.get_loc(df_oos.index[0])

    oos_trades = []
    for i in range(len(df_oos)):
        # Usamos df_feat (completo) para lookback, pero solo en periodo OOS
        # Offset: el bar i en df_oos corresponde a oos_start_bar+i en df_feat
        trade = detect_setup(df_feat, oos_start_bar + i)
        if trade is None:
            continue
        out = sim_trade_fixed(df_feat, oos_start_bar + i,
                              trade['entry'], trade['tp_pct'], trade['sl_pct'],
                              max_bars=MAX_BARS)
        oos_trades.append({
            'outcome': out[0], 'pnl_pct': out[2],
            'ts': df_oos.index[i], 'setup': trade['setup'],
        })

    m_oos = metrics(oos_trades, f'OOS {OOS_START}-{OOS_END}')
    print()
    print_metrics(m_oos)

    verdict = ('APROBADO' if wf['approved'] and m_oos['pf'] > 1.2 and m_oos['wr'] > 0.40
               else 'RECHAZADO')
    print(f"\n{'='*70}")
    print(f"VEREDICTO FINAL: {verdict}")
    print(f"  Walk-forward: {wf['folds_ok']}/12 folds | OOS WR={m_oos['wr']:.1%} PF={m_oos['pf']:.2f}")
    print(f"{'='*70}")

    _write_doc(wf, m_oos, verdict)
    return verdict, m_oos


def _write_doc(wf, m_oos, verdict):
    doc_dir = ROOT / 'docs'
    doc_dir.mkdir(exist_ok=True)
    with open(doc_dir / 'V15_B_BREAKOUT_results.md', 'w', encoding='utf-8') as f:
        f.write(f"# V15 Estrategia B: Momentum Breakout\n\n")
        f.write(f"**Rama**: `v15/momentum-breakout`\n")
        f.write(f"**Fecha**: {pd.Timestamp.now().date()}\n")
        f.write(f"**Veredicto**: {verdict}\n\n")
        f.write("## Lo nuevo vs V14\n")
        f.write("- V14 BREAKOUT_UP requeria bb_pct>1.0 en VOLATILE -> casi nunca disparaba\n")
        f.write("- Esta detecta consolidacion real (BB estrecho 3/5 bars) + ruptura con volumen\n")
        f.write("- SL bajo minimo de consolidacion (logica real)\n\n")
        f.write(f"## Walk-forward\n\n| Periodo | N | WR | PF | Anual | OK |\n|---------|---|----|----|-------|----|\n")
        for r in wf['folds']:
            ok_s = 'SI' if r['ok'] else 'NO'
            wr_s = f"{r['wr']:.1%}" if r['n'] > 0 else '-'
            pf_s = f"{r['pf']:.2f}" if r['n'] > 0 else '-'
            ann_s = f"{r.get('annual_pct',0):.0f}%" if r['n'] > 0 else '-'
            f.write(f"| {r['period']} | {r['n']} | {wr_s} | {pf_s} | {ann_s} | {ok_s} |\n")
        f.write(f"\n**Folds OK**: {wf['folds_ok']}/12\n\n")
        f.write(f"## OOS | N={m_oos['n']} | WR={m_oos['wr']:.1%} | PF={m_oos['pf']:.2f} | ~{m_oos['annual_pct']:.0f}%/yr\n\n")
        f.write(f"## Veredicto: {verdict}\n")


if __name__ == '__main__':
    main()
