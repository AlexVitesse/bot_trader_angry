"""
validate_v15_b_short.py  —  Branch: v15/momentum-breakout
==============================================================================
Validacion del equivalente SHORT de la Estrategia B (Momentum Breakout).

CONTEXTO:
  La Estrategia B LONG mostro WR=55-61%, PF=1.67-2.87 — excelente calidad.
  Pero LONG-only falla en bear markets (2022: -75%, 2025: bajista).
  Si SHORT tiene calidad similar en mercados bajistas, el comite V15 puede
  combinar: LONG en BULL + SHORT en BEAR → cobertura de todos los regimenes.

SETUP SHORT (inverso de LONG):
  1. Ruptura bajista: close < minimo de 20 barras anteriores
  2. Volumen confirma: vol_ratio >= 1.8 (igual que LONG)
  3. Vela de ruptura no demasiado grande: < 2.8%
  4. Consolidacion previa: BB estrecho (3/5 barras < 4.5%)
  5. ADX anterior bajo: < 28 (no ya en tendencia fuerte)
  6. MACRO BEAR: EMA20_1d < EMA50_1d (shifted 1 dia)

LOGICA TP/SL SHORT:
  - Entry: close de la vela de ruptura bajista
  - SL: max(high[-5:0]) * 1.003  (sobre el maximo de consolidacion)
  - TP: entry * (1 - tp_pct) donde tp_pct = sl_pct * 1.5  (RR 1.5:1)

CRITERIO DE APROBACION (mismo que LONG):
  WF: >= 7/12 folds OK (WR>40%, PF>1.2, n>=3)
  OOS: WR>=50%, PF>=1.3, >=2.5 trades/mes
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
    metrics, print_metrics,
    WF_FOLDS, OOS_START, OOS_END, COMMISSION
)

STRATEGY_NAME = 'B-SHORT: Momentum Breakdown + Macro BEAR'
BRANCH        = 'v15/momentum-breakout'
MAX_BARS      = 16

# Parametros (mismos que LONG v3 — simetricos)
VOL_RATIO_MIN   = 1.8
BB_WIDTH_MAX    = 4.5
BAR_MOVE_MAX    = 2.8
ADX_PREV_MAX    = 28
NARROW_BARS_MIN = 3
SL_PCT_MIN      = 0.005
SL_PCT_MAX      = 0.04


# ============================================================
# SIMULACION SHORT
# ============================================================
def sim_short_fixed(df: pd.DataFrame, entry_bar: int, entry_price: float,
                    tp_pct: float, sl_pct: float,
                    max_bars: int = MAX_BARS) -> tuple:
    """
    SHORT trade con TP y SL fijos.
    TP: precio baja tp_pct  (ganas)
    SL: precio sube sl_pct  (pierdes)
    Retorna: (outcome, exit_price, pnl_pct, bars)
    """
    tp = entry_price * (1 - tp_pct)
    sl = entry_price * (1 + sl_pct)
    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df):
            ep = float(df['close'].iloc[-1])
            pnl = (entry_price - ep) / entry_price - 2 * COMMISSION
            return ('TP' if ep < entry_price else 'SL'), ep, pnl, i
        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        # SL tocado primero (precio sube)
        if hi >= sl:
            pnl = -sl_pct - 2 * COMMISSION
            # Si en la misma vela tambien toca TP, usar precio de cierre como desempate
            if lo <= tp and float(df['close'].iloc[b]) < (sl + tp) / 2:
                pnl = tp_pct - 2 * COMMISSION
                return 'TP', tp, pnl, i
            return 'SL', sl, pnl, i
        # TP tocado (precio baja)
        if lo <= tp:
            pnl = tp_pct - 2 * COMMISSION
            return 'TP', tp, pnl, i
    ep  = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (entry_price - ep) / entry_price - 2 * COMMISSION
    return ('TP' if ep < entry_price else 'SL'), ep, pnl, max_bars


# ============================================================
# DETECTOR DE SETUP SHORT
# ============================================================
def detect_setup_short(df: pd.DataFrame, i: int) -> dict | None:
    if i < 25:
        return None
    row = df.iloc[i]

    # 0. Filtro macro — solo SHORT cuando tendencia diaria es BAJISTA
    bull_daily = row.get('bull_1d', 1)
    if bull_daily == 1:   # EMA20 > EMA50 → mercado BULL → no ir SHORT
        return None

    # 1. Ruptura bajista: close < minimo de 20 barras anteriores
    low20 = float(df['low'].iloc[i-20:i].min())
    if row['close'] >= low20:
        return None

    # 2. Volumen confirma
    if row.get('vol_ratio', 1) < VOL_RATIO_MIN:
        return None

    # 3. Vela de ruptura no demasiado grande
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > BAR_MOVE_MAX:
        return None

    # 4. Consolidacion previa (BB estrecho)
    recent_bb = df['bb_width'].iloc[i-5:i]
    narrow_bars = (recent_bb < BB_WIDTH_MAX).sum()
    if narrow_bars < NARROW_BARS_MIN:
        return None

    # 5. ADX anterior bajo (no ya en tendencia fuerte)
    prev_adx = df['adx14'].iloc[i-3:i].mean()
    if prev_adx > ADX_PREV_MAX:
        return None

    # 6. SL sobre el maximo de la consolidacion
    entry  = float(row['close'])
    sl_raw = float(df['high'].iloc[i-5:i].max()) * 1.003
    sl_pct = (sl_raw - entry) / entry

    if sl_pct < SL_PCT_MIN or sl_pct > SL_PCT_MAX:
        return None

    tp_pct = sl_pct * 1.5

    return {
        'setup': 'BREAKDOWN_SHORT',
        'entry': entry,
        'sl_pct': sl_pct,
        'tp_pct': tp_pct,
        'vol_ratio': float(row.get('vol_ratio', 1)),
        'bar_move': bar_move,
        'prev_adx': prev_adx,
        'narrow_bars': int(narrow_bars),
        'bull_daily': int(bull_daily),
    }


# ============================================================
# WALK-FORWARD (adaptado para SHORT)
# ============================================================
def walk_forward_short(df: pd.DataFrame, min_trades: int = 3) -> dict:
    results = []
    approved_min = 7

    for start_s, end_s in WF_FOLDS:
        mask = (df.index >= start_s) & (df.index <= end_s)
        df_fold = df[mask]
        if len(df_fold) == 0:
            results.append({'period': f"{start_s[:7]}/{end_s[5:7]}",
                           'n': 0, 'wr': 0, 'pf': 0, 'ok': False,
                           'annual_pct': 0})
            continue

        start_bar = df.index.get_loc(df_fold.index[0])
        trades = []
        for i in range(len(df_fold)):
            trade = detect_setup_short(df, start_bar + i)
            if trade is None:
                continue
            out = sim_short_fixed(df, start_bar + i,
                                  trade['entry'], trade['tp_pct'], trade['sl_pct'])
            trades.append({'outcome': out[0], 'pnl_pct': out[2],
                           'ts': df_fold.index[i]})

        period_label = f"{start_s[:7]}/{end_s[5:7]}"
        m = metrics(trades, period_label)
        days = (pd.Timestamp(end_s) - pd.Timestamp(start_s)).days
        annual = m['avg_pnl'] * m['n'] / days * 365 * 100 if days > 0 else 0

        ok = (m['n'] >= min_trades and m['wr'] > 0.40 and m['pf'] > 1.2)
        results.append({'period': period_label, 'n': m['n'],
                        'wr': m['wr'], 'pf': m['pf'],
                        'ok': ok, 'annual_pct': annual})

    folds_ok = sum(1 for r in results if r['ok'])
    return {
        'folds': results,
        'folds_ok': folds_ok,
        'approved': folds_ok >= approved_min
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print(f"ESTRATEGIA {STRATEGY_NAME}")
    print(f"Rama: {BRANCH}")
    print("=" * 70)
    print("\nEquivalente SHORT de la Estrategia B LONG:")
    print(f"  close < low20     (ruptura bajista vs close > high20)")
    print(f"  SL = max(high[-5:]) * 1.003  (sobre consolidacion)")
    print(f"  TP = SL * 1.5     (RR 1.5:1, igual que LONG)")
    print(f"  Macro: EMA20_1d < EMA50_1d  (solo en tendencia bajista)")
    print("\nFolds clave esperados: 2022-H1, 2022-H2, 2025 (bear markets)")

    # Cargar datos
    print("\nCargando datos...")
    df_raw   = load_btc_4h()
    df_feat  = compute_features_4h(df_raw)
    df_daily = compute_macro_daily(df_feat)
    df_feat  = merge_daily_to_4h(df_feat, df_daily)
    print(f"  4H: {len(df_feat)} velas | {df_feat.index[0].date()} - {df_feat.index[-1].date()}")

    # Cuantas barras tienen macro BEAR
    bear_pct = (df_feat.get('bull_1d', pd.Series([1]*len(df_feat))) == 0).mean()
    print(f"  Barras con macro BEAR: {bear_pct:.1%}")

    # Walk-forward
    print("\nWalk-forward: 12 semestres...")
    wf = walk_forward_short(df_feat, min_trades=3)
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
        trade = detect_setup_short(df_feat, oos_start_bar + i)
        if trade is None:
            continue
        out = sim_short_fixed(df_feat, oos_start_bar + i,
                              trade['entry'], trade['tp_pct'], trade['sl_pct'])
        oos_trades.append({'outcome': out[0], 'pnl_pct': out[2],
                           'ts': df_oos.index[i]})

    m_oos = metrics(oos_trades, f'OOS {OOS_START}-{OOS_END}')
    print()
    print_metrics(m_oos)

    # Comparacion LONG vs SHORT
    if oos_trades:
        print(f"\n  Comparacion B-LONG v3 vs B-SHORT:")
        print(f"    {'':20s}  {'LONG v3':>10}  {'SHORT':>10}")
        print(f"    {'N OOS':20s}  {'31':>10}  {m_oos['n']:>10}")
        print(f"    {'WR':20s}  {'61.3%':>10}  {m_oos['wr']:.1%}")
        print(f"    {'PF':20s}  {'2.87':>10}  {m_oos['pf']:.2f}")
        print(f"    {'Trades/mes':20s}  {'0.9':>10}  {m_oos['trades_pm']:.1f}")
        print(f"    {'WF folds OK':20s}  {'5/12':>10}  {wf['folds_ok']}/12")

    # Proyeccion combinada LONG+SHORT
    print(f"\n  Proyeccion combinada (LONG en BULL + SHORT en BEAR):")
    long_tpm = 0.9   # B-LONG v3
    short_tpm = m_oos['trades_pm']
    print(f"    LONG t/m:   {long_tpm:.1f}")
    print(f"    SHORT t/m:  {short_tpm:.1f}")
    print(f"    Total t/m:  {long_tpm + short_tpm:.1f}")

    verdict = ('APROBADO'
               if wf['approved'] and m_oos['pf'] > 1.2 and
                  m_oos['wr'] > 0.48 and m_oos['trades_pm'] >= 2.5
               else 'RECHAZADO')
    print(f"\n{'='*70}")
    print(f"VEREDICTO B-SHORT: {verdict}")
    print(f"  WF: {wf['folds_ok']}/12 | WR={m_oos['wr']:.1%} | "
          f"PF={m_oos['pf']:.2f} | {m_oos['trades_pm']:.1f}t/m")
    print(f"{'='*70}")

    _write_doc(wf, m_oos, verdict)
    return verdict, m_oos


def _write_doc(wf, m_oos, verdict):
    doc_dir = ROOT / 'docs'
    doc_dir.mkdir(exist_ok=True)
    with open(doc_dir / 'V15_B_SHORT_results.md', 'w', encoding='utf-8') as f:
        f.write("# V15 Estrategia B-SHORT: Momentum Breakdown\n\n")
        f.write(f"**Rama**: `v15/momentum-breakout`\n")
        f.write(f"**Direccion**: SHORT (equivalente inverso de B-LONG)\n")
        f.write(f"**Fecha**: {pd.Timestamp.now().date()}\n")
        f.write(f"**Veredicto**: {verdict}\n\n")
        f.write("## Setup\n\n")
        f.write("| Condicion | LONG | SHORT |\n|-----------|------|-------|\n")
        f.write("| Ruptura | close > high20 | close < low20 |\n")
        f.write("| Macro | EMA20>EMA50 (BULL) | EMA20<EMA50 (BEAR) |\n")
        f.write("| Vol | >= 1.8x | >= 1.8x |\n")
        f.write("| BB | <4.5% (estrecho) | <4.5% (estrecho) |\n")
        f.write("| SL | bajo min consolidacion | sobre max consolidacion |\n")
        f.write("| RR | 1.5:1 | 1.5:1 |\n\n")
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
