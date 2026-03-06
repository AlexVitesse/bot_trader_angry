"""
validate_v15_e_short_bear.py  —  Branch: v15/momentum-breakout
==============================================================================
Estrategia E: SHORT en Bear Market ("Failed Bounce at Resistance")

POR QUE B-SHORT FALLO:
  El setup "BB estrecho + ruptura bajo low20" es un patron de BULL market
  (acumulacion → breakout). En bear markets:
    - ADX alto (tendencia fuerte, no ADX<28)
    - Sin consolidacion (venta sostenida, no BB estrecho)
    - Sin vol spike al bajar (los bears son vendedores silenciosos)
  Resultado: 0 trades en 2022-H1 (el mejor bear market del historial).

PATRON CORRECTO PARA SHORT EN BEAR:
  "Rebote fallido a resistencia"
  En un mercado bajista, el precio:
    1. Cae fuertemente
    2. Rebota (RSI sube de oversold a zona media 42-62)
    3. Llega a resistencia (EMA20_4h = media movil de corto plazo)
    4. No puede recuperarla → vela cierra bajo EMA20_4h
    5. Es el momento de SHORT: el rebote se agoto

CONDICIONES DE ENTRADA SHORT:
  1. Macro BEAR: EMA20_1d < EMA50_1d  (tendencia diaria bajista)
  2. 4H bajista: EMA20_4h < EMA50_4h  (tendencia 4H confirmada)
  3. Rebote previo: RSI14 subio a zona 42-62 en las ultimas 6 velas
  4. Rechazo: close < EMA20_4h  (no puede recuperar la EMA de corto plazo)
  5. Vela no demasiado grande: < 2.5% (entrada calmada, no en panico)
  6. Volumen: >= 0.8x promedio (actividad minima, no vela de baja liquidez)

TP/SL:
  - SL: maximo de las ultimas 3 barras * 1.003  (sobre el swing high reciente)
  - TP: SL_pct * 2.0  (RR 2:1, mas generoso que B-LONG para compensar)
  - max_bars: 20 (80h = ~3.3 dias, bear moves son mas rapidos)
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

STRATEGY_NAME = 'E: SHORT Bear Market (Failed Bounce)'
BRANCH        = 'v15/momentum-breakout'
MAX_BARS      = 20
TP_RR         = 2.0   # TP = SL * 2.0 (RR 2:1)

SL_PCT_MIN    = 0.005
SL_PCT_MAX    = 0.04
BAR_MOVE_MAX  = 2.5
VOL_RATIO_MIN = 0.8   # Actividad minima (no vela de baja liquidez)
RSI_MIN       = 42    # RSI subio (rebote desde oversold)
RSI_MAX       = 62    # RSI no esta en overbought extremo


# ============================================================
# SIMULACION SHORT (precio baja = ganancia)
# ============================================================
def sim_short_fixed(df, entry_bar, entry_price, tp_pct, sl_pct, max_bars=MAX_BARS):
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
        if hi >= sl:
            pnl = -sl_pct - 2 * COMMISSION
            if lo <= tp and float(df['close'].iloc[b]) < (sl + tp) / 2:
                pnl = tp_pct - 2 * COMMISSION
                return 'TP', tp, pnl, i
            return 'SL', sl, pnl, i
        if lo <= tp:
            pnl = tp_pct - 2 * COMMISSION
            return 'TP', tp, pnl, i
    ep  = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (entry_price - ep) / entry_price - 2 * COMMISSION
    return ('TP' if ep < entry_price else 'SL'), ep, pnl, max_bars


# ============================================================
# DETECTOR
# ============================================================
def detect_setup(df, i):
    if i < 30:
        return None
    row = df.iloc[i]

    # 1. Macro BEAR diario
    if row.get('bull_1d', 1) == 1:
        return None

    # 2. 4H bajista: EMA20 < EMA50
    ema20_4h = float(df['ema20'].iloc[i])
    ema50_4h = float(df['ema50'].iloc[i])
    if ema20_4h >= ema50_4h:
        return None

    # 3. Rechazo FRESCO: la vela anterior toco o supero EMA20_4h, esta vela volvio a caer
    #    Este es el momento exacto del rechazo, no cualquier momento bajo la EMA20
    prev_high = float(df['high'].iloc[i-1])
    close = float(row['close'])
    if prev_high < ema20_4h * 0.998:   # la anterior ni se acerco a EMA20 → no es rechazo fresco
        return None
    if close >= ema20_4h:              # la actual sigue arriba → no hay rechazo aun
        return None

    # 4. Vela bajista (confirma el rechazo)
    if close >= float(row['open']):
        return None

    # 5. RSI en zona media — no en extremo oversold (rebote potencial)
    rsi_now = float(row['rsi14'])
    if rsi_now < RSI_MIN or rsi_now > RSI_MAX:
        return None

    # 6. Vela no demasiado grande (entrada calmada)
    bar_move = abs(close - float(row['open'])) / float(row['open']) * 100
    if bar_move > BAR_MOVE_MAX:
        return None

    # 7. Volumen minimo
    if row.get('vol_ratio', 1) < VOL_RATIO_MIN:
        return None

    # 8. SL sobre el swing high reciente
    entry  = close
    sl_raw = float(df['high'].iloc[i-3:i+1].max()) * 1.003
    sl_pct = (sl_raw - entry) / entry

    if sl_pct < SL_PCT_MIN or sl_pct > SL_PCT_MAX:
        return None

    tp_pct = sl_pct * TP_RR

    return {
        'setup': 'SHORT_FAILED_BOUNCE',
        'entry': entry,
        'sl_pct': sl_pct,
        'tp_pct': tp_pct,
        'rsi': rsi_now,
        'bar_move': bar_move,
        'ema20_dist': (close - ema20_4h) / ema20_4h * 100,
    }


# ============================================================
# WALK-FORWARD SHORT
# ============================================================
def walk_forward_short(df, min_trades=3):
    results = []
    for start_s, end_s in WF_FOLDS:
        mask = (df.index >= start_s) & (df.index <= end_s)
        df_fold = df[mask]
        if len(df_fold) == 0:
            results.append({'period': f"{start_s[:7]}/{end_s[5:7]}",
                           'n': 0, 'wr': 0, 'pf': 0, 'ok': False, 'annual_pct': 0})
            continue
        start_bar = df.index.get_loc(df_fold.index[0])
        trades = []
        for i in range(len(df_fold)):
            trade = detect_setup(df, start_bar + i)
            if trade is None:
                continue
            out = sim_short_fixed(df, start_bar + i, trade['entry'],
                                  trade['tp_pct'], trade['sl_pct'])
            trades.append({'outcome': out[0], 'pnl_pct': out[2],
                           'ts': df_fold.index[i]})
        period_label = f"{start_s[:7]}/{end_s[5:7]}"
        m = metrics(trades, period_label)
        days = (pd.Timestamp(end_s) - pd.Timestamp(start_s)).days
        annual = m['avg_pnl'] * m['n'] / days * 365 * 100 if days > 0 and m['n'] > 0 else 0
        ok = (m['n'] >= min_trades and m['wr'] > 0.40 and m['pf'] > 1.2)
        results.append({'period': period_label, 'n': m['n'],
                        'wr': m['wr'], 'pf': m['pf'],
                        'ok': ok, 'annual_pct': annual})
    folds_ok = sum(1 for r in results if r['ok'])
    return {'folds': results, 'folds_ok': folds_ok, 'approved': folds_ok >= 7}


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print(f"ESTRATEGIA {STRATEGY_NAME}")
    print(f"Rama: {BRANCH}")
    print("=" * 70)
    print("\nSetup: Rebote fallido a resistencia en mercado bajista")
    print(f"  Macro BEAR: EMA20_1d < EMA50_1d")
    print(f"  4H bajista: EMA20_4h < EMA50_4h")
    print(f"  Rebote previo: RSI subio a {RSI_MIN}-72 en ultimas 6 velas")
    print(f"  Rechazo: close < EMA20_4h + RSI actual {30}-{RSI_MAX}")
    print(f"  TP/SL: RR={TP_RR}:1 | SL sobre swing high 3 barras")

    print("\nCargando datos...")
    df_raw   = load_btc_4h()
    df_feat  = compute_features_4h(df_raw)
    df_daily = compute_macro_daily(df_feat)
    df_feat  = merge_daily_to_4h(df_feat, df_daily)
    bear_pct = (df_feat.get('bull_1d', pd.Series([1]*len(df_feat))) == 0).mean()
    print(f"  4H: {len(df_feat)} velas | {df_feat.index[0].date()} - {df_feat.index[-1].date()}")
    print(f"  Barras con macro BEAR: {bear_pct:.1%}")

    print("\nWalk-forward: 12 semestres...")
    wf = walk_forward_short(df_feat, min_trades=3)
    print(f"\n  Folds OK: {wf['folds_ok']}/12 | "
          f"{'APROBADO' if wf['approved'] else 'RECHAZADO'}")
    print(f"\n  {'Periodo':<14} | {'N':>4} | {'WR':>7} | {'PF':>6} | {'Anual':>8} | OK")
    print("  " + "-" * 55)
    for r in wf['folds']:
        ok_s  = '+' if r['ok'] else '-'
        ann_s = f"{r['annual_pct']:.0f}%" if r['n'] > 0 else 'n/a'
        wr_s  = f"{r['wr']:.1%}"  if r['n'] > 0 else 'n/a'
        pf_s  = f"{r['pf']:.2f}"  if r['n'] > 0 else 'n/a'
        print(f"  {r['period']:<14} | {r['n']:>4} | {wr_s:>7} | {pf_s:>6} | {ann_s:>8} | {ok_s}")

    # OOS
    print(f"\nOOS completo ({OOS_START} a {OOS_END})...")
    oos_mask = (df_feat.index >= OOS_START) & (df_feat.index <= OOS_END)
    df_oos   = df_feat[oos_mask]
    oos_start_bar = df_feat.index.get_loc(df_oos.index[0])

    oos_trades = []
    for i in range(len(df_oos)):
        trade = detect_setup(df_feat, oos_start_bar + i)
        if trade is None:
            continue
        out = sim_short_fixed(df_feat, oos_start_bar + i,
                              trade['entry'], trade['tp_pct'], trade['sl_pct'])
        oos_trades.append({'outcome': out[0], 'pnl_pct': out[2],
                           'ts': df_oos.index[i], 'rsi': trade['rsi']})

    m_oos = metrics(oos_trades, f'OOS {OOS_START}-{OOS_END}')
    print()
    print_metrics(m_oos)

    if oos_trades:
        # Breakdown por RSI al entrar
        low_rsi  = [t for t in oos_trades if t['rsi'] < 48]
        high_rsi = [t for t in oos_trades if t['rsi'] >= 48]
        if low_rsi and high_rsi:
            m_lr = metrics(low_rsi,  'RSI<48 (rechazo temprano)')
            m_hr = metrics(high_rsi, 'RSI>=48 (rechazo en resistencia)')
            print(f"\n  Breakdown por RSI al entrar:")
            print(f"    RSI<48  (rechazo temprano): N={m_lr['n']:>3} | WR={m_lr['wr']:.1%} | PF={m_lr['pf']:.2f}")
            print(f"    RSI>=48 (en resistencia):  N={m_hr['n']:>3} | WR={m_hr['wr']:.1%} | PF={m_hr['pf']:.2f}")

    # Proyeccion combinada con LONG
    long_tpm  = 0.9   # B-LONG v3
    short_tpm = m_oos['trades_pm']
    print(f"\n  Proyeccion combinada LONG+SHORT:")
    print(f"    B-LONG v3: WR=61.3%, PF=2.87, {long_tpm:.1f}t/m (solo BULL)")
    print(f"    E-SHORT:   WR={m_oos['wr']:.1%}, PF={m_oos['pf']:.2f}, {short_tpm:.1f}t/m (solo BEAR)")
    print(f"    Total:     {long_tpm + short_tpm:.1f}t/m")

    verdict = ('APROBADO'
               if wf['approved'] and m_oos['pf'] > 1.2 and
                  m_oos['wr'] > 0.45 and m_oos['trades_pm'] >= 2.0
               else 'RECHAZADO')
    print(f"\n{'='*70}")
    print(f"VEREDICTO E-SHORT: {verdict}")
    print(f"  WF: {wf['folds_ok']}/12 | WR={m_oos['wr']:.1%} | "
          f"PF={m_oos['pf']:.2f} | {m_oos['trades_pm']:.1f}t/m")
    print(f"{'='*70}")

    _write_doc(wf, m_oos, verdict)
    return verdict, m_oos


def _write_doc(wf, m_oos, verdict):
    doc_dir = ROOT / 'docs'
    doc_dir.mkdir(exist_ok=True)
    with open(doc_dir / 'V15_E_SHORT_BEAR_results.md', 'w', encoding='utf-8') as f:
        f.write("# V15 Estrategia E: SHORT Bear Market (Failed Bounce)\n\n")
        f.write(f"**Rama**: `v15/momentum-breakout`\n")
        f.write(f"**Fecha**: {pd.Timestamp.now().date()}\n")
        f.write(f"**Veredicto**: {verdict}\n\n")
        f.write("## Por que B-SHORT fallo\n\n")
        f.write("El patron 'BB estrecho + ruptura bajo low20' es de bull market.\n")
        f.write("En 2022-H1 (bear -75%), solo 10 barras cumplian todas las condiciones.\n\n")
        f.write("## Setup E: Failed Bounce\n\n")
        f.write("| Condicion | Valor |\n|-----------|-------|\n")
        f.write("| Macro | EMA20_1d < EMA50_1d (BEAR) |\n")
        f.write("| 4H trend | EMA20_4h < EMA50_4h |\n")
        f.write(f"| RSI reciente | subio a {RSI_MIN}-72 (hubo rebote) |\n")
        f.write(f"| RSI actual | 30-{RSI_MAX} (rechazo sin oversold extremo) |\n")
        f.write("| Entry | close < EMA20_4h (rechazado en resistencia) |\n")
        f.write(f"| Bar move | < {BAR_MOVE_MAX}% |\n")
        f.write(f"| SL | max(high[-3:]) * 1.003 |\n")
        f.write(f"| TP | SL_pct * {TP_RR} (RR {TP_RR}:1) |\n\n")
        f.write("## Walk-forward\n\n")
        f.write("| Periodo | N | WR | PF | Anual | OK |\n|---------|---|----|----|-------|----|\\n")
        for r in wf['folds']:
            ok_s = 'SI' if r['ok'] else 'NO'
            wr_s = f"{r['wr']:.1%}" if r['n'] > 0 else '-'
            pf_s = f"{r['pf']:.2f}" if r['n'] > 0 else '-'
            ann_s = f"{r['annual_pct']:.0f}%" if r['n'] > 0 else '-'
            f.write(f"| {r['period']} | {r['n']} | {wr_s} | {pf_s} | {ann_s} | {ok_s} |\n")
        f.write(f"\n**Folds OK**: {wf['folds_ok']}/12\n\n")
        f.write(f"## OOS | N={m_oos['n']} | WR={m_oos['wr']:.1%} | "
                f"PF={m_oos['pf']:.2f} | ~{m_oos['annual_pct']:.0f}%/yr\n\n")
        f.write(f"## Veredicto: {verdict}\n")


if __name__ == '__main__':
    main()
