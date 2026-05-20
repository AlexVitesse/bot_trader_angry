"""
revalidate_v15.py -- Re-validación honesta de los pares V15
============================================================
Motor de backtest CORREGIDO: una posición a la vez (ver
docs/revalidation/PASO0_lookahead.md). El motor anterior
(evaluate_new_pairs_v15.py) abría un trade en CADA vela con señal,
generando trades solapados que inflan PF a 8-20 y reducen DD a 1-4%.

Aquí, tras abrir un trade se salta hasta DESPUÉS de la vela en que cierra
-- exactamente lo que hace el bot en vivo (una posición por par).

Cubre los 20 pares rule_based_trailing (BTC-follower + breakout + trailing).
BTC (ML GBM) y ETH (rule_based propio) requieren motor dedicado -> se omiten
con aviso.

3 capas de validación:
  A. Forward-OOS  -- datos posteriores al cutoff (requiere refrescar datos)
  B. Walk-forward -- 12 semestres, motor corregido
  C. Bootstrap    -- ¿el retorno es edge o azar?

Uso:
  python revalidate_v15.py --pair ADA
  python revalidate_v15.py --all
Ejecutar con el Python de producción (ver CLAUDE.md) si falta pandas_ta.
"""

import sys
import json
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from v15_framework import (
    load_pair_4h, load_btc_4h,
    compute_features_4h, compute_macro_daily, merge_daily_to_4h,
    COMMISSION, WF_FOLDS,
)

# --- parámetros generales ---
REGIME_DEAD_ZONE = 0.02
MIN_TRADES_FOLD = 5        # Paso 0: subido de 3/1 a 5 (umbral honesto)
BOOTSTRAP_N = 3000
MIN_TRADES_OOS = 10        # menos -> PAPER-ONLY automático

# pares cubiertos por este motor (rule_based_trailing)
PAIR_DIR = {
    'ADA': 'ada_v15', 'SOL': 'sol_v15', 'DOGE': 'doge_v15', 'LINK': 'link_v15',
    'AVAX': 'avax_v15', 'DOT': 'dot_v15', 'NEAR': 'near_v15', 'XRP': 'xrp_v15',
    'ATOM': 'atom_v15', 'INJ': 'inj_v15', 'ALGO': 'algo_v15', 'FIL': 'fil_v15',
    '1000SHIB': '1000shib_v15', 'BNB': 'bnb_v15', 'LTC': 'ltc_v15',
    'ETC': 'etc_v15', 'BCH': 'bch_v15', 'UNI': 'uni_v15', 'AAVE': 'aave_v15',
    'OP': 'op_v15', 'BTC': 'btc_v15', 'ETH': 'eth_v15',
}


# =============================================================================
# DETECCIÓN (copiada de evaluate_new_pairs_v15.py -- Paso 0: sin look-ahead)
# =============================================================================
def detect_regime(row):
    ema20 = row.get('ema20_1d', None)
    ema50 = row.get('ema50_1d', None)
    ema200 = row.get('ema200_1d', None)
    close = row.get('close', None)
    if ema20 is None or ema50 is None or pd.isna(ema20) or pd.isna(ema50):
        return 'RANGE'
    dist = (ema20 - ema50) / ema50
    if dist > REGIME_DEAD_ZONE:
        return 'BULL'
    elif dist < -REGIME_DEAD_ZONE:
        if ema200 is not None and close is not None and not pd.isna(ema200):
            if close > ema200:
                return 'RANGE'
        if close is not None and not pd.isna(close) and close > ema50:
            return 'RANGE'
        return 'BEAR'
    return 'RANGE'


def detect_breakout(df, idx, vol_min, bb_max, bar_max=3.5):
    if idx < 25:
        return False
    row = df.iloc[idx]
    close = row['close']
    high20 = df['high'].iloc[idx - 20:idx].max()   # rango pasado, exclusivo
    vol_ratio = row.get('vol_ratio', 0)
    bb_width = row.get('bb_width', 99)
    ret_1 = row.get('ret_1', 0)
    if close <= high20 or vol_ratio < vol_min or bb_width > bb_max or abs(ret_1) > bar_max:
        return False
    return True


def detect_btc_breakout(df_btc, idx, vol_min=1.0):
    if idx < 25:
        return False
    close = float(df_btc.iloc[idx]['close'])
    high20 = float(df_btc['high'].iloc[max(0, idx - 20):idx].max())
    vol_ratio = float(df_btc.iloc[idx].get('vol_ratio', 0))
    return close > high20 and vol_ratio >= vol_min


def detect_btc_breakdown(df_btc, idx, lookback=20, vol_min=1.0):
    if idx < 25:
        return False
    close = float(df_btc.iloc[idx]['close'])
    low20 = float(df_btc['low'].iloc[max(0, idx - lookback):idx].min())
    vol_ratio = float(df_btc.iloc[idx].get('vol_ratio', 0))
    return close < low20 and vol_ratio >= vol_min


# =============================================================================
# SIMULADORES TRAILING (entran en vela cerrada, evalúan desde entry_bar+1)
# =============================================================================
def sim_long_trailing(df, entry_bar, entry_price, trail_dist, max_bars):
    """
    Trailing stop LONG, SIN look-ahead intrabar.
    El stop en la vela b se fija con el máximo de las velas ANTERIORES (<= b-1).
    Dentro de la vela b solo se comprueba la salida contra ese stop; el peak se
    actualiza DESPUÉS, para la vela siguiente.
    """
    sl_price = entry_price * (1 - trail_dist)
    peak = entry_price
    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df):
            exit_p = float(df['close'].iloc[-1])
            pnl = (exit_p - entry_price) / entry_price - 2 * COMMISSION
            return ('TP' if exit_p > entry_price else 'SL'), pnl, i
        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        # 1) salida contra el stop ya conocido (de velas previas)
        if lo <= sl_price:
            pnl = (sl_price - entry_price) / entry_price - 2 * COMMISSION
            return ('TP' if sl_price > entry_price else 'SL'), pnl, i
        # 2) recién ahora se actualiza el peak/stop para la vela siguiente
        if hi > peak:
            peak = hi
        sl_price = max(sl_price, peak * (1 - trail_dist))
    exit_p = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (exit_p - entry_price) / entry_price - 2 * COMMISSION
    return ('TP' if exit_p > entry_price else 'SL'), pnl, max_bars


def sim_short_trailing(df, entry_bar, entry_price, trail_dist, max_bars):
    """Trailing stop SHORT, SIN look-ahead intrabar (espejo de sim_long_trailing)."""
    sl_price = entry_price * (1 + trail_dist)
    trough = entry_price
    for i in range(1, max_bars + 1):
        b = entry_bar + i
        if b >= len(df):
            exit_p = float(df['close'].iloc[-1])
            pnl = (entry_price - exit_p) / entry_price - 2 * COMMISSION
            return ('TP' if exit_p < entry_price else 'SL'), pnl, i
        hi = float(df['high'].iloc[b])
        lo = float(df['low'].iloc[b])
        # 1) salida contra el stop ya conocido (de velas previas)
        if hi >= sl_price:
            pnl = (entry_price - sl_price) / entry_price - 2 * COMMISSION
            return ('TP' if sl_price < entry_price else 'SL'), pnl, i
        # 2) recién ahora se actualiza el trough/stop para la vela siguiente
        if lo < trough:
            trough = lo
        sl_price = min(sl_price, trough * (1 + trail_dist))
    exit_p = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (entry_price - exit_p) / entry_price - 2 * COMMISSION
    return ('TP' if exit_p < entry_price else 'SL'), pnl, max_bars


# =============================================================================
# MOTOR CORREGIDO: una posición a la vez
# =============================================================================
def run_engine(df, df_btc, regimes, params, start_i, end_i, btc_pos=None):
    """
    Recorre las velas [start_i, end_i). Al abrir un trade, SALTA hasta después
    de la vela en que cierra -> nunca solapa posiciones (igual que el bot real).
    Devuelve la lista de trades.

    btc_pos: array precalculado df.index -> posición en df_btc (evita O(n^2)).
    """
    if btc_pos is None:
        btc_pos = df_btc.index.get_indexer(df.index, method='nearest')
    vol_min   = params.get('breakout_vol_min', 1.2)
    bb_max    = params.get('breakout_bb_max', 7.0)
    bar_max   = params.get('breakout_bar_move_max', 3.5)
    corr_min  = params.get('follower_corr_min', 0.4)
    bd_look   = params.get('breakdown_lookback', 20)
    bd_vol    = params.get('breakdown_vol_min', 1.0)
    trail_f   = params.get('trail_atr_factor', 0.20)
    trail_fl  = params.get('trail_floor', 0.006)
    max_bars  = params.get('trail_max_bars', 30)

    trades = []
    i = max(start_i, 25)
    end_i = min(end_i, len(df) - 2)
    while i < end_i:
        ts = df.index[i]
        regime = regimes.iloc[i]
        entry = float(df['close'].iloc[i])
        atr_pct = float(df['atr_pct'].iloc[i]) / 100.0
        if not np.isfinite(atr_pct) or atr_pct <= 0:
            i += 1
            continue
        trail_dist = max(trail_fl, atr_pct * trail_f)

        side = None
        btc_idx = int(btc_pos[i])
        if regime != 'BEAR':
            if detect_breakout(df, i, vol_min, bb_max, bar_max):
                side = 'LONG'
            elif 0 <= btc_idx < len(df_btc) and detect_btc_breakout(df_btc, btc_idx):
                if float(df['pair_btc_corr'].iloc[i]) >= corr_min:
                    side = 'LONG'
        else:
            if 0 <= btc_idx < len(df_btc) and detect_btc_breakdown(df_btc, btc_idx, bd_look, bd_vol):
                if float(df['pair_btc_corr'].iloc[i]) >= corr_min:
                    side = 'SHORT'

        if side is None:
            i += 1
            continue

        if side == 'LONG':
            out = sim_long_trailing(df, i, entry, trail_dist, max_bars)
        else:
            out = sim_short_trailing(df, i, entry, trail_dist, max_bars)

        trades.append({'outcome': out[0], 'pnl_pct': out[1], 'ts': ts,
                       'side': side, 'bars': out[2]})
        # EL FIX: reanudar la búsqueda DESPUÉS de la vela de cierre
        i += out[2] + 1

    return trades


# =============================================================================
# MÉTRICAS + BOOTSTRAP
# =============================================================================
def calc_metrics(trades):
    if not trades:
        return {'n': 0, 'wr': 0.0, 'pf': 0.0, 'avg': 0.0, 'total': 0.0,
                'equity': 1.0, 'max_dd': 0.0}
    n = len(trades)
    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / n
    gw = sum(wins)
    gl = abs(sum(losses))
    pf = (gw / gl) if gl > 1e-9 else float('inf')
    # equity / drawdown sobre trades secuenciales (ya no solapados)
    cum, peak, max_dd = 1.0, 1.0, 0.0
    for t in sorted(trades, key=lambda x: x['ts']):
        cum *= (1 + t['pnl_pct'])
        peak = max(peak, cum)
        max_dd = max(max_dd, (peak - cum) / peak)
    return {'n': n, 'wr': wr, 'pf': pf, 'avg': sum(pnls) / n,
            'total': cum - 1.0, 'equity': cum, 'max_dd': max_dd}


def fold_ok(m):
    """Criterio honesto de fold (Paso 0): n>=5, PF finito y >=1.2, WR>break-even."""
    if m['n'] < MIN_TRADES_FOLD:
        return False
    if not np.isfinite(m['pf']):
        return False          # PF=inf (0 pérdidas) -> muestra insuficiente, no aprobar
    return m['pf'] >= 1.2 and m['total'] > 0


def bootstrap_pvalue(trades, n_iter=BOOTSTRAP_N, seed=42):
    """
    ¿El retorno total pudo salir por azar? Re-muestrea los trades con reemplazo.
    p = fracción de re-muestreos cuyo retorno total es <= 0.
    También devuelve el percentil 5 del retorno total.
    """
    if len(trades) < 3:
        return None
    rng = np.random.default_rng(seed)
    pnls = np.array([t['pnl_pct'] for t in trades])
    k = len(pnls)
    totals = np.empty(n_iter)
    for j in range(n_iter):
        sample = rng.choice(pnls, size=k, replace=True)
        totals[j] = np.prod(1 + sample) - 1.0
    p_value = float(np.mean(totals <= 0))
    return {'p_value': p_value,
            'pctl_5': float(np.percentile(totals, 5)),
            'pctl_50': float(np.percentile(totals, 50))}


# =============================================================================
# CARGA DE DATOS / META
# =============================================================================
def load_meta(pair):
    d = PAIR_DIR.get(pair.upper())
    if not d:
        return None
    p = ROOT / 'strategies' / d / 'models' / 'meta_v15.json'
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding='utf-8'))


def load_pair_data(pair, df_btc):
    df_raw = load_pair_4h(pair)
    df = compute_features_4h(df_raw.copy())
    try:
        from v15_framework import load_pair_1d
        pair_1d = load_pair_1d(pair)
    except Exception:
        pair_1d = df_raw.resample('1D').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum'}).dropna()
    macro = compute_macro_daily(pair_1d)
    df = merge_daily_to_4h(df, macro)
    regimes = df.apply(detect_regime, axis=1)

    common = df.index.intersection(df_btc.index)
    if len(common) > 100:
        pair_ret = df.loc[common, 'close'].pct_change()
        btc_ret = df_btc.loc[common, 'close'].pct_change()
        rc = pair_ret.rolling(168).corr(btc_ret)
        df['pair_btc_corr'] = rc.reindex(df.index).ffill().fillna(0.7)
    else:
        df['pair_btc_corr'] = 0.7
    return df, regimes


# =============================================================================
# CAPAS A / B / C
# =============================================================================
def layer_B_walkforward(df, df_btc, regimes, params, btc_pos):
    folds = []
    for start_s, end_s in WF_FOLDS:
        s = pd.Timestamp(start_s, tz='UTC')
        e = pd.Timestamp(end_s, tz='UTC')
        idxs = np.where((df.index >= s) & (df.index <= e))[0]
        if len(idxs) < 100:
            folds.append({'period': f'{start_s[:7]}', 'n': 0, 'ok': False, 'nodata': True})
            continue
        trades = run_engine(df, df_btc, regimes, params, idxs[0], idxs[-1] + 1, btc_pos)
        m = calc_metrics(trades)
        folds.append({'period': f'{start_s[:7]}', 'n': m['n'], 'wr': m['wr'],
                      'pf': m['pf'], 'total': m['total'], 'ok': fold_ok(m),
                      'nodata': False})
    ok = sum(1 for f in folds if f['ok'])
    total = sum(1 for f in folds if not f['nodata'])
    return {'folds': folds, 'folds_ok': ok, 'folds_total': total}


def layer_A_forward_oos(df, df_btc, regimes, params, cutoff, btc_pos):
    """Forward-OOS: trades sobre datos POSTERIORES al cutoff (nunca vistos)."""
    cut = pd.Timestamp(cutoff, tz='UTC')
    idxs = np.where(df.index > cut)[0]
    data_end = df.index.max()
    if len(idxs) < 30:
        return {'available': False, 'cutoff': str(cut.date()),
                'data_end': str(data_end.date()), 'bars': len(idxs)}
    trades = run_engine(df, df_btc, regimes, params, idxs[0], idxs[-1] + 1, btc_pos)
    m = calc_metrics(trades)
    return {'available': True, 'cutoff': str(cut.date()),
            'data_end': str(data_end.date()), 'bars': len(idxs),
            'metrics': m, 'trades': trades}


# =============================================================================
# FICHA + VEREDICTO
# =============================================================================
def verdict(layA, layB, layC):
    """KEEP / PAPER-ONLY / REJECT segun las 3 capas."""
    wf_pass = layB['folds_total'] >= 1 and layB['folds_ok'] >= max(7, 0) \
        and layB['folds_ok'] / max(1, layB['folds_total']) >= 0.58
    if not layA['available']:
        # sin datos OOS -> no se puede decidir KEEP
        if wf_pass:
            return 'PAPER-ONLY', 'WF ok pero falta forward-OOS (refrescar datos)'
        return 'REJECT (provisional)', 'falla WF con motor corregido'
    mA = layA['metrics']
    if mA['n'] < MIN_TRADES_OOS:
        return 'PAPER-ONLY', f"forward-OOS con solo {mA['n']} trades (<{MIN_TRADES_OOS})"
    boot_ok = layC is not None and layC['p_value'] < 0.05
    oos_ok = mA['pf'] >= 1.2 and mA['total'] > 0 and np.isfinite(mA['pf'])
    if wf_pass and oos_ok and boot_ok:
        return 'KEEP', 'cumple WF + forward-OOS + bootstrap'
    if oos_ok and boot_ok:
        return 'PAPER-ONLY', 'forward-OOS ok pero WF insuficiente'
    return 'REJECT', 'falla forward-OOS o bootstrap'


def fmt_pf(pf):
    return 'inf' if not np.isfinite(pf) else f'{pf:.2f}'


def write_ficha(pair, meta, layA, layB, layC, declared):
    lines = []
    lines.append(f'# Re-validación V15 — {pair}/USDT\n')
    lines.append(f'> Generado por `revalidate_v15.py` (motor corregido, una posición '
                 f'a la vez). Fecha: {pd.Timestamp.utcnow().date()}\n')

    v, reason = verdict(layA, layB, layC)
    lines.append(f'## VEREDICTO: **{v}**\n')
    lines.append(f'{reason}\n')

    # comparación declarado vs real
    lines.append('## Declarado en meta_v15.json vs. motor corregido\n')
    lines.append('| Métrica | Declarado (motor viejo) | Motor corregido |')
    lines.append('|---------|------------------------|-----------------|')
    dl = declared
    lines.append(f"| WF LONG | {dl.get('long_wf_ok','?')}/{dl.get('long_wf_total','?')} "
                 f"| {layB['folds_ok']}/{layB['folds_total']} (LONG+SHORT combinado) |")
    lines.append(f"| PF LONG | {dl.get('long_pf','?')} | ver capa B abajo |")
    lines.append(f"| PF SHORT | {dl.get('short_pf','?')} | — |")
    lines.append(f"| DD declarado | {dl.get('long_dd','?')} | — |\n")

    # Capa A
    lines.append('## Capa A — Forward-OOS (datos posteriores al cutoff)\n')
    if not layA['available']:
        lines.append(f"⚠️ **No disponible.** Cutoff = {layA['cutoff']}, los datos "
                     f"locales terminan en {layA['data_end']} "
                     f"({layA['bars']} velas post-cutoff). "
                     f"Refrescar datos con `download_new_pairs.py` para activar esta capa.\n")
    else:
        m = layA['metrics']
        lines.append(f"Ventana: {layA['cutoff']} → {layA['data_end']} ({layA['bars']} velas)\n")
        lines.append(f"- Trades: {m['n']} | WR: {m['wr']:.1%} | PF: {fmt_pf(m['pf'])} "
                     f"| Retorno: {m['total']:+.1%} | DD: {m['max_dd']:.1%}\n")

    # Capa B
    lines.append('## Capa B — Walk-forward (motor corregido)\n')
    lines.append(f"**{layB['folds_ok']}/{layB['folds_total']} folds aprobados** "
                 f"(criterio: n≥{MIN_TRADES_FOLD}, PF finito ≥1.2, retorno>0)\n")
    lines.append('| Semestre | N | WR | PF | Retorno | OK |')
    lines.append('|----------|---|----|----|---------|----|')
    for f in layB['folds']:
        if f['nodata']:
            lines.append(f"| {f['period']} | — | — | — | sin datos | — |")
        else:
            lines.append(f"| {f['period']} | {f['n']} | {f['wr']:.0%} | "
                         f"{fmt_pf(f['pf'])} | {f['total']:+.1%} | "
                         f"{'✅' if f['ok'] else '❌'} |")
    lines.append('')

    # Capa C
    lines.append('## Capa C — Bootstrap de significancia\n')
    if layC is None:
        lines.append('Muestra insuficiente para bootstrap.\n')
    else:
        sig = '✅ significativo' if layC['p_value'] < 0.05 else '❌ NO significativo'
        lines.append(f"- p-value (retorno ≤ 0 por azar): **{layC['p_value']:.3f}** — {sig}")
        lines.append(f"- Retorno mediano re-muestreado: {layC['pctl_50']:+.1%}")
        lines.append(f"- Percentil 5: {layC['pctl_5']:+.1%}\n")

    out = ROOT / 'docs' / 'revalidation' / f'{pair}.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    return out, v


# =============================================================================
# MAIN
# =============================================================================
def revalidate_pair(pair, df_btc):
    pair = pair.upper()
    meta = load_meta(pair)
    if meta is None:
        print(f'  [{pair}] sin meta_v15.json — omitido')
        return None
    mtype = meta.get('model_type', '')
    if mtype != 'rule_based_trailing':
        print(f'  [{pair}] model_type="{mtype}" — requiere motor dedicado '
              f'(BTC ML / ETH reglas). Omitido por este script.')
        return None

    try:
        df, regimes = load_pair_data(pair, df_btc)
    except Exception as e:
        print(f'  [{pair}] error cargando datos: {e}')
        return None

    cutoff = meta.get('training_date', '2026-03-25')
    params = meta
    declared = meta.get('backtest', {})

    # precalcular alineación BTC una sola vez (evita O(n^2))
    btc_pos = df_btc.index.get_indexer(df.index, method='nearest')

    layB = layer_B_walkforward(df, df_btc, regimes, params, btc_pos)
    layA = layer_A_forward_oos(df, df_btc, regimes, params, cutoff, btc_pos)

    # bootstrap sobre el conjunto de trades del walk-forward completo
    wf0 = np.where(df.index >= pd.Timestamp(WF_FOLDS[0][0], tz='UTC'))[0]
    all_trades = run_engine(df, df_btc, regimes, params,
                            int(wf0[0]) if len(wf0) else 0, len(df), btc_pos)
    layC = bootstrap_pvalue(all_trades)

    out, v = write_ficha(pair, meta, layA, layB, layC, declared)
    mB = f"{layB['folds_ok']}/{layB['folds_total']}"
    print(f'  [{pair}] WF(corregido)={mB}  veredicto={v}  -> {out}')
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', help='par individual, ej. ADA')
    ap.add_argument('--all', action='store_true', help='los 20 pares trailing')
    args = ap.parse_args()

    print('Cargando BTC 4h...')
    df_btc = compute_features_4h(load_btc_4h().copy())

    if args.pair:
        revalidate_pair(args.pair, df_btc)
    elif args.all:
        pairs = [p for p in PAIR_DIR if p not in ('BTC', 'ETH')]
        print(f'Re-validando {len(pairs)} pares rule_based_trailing...\n')
        for p in pairs:
            revalidate_pair(p, df_btc)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
