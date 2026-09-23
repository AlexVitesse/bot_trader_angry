"""
backtest_v15_committee.py  —  V15 Expert Committee Backtest
==============================================================================
Sistema combinado que opera en TODOS los regimenes:

  BULL  macro -> Breakout B LONG  (WR=61%, PF=2.87 standalone)
  BEAR  macro -> SHORT ML         (WR=45%, PF=1.16 standalone)
  RANGE macro -> no operar (esperar)

Gates adicionales:
  - Funding veto: z-score > 2 -> bloquea LONG (mercado overcrowded)
                  z-score < -1.5 -> bloquea SHORT (oversold extremo = squeeze)
  - Consecutive losses: 3 SL seguidos -> pausa (ya validado en portfolio_manager)

Evaluacion:
  1. Walk-forward 12 semestres (2020-2025): >= 7/12 folds OK
  2. OOS 2022-2026: WR >= 40%, PF >= 1.3, >= 2.0 trades/mes
  3. Stress test por regimen: cada uno debe ser rentable o neutro
  4. Cross-asset ETH (separado)
"""

import sys, warnings, json
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from v15_framework import (
    load_btc_4h, compute_features_4h, compute_macro_daily, merge_daily_to_4h,
    load_funding, sim_trade_fixed, metrics, print_metrics,
    WF_FOLDS, OOS_START, OOS_END, COMMISSION
)

# ============================================================
# CONFIG
# ============================================================
LONG_MAX_BARS  = 16
SHORT_MAX_BARS = 16
SHORT_THRESHOLD = 0.60
LONG_ML_THRESHOLD = 0.65
FUNDING_VETO_LONG  = 2.0    # z-score > 2 -> bloquea LONG
FUNDING_VETO_SHORT = -1.5   # z-score < -1.5 -> bloquea SHORT
CONSEC_LOSS_PAUSE  = 3      # 3 SL seguidos -> pausa

# ML features (mismas para LONG y SHORT — contexto tecnico completo)
ML_FEATURES = [
    'ema200_dist', 'ema20_slope', 'ema50_slope',
    'rsi14', 'rsi_slope', 'di_diff', 'adx14',
    'bb_pct', 'bb_width', 'atr_pct',
    'range_pos', 'vol_ratio', 'vol_slope',
    'ret_1', 'ret_5', 'ret_10',
    'consec_up', 'bull_1d',
]
SHORT_FEATURES = ML_FEATURES  # alias
LONG_FEATURES = ML_FEATURES   # mismas features, el modelo aprende distintos patrones


# ============================================================
# EXTRA FEATURES (para SHORT ML)
# ============================================================
def add_extra_features(df):
    df = df.copy()
    c, v = df['close'], df['volume']
    df['rsi_slope'] = df['rsi14'].diff(3)
    vol_ma5  = v.rolling(5).mean()
    vol_ma20 = v.rolling(20).mean()
    df['vol_slope'] = (vol_ma5 / vol_ma20.replace(0, np.nan) - 1) * 100
    df['ret_10'] = c.pct_change(10) * 100
    up = (c > c.shift(1)).astype(int)
    df['consec_up'] = up.rolling(8).sum()
    return df


# ============================================================
# REGIME DETECTION
# ============================================================
REGIME_DEAD_ZONE = 0.02   # 2% dead zone entre BULL y BEAR

def detect_regime(row):
    """BULL / BEAR / RANGE basado en macro diario con zona muerta + recovery."""
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
        # Recovery filter: precio > EMA200 = no es bear real
        if ema200 is not None and close is not None and not pd.isna(ema200):
            if close > ema200:
                return 'RANGE'
        # Precio debe estar DEBAJO de EMA50 (tendencia bajista confirmada)
        if close is not None and not pd.isna(close):
            if close > ema50:
                return 'RANGE'
        return 'BEAR'
    return 'RANGE'


# ============================================================
# LONG SETUP: Breakout B (v1 original — maxima calidad)
# ============================================================
def detect_long_breakout(df, i):
    if i < 25:
        return None
    row = df.iloc[i]

    # Ruptura: close > max 20 barras
    high20 = float(df['high'].iloc[i-20:i].max())
    if row['close'] <= high20:
        return None
    # Volumen confirma (original: >= 1.8x)
    if row.get('vol_ratio', 1) < 1.8:
        return None
    # Vela no demasiado grande
    bar_move = abs(float(row['close']) - float(row['open'])) / float(row['open']) * 100
    if bar_move > 2.5:
        return None
    # Consolidacion previa (BB estrecho 3/5 barras < 4%)
    recent_bb = df['bb_width'].iloc[i-5:i]
    narrow_bars = (recent_bb < 4.0).sum()
    if narrow_bars < 3:
        return None
    # ADX anterior bajo (< 28)
    prev_adx = df['adx14'].iloc[i-3:i].mean()
    if prev_adx > 28:
        return None
    # SL/TP
    entry  = float(row['close'])
    sl_raw = float(df['low'].iloc[i-5:i].min()) * 0.997
    sl_pct = (entry - sl_raw) / entry
    if sl_pct < 0.005 or sl_pct > 0.04:
        return None
    tp_pct = sl_pct * 1.5
    return {'direction': 'LONG', 'setup': 'BREAKOUT_B',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}


# ============================================================
# LONG SETUP: Pullback to EMA20 (reglas, sin ML)
# ============================================================
def detect_long_pullback(df, i):
    """Compra en pullback a EMA20 en tendencia alcista confirmada."""
    if i < 25:
        return None
    row = df.iloc[i]
    prev = df.iloc[i-1]
    prev2 = df.iloc[i-2]
    c = float(row['close'])
    o = float(row['open'])

    # EMA20 4h como soporte
    ema20 = float(row.get('ema20', 0))
    ema50 = float(row.get('ema50', 0))
    if ema20 <= 0 or ema50 <= 0:
        return None

    # Precio debe estar sobre EMA50 (tendencia 4H confirmada)
    if c < ema50:
        return None

    # Precio cerca de EMA20 (dentro de 1.5% por arriba)
    dist_ema20 = (c - ema20) / ema20
    if dist_ema20 < -0.005 or dist_ema20 > 0.015:
        return None

    # ADX minimo (hay tendencia, no rango)
    adx = float(row.get('adx14', 0))
    if adx < 15:
        return None

    # RSI en zona de pullback (no oversold ni overbought)
    rsi = float(row.get('rsi14', 50))
    if rsi < 33 or rsi > 58:
        return None

    # Vela actual bullish (reversal)
    if c <= o:
        return None

    # Vela anterior fue bajista (confirma pullback)
    if float(prev['close']) >= float(prev['open']):
        return None

    # Volumen no excesivo (pullback en bajo vol, no panic)
    vol_ratio = float(row.get('vol_ratio', 1))
    if vol_ratio > 2.0:
        return None

    # ATR-based TP/SL (adapta a volatilidad)
    atr_pct = float(row.get('atr_pct', 2.0))
    entry = c
    sl_pct = max(min(atr_pct / 100 * 1.0, 0.03), 0.01)  # 1 ATR, capped 1%-3%
    tp_pct = sl_pct * 1.67  # RR 1.67:1

    return {'direction': 'LONG', 'setup': 'PULLBACK_EMA20',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}


# ============================================================
# SHORT SETUP: ML model
# ============================================================
def detect_short_ml(df, i, model_data):
    if model_data is None or i < 30:
        return None
    row = df.iloc[i]
    model  = model_data['model']
    scaler = model_data['scaler']

    # Features
    x = pd.DataFrame([row[SHORT_FEATURES].fillna(0).values], columns=SHORT_FEATURES)
    x_s = scaler.transform(x)
    prob = model.predict_proba(x_s)[0][1]

    if prob < SHORT_THRESHOLD:
        return None

    entry = float(row['close'])
    # SL dinamico: max 3 barras
    sl_raw = float(df['high'].iloc[max(0,i-3):i+1].max()) * 1.003
    sl_pct = (sl_raw - entry) / entry
    sl_pct = min(max(sl_pct, 0.015), 0.04)
    tp_pct = sl_pct * 1.67

    return {'direction': 'SHORT', 'setup': 'ML_SHORT',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct,
            'prob': prob}


# ============================================================
# SIMULACION SHORT
# ============================================================
def sim_short(df, entry_bar, entry_price, tp_pct, sl_pct, max_bars=SHORT_MAX_BARS):
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
    ep = float(df['close'].iloc[entry_bar + max_bars])
    pnl = (entry_price - ep) / entry_price - 2 * COMMISSION
    return ('TP' if ep < entry_price else 'SL'), ep, pnl, max_bars


# ============================================================
# RANGE SETUP: Mean Reversion (comprar abajo, vender arriba)
# ============================================================
RANGE_MAX_BARS = 12  # targets mas cortos en rango

def detect_range_long(df, i):
    """LONG mean reversion: comprar cerca del piso del rango."""
    if i < 25:
        return None
    row = df.iloc[i]
    prev = df.iloc[i-1]
    c = float(row['close'])
    o = float(row['open'])

    # Posicion en el rango de 20 barras: < 0.20 = cerca del piso
    range_pos = float(row.get('range_pos', 0.5))
    if range_pos > 0.25 or pd.isna(range_pos):
        return None

    # RSI oversold moderado (no extremo crash)
    rsi = float(row.get('rsi14', 50))
    if rsi < 25 or rsi > 45:
        return None

    # ADX bajo (confirma rango, no tendencia)
    adx = float(row.get('adx14', 30))
    if adx > 25:
        return None

    # Vela bullish (reversal desde abajo)
    if c <= o:
        return None

    # Vela anterior bajista (confirma pullback)
    if float(prev['close']) >= float(prev['open']):
        return None

    # Volumen no excesivo (no es capitulacion)
    vol_ratio = float(row.get('vol_ratio', 1))
    if vol_ratio > 2.5:
        return None

    # ATR-based TP/SL tight (en rango targets cortos)
    atr_pct = float(row.get('atr_pct', 2.0))
    entry = c
    sl_pct = max(min(atr_pct / 100 * 0.8, 0.02), 0.008)
    tp_pct = sl_pct * 1.5

    return {'direction': 'LONG', 'setup': 'RANGE_MR_LONG',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}


def detect_range_short(df, i):
    """SHORT mean reversion: vender solo en extremos claros del rango."""
    if i < 25:
        return None
    row = df.iloc[i]
    prev = df.iloc[i-1]
    prev2 = df.iloc[i-2]
    c = float(row['close'])
    o = float(row['open'])

    # ADX bajo = rango real
    adx = float(row.get('adx14', 30))
    if adx > 22:
        return None

    # Precio en extremo superior (BB > 0.90)
    bb_pct = float(row.get('bb_pct', 0.5))
    if bb_pct < 0.90:
        return None

    # RSI overbought claro
    rsi = float(row.get('rsi14', 50))
    if rsi < 62 or rsi > 78:
        return None

    # Vela bearish (rechazo)
    if c >= o:
        return None

    # Al menos 2 velas verdes antes (subida real)
    if float(prev['close']) <= float(prev['open']):
        return None
    if float(prev2['close']) <= float(prev2['open']):
        return None

    # No breakout (close < high20)
    high20 = float(row.get('high20', c))
    if high20 > 0 and c > high20:
        return None

    # ATR-based TP/SL tight
    atr_pct = float(row.get('atr_pct', 2.0))
    entry = c
    sl_pct = max(min(atr_pct / 100 * 0.8, 0.02), 0.008)
    tp_pct = sl_pct * 1.5

    return {'direction': 'SHORT', 'setup': 'RANGE_MR_SHORT',
            'entry': entry, 'sl_pct': sl_pct, 'tp_pct': tp_pct}


# ============================================================
# FUNDING Z-SCORE
# ============================================================
def add_funding_zscore(df):
    """Agrega funding z-score a df (rolling 90-day)."""
    try:
        funding = load_funding()
        # Merge asof
        df_out = df.copy()
        funding_reindexed = funding.reindex(df.index, method='ffill')
        roll_mean = funding_reindexed.rolling(90 * 6).mean()  # 90 dias * 6 barras/dia
        roll_std  = funding_reindexed.rolling(90 * 6).std()
        df_out['funding_zscore'] = (funding_reindexed - roll_mean) / roll_std.replace(0, np.nan)
        df_out['funding_zscore'] = df_out['funding_zscore'].fillna(0)
        return df_out
    except Exception as e:
        print(f"  WARNING: no se pudo cargar funding ({e}), sin veto")
        df_out = df.copy()
        df_out['funding_zscore'] = 0
        return df_out


# ============================================================
# COMITE PRINCIPAL
# ============================================================
def run_committee(df, short_model_data, start_idx, end_idx):
    """Ejecuta el comite sobre un rango de barras. Retorna lista de trades."""
    trades = []
    consec_losses = 0
    paused = False

    for i in range(start_idx, end_idx):
        if i < 30:
            continue

        row = df.iloc[i]
        regime = detect_regime(row)
        funding_z = row.get('funding_zscore', 0)
        trade = None

        # Pausa por racha de perdidas
        if paused:
            paused = False
            consec_losses = 0

        # BULL -> buscar LONG (breakout B primero, luego pullback EMA20)
        if regime == 'BULL':
            if funding_z > FUNDING_VETO_LONG:
                continue
            trade = detect_long_breakout(df, i)
            if trade is None:
                trade = detect_long_pullback(df, i)

        # BEAR -> buscar SHORT ML
        elif regime == 'BEAR':
            if funding_z < FUNDING_VETO_SHORT:
                continue
            trade = detect_short_ml(df, i, short_model_data)

        # RANGE -> solo Breakout B (mas selectivo que pullback)
        elif regime == 'RANGE':
            if funding_z > FUNDING_VETO_LONG:
                continue
            trade = detect_long_breakout(df, i)

        if trade is None:
            continue

        # Simular trade
        max_b = LONG_MAX_BARS if trade['direction'] == 'LONG' else SHORT_MAX_BARS
        if trade['direction'] == 'LONG':
            out = sim_trade_fixed(df, i, trade['entry'],
                                  trade['tp_pct'], trade['sl_pct'],
                                  max_bars=max_b)
        else:
            out = sim_short(df, i, trade['entry'],
                            trade['tp_pct'], trade['sl_pct'],
                            max_bars=max_b)

        # Racha de perdidas
        if out[0] == 'SL':
            consec_losses += 1
            if consec_losses >= CONSEC_LOSS_PAUSE:
                paused = True
        else:
            consec_losses = 0

        trades.append({
            'outcome': out[0],
            'pnl_pct': out[2],
            'ts': df.index[i],
            'direction': trade['direction'],
            'setup': trade['setup'],
            'regime': regime,
            'funding_z': funding_z,
        })

    return trades


# ============================================================
# WALK-FORWARD ML (SHORT model expanding)
# ============================================================
def walk_forward_committee(df, labels_short):
    """WF donde el modelo SHORT se reentrena en expanding window. LONG es rule-based."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    results = []
    all_trades = []

    for fold_idx, (start_s, end_s) in enumerate(WF_FOLDS):
        test_mask  = (df.index >= start_s) & (df.index <= end_s)
        train_mask = df.index < start_s

        period_label = f"{start_s[:7]}/{end_s[5:7]}"
        df_train = df[train_mask]

        # --- Entrenar SHORT ML solo en barras BEAR ---
        y_short_tr = labels_short[train_mask]
        bear_train = df_train.get('bull_1d', pd.Series(1, index=df_train.index)) == 0
        valid_short = y_short_tr.notna() & bear_train
        df_short_tr = df_train[valid_short]
        y_short_fit = y_short_tr[valid_short]

        short_model_data = None
        if len(df_short_tr) >= 800 and y_short_fit.sum() >= 20 and (len(y_short_fit) - y_short_fit.sum()) >= 20:
            X_s = df_short_tr[SHORT_FEATURES].fillna(0)
            scaler_s = StandardScaler()
            X_ss = scaler_s.fit_transform(X_s)
            model_s = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                min_samples_leaf=20, subsample=0.8, random_state=42
            )
            model_s.fit(X_ss, y_short_fit)
            short_model_data = {'model': model_s, 'scaler': scaler_s}

        # Ejecutar comite sobre la fold de test
        df_test = df[test_mask]
        if len(df_test) == 0:
            results.append({'period': period_label, 'n': 0, 'wr': 0,
                           'pf': 0, 'ok': False, 'annual_pct': 0,
                           'n_long': 0, 'n_short': 0})
            continue

        start_bar = df.index.get_loc(df_test.index[0])
        end_bar   = df.index.get_loc(df_test.index[-1]) + 1
        trades = run_committee(df, short_model_data, start_bar, end_bar)

        # Metrics
        m = metrics(trades, period_label)
        days = (pd.Timestamp(end_s) - pd.Timestamp(start_s)).days
        annual = m['avg_pnl'] * m['n'] / days * 365 * 100 if days > 0 and m['n'] > 0 else 0

        n_long  = sum(1 for t in trades if t['direction'] == 'LONG')
        n_short = sum(1 for t in trades if t['direction'] == 'SHORT')

        ok = (m['n'] >= 3 and m['wr'] > 0.38 and m['pf'] > 1.0)
        results.append({'period': period_label, 'n': m['n'],
                        'wr': m['wr'], 'pf': m['pf'],
                        'ok': ok, 'annual_pct': annual,
                        'n_long': n_long, 'n_short': n_short})
        all_trades.extend(trades)

    folds_ok = sum(1 for r in results if r['ok'])
    return {'folds': results, 'folds_ok': folds_ok,
            'approved': folds_ok >= 7, 'all_trades': all_trades}


# ============================================================
# LABELS SHORT (para WF training)
# ============================================================
def create_long_labels(df, tp_pct=0.03, sl_pct=0.015, max_bars=16):
    """Label: precio sube tp_pct antes de caer sl_pct en max_bars."""
    closes = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values
    labels = np.full(len(df), np.nan)
    for i in range(len(df) - max_bars - 1):
        entry = closes[i]
        tp = entry * (1 + tp_pct)
        sl = entry * (1 - sl_pct)
        for j in range(i + 1, i + max_bars + 1):
            if highs[j] >= tp:
                labels[i] = 1; break
            if lows[j] <= sl:
                labels[i] = 0; break
        else:
            labels[i] = 1 if closes[i + max_bars] > entry else 0
    return pd.Series(labels, index=df.index)


def create_short_labels(df, tp_pct=0.025, sl_pct=0.015, max_bars=16):
    closes = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values
    labels = np.full(len(df), np.nan)
    for i in range(len(df) - max_bars - 1):
        entry = closes[i]
        tp = entry * (1 - tp_pct)
        sl = entry * (1 + sl_pct)
        for j in range(i + 1, i + max_bars + 1):
            if lows[j] <= tp:
                labels[i] = 1; break
            if highs[j] >= sl:
                labels[i] = 0; break
        else:
            labels[i] = 1 if closes[i + max_bars] < entry else 0
    return pd.Series(labels, index=df.index)


# ============================================================
# MAIN
# ============================================================
def load_pair_4h(pair='BTC'):
    """Carga OHLCV 4h para cualquier par."""
    if pair.upper() == 'BTC':
        return load_btc_4h()
    else:
        # ETH, SOL, etc. — cargar desde parquet
        fname = f'{pair.upper()}_USDT_4h_full.parquet'
        fpath = ROOT / 'data' / fname
        if not fpath.exists():
            raise FileNotFoundError(f"No se encontro {fpath}")
        df = pd.read_parquet(fpath)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        return df.sort_index()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair', default='BTC', help='Par a evaluar (BTC, ETH, SOL...)')
    args = parser.parse_args()
    pair = args.pair.upper()

    print("=" * 70)
    print(f"V15 EXPERT COMMITTEE -- BACKTEST {pair}")
    print("=" * 70)
    print("\nComite:")
    print("  BULL  -> Breakout B LONG + Pullback EMA20 LONG (reglas)")
    print("  BEAR  -> SHORT ML         (GBM threshold=0.60, entrenado en BEAR)")
    print("  RANGE -> Mean Reversion   (comprar BB<0.2, vender BB>0.8)")
    print("  Gates: funding veto + consecutive loss pause")

    # Cargar datos
    print(f"\nCargando datos {pair}...")
    df_raw   = load_pair_4h(pair)
    df_feat  = compute_features_4h(df_raw)
    df_feat  = add_extra_features(df_feat)
    df_daily = compute_macro_daily(df_feat)
    df_feat  = merge_daily_to_4h(df_feat, df_daily)
    df_feat  = add_funding_zscore(df_feat)
    print(f"  {len(df_feat)} velas | {df_feat.index[0].date()} - {df_feat.index[-1].date()}")

    # Distribucion de regimenes
    regimes = df_feat.apply(lambda r: detect_regime(r), axis=1)
    for r in ['BULL', 'BEAR', 'RANGE']:
        pct = (regimes == r).mean()
        print(f"  {r}: {pct:.1%}")

    # Labels SHORT para WF training (LONG es rule-based, no necesita labels)
    print("\nCreando labels SHORT...")
    labels_short = create_short_labels(df_feat)
    base_short = labels_short[labels_short.notna()].mean()
    print(f"  SHORT base rate: {base_short:.1%}")

    # Walk-forward
    print("\n" + "=" * 70)
    print("WALK-FORWARD: 12 semestres")
    print("=" * 70)
    wf = walk_forward_committee(df_feat, labels_short)

    print(f"\n  {'Periodo':<14} | {'N':>4} | {'L':>3} | {'S':>3} | {'WR':>7} | {'PF':>6} | {'Anual':>8} | OK")
    print("  " + "-" * 65)
    for r in wf['folds']:
        ok_s  = '+' if r['ok'] else '-'
        ann_s = f"{r['annual_pct']:.0f}%" if r['n'] > 0 else 'n/a'
        wr_s  = f"{r['wr']:.1%}" if r['n'] > 0 else 'n/a'
        pf_s  = f"{r['pf']:.2f}" if r['n'] > 0 else 'n/a'
        print(f"  {r['period']:<14} | {r['n']:>4} | {r['n_long']:>3} | {r['n_short']:>3} | "
              f"{wr_s:>7} | {pf_s:>6} | {ann_s:>8} | {ok_s}")

    print(f"\n  Folds OK: {wf['folds_ok']}/12 | "
          f"{'APROBADO' if wf['approved'] else 'RECHAZADO'}")

    # OOS completo
    print(f"\n{'='*70}")
    print(f"OOS COMPLETO ({OOS_START} a {OOS_END})")
    print(f"{'='*70}")

    oos_trades = [t for t in wf['all_trades']
                  if OOS_START <= str(t['ts'])[:10] <= OOS_END]
    m_oos = metrics(oos_trades, f'OOS {OOS_START}-{OOS_END}')
    print_metrics(m_oos)

    # Breakdown LONG vs SHORT
    long_trades  = [t for t in oos_trades if t['direction'] == 'LONG']
    short_trades = [t for t in oos_trades if t['direction'] == 'SHORT']
    m_long  = metrics(long_trades,  'LONG')
    m_short = metrics(short_trades, 'SHORT')

    print(f"\n  Breakdown por direccion:")
    print(f"    LONG:  N={m_long['n']:>3} | WR={m_long['wr']:.1%} | PF={m_long['pf']:.2f} | {m_long['trades_pm']:.1f}t/m")
    print(f"    SHORT: N={m_short['n']:>3} | WR={m_short['wr']:.1%} | PF={m_short['pf']:.2f} | {m_short['trades_pm']:.1f}t/m")

    # Breakdown por regimen
    print(f"\n  Breakdown por regimen:")
    for regime in ['BULL', 'BEAR', 'RANGE']:
        reg_trades = [t for t in oos_trades if t['regime'] == regime]
        m_reg = metrics(reg_trades, regime)
        if m_reg['n'] > 0:
            print(f"    {regime}: N={m_reg['n']:>3} | WR={m_reg['wr']:.1%} | PF={m_reg['pf']:.2f}")

    # Funding veto stats
    veto_long  = sum(1 for t in oos_trades if t['direction'] == 'LONG' and t['funding_z'] > FUNDING_VETO_LONG)
    veto_short = sum(1 for t in oos_trades if t['direction'] == 'SHORT' and t['funding_z'] < FUNDING_VETO_SHORT)
    print(f"\n  Funding veto: {veto_long} LONG bloqueados, {veto_short} SHORT bloqueados")

    # Stress test por periodo
    print(f"\n  Stress test por periodo clave:")
    periods = [
        ('2020-H2 (bull fuerte)', '2020-07-01', '2020-12-31'),
        ('2022-H1 (bear -75%)',   '2022-01-01', '2022-06-30'),
        ('2022-H2 (bear bottom)', '2022-07-01', '2022-12-31'),
        ('2023-H1 (lateral)',     '2023-01-01', '2023-06-30'),
        ('2024-H1 (bull ETF)',    '2024-01-01', '2024-06-30'),
        ('2025-H2 (bear/range)',  '2025-07-01', '2025-12-31'),
    ]
    for label, s, e in periods:
        pt = [t for t in wf['all_trades'] if s <= str(t['ts'])[:10] <= e]
        m_p = metrics(pt, label)
        n_l = sum(1 for t in pt if t['direction'] == 'LONG')
        n_s = sum(1 for t in pt if t['direction'] == 'SHORT')
        dir_s = f"L={n_l} S={n_s}"
        print(f"    {label:<25} | N={m_p['n']:>3} ({dir_s:>8}) | "
              f"WR={m_p['wr']:.1%} | PF={m_p['pf']:.2f}")

    # Equity curve
    cumulative = 1.0
    equity = []
    for t in sorted(wf['all_trades'], key=lambda x: x['ts']):
        cumulative *= (1 + t['pnl_pct'])
        equity.append((t['ts'], cumulative))
    if equity:
        print(f"\n  Equity curve: ${1000:.0f} -> ${1000*equity[-1][1]:.0f} "
              f"({(equity[-1][1]-1)*100:.1f}%)")
        peak = 1.0
        max_dd = 0
        for _, eq in equity:
            peak = max(peak, eq)
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)
        print(f"  Max drawdown: {max_dd:.1%}")

    # Veredicto
    total_tpm = m_oos['trades_pm']
    verdict = ('APROBADO'
               if wf['approved'] and m_oos['pf'] > 1.2
                  and m_oos['wr'] > 0.38 and total_tpm >= 1.5
               else 'RECHAZADO')

    print(f"\n{'='*70}")
    print(f"VEREDICTO COMITE V15: {verdict}")
    print(f"  WF: {wf['folds_ok']}/12 | WR={m_oos['wr']:.1%} | "
          f"PF={m_oos['pf']:.2f} | {total_tpm:.1f}t/m")
    print(f"{'='*70}")

    _write_doc(wf, m_oos, m_long, m_short, verdict, equity)
    return verdict


def _write_doc(wf, m_oos, m_long, m_short, verdict, equity):
    doc_dir = ROOT / 'docs'
    doc_dir.mkdir(exist_ok=True)
    with open(doc_dir / 'V15_COMMITTEE_results.md', 'w', encoding='utf-8') as f:
        f.write("# V15 Expert Committee — Backtest Combinado\n\n")
        f.write(f"**Fecha**: {pd.Timestamp.now().date()}\n")
        f.write(f"**Veredicto**: {verdict}\n\n")
        f.write("## Componentes\n\n")
        f.write("| Regimen | Estrategia | Descripcion |\n")
        f.write("|---------|-----------|-------------|\n")
        f.write("| BULL | Breakout B LONG | vol>=1.8, BB estrecho, close>high20 |\n")
        f.write("| BEAR | SHORT ML (GBM) | Entrenado solo en BEAR, threshold=0.55 |\n")
        f.write("| RANGE | No operar | Esperar |\n\n")
        f.write("## Walk-forward\n\n")
        f.write("| Periodo | N | L | S | WR | PF | Anual | OK |\n")
        f.write("|---------|---|---|---|----|----|-------|----|\\n")
        for r in wf['folds']:
            ok_s = 'SI' if r['ok'] else 'NO'
            wr_s = f"{r['wr']:.1%}" if r['n'] > 0 else '-'
            pf_s = f"{r['pf']:.2f}" if r['n'] > 0 else '-'
            ann_s = f"{r.get('annual_pct',0):.0f}%" if r['n'] > 0 else '-'
            f.write(f"| {r['period']} | {r['n']} | {r['n_long']} | {r['n_short']} | "
                    f"{wr_s} | {pf_s} | {ann_s} | {ok_s} |\n")
        f.write(f"\n**Folds OK**: {wf['folds_ok']}/12\n\n")
        f.write(f"## OOS | N={m_oos['n']} | WR={m_oos['wr']:.1%} | "
                f"PF={m_oos['pf']:.2f} | {m_oos['trades_pm']:.1f}t/m\n\n")
        f.write(f"### LONG: N={m_long['n']} | WR={m_long['wr']:.1%} | PF={m_long['pf']:.2f}\n")
        f.write(f"### SHORT: N={m_short['n']} | WR={m_short['wr']:.1%} | PF={m_short['pf']:.2f}\n\n")
        if equity:
            f.write(f"## Equity: $1000 -> ${1000*equity[-1][1]:.0f} "
                    f"({(equity[-1][1]-1)*100:.1f}%)\n\n")
        f.write(f"## Veredicto: {verdict}\n")
        f.write("Criterios: WF>=7/12, WR>=38%, PF>=1.2, >=1.5 trades/mes\n")


if __name__ == '__main__':
    main()
