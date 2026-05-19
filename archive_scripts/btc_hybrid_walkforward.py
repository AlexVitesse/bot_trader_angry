"""
BTC Hybrid System - Walk-Forward Validation
============================================
Prueba el sistema hibrido con validacion walk-forward.
Entrena en ventana rolling, testea en siguiente periodo.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')

TP_PCT = 0.03
SL_PCT = 0.015


def load_data():
    df = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_full.parquet')
    return df.sort_index()


def compute_features(df):
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    feat = pd.DataFrame(index=df.index)

    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)

    bb = ta.bbands(c, length=20)
    if bb is not None:
        feat['bb_pct'] = (c - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0])
        feat['bb_width'] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / bb.iloc[:, 1] * 100

    ema20 = ta.ema(c, length=20)
    ema50 = ta.ema(c, length=50)
    ema200 = ta.ema(c, length=200)
    feat['ema20_dist'] = (c - ema20) / ema20 * 100
    feat['ema50_dist'] = (c - ema50) / ema50 * 100
    feat['ema200_dist'] = (c - ema200) / ema200 * 100

    adx = ta.adx(h, l, c, length=14)
    if adx is not None:
        feat['adx'] = adx.iloc[:, 0]
        feat['di_diff'] = adx.iloc[:, 1] - adx.iloc[:, 2]

    feat['atr_pct'] = ta.atr(h, l, c, length=14) / c * 100
    feat['vol_ratio'] = v / v.rolling(20).mean()

    chop = ta.chop(h, l, c, length=14)
    if chop is not None:
        feat['chop'] = chop

    feat['ret_5'] = c.pct_change(5) * 100
    feat['ret_20'] = c.pct_change(20) * 100
    feat['consec_down'] = (c < c.shift(1)).rolling(10).sum()
    feat['hour'] = df.index.hour
    feat['hour_sin'] = np.sin(2 * np.pi * feat['hour'] / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * feat['hour'] / 24)

    return feat


def detect_setups(feat):
    """Detecta setups tecnicos."""
    setups = pd.DataFrame(index=feat.index)
    setups['has_setup'] = False
    setups['setup_type'] = ''

    for idx in feat.index:
        row = feat.loc[idx]
        rsi14 = row.get('rsi14', 50)
        rsi7 = row.get('rsi7', 50)
        bb_pct = row.get('bb_pct', 0.5)
        consec_down = row.get('consec_down', 0)
        vol_ratio = row.get('vol_ratio', 1)
        ema200_dist = row.get('ema200_dist', 0)
        ema20_dist = row.get('ema20_dist', 0)
        adx = row.get('adx', 20)

        if pd.isna(rsi14) or pd.isna(bb_pct):
            continue

        if rsi14 < 25 and bb_pct < 0.3:
            setups.loc[idx, 'has_setup'] = True
            setups.loc[idx, 'setup_type'] = 'RSI_EXTREME'
        elif consec_down >= 4 and rsi14 < 35 and vol_ratio > 1.3:
            setups.loc[idx, 'has_setup'] = True
            setups.loc[idx, 'setup_type'] = 'CAPITULATION'
        elif ema200_dist > 5 and adx > 25 and rsi14 < 40 and ema20_dist < -1:
            setups.loc[idx, 'has_setup'] = True
            setups.loc[idx, 'setup_type'] = 'TREND_PULLBACK'
        elif rsi7 < 25 and rsi14 > rsi7 + 5 and bb_pct < 0.2:
            setups.loc[idx, 'has_setup'] = True
            setups.loc[idx, 'setup_type'] = 'DIVERGENCE'

    return setups


def train_ml_filter(feat, setups, df, train_mask):
    """Entrena filtro ML en periodo de training."""
    context_cols = ['adx', 'di_diff', 'chop', 'atr_pct', 'vol_ratio',
                    'bb_width', 'ret_5', 'ret_20', 'hour_sin', 'hour_cos']

    train_setups = setups[train_mask & setups['has_setup']]

    setup_results = []
    for idx in train_setups.index:
        if idx not in df.index:
            continue
        entry_price = df.loc[idx, 'close']
        future = df.loc[idx:].head(30)
        result = None
        for future_idx in future.index[1:]:
            future_price = df.loc[future_idx, 'close']
            pnl = (future_price - entry_price) / entry_price
            if pnl >= TP_PCT:
                result = 1
                break
            elif pnl <= -SL_PCT:
                result = 0
                break
        if result is not None:
            setup_results.append({'idx': idx, 'result': result})

    if len(setup_results) < 20:
        return None

    results_df = pd.DataFrame(setup_results).set_index('idx')
    available_cols = [c for c in context_cols if c in feat.columns]
    X = feat.loc[results_df.index, available_cols].dropna()
    y = results_df.loc[X.index, 'result']

    if len(y) < 20:
        return None

    model = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    model.fit(X, y)
    return model


def backtest_period(df, feat, setups, model=None, threshold=0.45):
    """Backtest de un periodo con o sin filtro ML."""
    trades = []
    position = None
    context_cols = ['adx', 'di_diff', 'chop', 'atr_pct', 'vol_ratio',
                    'bb_width', 'ret_5', 'ret_20', 'hour_sin', 'hour_cos']

    for idx in df.index:
        price = df.loc[idx, 'close']

        if position is not None:
            pnl_pct = (price - position['entry']) / position['entry']
            if pnl_pct >= TP_PCT:
                trades.append({'pnl': pnl_pct, 'result': 'WIN'})
                position = None
            elif pnl_pct <= -SL_PCT:
                trades.append({'pnl': pnl_pct, 'result': 'LOSS'})
                position = None

        if position is None and idx in setups.index and setups.loc[idx, 'has_setup']:
            should_enter = True

            if model is not None:
                available_cols = [c for c in context_cols if c in feat.columns]
                X = feat.loc[[idx], available_cols]
                if not X.isna().any().any():
                    prob = model.predict_proba(X)[0, 1]
                    should_enter = prob >= threshold

            if should_enter:
                position = {'entry': price}

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    n = len(trades_df)
    wins = (trades_df['result'] == 'WIN').sum()
    total_pnl = trades_df['pnl'].sum() * 100
    gp = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gl = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    pf = gp / gl if gl > 0 else 999

    return {
        'trades': n,
        'wins': wins,
        'wr': wins / n * 100,
        'pnl': total_pnl,
        'pf': pf
    }


def walk_forward_validation(df, feat, setups, train_months=12, test_months=3):
    """
    Walk-forward validation.
    - Entrena en ventana de N meses
    - Testea en siguientes M meses
    - Avanza M meses y repite
    """
    print("=" * 70)
    print("WALK-FORWARD VALIDATION - SISTEMA HIBRIDO BTC")
    print("=" * 70)
    print(f"\nConfiguracion:")
    print(f"  Ventana training: {train_months} meses")
    print(f"  Ventana test: {test_months} meses")
    print(f"  Threshold ML: 0.45")

    # Generar folds
    start_date = df.index.min()
    end_date = df.index.max()

    # Empezar despues de tener suficientes datos para entrenar
    first_test_start = start_date + pd.DateOffset(months=train_months)

    folds = []
    current_test_start = first_test_start

    while current_test_start + pd.DateOffset(months=test_months) <= end_date:
        train_start = current_test_start - pd.DateOffset(months=train_months)
        train_end = current_test_start - pd.Timedelta(days=1)
        test_start = current_test_start
        test_end = current_test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)

        folds.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end
        })

        current_test_start += pd.DateOffset(months=test_months)

    print(f"  Total folds: {len(folds)}")

    # Ejecutar cada fold
    results_no_filter = []
    results_with_filter = []

    print(f"\n{'Fold':<6} {'Test Period':<25} {'Sin Filtro':^20} {'Con Filtro':^20}")
    print(f"{'':6} {'':25} {'Trades WR% PnL%':^20} {'Trades WR% PnL%':^20}")
    print("-" * 75)

    for i, fold in enumerate(folds):
        train_mask = (df.index >= fold['train_start']) & (df.index <= fold['train_end'])
        test_mask = (df.index >= fold['test_start']) & (df.index <= fold['test_end'])

        df_test = df[test_mask]
        feat_test = feat[test_mask]
        setups_test = setups[test_mask]

        if len(df_test) < 10:
            continue

        # Sin filtro
        result_no_filter = backtest_period(df_test, feat_test, setups_test, model=None)

        # Con filtro
        model = train_ml_filter(feat, setups, df, train_mask)
        result_with_filter = backtest_period(df_test, feat_test, setups_test, model=model, threshold=0.45)

        period_str = f"{fold['test_start'].strftime('%Y-%m')} to {fold['test_end'].strftime('%Y-%m')}"

        if result_no_filter:
            results_no_filter.append(result_no_filter)
            nf_str = f"{result_no_filter['trades']:>3} {result_no_filter['wr']:>5.1f} {result_no_filter['pnl']:>+6.1f}"
        else:
            nf_str = "  -     -      -"

        if result_with_filter:
            results_with_filter.append(result_with_filter)
            wf_str = f"{result_with_filter['trades']:>3} {result_with_filter['wr']:>5.1f} {result_with_filter['pnl']:>+6.1f}"
        else:
            wf_str = "  -     -      -"

        print(f"{i+1:<6} {period_str:<25} {nf_str:^20} {wf_str:^20}")

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN WALK-FORWARD")
    print("=" * 70)

    if results_no_filter:
        total_trades_nf = sum(r['trades'] for r in results_no_filter)
        total_wins_nf = sum(r['wins'] for r in results_no_filter)
        total_pnl_nf = sum(r['pnl'] for r in results_no_filter)
        avg_wr_nf = total_wins_nf / total_trades_nf * 100 if total_trades_nf > 0 else 0
        positive_folds_nf = sum(1 for r in results_no_filter if r['pnl'] > 0)

        print(f"\nSIN FILTRO ML:")
        print(f"  Folds: {len(results_no_filter)} | Positivos: {positive_folds_nf} ({positive_folds_nf/len(results_no_filter)*100:.0f}%)")
        print(f"  Total trades: {total_trades_nf}")
        print(f"  WR global: {avg_wr_nf:.1f}%")
        print(f"  PnL total: {total_pnl_nf:+.1f}%")

    if results_with_filter:
        total_trades_wf = sum(r['trades'] for r in results_with_filter)
        total_wins_wf = sum(r['wins'] for r in results_with_filter)
        total_pnl_wf = sum(r['pnl'] for r in results_with_filter)
        avg_wr_wf = total_wins_wf / total_trades_wf * 100 if total_trades_wf > 0 else 0
        positive_folds_wf = sum(1 for r in results_with_filter if r['pnl'] > 0)

        print(f"\nCON FILTRO ML:")
        print(f"  Folds: {len(results_with_filter)} | Positivos: {positive_folds_wf} ({positive_folds_wf/len(results_with_filter)*100:.0f}%)")
        print(f"  Total trades: {total_trades_wf}")
        print(f"  WR global: {avg_wr_wf:.1f}%")
        print(f"  PnL total: {total_pnl_wf:+.1f}%")

    # Veredicto
    print("\n" + "=" * 70)
    if results_with_filter and results_no_filter:
        mejora_pnl = total_pnl_wf - total_pnl_nf
        print(f"Mejora del filtro ML: {mejora_pnl:+.1f}%")

        if total_pnl_wf > 0 and positive_folds_wf >= len(results_with_filter) * 0.5:
            print("[APROBADO] Sistema hibrido consistente en walk-forward")
        elif total_pnl_wf > total_pnl_nf:
            print("[MARGINAL] Filtro mejora pero no consistente")
        else:
            print("[RECHAZADO] Sistema hibrido NO funciona en walk-forward")

    print("=" * 70)

    return results_no_filter, results_with_filter


def main():
    print("Cargando datos BTC...")
    df = load_data()
    print(f"Total: {len(df):,} candles ({df.index.min().date()} a {df.index.max().date()})")

    print("\nCalculando features...")
    feat = compute_features(df)
    feat = feat.replace([np.inf, -np.inf], np.nan)

    print("Detectando setups...")
    setups = detect_setups(feat)
    n_setups = setups['has_setup'].sum()
    print(f"Setups: {n_setups} ({n_setups/len(setups)*100:.1f}%)")

    # Walk-forward con ventanas de 12 meses train, 3 meses test
    walk_forward_validation(df, feat, setups, train_months=12, test_months=3)


if __name__ == '__main__':
    main()
