"""
SOL/USDT - Experimentos Avanzados de Mejora
Ideas:
1. Clasificador binario WIN/LOSS
2. Features de correlacion BTC/ETH
3. XGBoost y CatBoost
4. Ensemble de modelos
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import ccxt
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_DIR = Path('data')
MODELS_DIR = Path('models')


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """54 features base."""
    feat = pd.DataFrame(index=df.index)
    c, h, l, o, v = df['close'], df['high'], df['low'], df['open'], df['volume']

    for p in [1, 2, 3, 5, 10, 20, 50]:
        feat[f'ret_{p}'] = c.pct_change(p)

    feat['atr14'] = ta.atr(h, l, c, length=14)
    feat['atr_r'] = feat['atr14'] / feat['atr14'].rolling(50).mean()
    feat['vol5'] = c.pct_change().rolling(5).std()
    feat['vol20'] = c.pct_change().rolling(20).std()
    feat['vol_ratio'] = feat['vol5'] / (feat['vol20'] + 1e-10)

    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)
    feat['rsi21'] = ta.rsi(c, length=21)

    sr = ta.stochrsi(c, length=14, rsi_length=14, k=3, d=3)
    if sr is not None:
        feat['srsi_k'] = sr.iloc[:, 0]
        feat['srsi_d'] = sr.iloc[:, 1]

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        feat['macd'] = macd.iloc[:, 0]
        feat['macd_h'] = macd.iloc[:, 1]
        feat['macd_s'] = macd.iloc[:, 2]

    feat['roc5'] = ta.roc(c, length=5)
    feat['roc10'] = ta.roc(c, length=10)
    feat['roc20'] = ta.roc(c, length=20)

    for el in [8, 21, 55, 100, 200]:
        e = ta.ema(c, length=el)
        feat[f'ema{el}_d'] = (c - e) / e * 100

    feat['ema8_sl'] = ta.ema(c, length=8).pct_change(3) * 100
    feat['ema21_sl'] = ta.ema(c, length=21).pct_change(5) * 100
    feat['ema55_sl'] = ta.ema(c, length=55).pct_change(5) * 100

    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        bw = bb.iloc[:, 2] - bb.iloc[:, 0]
        feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw
        feat['bb_w'] = bw / bb.iloc[:, 1] * 100

    feat['vr'] = v / v.rolling(20).mean()
    feat['vr5'] = v.rolling(5).mean() / v.rolling(20).mean()

    feat['spr'] = (h - l) / c * 100
    feat['body'] = abs(c - o) / (h - l + 1e-10)
    feat['upper_wick'] = (h - np.maximum(c, o)) / (h - l + 1e-10)
    feat['lower_wick'] = (np.minimum(c, o) - l) / (h - l + 1e-10)

    ax = ta.adx(h, l, c, length=14)
    if ax is not None:
        feat['adx'] = ax.iloc[:, 0]
        feat['dip'] = ax.iloc[:, 1]
        feat['dim'] = ax.iloc[:, 2]
        feat['di_diff'] = feat['dip'] - feat['dim']

    chop = ta.chop(h, l, c, length=14)
    if chop is not None:
        feat['chop'] = chop

    hr = df.index.hour
    dw = df.index.dayofweek
    feat['h_s'] = np.sin(2 * np.pi * hr / 24)
    feat['h_c'] = np.cos(2 * np.pi * hr / 24)
    feat['d_s'] = np.sin(2 * np.pi * dw / 7)
    feat['d_c'] = np.cos(2 * np.pi * dw / 7)

    feat['ret1_lag1'] = feat['ret_1'].shift(1)
    feat['rsi14_lag1'] = feat['rsi14'].shift(1)
    feat['ret1_lag2'] = feat['ret_1'].shift(2)
    feat['rsi14_lag2'] = feat['rsi14'].shift(2)
    feat['ret1_lag3'] = feat['ret_1'].shift(3)
    feat['rsi14_lag3'] = feat['rsi14'].shift(3)

    return feat


def detect_regime(df: pd.DataFrame) -> pd.Series:
    c = df['close']
    ema20 = ta.ema(c, length=20)
    ema50 = ta.ema(c, length=50)
    ret20 = c.pct_change(20)

    regime = pd.Series('RANGE', index=df.index)
    bull = (c > ema50) & (ema20 > ema50) & (ret20 > 0.05)
    bear = (c < ema50) & (ema20 < ema50) & (ret20 < -0.05)
    regime[bull] = 'BULL'
    regime[bear] = 'BEAR'

    return regime


def download_btc_eth_data():
    """Descarga datos de BTC y ETH para correlacion."""
    exchange = ccxt.binance({'enableRateLimit': True})

    for pair in ['BTC/USDT', 'ETH/USDT']:
        safe = pair.replace('/', '_')
        path = DATA_DIR / f'{safe}_4h_full.parquet'

        if path.exists():
            print(f"   {pair} ya existe")
            continue

        print(f"   Descargando {pair}...")
        all_data = []
        since = exchange.parse8601('2019-01-01T00:00:00Z')

        while True:
            ohlcv = exchange.fetch_ohlcv(pair, '4h', since=since, limit=1000)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(ohlcv) < 1000:
                break

        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.to_parquet(path)
        print(f"   {pair}: {len(df)} velas")


def add_correlation_features(sol_feat: pd.DataFrame, sol_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features de correlacion con BTC y ETH."""
    # Cargar BTC y ETH
    btc = pd.read_parquet(DATA_DIR / 'BTC_USDT_4h_full.parquet')
    eth = pd.read_parquet(DATA_DIR / 'ETH_USDT_4h_full.parquet')

    # Normalizar timezones (remover timezone info si existe)
    if btc.index.tz is not None:
        btc.index = btc.index.tz_localize(None)
    if eth.index.tz is not None:
        eth.index = eth.index.tz_localize(None)
    if sol_df.index.tz is not None:
        sol_df = sol_df.copy()
        sol_df.index = sol_df.index.tz_localize(None)
    if sol_feat.index.tz is not None:
        sol_feat = sol_feat.copy()
        sol_feat.index = sol_feat.index.tz_localize(None)

    # Alinear indices
    common_idx = sol_df.index.intersection(btc.index).intersection(eth.index)

    feat = sol_feat.loc[common_idx].copy()

    # Returns de BTC y ETH
    btc_ret = btc['close'].pct_change()
    eth_ret = eth['close'].pct_change()
    sol_ret = sol_df['close'].pct_change()

    # Features de correlacion
    feat['btc_ret_1'] = btc_ret.loc[common_idx]
    feat['btc_ret_5'] = btc['close'].pct_change(5).loc[common_idx]
    feat['btc_ret_20'] = btc['close'].pct_change(20).loc[common_idx]

    feat['eth_ret_1'] = eth_ret.loc[common_idx]
    feat['eth_ret_5'] = eth['close'].pct_change(5).loc[common_idx]

    # RSI de BTC
    feat['btc_rsi14'] = ta.rsi(btc['close'], length=14).loc[common_idx]

    # Correlacion rolling SOL-BTC
    corr_20 = sol_ret.rolling(20).corr(btc_ret)
    feat['sol_btc_corr_20'] = corr_20.loc[common_idx]

    # Beta SOL vs BTC (sensibilidad)
    cov = sol_ret.rolling(20).cov(btc_ret)
    var = btc_ret.rolling(20).var()
    feat['sol_btc_beta'] = (cov / var).loc[common_idx]

    # Ratio SOL/BTC
    sol_btc_ratio = sol_df['close'] / btc['close']
    feat['sol_btc_ratio_ret'] = sol_btc_ratio.pct_change(5).loc[common_idx]

    # Divergencia: SOL sube cuando BTC baja
    feat['sol_btc_divergence'] = (sol_ret - btc_ret).rolling(5).mean().loc[common_idx]

    return feat


def backtest_classifier(df, feat, model, scaler, feature_cols,
                        tp_pct, sl_pct, prob_threshold=0.5,
                        only_short=False, start_date=None):
    """Backtest con clasificador."""

    X = feat[feature_cols].copy()
    if start_date:
        X = X[X.index >= start_date]

    valid = X.notna().all(axis=1)
    X = X[valid]
    df_bt = df.loc[X.index].copy()

    X_scaled = scaler.transform(X)

    # Clasificador predice probabilidad de WIN
    probs = model.predict_proba(X_scaled)[:, 1]

    regime = detect_regime(df_bt)

    trades = []
    for i, (idx, row) in enumerate(X.iterrows()):
        if i >= len(X) - 5:
            break

        prob = probs[i]
        reg = regime.iloc[i]

        if prob < prob_threshold:
            continue

        # Direccion basada en features (usar ret_1 como proxy)
        direction = -1  # Default SHORT para SOL

        if only_short and direction == 1:
            continue

        # Regime filter
        if reg == 'BULL' and direction == -1:
            continue
        if reg == 'BEAR' and direction == 1:
            continue

        entry_price = df_bt.loc[idx, 'close']

        if direction == 1:
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
        else:
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)

        exit_price = None
        exit_reason = 'timeout'

        future_idx = X.index.get_loc(idx)
        for j in range(1, min(21, len(df_bt) - future_idx)):
            bar = df_bt.iloc[future_idx + j]
            high, low = bar['high'], bar['low']

            if direction == 1:
                if low <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'sl'
                    break
                elif high >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'
                    break
            else:
                if high >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'sl'
                    break
                elif low <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'
                    break

        if exit_price is None:
            exit_price = df_bt.iloc[min(future_idx + 20, len(df_bt) - 1)]['close']

        if direction == 1:
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        trades.append({
            'entry_time': idx,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'prob': prob,
            'regime': reg,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_pct * 100,
        })

    if not trades:
        return {'trades': 0, 'wr': 0, 'pnl': 0}

    trades_df = pd.DataFrame(trades)
    wins = (trades_df['pnl_pct'] > 0).sum()
    total = len(trades_df)
    wr = wins / total
    pnl = trades_df['pnl_usd'].sum()

    return {'trades': total, 'wr': wr, 'pnl': pnl, 'trades_df': trades_df}


def backtest_regressor(df, feat, model, scaler, feature_cols, pred_std,
                       tp_pct, sl_pct, conv_min=0.5,
                       only_short=False, start_date=None):
    """Backtest con regresor."""

    X = feat[feature_cols].copy()
    if start_date:
        X = X[X.index >= start_date]

    valid = X.notna().all(axis=1)
    X = X[valid]
    df_bt = df.loc[X.index].copy()

    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    conv = np.abs(preds) / pred_std

    regime = detect_regime(df_bt)

    trades = []
    for i, (idx, row) in enumerate(X.iterrows()):
        if i >= len(X) - 5:
            break

        pred = preds[i]
        c = conv[i]
        reg = regime.iloc[i]

        if c < conv_min:
            continue

        direction = 1 if pred > 0 else -1

        if only_short and direction == 1:
            continue

        if reg == 'BULL' and direction == -1:
            continue
        if reg == 'BEAR' and direction == 1:
            continue

        entry_price = df_bt.loc[idx, 'close']

        if direction == 1:
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
        else:
            tp_price = entry_price * (1 - tp_pct)
            sl_price = entry_price * (1 + sl_pct)

        exit_price = None
        exit_reason = 'timeout'

        future_idx = X.index.get_loc(idx)
        for j in range(1, min(21, len(df_bt) - future_idx)):
            bar = df_bt.iloc[future_idx + j]
            high, low = bar['high'], bar['low']

            if direction == 1:
                if low <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'sl'
                    break
                elif high >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'
                    break
            else:
                if high >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'sl'
                    break
                elif low <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'
                    break

        if exit_price is None:
            exit_price = df_bt.iloc[min(future_idx + 20, len(df_bt) - 1)]['close']

        if direction == 1:
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        trades.append({
            'entry_time': idx,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'conviction': c,
            'regime': reg,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_pct * 100,
        })

    if not trades:
        return {'trades': 0, 'wr': 0, 'pnl': 0}

    trades_df = pd.DataFrame(trades)
    wins = (trades_df['pnl_pct'] > 0).sum()
    total = len(trades_df)
    wr = wins / total
    pnl = trades_df['pnl_usd'].sum()

    return {'trades': total, 'wr': wr, 'pnl': pnl}


# =============================================================================
# EXPERIMENTO 1: Clasificador WIN/LOSS
# =============================================================================
def experiment_classifier():
    """Entrena clasificador binario WIN/LOSS."""
    print("\n" + "="*70)
    print("EXPERIMENTO 1: Clasificador WIN/LOSS")
    print("="*70)

    df = pd.read_parquet(DATA_DIR / 'SOL_USDT_4h_full.parquet')
    feat = compute_features(df)

    # Target: WIN (1) o LOSS (0) basado en retorno 5 velas
    # Consideramos WIN si el trade SHORT gana (precio baja)
    future_ret = df['close'].pct_change(5).shift(-5)

    # Para SHORT: WIN si precio baja (future_ret < 0)
    target = (future_ret < 0).astype(int)

    valid = feat.notna().all(axis=1) & target.notna()
    feat = feat[valid]
    target = target[valid]
    df_clean = df[valid].copy()

    # Split
    train_end = '2025-10-31'
    train_mask = feat.index <= train_end
    test_mask = feat.index > train_end

    X_train, y_train = feat[train_mask], target[train_mask]
    X_test, y_test = feat[test_mask], target[test_mask]

    print(f"\n   Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"   Train WIN rate: {y_train.mean()*100:.1f}%")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Probar varios clasificadores
    classifiers = {
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_split=20,
            random_state=42
        ),
    }

    results = []
    for name, clf in classifiers.items():
        print(f"\n   Entrenando {name}...")
        clf.fit(X_train_s, y_train)

        # Metricas
        train_pred = clf.predict(X_train_s)
        test_pred = clf.predict(X_test_s)
        test_prob = clf.predict_proba(X_test_s)[:, 1]

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        try:
            test_auc = roc_auc_score(y_test, test_prob)
        except:
            test_auc = 0.5

        print(f"   {name}: Train Acc={train_acc*100:.1f}%, Test Acc={test_acc*100:.1f}%, AUC={test_auc:.3f}")

        # Backtest con diferentes thresholds
        print(f"   Backtest (ultimos 3 meses):")
        for prob_thresh in [0.5, 0.55, 0.6, 0.65]:
            res = backtest_classifier(
                df_clean, feat, clf, scaler, list(feat.columns),
                tp_pct=0.05, sl_pct=0.025, prob_threshold=prob_thresh,
                only_short=True, start_date='2025-12-01'
            )
            print(f"      P>{prob_thresh}: {res['trades']} trades | {res['wr']*100:.1f}% WR | ${res['pnl']:+.2f}")
            results.append({
                'model': name,
                'prob_thresh': prob_thresh,
                'trades': res['trades'],
                'wr': res['wr'],
                'pnl': res['pnl'],
            })

    return pd.DataFrame(results)


# =============================================================================
# EXPERIMENTO 2: Features de Correlacion BTC/ETH
# =============================================================================
def experiment_correlation_features():
    """Agrega features de correlacion con BTC y ETH."""
    print("\n" + "="*70)
    print("EXPERIMENTO 2: Features de Correlacion BTC/ETH")
    print("="*70)

    # Descargar datos BTC/ETH si no existen
    print("\n   Verificando datos BTC/ETH...")
    download_btc_eth_data()

    df = pd.read_parquet(DATA_DIR / 'SOL_USDT_4h_full.parquet')
    feat_base = compute_features(df)

    # Agregar features de correlacion
    print("\n   Agregando features de correlacion...")
    feat = add_correlation_features(feat_base, df)
    print(f"   Features totales: {len(feat.columns)}")

    # Target
    future_ret = df['close'].pct_change(5).shift(-5)

    valid = feat.notna().all(axis=1) & future_ret.notna()
    feat = feat[valid]
    target = future_ret[valid]
    df_clean = df.loc[feat.index].copy()

    # Split
    train_end = '2025-10-31'
    train_mask = feat.index <= train_end
    test_mask = feat.index > train_end

    X_train, y_train = feat[train_mask], target[train_mask]
    X_test, y_test = feat[test_mask], target[test_mask]

    print(f"\n   Train: {len(X_train)} | Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Entrenar modelo
    model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=42
    )
    model.fit(X_train_s, y_train)

    pred_train = model.predict(X_train_s)
    pred_std = np.std(pred_train)

    # Feature importance de features nuevos
    importance = pd.DataFrame({
        'feature': feat.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n   Top 15 features (incluyendo correlacion):")
    for _, row in importance.head(15).iterrows():
        marker = " <-- NUEVO" if 'btc' in row['feature'] or 'eth' in row['feature'] or 'sol_btc' in row['feature'] else ""
        print(f"      {row['feature']}: {row['importance']:.4f}{marker}")

    # Backtest
    print("\n   Backtest (ultimos 3 meses):")
    for conv in [0.3, 0.5, 0.7, 1.0]:
        res = backtest_regressor(
            df_clean, feat, model, scaler, list(feat.columns), pred_std,
            tp_pct=0.05, sl_pct=0.025, conv_min=conv,
            only_short=True, start_date='2025-12-01'
        )
        print(f"      Conv >= {conv}: {res['trades']} trades | {res['wr']*100:.1f}% WR | ${res['pnl']:+.2f}")

    # Guardar modelo si es bueno
    return model, scaler, list(feat.columns), pred_std


# =============================================================================
# EXPERIMENTO 3: XGBoost y CatBoost
# =============================================================================
def experiment_other_algorithms():
    """Prueba XGBoost y CatBoost."""
    print("\n" + "="*70)
    print("EXPERIMENTO 3: XGBoost y CatBoost")
    print("="*70)

    df = pd.read_parquet(DATA_DIR / 'SOL_USDT_4h_full.parquet')
    feat = compute_features(df)

    future_ret = df['close'].pct_change(5).shift(-5)

    valid = feat.notna().all(axis=1) & future_ret.notna()
    feat = feat[valid]
    target = future_ret[valid]
    df_clean = df[valid].copy()

    train_end = '2025-10-31'
    train_mask = feat.index <= train_end

    X_train, y_train = feat[train_mask], target[train_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    results = []

    # XGBoost
    try:
        from xgboost import XGBRegressor
        print("\n   Entrenando XGBoost...")

        xgb = XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42, verbosity=0
        )
        xgb.fit(X_train_s, y_train)
        pred_std = np.std(xgb.predict(X_train_s))

        print("   Backtest XGBoost:")
        for conv in [0.3, 0.5, 0.7]:
            res = backtest_regressor(
                df_clean, feat, xgb, scaler, list(feat.columns), pred_std,
                tp_pct=0.05, sl_pct=0.025, conv_min=conv,
                only_short=True, start_date='2025-12-01'
            )
            print(f"      Conv >= {conv}: {res['trades']} trades | {res['wr']*100:.1f}% WR | ${res['pnl']:+.2f}")
            results.append({'model': 'XGBoost', 'conv': conv, **res})

    except ImportError:
        print("   XGBoost no instalado")

    # CatBoost
    try:
        from catboost import CatBoostRegressor
        print("\n   Entrenando CatBoost...")

        cat = CatBoostRegressor(
            iterations=200, learning_rate=0.05, depth=4,
            random_state=42, verbose=0
        )
        cat.fit(X_train_s, y_train)
        pred_std = np.std(cat.predict(X_train_s))

        print("   Backtest CatBoost:")
        for conv in [0.3, 0.5, 0.7]:
            res = backtest_regressor(
                df_clean, feat, cat, scaler, list(feat.columns), pred_std,
                tp_pct=0.05, sl_pct=0.025, conv_min=conv,
                only_short=True, start_date='2025-12-01'
            )
            print(f"      Conv >= {conv}: {res['trades']} trades | {res['wr']*100:.1f}% WR | ${res['pnl']:+.2f}")
            results.append({'model': 'CatBoost', 'conv': conv, **res})

    except ImportError:
        print("   CatBoost no instalado")

    return results


# =============================================================================
# EXPERIMENTO 4: Ensemble de Modelos
# =============================================================================
def experiment_ensemble():
    """Ensemble de multiples modelos."""
    print("\n" + "="*70)
    print("EXPERIMENTO 4: Ensemble de Modelos")
    print("="*70)

    df = pd.read_parquet(DATA_DIR / 'SOL_USDT_4h_full.parquet')
    feat = compute_features(df)

    future_ret = df['close'].pct_change(5).shift(-5)

    valid = feat.notna().all(axis=1) & future_ret.notna()
    feat = feat[valid]
    target = future_ret[valid]
    df_clean = df[valid].copy()

    train_end = '2025-10-31'
    train_mask = feat.index <= train_end
    test_mask = feat.index > train_end

    X_train, y_train = feat[train_mask], target[train_mask]
    X_test = feat[test_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Entrenar multiples modelos
    print("\n   Entrenando modelos base...")

    models = {
        'GB_deep': GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42
        ),
        'GB_shallow': GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=42
        ),
        'GB_slow': GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.02, max_depth=4,
            subsample=0.8, random_state=42
        ),
    }

    predictions = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        predictions[name] = model.predict(X_test_s)
        print(f"      {name} entrenado")

    # Ensemble promedio
    print("\n   Probando estrategias de ensemble...")

    # 1. Promedio simple
    pred_avg = np.mean([predictions[m] for m in models], axis=0)
    pred_std = np.std(pred_avg)

    # Crear modelo wrapper para backtest
    class EnsembleModel:
        def __init__(self, models, scaler):
            self.models = models
            self.scaler = scaler

        def predict(self, X):
            preds = []
            for model in self.models.values():
                preds.append(model.predict(X))
            return np.mean(preds, axis=0)

    ensemble = EnsembleModel(models, scaler)

    print("\n   Backtest Ensemble Promedio:")
    for conv in [0.3, 0.5, 0.7]:
        res = backtest_regressor(
            df_clean, feat, ensemble, scaler, list(feat.columns), pred_std,
            tp_pct=0.05, sl_pct=0.025, conv_min=conv,
            only_short=True, start_date='2025-12-01'
        )
        print(f"      Conv >= {conv}: {res['trades']} trades | {res['wr']*100:.1f}% WR | ${res['pnl']:+.2f}")

    # 2. Votacion: solo operar si todos los modelos coinciden en direccion
    print("\n   Backtest Votacion (todos coinciden):")
    test_df = df_clean[test_mask].copy()
    test_feat = feat[test_mask].copy()

    # Simular votacion
    votes = []
    for name, preds in predictions.items():
        votes.append(np.sign(preds))

    votes = np.array(votes)
    agreement = np.abs(votes.sum(axis=0)) == len(models)  # Todos coinciden

    # Solo considerar donde hay acuerdo
    agreed_idx = test_feat.index[agreement]
    print(f"      Trades con acuerdo total: {len(agreed_idx)} de {len(test_feat)}")

    if len(agreed_idx) > 0:
        # Filtrar por fecha reciente
        recent_agreed = [idx for idx in agreed_idx if idx >= pd.Timestamp('2025-12-01')]
        print(f"      Trades recientes con acuerdo: {len(recent_agreed)}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("SOL/USDT - EXPERIMENTOS AVANZADOS")
    print("="*70)

    # Baseline para comparar
    print("\n[BASELINE] Modelo V2 actual (conv >= 0.5, solo SHORT):")
    df = pd.read_parquet(DATA_DIR / 'SOL_USDT_4h_full.parquet')
    feat = compute_features(df)
    model_data = joblib.load(MODELS_DIR / 'sol_usdt_v2_gradientboosting.pkl')

    res = backtest_regressor(
        df, feat, model_data['model'], model_data['scaler'],
        model_data['feature_cols'], model_data['pred_std'],
        tp_pct=0.05, sl_pct=0.025, conv_min=0.5,
        only_short=True, start_date='2025-12-01'
    )
    print(f"   {res['trades']} trades | {res['wr']*100:.1f}% WR | ${res['pnl']:+.2f}")
    baseline_pnl = res['pnl']

    # Experimentos
    clf_results = experiment_classifier()
    corr_results = experiment_correlation_features()
    algo_results = experiment_other_algorithms()
    experiment_ensemble()

    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DE EXPERIMENTOS")
    print("="*70)
    print(f"\nBaseline (V2 conv>=0.5 SHORT): ${baseline_pnl:+.2f}")

    print("\nMejores resultados por experimento:")
    print("-" * 50)
