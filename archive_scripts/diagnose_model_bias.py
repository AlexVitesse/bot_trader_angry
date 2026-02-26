"""
Diagnóstico del Sesgo en Modelos V95_V7
=======================================
Verificar por qué los modelos solo predicen SHORTS (retornos negativos)
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import joblib
from pathlib import Path

DATA_DIR = Path(__file__).parent / 'data'
MODELS_DIR = Path(__file__).parent / 'models'

PAIRS = ['ADA/USDT', 'XRP/USDT', 'DOT/USDT', 'DOGE/USDT',
         'ETH/USDT', 'AVAX/USDT', 'NEAR/USDT', 'LINK/USDT']


def load_data_4h(pair):
    safe = pair.replace('/', '_')
    for suffix in ['_4h_v95.parquet', '_4h_history.parquet']:
        cache = DATA_DIR / f'{safe}{suffix}'
        if cache.exists():
            df = pd.read_parquet(cache)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            return df
    return None


def load_model(pair):
    safe = pair.replace('/', '').replace('_', '')
    try:
        return joblib.load(MODELS_DIR / f'v95_v7_{safe}.pkl')
    except:
        return None


def compute_features(df):
    feat = pd.DataFrame(index=df.index)
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']

    for p in [1, 3, 5, 10, 20]:
        feat[f'ret_{p}'] = c.pct_change(p)

    feat['atr14'] = ta.atr(h, l, c, length=14)
    feat['atr_r'] = feat['atr14'] / feat['atr14'].rolling(50).mean()
    feat['vol5'] = c.pct_change().rolling(5).std()
    feat['vol20'] = c.pct_change().rolling(20).std()
    feat['rsi14'] = ta.rsi(c, length=14)
    feat['rsi7'] = ta.rsi(c, length=7)

    sr = ta.stochrsi(c, length=14, rsi_length=14, k=3, d=3)
    if sr is not None:
        feat['srsi_k'] = sr.iloc[:, 0]
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        feat['macd_h'] = macd.iloc[:, 1]

    feat['roc5'] = ta.roc(c, length=5)
    feat['roc20'] = ta.roc(c, length=20)

    for el in [8, 21, 55, 100, 200]:
        e = ta.ema(c, length=el)
        feat[f'ema{el}_d'] = (c - e) / e * 100
    feat['ema8_sl'] = ta.ema(c, length=8).pct_change(3) * 100
    feat['ema55_sl'] = ta.ema(c, length=55).pct_change(5) * 100

    bb = ta.bbands(c, length=20, std=2.0)
    if bb is not None:
        bw = bb.iloc[:, 2] - bb.iloc[:, 0]
        feat['bb_pos'] = (c - bb.iloc[:, 0]) / bw
        feat['bb_w'] = bw / bb.iloc[:, 1] * 100

    feat['vr'] = v / v.rolling(20).mean()
    feat['spr'] = (h - l) / c * 100
    feat['body'] = abs(c - df['open']) / (h - l + 1e-10)

    ax = ta.adx(h, l, c, length=14)
    if ax is not None:
        feat['adx'] = ax.iloc[:, 0]
        feat['dip'] = ax.iloc[:, 1]
        feat['dim'] = ax.iloc[:, 2]

    hr = df.index.hour
    dw = df.index.dayofweek
    feat['h_s'] = np.sin(2 * np.pi * hr / 24)
    feat['h_c'] = np.cos(2 * np.pi * hr / 24)
    feat['d_s'] = np.sin(2 * np.pi * dw / 7)
    feat['d_c'] = np.cos(2 * np.pi * dw / 7)

    return feat


def main():
    print("=" * 80, flush=True)
    print("DIAGNÓSTICO: SESGO EN MODELOS V95_V7", flush=True)
    print("=" * 80, flush=True)

    print("\nObjetivo: Entender por qué los modelos solo predicen SHORTS\n", flush=True)

    all_preds = []
    all_actual = []

    for pair in PAIRS:
        print(f"\n{'='*60}", flush=True)
        print(f"PAR: {pair}", flush=True)
        print("=" * 60, flush=True)

        df = load_data_4h(pair)
        model = load_model(pair)

        if df is None or model is None:
            print(f"  [!] Sin datos o modelo", flush=True)
            continue

        feat = compute_features(df)
        feat = feat.replace([np.inf, -np.inf], np.nan)

        fcols = [c for c in model.feature_name_ if c in feat.columns]
        valid_idx = feat.dropna().index

        X = feat.loc[valid_idx, fcols].fillna(0)
        preds = model.predict(X)

        # Retornos reales a 5 velas
        fwd_5 = (df['close'].shift(-5) - df['close']) / df['close']
        actual = fwd_5.loc[valid_idx].dropna()
        preds_aligned = pd.Series(preds, index=valid_idx).loc[actual.index]

        # Estadísticas de predicciones
        print(f"\n  PREDICCIONES DEL MODELO:", flush=True)
        print(f"    Media:     {preds.mean():.6f}", flush=True)
        print(f"    Mediana:   {np.median(preds):.6f}", flush=True)
        print(f"    Std:       {preds.std():.6f}", flush=True)
        print(f"    Min:       {preds.min():.6f}", flush=True)
        print(f"    Max:       {preds.max():.6f}", flush=True)

        # Conteo de direcciones
        n_long = (preds > 0).sum()
        n_short = (preds <= 0).sum()
        pct_long = n_long / len(preds) * 100
        pct_short = n_short / len(preds) * 100
        print(f"\n  DIRECCIONES PREDICHAS:", flush=True)
        print(f"    LONG (pred > 0):  {n_long:,} ({pct_long:.1f}%)", flush=True)
        print(f"    SHORT (pred <= 0): {n_short:,} ({pct_short:.1f}%)", flush=True)

        # Comparar con retornos reales
        print(f"\n  RETORNOS REALES (fwd_5):", flush=True)
        print(f"    Media:     {actual.mean():.6f}", flush=True)
        print(f"    Mediana:   {actual.median():.6f}", flush=True)
        print(f"    % Positivos: {(actual > 0).sum() / len(actual) * 100:.1f}%", flush=True)
        print(f"    % Negativos: {(actual <= 0).sum() / len(actual) * 100:.1f}%", flush=True)

        # Correlación predicción vs real
        corr = np.corrcoef(preds_aligned.values, actual.values)[0, 1]
        print(f"\n  CORRELACIÓN pred vs real: {corr:.3f}", flush=True)

        # Problema: el modelo predice correlacionado pero con offset
        # Calcular offset
        offset = preds_aligned.values.mean() - actual.values.mean()
        print(f"  OFFSET (pred_mean - real_mean): {offset:.6f}", flush=True)

        # Si corregimos el offset, ¿cuántos LONGS habría?
        preds_corrected = preds_aligned.values - preds_aligned.values.mean() + actual.values.mean()
        n_long_corrected = (preds_corrected > 0).sum()
        pct_long_corrected = n_long_corrected / len(preds_corrected) * 100
        print(f"\n  SI CORREGIMOS OFFSET:", flush=True)
        print(f"    LONG (pred_corr > 0): {n_long_corrected:,} ({pct_long_corrected:.1f}%)", flush=True)

        # Verificar correlación de signos (cuando pred > 0, ¿real > 0?)
        correct_sign = ((preds_aligned.values > 0) == (actual.values > 0)).sum()
        sign_accuracy = correct_sign / len(preds_aligned) * 100
        print(f"\n  PRECISIÓN DE SIGNOS: {sign_accuracy:.1f}%", flush=True)

        all_preds.extend(preds)
        all_actual.extend(actual.values)

    # Resumen global
    print("\n" + "=" * 80, flush=True)
    print("RESUMEN GLOBAL", flush=True)
    print("=" * 80, flush=True)

    all_preds = np.array(all_preds)
    all_actual = np.array(all_actual)

    print(f"\n  PREDICCIONES GLOBALES:", flush=True)
    print(f"    Media:     {all_preds.mean():.6f}", flush=True)
    print(f"    % LONG:    {(all_preds > 0).sum() / len(all_preds) * 100:.1f}%", flush=True)
    print(f"    % SHORT:   {(all_preds <= 0).sum() / len(all_preds) * 100:.1f}%", flush=True)

    print(f"\n  RETORNOS REALES GLOBALES:", flush=True)
    print(f"    Media:     {all_actual.mean():.6f}", flush=True)
    print(f"    % Positivos: {(all_actual > 0).sum() / len(all_actual) * 100:.1f}%", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("DIAGNÓSTICO", flush=True)
    print("=" * 80, flush=True)

    if all_preds.mean() < -0.001:
        print("\n  [!] SESGO CONFIRMADO: Predicciones tienen media negativa", flush=True)
        print("      El modelo predice retornos consistentemente negativos.", flush=True)
        print("\n  CAUSA PROBABLE:", flush=True)
        print("      - El modelo optimiza CORRELACIÓN, no magnitud", flush=True)
        print("      - Puede estar 'centrado' en datos con tendencia bajista", flush=True)
        print("      - El offset negativo no afecta la correlación", flush=True)

        print("\n  SOLUCIÓN PROPUESTA:", flush=True)
        print("      1. CENTRAR predicciones (restar media)", flush=True)
        print("      2. O usar threshold diferente de 0", flush=True)
        print("      3. O entrenar modelo CLASIFICADOR (1=up, 0=down)", flush=True)

        # Calcular threshold óptimo
        print("\n  ANÁLISIS DE THRESHOLD ÓPTIMO:", flush=True)
        for thresh in [0, -0.001, -0.002, -0.003, -0.005, -0.01]:
            n_long = (all_preds > thresh).sum()
            pct = n_long / len(all_preds) * 100
            print(f"      thresh={thresh:.3f}: {pct:.1f}% LONG", flush=True)
    else:
        print("\n  [?] No hay sesgo claro en las predicciones", flush=True)

    print("\n", flush=True)


if __name__ == '__main__':
    main()
