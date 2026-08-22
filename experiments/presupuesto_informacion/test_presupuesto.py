"""
Presupuesto de informacion — por que NINGUN modelo mas complejo puede funcionar.

Los seis (siete) negativos del proyecto explican QUE no funciona. Este explica
POR QUE no puede funcionar, que es lo que evita el octavo intento.

Tres bloques:
  A. Cuanta informacion independiente hay de verdad en 6,5 anos de BTC 4h.
  B. Los 5 meses posteriores al parquet (2026-03-05 -> hoy): son distintos?
     Si. Pero cuanta informacion NUEVA aportan? Ninguna.
  C. Siguen calibrados los parametros congelados con la volatilidad nueva?

Necesita red (baja los 5 meses vivos de Binance y los pega al parquet).

Uso: python experiments/presupuesto_informacion/test_presupuesto.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src import v2_engine as v2

P = {**v2.PARAMS_V2, 'f_enable_short': False}
CUT = pd.Timestamp('2026-03-05', tz='UTC')   # fin del parquet de ajuste
COLS = ['open', 'high', 'low', 'close', 'volume']


def cargar():
    """Parquet de ajuste + los meses vivos posteriores, sin duplicados."""
    import ccxt
    ex = ccxt.binanceusdm({'enableRateLimit': True})

    def bajar(tf, lim):
        r = ex.fetch_ohlcv('BTC/USDT:USDT', tf, limit=lim)
        d = pd.DataFrame(r, columns=['ts'] + COLS)
        d['ts'] = pd.to_datetime(d['ts'], unit='ms', utc=True)
        return d.set_index('ts')

    root = Path(__file__).resolve().parents[2] / 'data'
    p4 = pd.read_parquet(root / 'btcusdt_4h_v15.parquet')[COLS]
    p1 = pd.read_parquet(root / 'btcusdt_1d_v15.parquet')[COLS]
    l4, l1 = bajar('4h', 1500), bajar('1d', 700)
    return (pd.concat([p4, l4[~l4.index.isin(p4.index)]]).sort_index(),
            pd.concat([p1, l1[~l1.index.isin(p1.index)]]).sort_index())


def n_efectivo(serie, n):
    """N_eff = N*(1-rho)/(1+rho). Una serie casi congelada no aporta N muestras."""
    rho = serie.autocorr(1)
    return rho, n * (1 - rho) / (1 + rho)


def bloque_a(f, df4, df1):
    print("=== A) Cuanta informacion INDEPENDIENTE hay en 6,5 anos de BTC 4h ===\n")
    n = len(f)
    print(f"  velas 4h brutas (lo que 've' un modelo ML):   {n:,}\n")
    print(f"  pero las features estan casi congeladas:")
    for col in ['bull_1d', 'adx', 'atr_pct', 'bb_width', 'vol_ratio']:
        rho, neff = n_efectivo(f[col], n)
        print(f"    {col:<11} rho={rho:5.3f}  ->  N_eff = {neff:9,.0f}")

    ep = int((f['bull_1d'] != f['bull_1d'].shift()).sum())
    trades = v2.run_v2_backtest(df4, df1, None, P)
    gan = sum(1 for t in trades if t['pnl_pct'] > 0)
    print(f"\n  episodios de regimen distintos:              {ep}")
    print(f"  TRADES (los eventos a predecir):             {len(trades)}"
          f"   ({gan} ganan, {len(trades) - gan} pierden)")

    print(f"\n  {'':<22} {'parametros':>11} {'eventos/parametro':>19}")
    print(f"  {'V2 (PARAMS_V2)':<22} {10:>11,} {len(trades) / 10:>19.1f}")
    for arb, hojas in [(100, 31), (500, 31), (1000, 63)]:
        k = arb * hojas * 2
        print(f"  {'LightGBM %dx%d' % (arb, hojas):<22} {k:>11,} "
              f"{len(trades) / k:>19.4f}")
    return trades


def bloque_b(f, trades):
    print("\n=== B) Los 5 meses que faltaban: distintos, pero informativos? ===\n")
    tr_, te = f[f.index < CUT], f[f.index >= CUT]
    print(f"  ya visto: {len(tr_):,} velas | nuevo: {len(te):,} velas\n")
    print(f"  {'feature':<12} {'mediana antes':>14} {'mediana ahora':>14} "
          f"{'KS p':>10}  veredicto")
    for col in ['atr_pct', 'adx', 'bb_width', 'vol_ratio']:
        a, b = tr_[col].dropna(), te[col].dropna()
        pv = stats.ks_2samp(a, b).pvalue
        print(f"  {col:<12} {a.median():14.4f} {b.median():14.4f} {pv:10.2e}  "
              f"{'DISTINTO' if pv < 0.01 else 'igual'}")
    ra, rb = tr_['close'].pct_change().dropna(), te['close'].pct_change().dropna()
    ann = np.sqrt(6 * 365) * 100
    print(f"\n  vol 4h anualizada: {ra.std() * ann:.1f}% antes -> "
          f"{rb.std() * ann:.1f}% ahora")

    ep_new = int((te['bull_1d'] != te['bull_1d'].shift()).sum())
    t_new = [t for t in trades if t['ts_entry'] >= str(CUT.date())]
    print(f"\n  PERO la informacion nueva es:")
    print(f"    episodios de regimen nuevos:  {ep_new}")
    print(f"    trades nuevos de V2:          {len(t_new)}")


def bloque_c(f):
    print("\n=== C) Siguen calibrados los parametros congelados? ===\n")
    print(f"  trailing A = min(max(atr_pct*{P['a_trail_atr_mult']}, "
          f"{P['a_trail_floor_pct']:.1%}), {P['a_trail_ceiling_pct']:.1%})\n")
    print(f"  {'periodo':<22} {'atr med':>9} {'trail bruto':>12} "
          f"{'% SUELO':>9} {'% TECHO':>9}")
    for lab, sub in [('entrenamiento', f[f.index < CUT]),
                     ('los 5 meses nuevos', f[f.index >= CUT])]:
        a = sub['atr_pct'].dropna()
        raw = a * P['a_trail_atr_mult']
        print(f"  {lab:<22} {a.median() * 100:8.2f}% {raw.median() * 100:11.2f}% "
              f"{100 * (raw < P['a_trail_floor_pct']).mean():8.1f}% "
              f"{100 * (raw > P['a_trail_ceiling_pct']).mean():8.1f}%")

    print(f"\n  umbrales fijos vs la distribucion nueva:")
    a_, b_ = f[f.index < CUT], f[f.index >= CUT]
    for col, thr, name in [('adx', P['a_adx_min'], 'a_adx_min'),
                           ('vol_ratio', P['a_vol_ratio_min'], 'a_vol_ratio_min')]:
        x, y = a_[col].dropna(), b_[col].dropna()
        print(f"    {name:<16}={thr:<5} pasa {100 * (x >= thr).mean():5.1f}% "
              f"-> {100 * (y >= thr).mean():5.1f}%")
    print(f"    {'compression':<16}       activa "
          f"{a_['compression_sustained'].mean() * 100:5.1f}% -> "
          f"{b_['compression_sustained'].mean() * 100:5.1f}% (cuantil movil)")
    print(f"    {'Donchian-55':<16}       rompe "
          f"{(a_['close'] > a_['donchian_high']).mean() * 100:5.1f}% -> "
          f"{(b_['close'] > b_['donchian_high']).mean() * 100:5.1f}%")


def main() -> None:
    df4, df1 = cargar()
    print(f"Serie unida 4h: {df4.index[0].date()} -> {df4.index[-1].date()} "
          f"({len(df4):,} velas)\n")
    f = v2.build_features(df4, df1, None, P)
    trades = bloque_a(f, df4, df1)
    bloque_b(f, trades)
    bloque_c(f)


if __name__ == '__main__':
    main()
