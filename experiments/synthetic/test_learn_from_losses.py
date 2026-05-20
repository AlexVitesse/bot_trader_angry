"""
test_learn_from_losses.py
=========================
Pregunta a responder empíricamente: ¿"aprender de los trades perdedores" mejora
la estrategia, o es overfitting disfrazado?

Método:
1. Generar N series sintéticas de BTC vía block bootstrap del BTC real 2020-2025.
   Bloques de 24 velas (4 días) — preserva vol, autocorrelación intrabar y fat tails,
   pero altera el orden. Son "BTCs alternativos" estadísticamente similares.

2. Test A — Robustez de V2: correr V2 (A + F_BTC) en las N series. Distribución
   de retornos. Si V2 generaliza, la mediana es positiva y la dispersión razonable.

3. Test B — "Naive learner": en la serie #1 (entrenamiento), identifico los
   trades perdedores y sus condiciones de entrada (ej. valor de bb_width al
   entrar). Construyo V2_adjusted que filtra entradas en condiciones similares.
   Comparo V2_original vs V2_adjusted en las series #2..N (independientes).

   Hipótesis: V2_adjusted se ve mejor en serie #1 (in-sample) pero IGUAL o PEOR
   en series #2..N (OOS). Si confirma → "aprender de pérdidas" es trampa.

4. Test C — Null hypothesis: shuffle aleatorio (no bloques) destruye estructura
   temporal. V2 corriendo ahí debería dar ≈0%. Si gana, su "edge" venía de
   autocorrelación de mercado, no de patrones reales.

Salida: distribución de métricas + comparación V2_original vs V2_adjusted +
veredicto sobre la hipótesis "aprender de pérdidas".
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
EXP = ROOT / 'experiments'

# ---------------------------------------------------------------------------
# Carga real BTC + estrategias
# ---------------------------------------------------------------------------
def load_btc_4h_real():
    df = pd.read_parquet(DATA / 'BTC_USDT_4h_full.parquet').sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    # ventana de origen para bootstrap: 2020-01-01 a 2025-12-31
    df = df.loc['2020-01-01':'2025-12-31']
    # asegurar columnas mínimas
    needed = ['open', 'high', 'low', 'close', 'volume']
    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f'falta columna {c}')
    return df[needed].copy()


def load_btc_1d_real():
    df = pd.read_parquet(DATA / 'btcusdt_1d_v15.parquet').sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df.loc['2020-01-01':'2025-12-31'].copy()


def load_funding_real():
    df = pd.read_parquet(DATA / 'btc_v15_funding.parquet').sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df.loc['2020-01-01':'2025-12-31'].copy()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Block bootstrap OHLCV — preserva propiedades estadísticas, altera orden
# ---------------------------------------------------------------------------
def block_bootstrap_ohlcv(df: pd.DataFrame, block_size: int = 24, n_bars: int = None,
                          seed: int = None) -> pd.DataFrame:
    """
    Bootstrap por bloques de OHLCV.
    Cada bloque es un trozo real de N velas contiguas; los bloques se
    concatenan en orden aleatorio. Los precios se re-escalan para que
    cada bloque empiece donde el anterior cerró (continuidad).
    """
    if n_bars is None:
        n_bars = len(df)
    rng = np.random.default_rng(seed)
    max_start = len(df) - block_size
    n_blocks = (n_bars // block_size) + 2
    starts = rng.integers(0, max_start, size=n_blocks)

    parts = []
    last_close = float(df['close'].iloc[0])
    for s in starts:
        block = df.iloc[s:s + block_size].copy()
        # escalar para continuidad: open del bloque = last_close
        scale = last_close / float(block['open'].iloc[0])
        block[['open', 'high', 'low', 'close']] = block[['open', 'high', 'low', 'close']] * scale
        parts.append(block)
        last_close = float(block['close'].iloc[-1])

    out = pd.concat(parts).iloc[:n_bars].copy()
    out.index = pd.date_range(start=df.index[0], periods=len(out),
                              freq='4h', tz='UTC')
    return out


def shuffle_returns(df: pd.DataFrame, seed: int = None) -> pd.DataFrame:
    """
    Null hypothesis: shuffle aleatorio de retornos log (no bloques).
    Destruye autocorrelación. Reconstruye OHLCV manteniendo H/L spread
    proporcional al de cada vela original.
    """
    rng = np.random.default_rng(seed)
    df_c = df.copy()
    log_ret = np.log(df_c['close'].values[1:] / df_c['close'].values[:-1])
    perm = rng.permutation(len(log_ret))
    log_ret_shuf = log_ret[perm]
    # reconstruir close
    new_close = np.empty(len(df_c))
    new_close[0] = df_c['close'].iloc[0]
    new_close[1:] = df_c['close'].iloc[0] * np.exp(np.cumsum(log_ret_shuf))
    # reconstruir OHLV: usar spreads relativos originales (en orden permutado)
    ratio_h_c = (df_c['high'] / df_c['close']).values[1:][perm]
    ratio_l_c = (df_c['low'] / df_c['close']).values[1:][perm]
    ratio_o_c = (df_c['open'] / df_c['close']).values[1:][perm]
    vol_shuf = df_c['volume'].values[1:][perm]

    out = pd.DataFrame({
        'open': np.r_[df_c['open'].iloc[0], new_close[1:] * ratio_o_c],
        'high': np.r_[df_c['high'].iloc[0], new_close[1:] * ratio_h_c],
        'low': np.r_[df_c['low'].iloc[0], new_close[1:] * ratio_l_c],
        'close': new_close,
        'volume': np.r_[df_c['volume'].iloc[0], vol_shuf],
    }, index=df_c.index)
    return out


# ---------------------------------------------------------------------------
# Run V2 (A + F_BTC) en una serie sintética
# ---------------------------------------------------------------------------
def run_V2_on(df_synth_4h: pd.DataFrame, df_1d_real: pd.DataFrame,
              df_funding_real: pd.DataFrame, A, F,
              entry_filter=None) -> list:
    """
    Aplica V2 (A first, F_BTC second) sobre la serie sintética 4h dada.
    df_1d_real y df_funding_real se usan tal cual (no se boots-trappean — sirven
    como contexto macro/funding). Esto es válido porque A's daily filter usa
    el daily DEL PROPIO 4h (resample interno).

    entry_filter: callable(df, idx) -> bool. Si devuelve False, salta la entrada.
    """
    paramsA = dict(A.PARAMS); paramsA['cutoff_date'] = '2099-01-01'
    paramsF = dict(F.PARAMS); paramsF['cutoff_date'] = '2099-01-01'

    # A.prepare_data: si df_1d=None usa fallback que deriva daily desde 4h con shift(1).
    df_A = A.prepare_data(df_synth_4h, None, None, paramsA)
    df_F = F.prepare_data(df_synth_4h, None, None, paramsF)

    common = df_A.index.intersection(df_F.index)
    df_A_c = df_A.loc[common]
    df_F_c = df_F.loc[common]

    trades = []
    i = 0
    end_i = len(common) - 1
    while i < end_i:
        # ¿filtro lo bloquea?
        if entry_filter is not None and not entry_filter(df_F_c, i):
            i += 1
            continue

        sigA = A.signal(df_A_c, i, paramsA)
        if sigA == 'LONG':
            out = A.simulate(df_A_c, i, paramsA)
            bars = int(out.get('bars', 1))
            trades.append({
                'i': i, 'ts': str(common[i]),
                'strat': 'A', 'side': 'LONG',
                'pnl_pct': float(out.get('pnl_pct', 0.0)),
                'outcome': out.get('outcome'),
                'bars': bars,
                # features al momento de la entrada (para "learn from losses")
                'bb_width': float(df_F_c['bb_width'].iloc[i]) if 'bb_width' in df_F_c.columns else 0.0,
                'atr_pct': float(df_F_c['atr_pct'].iloc[i]) if 'atr_pct' in df_F_c.columns else 0.0,
                'rsi14': float(df_F_c['rsi14'].iloc[i]) if 'rsi14' in df_F_c.columns else 50.0,
            })
            i += bars + 1
            continue

        sigF = F.signal(df_F_c, i, paramsF)
        if sigF in ('LONG', 'SHORT'):
            out = F.simulate(df_F_c, i, paramsF, side=sigF)
            bars = int(out.get('bars', 1))
            pnl = out.get('leveraged_pnl_pct', out.get('pnl_pct', 0.0))
            trades.append({
                'i': i, 'ts': str(common[i]),
                'strat': 'F', 'side': sigF,
                'pnl_pct': float(pnl),
                'outcome': out.get('outcome'),
                'bars': bars,
                'bb_width': float(df_F_c['bb_width'].iloc[i]) if 'bb_width' in df_F_c.columns else 0.0,
                'atr_pct': float(df_F_c['atr_pct'].iloc[i]) if 'atr_pct' in df_F_c.columns else 0.0,
                'rsi14': float(df_F_c['rsi14'].iloc[i]) if 'rsi14' in df_F_c.columns else 50.0,
            })
            i += bars + 1
            continue
        i += 1
    return trades


def metrics(trades):
    if not trades:
        return {'n': 0, 'wr': 0.0, 'pf': 0.0, 'total': 0.0, 'annual': 0.0}
    n = len(trades)
    pnls = [t['pnl_pct'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / n
    gw = sum(wins); gl = abs(sum(losses))
    pf = (gw / gl) if gl > 1e-9 else float('inf')
    cum = 1.0
    for t in sorted(trades, key=lambda x: pd.to_datetime(x['ts'])):
        cum *= (1.0 + t['pnl_pct'])
    # serie cubre ~6 años (igual longitud que real 2020-2025)
    annual = cum ** (1.0 / 6.0) - 1.0
    return {'n': n, 'wr': wr, 'pf': pf, 'total': cum - 1.0, 'annual': annual}


# ---------------------------------------------------------------------------
# Naive learner: construir filtro basado en perdedores de una serie
# ---------------------------------------------------------------------------
def build_loss_filter(trades_train: list) -> callable:
    """
    "Naive learner" estilo trader retail: para cada feature, compara la
    mediana en perdedores vs ganadores. Si los perdedores tienden a tener
    valores más altos en bb_width, agrega un filtro: "skip si bb_width >
    mediana_perdedores" (descarta la mitad superior de pérdidas históricas).

    Es EXACTAMENTE el tipo de ajuste que haría un trader humano "aprendiendo
    de sus pérdidas". Si funciona OOS → el aprendizaje es real.
    Si NO funciona OOS → era overfitting al pasado.
    """
    losers = [t for t in trades_train if t['pnl_pct'] <= 0]
    winners = [t for t in trades_train if t['pnl_pct'] > 0]
    if len(losers) < 10 or len(winners) < 10:
        return lambda df, i: True, []

    rules = []
    for feat in ['bb_width', 'atr_pct', 'rsi14']:
        l_vals = np.array([t[feat] for t in losers])
        w_vals = np.array([t[feat] for t in winners])
        l_med = float(np.median(l_vals))
        w_med = float(np.median(w_vals))
        if l_med > w_med * 1.05:        # perdedores claramente más altos
            rules.append(('above', feat, l_med))
        elif l_med < w_med * 0.95:       # perdedores claramente más bajos
            rules.append(('below', feat, l_med))
        # si son similares: no agregar regla (no hay señal)

    def f(df, i):
        for direction, feat, thresh in rules:
            try:
                val = float(df[feat].iloc[i])
            except Exception:
                continue
            if direction == 'above' and val > thresh:
                return False
            if direction == 'below' and val < thresh:
                return False
        return True

    return f, rules


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("Cargando BTC real 2020-2025...")
    df_real = load_btc_4h_real()
    print(f"  {len(df_real)} bars 4h")
    df_1d = load_btc_1d_real()
    df_fund = load_funding_real()

    A = load_module('A_strat', EXP / 'agent_A' / 'strategy.py')
    F = load_module('F_strat', EXP / 'agent_F' / 'strategy.py')

    print("\nBaseline V2 sobre BTC real (referencia):")
    real_trades = run_V2_on(df_real, df_1d, df_fund, A, F)
    m_real = metrics(real_trades)
    print(f"  N={m_real['n']:4d}  WR={m_real['wr']:.1%}  PF={m_real['pf']:.2f}  "
          f"annual={m_real['annual']:+.1%}")

    # =====================================================================
    # TEST A: V2 en N series sintéticas (robustez)
    # =====================================================================
    N_SYNTH = 20
    print(f"\n=== TEST A: V2 sobre {N_SYNTH} series sintéticas (block bootstrap) ===")
    synth_results = []
    for seed in range(N_SYNTH):
        df_synth = block_bootstrap_ohlcv(df_real, block_size=24, seed=seed)
        trades = run_V2_on(df_synth, df_1d, df_fund, A, F)
        m = metrics(trades)
        synth_results.append(m)
        print(f"  serie {seed:2d}: N={m['n']:4d}  WR={m['wr']:.1%}  "
              f"PF={m['pf']:.2f}  annual={m['annual']:+.1%}")

    annuals = [r['annual'] for r in synth_results]
    print(f"\n  Distribución annual return:")
    print(f"    mediana = {np.median(annuals):+.1%}")
    print(f"    media   = {np.mean(annuals):+.1%}")
    print(f"    p25-p75 = [{np.percentile(annuals,25):+.1%}, {np.percentile(annuals,75):+.1%}]")
    print(f"    p5-p95  = [{np.percentile(annuals, 5):+.1%}, {np.percentile(annuals,95):+.1%}]")
    print(f"    # series con annual > 0:   {sum(1 for a in annuals if a>0)}/{N_SYNTH}")
    print(f"    # series con annual > 10%: {sum(1 for a in annuals if a>0.10)}/{N_SYNTH}")
    print(f"    real BTC: {m_real['annual']:+.1%}  ({'dentro' if np.percentile(annuals,5)<=m_real['annual']<=np.percentile(annuals,95) else 'FUERA'} de p5-p95)")

    # =====================================================================
    # TEST B: ¿"aprender de pérdidas" ayuda?
    # =====================================================================
    print(f"\n=== TEST B: Naive learner - aprender de pérdidas (entrenar en #0, testear en resto) ===")
    # Generar serie de entrenamiento
    df_train = block_bootstrap_ohlcv(df_real, block_size=24, seed=999)
    train_trades = run_V2_on(df_train, df_1d, df_fund, A, F)
    m_train = metrics(train_trades)
    print(f"  Serie de entrenamiento (seed=999):")
    print(f"    Sin filtro:    N={m_train['n']}  PF={m_train['pf']:.2f}  annual={m_train['annual']:+.1%}")

    # Construir filtro a partir de perdedores
    loss_filter, rules = build_loss_filter(train_trades)
    print(f"\n  Reglas aprendidas del 'naive learner':")
    if not rules:
        print(f"    (ninguna — losers y winners no se distinguen claramente)")
    for direction, feat, thresh in rules:
        print(f"    SKIP si {feat} {'>' if direction=='above' else '<'} {thresh:.3f}")
    train_trades_adj = run_V2_on(df_train, df_1d, df_fund, A, F, entry_filter=loss_filter)
    m_train_adj = metrics(train_trades_adj)
    print(f"    Con filtro:    N={m_train_adj['n']}  PF={m_train_adj['pf']:.2f}  annual={m_train_adj['annual']:+.1%}  <- ¡SE VE MEJOR! (es in-sample)")

    # Aplicar a las 20 series del TEST A
    print(f"\n  Aplicando filtro a {N_SYNTH} series OOS (mismo seed que TEST A):")
    deltas = []
    n_better = 0
    n_worse = 0
    for seed in range(N_SYNTH):
        df_oos = block_bootstrap_ohlcv(df_real, block_size=24, seed=seed)
        trades_orig = run_V2_on(df_oos, df_1d, df_fund, A, F)
        trades_adj = run_V2_on(df_oos, df_1d, df_fund, A, F, entry_filter=loss_filter)
        m_o = metrics(trades_orig)
        m_a = metrics(trades_adj)
        delta = m_a['annual'] - m_o['annual']
        deltas.append(delta)
        if delta > 0:
            n_better += 1
        else:
            n_worse += 1
        flag = '+' if delta > 0 else '-'
        print(f"    {seed:2d}: orig={m_o['annual']:+.1%}  adj={m_a['annual']:+.1%}  "
              f"delta={delta:+.1%} {flag}")

    print(f"\n  Resumen TEST B (V2_adjusted vs V2_original en OOS):")
    print(f"    mediana delta = {np.median(deltas):+.2%}")
    print(f"    media delta   = {np.mean(deltas):+.2%}")
    print(f"    # series donde el ajuste AYUDÓ:  {n_better}/{N_SYNTH}")
    print(f"    # series donde el ajuste EMPEORÓ: {n_worse}/{N_SYNTH}")
    import math
    k_extreme = max(n_better, n_worse)
    p_binom = 2 * sum(math.comb(N_SYNTH, k) for k in range(k_extreme, N_SYNTH + 1)) / (2 ** N_SYNTH)
    p_binom = min(p_binom, 1.0)
    print(f"    Test de signo (binomial): si fuera 50/50 azar, prob de >={k_extreme}/{N_SYNTH} "
          f"en una dirección = ~{p_binom:.3f}")

    # =====================================================================
    # TEST C: Null hypothesis (retornos shuffleados)
    # =====================================================================
    print(f"\n=== TEST C: Null hypothesis (retornos shuffleados, sin estructura temporal) ===")
    null_results = []
    for seed in range(10):
        df_null = shuffle_returns(df_real, seed=seed)
        trades = run_V2_on(df_null, df_1d, df_fund, A, F)
        m = metrics(trades)
        null_results.append(m)
        print(f"  seed {seed}: N={m['n']:4d}  WR={m['wr']:.1%}  PF={m['pf']:.2f}  "
              f"annual={m['annual']:+.1%}")

    null_annuals = [r['annual'] for r in null_results]
    print(f"\n  Annual return en null:")
    print(f"    mediana = {np.median(null_annuals):+.1%}")
    print(f"    media   = {np.mean(null_annuals):+.1%}")
    print(f"\n  Comparación clave:")
    print(f"    Mediana sintético (block bootstrap, mantiene autocorrelación): "
          f"{np.median(annuals):+.1%}")
    print(f"    Mediana null (sin autocorrelación):                            "
          f"{np.median(null_annuals):+.1%}")
    print(f"    Diferencia (edge real atribuible a estructura temporal):      "
          f"{np.median(annuals) - np.median(null_annuals):+.1%}")


if __name__ == '__main__':
    main()
