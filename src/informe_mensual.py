"""
Informe mensual por Telegram (lo envía el proceso V2, que lee también los
archivos del agresivo). Compara cada cosa en vivo con lo que predijo su
backtest. Solo lee, no cambia nada.

El día 1 de cada mes sale automáticamente el mes anterior; /informe da el mes
en curso hasta ahora.
"""
import json
import sqlite3
from pathlib import Path

import pandas as pd

V2_DESDE = pd.Timestamp('2026-08-11', tz='UTC')       # V2 solo BTC desplegado
ALERTA_DESDE = pd.Timestamp('2026-10-20', tz='UTC')   # alerta grabándose en vivo
ESPERADO_V2 = "~1,9 trades/mes, WR 44%, PF 1,65, techo histórico +23,9%/año (DD 43%)"
ESPERADO_AGR = "~32 trades/mes y −12,6%/año (su backtest pierde)"


def _trades(db: Path) -> pd.DataFrame:
    try:
        con = sqlite3.connect(db)
        df = pd.read_sql("SELECT * FROM ml_trades WHERE exit_time IS NOT NULL", con)
        con.close()
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    for c in ('entry_time', 'exit_time'):
        df[c] = pd.to_datetime(df[c], utc=True, format='ISO8601')
    # verdad = income de Binance cuando existe (pnl_real), si no el pnl del bot
    df['pnl_v'] = df['pnl_real'].fillna(df['pnl']) if 'pnl_real' in df else df['pnl']
    return df


def _stats(df: pd.DataFrame) -> str:
    if df.empty:
        return "0 trades"
    p = df['pnl_v']
    perd = -p[p <= 0].sum()
    pf = f"{p[p > 0].sum() / perd:.2f}" if perd > 0 else "∞"
    return f"{len(p)} trades | WR {(p > 0).mean():.0%} | PF {pf} | PnL ${p.sum():+,.2f}"


def _estado(db: Path) -> dict:
    try:
        con = sqlite3.connect(db)
        s = dict(con.execute("SELECT key, value FROM ml_state").fetchall())
        con.close()
        return s
    except Exception:
        return {}


def generar(data_dir: Path, ini: pd.Timestamp, fin: pd.Timestamp, titulo: str) -> str:
    data_dir = Path(data_dir)
    mes = lambda df: df[(df['exit_time'] >= ini) & (df['exit_time'] < fin)] if len(df) else df  # noqa: E731
    out = [f"<b>{titulo}</b>", "━━━━━━━━━━━━━━━"]

    # ---------------- V2
    v2 = _trades(data_dir / 'ml_bot.db')
    v2 = v2[v2['entry_time'] >= V2_DESDE] if len(v2) else v2
    st = _estado(data_dir / 'ml_bot.db')
    bal, pico = float(st.get('balance', 0) or 0), float(st.get('peak_balance', 0) or 0)
    out += ["🔵 <b>V2 · BTC · demo</b>",
            f"  Mes: {_stats(mes(v2))}",
            f"  Desde {V2_DESDE.date()}: {_stats(v2)}"]
    if bal:
        out.append(f"  Balance ${bal:,.2f} | DD {1 - bal / pico:.1%}" if pico else f"  Balance ${bal:,.2f}")
    if len(v2) and 'pnl_sim_pct' in v2:
        con_sim = v2.dropna(subset=['pnl_sim_pct', 'pnl_real'])
        if len(con_sim):
            dif = (con_sim['pnl_real'] / con_sim['notional'] - con_sim['pnl_sim_pct']).abs()
            out.append(f"  Real vs simulado: {len(con_sim)} trades, diferencia mediana {dif.median():.2%}")
    out.append(f"  Esperado (backtest): {ESPERADO_V2}")

    # ---------------- mezcla
    try:
        from src import pata_btc
        s = pata_btc.cargar(data_dir / 'pata_btc_state.json')
        if s:
            out += ["", pata_btc.resumen(s)]
    except Exception:
        pass

    # ---------------- agresivo
    ag = _trades(data_dir / 'ml_agresivo.db')
    out += ["", "🟣 <b>AGRESIVO · paper $100</b>",
            f"  Mes: {_stats(mes(ag))}",
            f"  Total: {_stats(ag)}"]
    try:
        bal_a = json.loads((data_dir / 'paper_agresivo_state.json').read_text())['balance']
        out.append(f"  Balance ${bal_a:,.2f} ({bal_a / 100 - 1:+.1%} desde $100)")
    except Exception:
        pass
    if len(ag):
        por_dir = ag.groupby('side')['pnl_v'].agg(['size', 'sum'])
        out.append("  Por dirección: " + " | ".join(f"{d} {int(n)}t ${s_:+,.2f}"
                                                    for d, (n, s_) in por_dir.iterrows()))
    out.append(f"  Esperado (backtest): {ESPERADO_AGR}")

    # ---------------- alerta de caidas
    try:
        filas = [json.loads(x) for x in (data_dir / 'alerta_caidas.jsonl').read_text().splitlines() if x]
        al = pd.DataFrame(filas)
        al['dia'] = pd.to_datetime(al['dia'], utc=True)
        en_mes = al[(al['dia'] >= ini - pd.Timedelta(days=1)) & (al['dia'] < fin)]
        activa = set(al.loc[al['activa'], 'dia'])
        out += ["", "⚠️ <b>Alerta de caídas (solo registro)</b>",
                f"  Días con alerta este mes: {int(en_mes['activa'].sum())} de {len(en_mes)}"]
        if len(v2):
            nuevos = v2[v2['entry_time'] >= ALERTA_DESDE]
            en_al = nuevos['entry_time'].dt.normalize().sub(pd.Timedelta(days=1)).isin(activa)
            out.append(f"  Trades V2 desde {ALERTA_DESDE.date()} en día de alerta: {_stats(nuevos[en_al])}")
            out.append(f"  Resto: {_stats(nuevos[~en_al])}")
    except Exception:
        pass
    return "\n".join(out)


def mes_anterior(ahora: pd.Timestamp):
    fin = ahora.normalize().replace(day=1)
    ini = (fin - pd.Timedelta(days=1)).replace(day=1)
    return ini, fin, f"📅 INFORME MENSUAL {ini.strftime('%Y-%m')}"


def mes_en_curso(ahora: pd.Timestamp):
    ini = ahora.normalize().replace(day=1)
    return ini, ahora, f"📅 INFORME {ini.strftime('%Y-%m')} (mes en curso)"
