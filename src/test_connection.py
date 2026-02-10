"""
Test de Conexion a Binance
===========================
Script para verificar que las API Keys funcionan correctamente.
Ejecutar: poetry run python src/test_connection.py
"""

import sys
from pathlib import Path

# Agregar el directorio raiz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ccxt
from config.settings import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    TRADING_MODE,
    SYMBOL,
    LEVERAGE,
    print_config,
    validate_config
)


def test_public_api():
    """Test de API publica (no requiere keys)."""
    print("\n[TEST 1] Conexion a API publica de Binance...")

    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })

        # Obtener precio actual
        ticker = exchange.fetch_ticker(SYMBOL)
        price = ticker['last']
        print(f"   [OK] Precio actual de {SYMBOL}: ${price:,.2f}")

        # Obtener ultimas velas
        ohlcv = exchange.fetch_ohlcv(SYMBOL, '5m', limit=5)
        print(f"   [OK] Ultimas 5 velas de 5m obtenidas")

        return True

    except Exception as e:
        print(f"   [ERROR] {e}")
        return False


def test_private_api():
    """Test de API privada (requiere keys)."""
    print("\n[TEST 2] Conexion a API privada de Binance...")

    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        print("   [SKIP] API Keys no configuradas")
        print("   Instrucciones:")
        print("   1. Copia .env.example a .env")
        print("   2. Agrega tus API Keys de Binance")
        print("   3. Ejecuta este script de nuevo")
        return False

    try:
        import time
        import requests
        import hmac
        import hashlib

        # Determinar URLs segun modo
        if TRADING_MODE == "testnet":
            base_url = 'https://testnet.binancefuture.com'
            print("   [INFO] Modo: TESTNET (Paper Trading)")
        else:
            base_url = 'https://fapi.binance.com'
            print("   [INFO] Modo: LIVE")

        # Sincronizar tiempo
        print("   [INFO] Sincronizando tiempo con Binance...")
        server_time = requests.get(f'{base_url}/fapi/v1/time', timeout=5).json()['serverTime']
        local_time = int(time.time() * 1000)
        time_offset = server_time - local_time
        print(f"   [INFO] Offset de tiempo: {time_offset}ms")

        # Usar requests directamente (ccxt tiene problemas con testnet)
        def signed_request(endpoint, params=None):
            """Hacer request firmado a Binance."""
            if params is None:
                params = {}

            # Usar tiempo del servidor
            timestamp = int(time.time() * 1000) + time_offset
            params['timestamp'] = timestamp

            query = '&'.join([f'{k}={v}' for k, v in params.items()])
            signature = hmac.new(
                BINANCE_API_SECRET.encode(),
                query.encode(),
                hashlib.sha256
            ).hexdigest()

            url = f'{base_url}{endpoint}?{query}&signature={signature}'
            headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
            return requests.get(url, headers=headers, timeout=10)

        print("   [INFO] Conectando a Binance Futures...")

        # Test: Obtener balance
        response = signed_request('/fapi/v2/balance')
        if response.status_code != 200:
            raise Exception(f"Error {response.status_code}: {response.text}")

        balances = response.json()
        usdt_balance = next((b for b in balances if b['asset'] == 'USDT'), None)

        if usdt_balance:
            available = float(usdt_balance.get('availableBalance', 0))
            total = float(usdt_balance.get('balance', 0))
            print(f"   [OK] Balance USDT:")
            print(f"        - Disponible: ${available:,.2f}")
            print(f"        - Total: ${total:,.2f}")
        else:
            print(f"   [OK] Balance USDT: $0.00")

        # Test: Obtener posiciones abiertas
        response = signed_request('/fapi/v2/positionRisk')
        if response.status_code == 200:
            positions = response.json()
            open_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]

            if open_positions:
                print(f"   [OK] Posiciones abiertas: {len(open_positions)}")
                for pos in open_positions:
                    symbol = pos.get('symbol', 'N/A')
                    amt = float(pos.get('positionAmt', 0))
                    pnl = float(pos.get('unRealizedProfit', 0))
                    side = 'LONG' if amt > 0 else 'SHORT'
                    print(f"        - {symbol} {side}: {abs(amt)} contratos, PnL: ${pnl:,.2f}")
            else:
                print(f"   [OK] Sin posiciones abiertas")

        # Test: Configurar leverage
        try:
            symbol_binance = SYMBOL.replace('/', '')
            leverage_params = {'symbol': symbol_binance, 'leverage': LEVERAGE}

            # POST request para leverage
            timestamp = int(time.time() * 1000) + time_offset
            leverage_params['timestamp'] = timestamp
            query = '&'.join([f'{k}={v}' for k, v in leverage_params.items()])
            signature = hmac.new(
                BINANCE_API_SECRET.encode(),
                query.encode(),
                hashlib.sha256
            ).hexdigest()

            url = f'{base_url}/fapi/v1/leverage?{query}&signature={signature}'
            headers = {'X-MBX-APIKEY': BINANCE_API_KEY}
            lev_response = requests.post(url, headers=headers, timeout=10)

            if lev_response.status_code == 200:
                print(f"   [OK] Leverage configurado a {LEVERAGE}x")
            else:
                print(f"   [WARN] Leverage: {lev_response.text[:100]}")
        except Exception as e:
            print(f"   [WARN] No se pudo configurar leverage: {e}")

        return True

    except Exception as e:
        print(f"   [ERROR] {e}")
        return False


def test_order_placement():
    """Test de colocacion de orden (solo en testnet)."""
    print("\n[TEST 3] Test de orden (solo informativo)...")

    if TRADING_MODE != "testnet":
        print("   [SKIP] Solo disponible en modo testnet")
        return True

    if not BINANCE_API_KEY or not BINANCE_API_SECRET:
        print("   [SKIP] API Keys no configuradas")
        return False

    print("   [INFO] En modo testnet, podrias colocar ordenes de prueba")
    print("   [INFO] Este test NO coloca ordenes reales")
    print("   [OK] Sistema listo para paper trading")

    return True


def main():
    """Ejecuta todos los tests de conexion."""
    print("\n" + "="*60)
    print("TEST DE CONEXION A BINANCE")
    print("="*60)

    # Mostrar configuracion
    print_config()

    # Validar configuracion
    if not validate_config():
        print("\n[!] Configuracion incompleta. Revisa el archivo .env")

    # Ejecutar tests
    results = []

    results.append(("API Publica", test_public_api()))
    results.append(("API Privada", test_private_api()))
    results.append(("Ordenes", test_order_placement()))

    # Resumen
    print("\n" + "="*60)
    print("RESUMEN DE TESTS")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"   {status} {name}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n[SUCCESS] Todos los tests pasaron!")
        print("Siguiente paso: Crear el bot de trading\n")
    else:
        print("\n[WARNING] Algunos tests fallaron.")
        print("Revisa la configuracion y vuelve a intentar.\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
