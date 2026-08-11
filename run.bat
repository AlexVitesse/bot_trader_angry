@echo off
REM ============================================================
REM  run.bat - Lanza el paper trade local (aislado de produccion)
REM ============================================================
REM  Uso:
REM    run                              perfil conservador (BTC, sin F_SHORT)
REM    run --perfil agresivo --con-short  el que si opera en bear
REM    run --replay-only --days 180       solo el historico, sin vigilar
REM
REM  Se puede doble-clicar desde el explorador.
REM  Cualquier flag que pases se reenvia a paper_local.py.
REM ============================================================

cd /d "%~dp0"

set "PY=C:\Users\pcdec\AppData\Local\pypoetry\Cache\virtualenvs\binance-scalper-bot-ofXWUGOe-py3.12\Scripts\python.exe"

if not exist "%PY%" (
    echo [ERROR] No encuentro el python del venv de produccion:
    echo         %PY%
    echo.
    echo Si el venv cambio de sitio, saca la ruta con:
    echo         poetry env info --executable
    echo y actualiza la variable PY de este fichero.
    pause
    exit /b 1
)

if not exist "logs" mkdir logs

echo Lanzando paper trade local. Ctrl-C para parar.
echo   trades  -^> logs\paper_^<perfil^>.jsonl
echo   estado  -^> data\paper_^<perfil^>.json  (se retoma al relanzar)
echo.

REM -u = salida sin buffer, si no en Windows no se ve nada en tiempo real
"%PY%" -u paper_local.py %*

echo.
echo Proceso terminado.
pause
