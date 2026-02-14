# =============================================================
# ML Bot Wrapper - Auto-restart, update, and retrain handler
# =============================================================
# Usage: powershell -ExecutionPolicy Bypass -File run_bot.ps1
#
# Exit codes from bot:
#   0  = Normal stop (user pressed Ctrl+C)
#   42 = Update requested (/update) -> git pull + pip install + restart
#   43 = Restart requested (/restart) -> just restart
#   44 = Retrain requested (/retrain) -> retrain models + restart
#   *  = Crash -> wait 30s + restart
# =============================================================

Set-Location $PSScriptRoot

Write-Host "[WRAPPER] ML Bot Wrapper iniciado"
Write-Host "[WRAPPER] Directorio: $(Get-Location)"

while ($true) {
    Write-Host "[WRAPPER] Iniciando bot..."
    python -u -m src.ml_bot
    $exitCode = $LASTEXITCODE

    Write-Host "[WRAPPER] Bot termino con codigo: $exitCode"

    if ($exitCode -eq 0) {
        Write-Host "[WRAPPER] Bot detenido normalmente. Saliendo."
        break
    }
    elseif ($exitCode -eq 42) {
        Write-Host "[WRAPPER] Update solicitado. Ejecutando git pull..."
        git pull
        Write-Host "[WRAPPER] Instalando dependencias..."
        pip install -e . 2>$null
        Write-Host "[WRAPPER] Reiniciando bot..."
    }
    elseif ($exitCode -eq 43) {
        Write-Host "[WRAPPER] Restart solicitado."
        Write-Host "[WRAPPER] Reiniciando bot..."
    }
    elseif ($exitCode -eq 44) {
        Write-Host "[WRAPPER] Retrain solicitado. Actualizando codigo primero..."
        git pull
        Write-Host "[WRAPPER] Instalando dependencias..."
        pip install -e . 2>$null
        Write-Host "[WRAPPER] Reentrenando modelos..."
        python -u ml_export_models.py
        $retrainCode = $LASTEXITCODE
        if ($retrainCode -eq 0) {
            Write-Host "[WRAPPER] Reentrenamiento exitoso."
        }
        else {
            Write-Host "[WRAPPER] ERROR en reentrenamiento (codigo $retrainCode)."
        }
        Write-Host "[WRAPPER] Reiniciando bot..."
    }
    else {
        Write-Host "[WRAPPER] Bot crasheo (codigo $exitCode). Reiniciando en 30s..."
        Start-Sleep -Seconds 30
    }

    Write-Host "---"
}

Write-Host "[WRAPPER] Wrapper finalizado."
