# =============================================================
# ML Bot Wrapper - Auto-restart on crash or /restart command
# =============================================================
# Usage: powershell -ExecutionPolicy Bypass -File run_bot.ps1
#
# Exit codes from bot:
#   0  = Normal stop (Ctrl+C)
#   43 = /restart  -> restart bot
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
    elseif ($exitCode -eq 43) {
        Write-Host "[WRAPPER] Restart solicitado. Reiniciando bot..."
    }
    else {
        Write-Host "[WRAPPER] Bot crasheo (codigo $exitCode). Reiniciando en 30s..."
        Start-Sleep -Seconds 30
    }

    Write-Host "---"
}

Write-Host "[WRAPPER] Wrapper finalizado."
