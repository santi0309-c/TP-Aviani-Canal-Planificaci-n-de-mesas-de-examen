@echo off
:: ============================================================
:: arrancar.bat — levanta backend (FastAPI) y frontend (Vite)
:: Uso: doble click o ejecutar desde la raiz del repo
:: ============================================================

echo.
echo  Planificacion de Mesas de Examen — UTN FRRo
echo  Backend: http://localhost:8000
echo  Frontend: http://localhost:5173
echo  Docs API: http://localhost:8000/docs
echo.

:: Levantar backend en nueva ventana
start "Backend FastAPI" cmd /k "cd /d %~dp0backend && .venv\Scripts\uvicorn app.main:app --reload --port 8000"

:: Esperar 2 segundos y levantar frontend en nueva ventana
timeout /t 2 /nobreak >nul
start "Frontend Vite" cmd /k "cd /d %~dp0frontend && npm run dev"

:: Esperar 3 segundos y abrir el navegador
timeout /t 3 /nobreak >nul
start http://localhost:5173

echo Listo! Se abrieron dos terminales y el navegador.
echo Cerrar este script no apaga los servidores.
