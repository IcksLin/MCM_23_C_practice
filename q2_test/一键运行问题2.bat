@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
set "PYTHONUTF8=1"

set "PYTHON_EXE=python"
where python >nul 2>&1
if errorlevel 1 (
    if exist "E:\anaconda\envs\yolov_env\python.exe" (
        set "PYTHON_EXE=E:\anaconda\envs\yolov_env\python.exe"
    ) else (
        echo [ERROR] Python was not found. Activate the project environment first.
        pause
        exit /b 1
    )
)

echo ============================================================
echo             Q2 One-Click Solver
echo ============================================================
echo Starting the formal Q2 scenario-based stochastic MILP...
"%PYTHON_EXE%" "scripts\run_q2.py" --seed 2024 --raw-scenarios 1000 --reduced-scenarios 30 --beta 0.90 --lambda-grid 0:1:0.1 --out-sample 5000 --mip-gap 0.001 --time-limit 600 --figures --reports
set "RUN_EXIT=%ERRORLEVEL%"

echo.
if "%RUN_EXIT%"=="0" goto :success
echo Q2 finished but did not reach the delivery threshold.
echo Exit code: %RUN_EXIT%
goto :finish

:success
echo Q2 solve completed successfully.

:finish
echo.
pause
exit /b %RUN_EXIT%
