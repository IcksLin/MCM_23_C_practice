@echo off
setlocal EnableExtensions
cd /d "%~dp0"

REM 优先使用 PATH 中的 python（conda/venv 激活后即可）
set "PYTHON_EXE=python"
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到 python，请先激活 Anaconda/Miniconda 环境
    pause
    exit /b 1
)

echo ============================================================
echo             Q1 One-Click Solver
echo ============================================================
echo [Stage 1/2] Running P1 validation...
"%PYTHON_EXE%" "scripts\p1_test.py"
if errorlevel 1 goto :p1_failed

echo.
echo [Stage 2/2] Solving, sensitivity analysis, figures and reports...
"%PYTHON_EXE%" "scripts\run_q1.py" --scenario both --eta 0.5 --delta 0.001 --demand-scale 1.0 --mip-gap 0.001 --time-limit 600 --seed 2024 --sensitivity --figures --reports
set "RUN_EXIT=%ERRORLEVEL%"

echo.
if "%RUN_EXIT%"=="0" goto :success
echo Q1 finished but did not reach the delivery threshold.
echo Exit code: %RUN_EXIT%
goto :finish

:success
echo Local readiness checks passed. Independent P2 review is still required.
goto :finish

:p1_failed
set "RUN_EXIT=1"
echo.
echo P1 validation failed. Formal solving was not started.
echo Review the error messages above.

:finish
echo.
pause
exit /b %RUN_EXIT%
