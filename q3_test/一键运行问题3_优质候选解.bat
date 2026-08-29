@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
set "PYTHON_EXE=E:\anaconda\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
"%PYTHON_EXE%" -c "import numpy,pandas,scipy,openpyxl,yaml" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python环境缺少必需依赖
    pause
    exit /b 1
)
echo Q3优质候选解实验：3个lambda，单阶段MILP，未认证候选方案
"%PYTHON_EXE%" -u scripts\run_q3.py --config configs\q3_quality_candidate.yaml --allow-uncertified --no-checkpoint
set "EXITCODE=%errorlevel%"
echo 实验退出码: %EXITCODE%
pause
exit /b %EXITCODE%
