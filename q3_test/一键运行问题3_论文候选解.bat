@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"
set "PYTHON_EXE=E:\anaconda\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
"%PYTHON_EXE%" -c "import numpy,pandas,scipy,openpyxl,yaml" >nul 2>&1
if errorlevel 1 exit /b 1
"%PYTHON_EXE%" -u scripts\run_q3.py --config configs\q3_paper_candidate.yaml --allow-uncertified --no-checkpoint
set "EXITCODE=%errorlevel%"
echo 论文候选解实验退出码: %EXITCODE%
pause
exit /b %EXITCODE%
