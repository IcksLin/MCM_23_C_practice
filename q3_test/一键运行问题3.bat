@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo ============================================================
echo  Q3 一键运行 — 相关性·替代性·互补性
echo ============================================================

REM 优先选择已验证具备 numpy/pandas/scipy/openpyxl/yaml 的环境
set "PYTHON_EXE=E:\anaconda\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"
"%PYTHON_EXE%" -c "import numpy,pandas,scipy,openpyxl,yaml" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python环境缺少 numpy pandas scipy openpyxl 或 pyyaml
    pause
    exit /b 1
)

REM 运行Q3主程序
"%PYTHON_EXE%" -u scripts\run_q3.py --config configs\q3_default.yaml --figures --reports

REM 透传退出码
set EXITCODE=%errorlevel%
if %EXITCODE% equ 0 (
    echo.
    echo [完成] Q3求解完成，全部硬门槛通过
) else if %EXITCODE% equ 2 (
    echo.
    echo [警告] Q3有可行方案但未通过认证（退出码2）
) else if %EXITCODE% equ 3 (
    echo.
    echo [错误] Q3无可行方案（退出码3）
) else (
    echo.
    echo [错误] Q3运行异常（退出码%EXITCODE%）
)

pause
exit /b %EXITCODE%
