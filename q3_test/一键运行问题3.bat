@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo ============================================================
echo  Q3 一键运行 — 相关性·替代性·互补性
echo ============================================================

REM 探测Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] 未找到python，请确认Python已安装并在PATH中
    pause
    exit /b 1
)

REM 运行Q3主程序
python scripts\run_q3.py --config configs\q3_default.yaml --figures --reports

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
