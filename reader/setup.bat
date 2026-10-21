@echo off
:: Auris - Windows setup wrapper
:: Usage: double-click or run from Command Prompt

cd /d "%~dp0"

set "BOOTSTRAP="

:: The py launcher always points at a real interpreter, so it comes first.
:: The versions are tried in order of how well they are tested on ARM64.
for %%V in (3.12 3.11 3.13 3) do (
    if not defined BOOTSTRAP (
        py -%%V -c "import sys" >nul 2>&1
        if not errorlevel 1 set "BOOTSTRAP=py -%%V"
    )
)

:: Plain python is the fallback, but the Microsoft Store alias in
:: WindowsApps is a stub: it opens the Store instead of running Python,
:: and the setup would hang there.
if not defined BOOTSTRAP (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        if not defined BOOTSTRAP (
            echo %%P | find /i "\WindowsApps\" >nul || set "BOOTSTRAP=python"
        )
    )
)

if not defined BOOTSTRAP (
    echo Python not found. Please install Python 3.10+ from https://python.org
    echo ^(the "py" launcher is enough - tick "Add python.exe to PATH"^)
    pause
    exit /b 1
)

echo Using Python: %BOOTSTRAP%
%BOOTSTRAP% --version

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv...
    call %BOOTSTRAP% -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" setup.py
if errorlevel 1 (
    echo.
    echo Setup failed. See output above for details.
    pause
    exit /b 1
)

pause
