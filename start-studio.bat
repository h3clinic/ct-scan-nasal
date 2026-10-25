@echo off
rem One-step start for the Nasal0 studio on Windows: double-click, or run from a terminal.
rem First run: makes a Python environment in %USERPROFILE%\.venvs\ct-scan-nasal and installs the studio.
rem Every run: prints the setup check, then opens the studio in the browser (kept awake while it runs).
rem Optional: set NASALSEG_ROOT to a NasalSeg folder (with images\ and labels\) to list its patients.
rem           start-studio.bat --update   reinstalls after pulling new code.
rem           start-studio.bat --no-browser   starts the studio without opening a browser.
setlocal
cd /d "%~dp0"
set "VENV=%USERPROFILE%\.venvs\ct-scan-nasal"
set "PY=%VENV%\Scripts\python.exe"

if not exist "%PY%" (
    echo Creating the Python environment in %VENV% ...
    py -3.12 -m venv "%VENV%" 2>nul || python -m venv "%VENV%"
    if not exist "%PY%" (
        echo Python 3.12 is needed: https://www.python.org/downloads/  ^(tick "Add python.exe to PATH"^)
        pause
        exit /b 1
    )
    set "UPDATE=1"
)
set "OPEN=--open"
for %%A in (%*) do (
    if /i "%%~A"=="--update" set "UPDATE=1"
    if /i "%%~A"=="--no-browser" set "OPEN="
)
"%PY%" -c "import nasal, gradio, pydicom, nibabel" 2>nul || set "UPDATE=1"
if defined UPDATE (
    echo Installing the studio ^(a few minutes the first time^) ...
    "%PY%" -m pip install --upgrade pip -q
    "%PY%" -m pip install -q -e ".[demo,segment,agents,patients]"
    if errorlevel 1 (
        echo Installing failed; see the messages above.
        pause
        exit /b 1
    )
)

"%PY%" -m nasal.setup_check

set "ARGS="
if defined NASALSEG_ROOT set ARGS=--images "%NASALSEG_ROOT%\images" --labels "%NASALSEG_ROOT%\labels"
echo Opening the studio at http://127.0.0.1:7861 ^(close this window to stop it^) ...
"%PY%" scripts\keep_awake.py -- "%PY%" -m nasal.studio %OPEN% %ARGS%
pause
