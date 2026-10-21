@echo off
rem Runs Vision Assistant with this folder's virtual environment (Python 3.10).
rem Usage:  run.bat            or   run.bat --mute --camera 0   (see run.bat --help)
cd /d "%~dp0"
if not exist "Scripts\python.exe" (
    echo Could not find Scripts\python.exe in %~dp0
    pause
    exit /b 1
)
"Scripts\python.exe" object_detection.py %*
if errorlevel 1 (
    echo.
    echo Vision Assistant exited with an error - see the messages above.
    pause
)
