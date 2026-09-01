@echo off
chcp 65001 >nul
cd /d "%~dp0"

:: ?? Node.js ????? PyCharm ????
where node >nul 2>nul
if errorlevel 1 if exist "C:\Users\hemen\AppData\Roaming\JetBrains\PyCharm2026.1\node\versions\24.19.0" set "PATH=C:\Users\hemen\AppData\Roaming\JetBrains\PyCharm2026.1\node\versions\24.19.0;%PATH%"

:: ?? Python ????
if exist "%~dp0venv\Scripts\activate.bat" call "%~dp0venv\Scripts\activate.bat"

:: ?????????????????????
start "" /b cmd /c "npm start" >nul 2>&1
exit /b 0
