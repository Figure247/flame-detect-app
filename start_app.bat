@echo off
chcp 65001 >nul
echo ========================================
echo   🔥 FlameDetect Pro 正在启动...
echo ========================================
echo.

cd /d "C:\Users\hemen\Desktop\flame-detect-app"

:: 设置 Node.js 路径（使用 PyCharm 自带的）
set PATH=C:\Users\hemen\AppData\Roaming\JetBrains\PyCharm2026.1\node\versions\24.19.0;%PATH%

:: 激活 Python 虚拟环境
call venv\Scripts\activate.bat

:: 启动应用
npm start

pause