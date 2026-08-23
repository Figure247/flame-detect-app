#!/usr/bin/env python3
"""
启动脚本 - 用于开发环境单独启动后端
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    # 安装依赖
    subprocess.run([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn", "ultralytics", "opencv-python", "torch", "psutil"], check=False)

    # 启动后端
    subprocess.run([sys.executable, "main.py"])