#!/usr/bin/env python3
"""
启动脚本 - 用于开发环境单独启动后端
"""

import subprocess
import sys
import os

if __name__ == "__main__":
    subprocess.run([sys.executable, "main.py"], check=False)