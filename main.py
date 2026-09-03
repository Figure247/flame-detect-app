#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ================================================================
# main.py - FastAPI 后端服务
# 优化: 更好的内存管理、缓存策略、错误处理、日志系统
# ================================================================

from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Query, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
import numpy as np
import cv2
from ultralytics import YOLO
import uvicorn
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import torch
import hashlib
from collections import deque, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime, timedelta
import base64
import psutil
import gc
import re
import logging
import sys
import errno
import html
import atexit
from threading import RLock

# ================================================================
# 日志配置
# ================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# ================================================================
# 目录配置
# ================================================================
def choose_writable_data_dir() -> str:
    """优先使用项目目录；如果磁盘已满则回退到用户目录。"""
    env_dir = os.environ.get('FLAME_DETECT_DATA_DIR')
    candidates = []
    if env_dir:
        candidates.append(Path(env_dir).expanduser())

    project_dir = (Path(__file__).resolve().parent / 'data').resolve()
    candidates.extend([project_dir, Path.home() / '.flame-detect-app' / 'data'])

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            free_gb = shutil.disk_usage(str(candidate)).free / (1024 ** 3)
            if free_gb >= 0.5:
                return str(candidate)
        except Exception:
            continue

    return str(project_dir)


DATA_DIR = choose_writable_data_dir()
STATIC_DIR = Path(__file__).parent
UPLOAD_DIR = Path(DATA_DIR) / 'uploads'
MODELS_DIR = Path(DATA_DIR) / 'models'
REPORTS_DIR = Path(DATA_DIR) / 'reports'
LOGS_DIR = Path(DATA_DIR) / 'logs'
HISTORY_DIR = Path(DATA_DIR) / 'history'
FALLBACK_REPORTS_DIR = Path.home() / '.flame-detect-app' / 'reports'

for d in [UPLOAD_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR, HISTORY_DIR, FALLBACK_REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logger.info(f"📁 数据目录: {DATA_DIR}")
logger.info(f"📁 模型目录: {MODELS_DIR}")

# ================================================================
# 配置
# ================================================================
def find_default_model() -> Optional[str]:
    """自动查找可用的默认模型，只允许使用应用自己的数据目录和项目模型目录。"""
    candidate_dirs = [
        MODELS_DIR,
        Path(__file__).resolve().parent / 'models',
        Path(__file__).resolve().parent / 'data' / 'models',
        Path.home() / '.flame-detect-app' / 'data' / 'models',
    ]

    unique_dirs: List[Path] = []
    seen = set()
    for directory in candidate_dirs:
        try:
            resolved = directory.resolve()
        except Exception:
            resolved = directory
        if str(resolved) not in seen:
            seen.add(str(resolved))
            unique_dirs.append(resolved)

    for directory in unique_dirs:
        try:
            if not directory.exists():
                continue
        except Exception:
            continue

        for ext in ['.pt', '.onnx', '.pth', '.weights']:
            candidates = sorted(
                directory.glob(f'*{ext}'),
                key=lambda p: p.stat().st_mtime if p.is_file() else 0,
                reverse=True,
            )
            for file_path in candidates:
                try:
                    if file_path.is_file() and file_path.stat().st_size > 1024 * 1024:
                        logger.info(f"✅ 找到默认模型: {file_path}")
                        return str(file_path)
                except Exception:
                    continue

    logger.warning("⚠️ 未找到默认模型（未在应用数据目录或项目模型目录中发现有效模型）")
    return None


DEFAULT_MODEL_PATH = find_default_model()
MAX_IMAGE_SIZE = 4096
MAX_HISTORY_SIZE = 100
SUPPORTED_EXTENSIONS = {'.pt', '.onnx', '.pth', '.weights'}
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
BACKEND_PORT = int(os.environ.get('FLAME_DETECT_PORT', '8000'))
MAX_IMAGE_UPLOAD_SIZE = 50 * 1024 * 1024
MAX_MODEL_UPLOAD_SIZE = 500 * 1024 * 1024


def safe_model_filename(filename: Optional[str]) -> str:
    """Return a model filename that cannot escape the models directory."""
    safe_name = Path(filename or '').name
    if not safe_name or safe_name != filename or Path(safe_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError('模型文件名无效或格式不受支持')
    return safe_name


async def read_upload_with_limit(file: UploadFile, max_size: int) -> bytes:
    """Read an upload in bounded chunks to avoid unbounded memory use."""
    chunks = []
    total_size = 0
    while chunk := await file.read(1024 * 1024):
        total_size += len(chunk)
        if total_size > max_size:
            raise HTTPException(status_code=413, detail=f"文件过大，请上传小于{max_size // (1024 * 1024)}MB的文件")
        chunks.append(chunk)
    return b''.join(chunks)

# ================================================================
# 缓存管理器 (LRU)
# ================================================================
class LRUCache:
    """线程安全的 LRU 缓存"""
    def __init__(self, max_size=30):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return round(self.hits / total * 100, 1) if total > 0 else 0

# ================================================================
# 模型管理器 (改进版)
# ================================================================
class ModelManager:
    """模型管理器 - 支持多模型、缓存、统计"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        self._model_lock = RLock()

        self.current_model = None
        self.current_model_path = None
        self.current_model_name = "未加载"
        self.available_models = []
        self.model_classes = {}
        self.is_loading = False
        self.load_time = 0.0
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 缓存
        self.cache = LRUCache(max_size=30)

        # 性能统计 (使用 deque 限制大小)
        self.total_inferences = 0
        self.inference_times = deque(maxlen=100)
        self.detection_counts = deque(maxlen=100)
        self.min_inference_time = float('inf')
        self.max_inference_time = 0.0
        self.start_time = time.time()

        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=2)

        # 初始化
        self.scan_models()

        # 尝试立即加载默认模型，确保启动后可直接使用
        if DEFAULT_MODEL_PATH and os.path.exists(DEFAULT_MODEL_PATH):
            self.executor.submit(self._load_model_async, DEFAULT_MODEL_PATH)
        else:
            logger.warning("⚠️ 未找到默认模型，请上传模型文件")

        # 注册退出清理
        atexit.register(self.cleanup)

    def _load_model_async(self, model_path):
        """异步加载模型"""
        success, message = self.load_model(model_path)
        if success:
            logger.info(f"✅ 模型异步加载成功: {self.current_model_name}")
        else:
            logger.error(f"❌ 模型异步加载失败: {message}")

    def cleanup(self):
        """清理资源"""
        logger.info("🧹 清理模型管理器资源...")
        if self.executor:
            self.executor.shutdown(wait=False)
        self.cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def scan_models(self) -> List[str]:
        """扫描 models 目录"""
        self.available_models = []
        for ext in SUPPORTED_EXTENSIONS:
            for file_path in MODELS_DIR.glob(f'*{ext}'):
                if file_path.is_file() and file_path.stat().st_size > 1024 * 1024:
                    self.available_models.append(file_path.name)

        # 检查默认模型
        if DEFAULT_MODEL_PATH and os.path.exists(DEFAULT_MODEL_PATH):
            default_name = os.path.basename(DEFAULT_MODEL_PATH)
            if default_name not in self.available_models:
                self.available_models.append(default_name)

        self.available_models.sort()
        logger.info(f"📋 找到 {len(self.available_models)} 个模型")
        return self.available_models

    def load_model(self, model_path: str, warmup: bool = False) -> tuple:
        """加载指定模型。默认不执行启动预热，因此首次启动更快。"""
        with self._model_lock:
            if self.is_loading:
                return False, "模型正在加载中，请稍后..."

            if not os.path.exists(model_path):
                return False, f"模型文件不存在: {model_path}"

            if os.path.getsize(model_path) < 1024 * 1024:
                return False, f"模型文件太小 (可能损坏): {os.path.getsize(model_path)} bytes"

            self.is_loading = True
        try:
            start_time = time.time()
            logger.info(f"🔄 正在加载模型: {model_path}")

            # 严格优先使用 GPU（如可用）
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # 清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 加载模型并显式放到目标设备
            model = YOLO(model_path)
            if hasattr(model, 'to'):
                try:
                    model.to(self.device)
                except Exception as exc:
                    logger.warning(f"⚠️ 模型显式切换设备失败: {exc}")

            # 避免在启动阶段执行一次额外的空图推理，减少启动延迟
            if warmup:
                try:
                    test_img = np.zeros((320, 320, 3), dtype=np.uint8)
                    _ = model(test_img, verbose=False, device=self.device)
                except Exception as exc:
                    logger.warning(f"⚠️ 模型预热失败，忽略继续启动: {exc}")

            # 更新状态
            with self._model_lock:
                self.current_model = model
                self.current_model_path = model_path
                self.current_model_name = os.path.basename(model_path)
                self.model_classes = model.names
                self.load_time = time.time() - start_time
                self.cache.clear()

            logger.info(f"✅ 模型加载成功: {self.current_model_name} ({self.load_time:.2f}s)")
            logger.info(f"💻 设备: {self.device}")

            return True, f"模型加载成功 (耗时: {self.load_time:.2f}s)"

        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            return False, f"模型加载失败: {str(e)}"
        finally:
            with self._model_lock:
                self.is_loading = False

    def load_model_by_name(self, model_name: str) -> tuple:
        """通过模型名称加载"""
        try:
            model_name = safe_model_filename(model_name)
        except ValueError as e:
            return False, str(e)

        model_path = MODELS_DIR / model_name
        if model_path.exists():
            return self.load_model(str(model_path))

        if DEFAULT_MODEL_PATH and model_name == os.path.basename(DEFAULT_MODEL_PATH):
            if os.path.exists(DEFAULT_MODEL_PATH):
                return self.load_model(DEFAULT_MODEL_PATH)

        return False, f"模型不存在: {model_name}"

    def _get_cache_key(self, image_data, conf_threshold):
        """生成缓存键"""
        if isinstance(image_data, np.ndarray):
            hash_obj = hashlib.md5(image_data.tobytes())
        elif isinstance(image_data, str):
            hash_obj = hashlib.md5(image_data.encode())
        else:
            return None
        return f"{hash_obj.hexdigest()}_{conf_threshold:.2f}"

    def predict(self, image_path: str, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """执行检测"""
        start_time = time.time()

        # 执行推理
        with self._model_lock:
            model = self.current_model
            model_name = self.current_model_name
            model_names = model.names if model else {}
            if model is None:
                raise ValueError("模型未加载")
            device = 0 if torch.cuda.is_available() else 'cpu'
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            results = model(image_path, conf=conf_threshold, verbose=False, device=device)[0]

        inference_time = (time.time() - start_time) * 1000

        boxes = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                boxes.append([int(x1), int(y1), int(x2), int(y2), conf, cls_id])

        # 更新统计
        self.total_inferences += 1
        self.inference_times.append(inference_time)
        self.detection_counts.append(len(boxes))

        # 更新极值
        self.min_inference_time = min(self.min_inference_time, inference_time)
        self.max_inference_time = max(self.max_inference_time, inference_time)

        return {
            "boxes": boxes,
            "names": model_names,
            "total": len(boxes),
            "model_name": model_name,
            "inference_time_ms": round(inference_time, 2),
            "conf_threshold": conf_threshold
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取当前模型信息"""
        base_info = {
            "status": "unloaded" if self.current_model is None else "loaded",
            "model_name": self.current_model_name,
            "classes": self.current_model.names if self.current_model else {},
            "class_count": len(self.current_model.names) if self.current_model else 0,
            "available_models": self.available_models,
            "total_inferences": self.total_inferences,
            "device": self.device,
            "avg_inference_time_ms": 0,
            "load_time": round(self.load_time, 2) if self.load_time > 0 else 0,
            "min_inference_time_ms": 0,
            "max_inference_time_ms": 0,
            "cache_hit_rate": self.cache.hit_rate,
            "uptime": round(time.time() - self.start_time, 0)
        }

        if self.inference_times:
            avg_time = sum(self.inference_times) / len(self.inference_times)
            base_info["avg_inference_time_ms"] = round(avg_time, 2)
            base_info["min_inference_time_ms"] = round(self.min_inference_time, 2) if self.min_inference_time != float('inf') else 0
            base_info["max_inference_time_ms"] = round(self.max_inference_time, 2)

        return base_info

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        info = self.get_model_info()
        return {
            "total_inferences": self.total_inferences,
            "avg_inference_time_ms": info.get("avg_inference_time_ms", 0),
            "min_inference_time_ms": info.get("min_inference_time_ms", 0),
            "max_inference_time_ms": info.get("max_inference_time_ms", 0),
            "current_model": self.current_model_name,
            "device": self.device,
            "cache_hit_rate": info.get("cache_hit_rate", 0),
            "model_loaded": self.current_model is not None,
            "uptime": info.get("uptime", 0),
            "memory_usage_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 1),
            "cpu_usage": psutil.cpu_percent()
        }


# 全局模型管理器
model_manager = ModelManager()

# ================================================================
# 图片处理工具
# ================================================================
def process_image(image_data: bytes, max_size: int = MAX_IMAGE_SIZE) -> np.ndarray:
    """处理图片，确保尺寸合适"""
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("无法解码图片")

    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return img


def draw_boxes_on_image(image: np.ndarray, boxes: List, names: Dict) -> np.ndarray:
    """在图片上绘制检测框"""
    img = image.copy()
    for box in boxes:
        if len(box) >= 5:
            x1, y1, x2, y2, conf = box[:5]
            cls_id = box[5] if len(box) > 5 else 0
            label = names.get(cls_id, 'fire')

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            text = f"{label} {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

            cv2.rectangle(img, (int(x1), int(y1) - text_size[1] - 6),
                          (int(x1) + text_size[0] + 6, int(y1)), (0, 0, 255), -1)
            cv2.putText(img, text, (int(x1) + 3, int(y1) - 3),
                        font, font_scale, (255, 255, 255), thickness)

    return img


def ensure_default_model_loaded() -> bool:
    """确保默认模型已加载；若未加载则自动尝试加载"""
    if model_manager.current_model is not None:
        return True
    if not DEFAULT_MODEL_PATH or not os.path.exists(DEFAULT_MODEL_PATH):
        return False
    success, message = model_manager.load_model(DEFAULT_MODEL_PATH)
    if success:
        logger.info(f"✅ 自动加载默认模型成功: {model_manager.current_model_name}")
        return True
    logger.warning(f"⚠️ 自动加载默认模型失败: {message}")
    return False


def make_report_summary(results: List[Dict[str, Any]], report_name: str = "batch_report") -> Dict[str, Any]:
    """生成结构化检测报告摘要"""
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]+', '_', report_name).strip('_') or 'batch_report'
    generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    total = len(results)
    fire_count = sum(1 for item in results if item.get('fire_count', 0) > 0)
    safe_count = sum(1 for item in results if item.get('fire_count', 0) == 0 and 'error' not in item)
    all_boxes = sum(int(item.get('total', 0)) for item in results)
    avg_time = round(sum(float(item.get('inference_time_ms', 0)) for item in results) / total, 2) if total else 0
    report = {
        "report_name": safe_name,
        "generated_at": generated_at,
        "model": results[0].get('model', model_manager.current_model_name) if results else model_manager.current_model_name,
        "conf_threshold": results[0].get('conf_threshold', 0.25) if results else 0.25,
        "total_images": total,
        "fire_images": fire_count,
        "safe_images": safe_count,
        "total_boxes": all_boxes,
        "avg_inference_time_ms": avg_time,
        "items": [
            {
                "file_name": item.get('file_name', 'unknown'),
                "fire_count": int(item.get('fire_count', 0)),
                "total_boxes": int(item.get('total', 0)),
                "inference_time_ms": float(item.get('inference_time_ms', 0)),
                "status": 'fire' if int(item.get('fire_count', 0)) > 0 else 'safe',
                "model": item.get('model', model_manager.current_model_name),
                "conf_threshold": float(item.get('conf_threshold', 0.25)),
                "image": item.get('image', '')
            }
            for item in results
        ]
    }
    return report


def cleanup_old_report_artifacts(min_free_mb: int = 256) -> bool:
    """删除最旧的报告与日志，释放磁盘空间。"""
    for directory in [REPORTS_DIR, LOGS_DIR, HISTORY_DIR]:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            free_gb = shutil.disk_usage(str(directory)).free / (1024 ** 3)
            if free_gb >= min_free_mb / 1024:
                return True
        except OSError:
            pass

        try:
            for path in sorted(directory.glob('*'), key=lambda p: p.stat().st_mtime):
                if path.is_file():
                    path.unlink(missing_ok=True)
                elif path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                if shutil.disk_usage(str(directory)).free >= min_free_mb * 1024 * 1024:
                    return True
        except Exception as exc:
            logger.warning(f"⚠️ 清理旧报告失败: {exc}")

    return False


def save_report_files(report_payload: Dict[str, Any], report_name: str = "batch_report") -> Dict[str, str]:
    """将报告保存为 JSON 和 HTML 文件，并返回路径信息。若磁盘已满则清理旧报告并回退到用户目录。"""
    global REPORTS_DIR
    safe_name = re.sub(r'[^a-zA-Z0-9_\-]+', '_', report_name).strip('_') or 'batch_report'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_id = f"{safe_name}_{timestamp}"

    def write_report_file(path: Path, content: str, *, is_json: bool = False):
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(content)
            return
        except OSError as exc:
            if exc.errno == errno.ENOSPC or 'No space left' in str(exc):
                raise
            # Avoid depending on Path.write_text, which may be monkeypatched in tests.
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(content)

    def write_pair(target_dir: Path):
        target_dir.mkdir(parents=True, exist_ok=True)
        json_path = target_dir / f"{report_id}.json"
        html_path = target_dir / f"{report_id}.html"

        json_content = json.dumps(report_payload, ensure_ascii=False, indent=2)
        write_report_file(json_path, json_content, is_json=True)

        fire_rows = ''
        for item in report_payload.get('items', []):
            status = item.get('status', 'safe')
            status_class = 'status-fire' if status == 'fire' else ('status-error' if status == 'error' else 'status-safe')
            status_text = '火情' if status == 'fire' else ('异常' if status == 'error' else '安全')
            image_data = item.get('image', '')
            image_html = (
                f'<img class="result-thumb" src="data:image/jpeg;base64,{html.escape(image_data)}" '
                f'alt="{html.escape(str(item.get("file_name", "检测图片")))}" loading="lazy" />'
                if image_data else '<div class="result-thumb empty">暂无图片</div>'
            )
            fire_rows += (
                f'<tr class="{status_class}">'
                f'<td><div class="result-image">{image_html}</div><span class="file-name">{html.escape(str(item.get("file_name", "unknown")))}</span></td>'
                f'<td>{int(item.get("fire_count", 0))}</td>'
                f'<td>{int(item.get("total_boxes", 0))}</td>'
                f'<td>{float(item.get("inference_time_ms", 0))} ms</td>'
                f'<td><span class="status-badge">{status_text}</span></td>'
                '</tr>'
            )

        html_content = f"""<!DOCTYPE html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>{report_payload['report_name']} 检测报告</title>
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 32px; }}
    .card {{ max-width: 1100px; margin: 0 auto; background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(148,163,184,0.2); border-radius: 16px; padding: 24px; box-shadow: 0 16px 40px rgba(15,23,42,0.25); }}
    h1 {{ margin-top: 0; font-size: 2rem; }}
    .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 20px 0; }}
    .meta div {{ background: #111827; border: 1px solid #334155; border-radius: 12px; padding: 12px 14px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #334155; text-align: left; }}
    th {{ background: #111827; }}
    tr:nth-child(even) {{ background: rgba(15, 23, 42, 0.4); }}
    tr.status-fire {{ border-left: 4px solid #ef4444; }}
    tr.status-safe {{ border-left: 4px solid #22c55e; }}
    tr.status-error {{ border-left: 4px solid #f59e0b; }}
    .result-image {{ width: 132px; height: 82px; margin-bottom: 7px; }}
    .result-thumb {{ width: 100%; height: 100%; object-fit: cover; display: block; border-radius: 8px; border: 1px solid #475569; background: #1e293b; }}
    .result-thumb.empty {{ display: flex; align-items: center; justify-content: center; color: #94a3b8; font-size: 0.75rem; }}
    .file-name {{ display: block; max-width: 260px; overflow-wrap: anywhere; }}
    .status-badge {{ display: inline-block; padding: 4px 10px; border-radius: 999px; font-weight: 700; }}
    .status-fire .status-badge {{ color: #fecaca; background: rgba(239, 68, 68, 0.2); }}
    .status-safe .status-badge {{ color: #bbf7d0; background: rgba(34, 197, 94, 0.2); }}
    .status-error .status-badge {{ color: #fde68a; background: rgba(245, 158, 11, 0.2); }}
    @media (max-width: 640px) {{ body {{ padding: 14px; }} .card {{ padding: 14px; }} table {{ font-size: 0.8rem; }} .result-image {{ width: 96px; height: 60px; }} th, td {{ padding: 8px 6px; }} }}
  </style>
</head>
<body>
  <div class=\"card\">
    <h1>{report_payload['report_name']} 检测报告</h1>
    <div class=\"meta\">
      <div><strong>生成时间</strong><br>{report_payload['generated_at']}</div>
      <div><strong>模型</strong><br>{report_payload['model']}</div>
      <div><strong>置信度阈值</strong><br>{report_payload['conf_threshold']}</div>
      <div><strong>总图片数</strong><br>{report_payload['total_images']}</div>
      <div><strong>火情图片</strong><br>{report_payload['fire_images']}</div>
      <div><strong>安全图片</strong><br>{report_payload['safe_images']}</div>
      <div><strong>总框数</strong><br>{report_payload['total_boxes']}</div>
      <div><strong>平均推理耗时</strong><br>{report_payload['avg_inference_time_ms']} ms</div>
    </div>
    <table>
      <thead>
        <tr><th>图片</th><th>火焰数</th><th>框数</th><th>耗时</th><th>状态</th></tr>
      </thead>
      <tbody>
        {fire_rows}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
        write_report_file(html_path, html_content)
        return json_path, html_path

    target_dirs = [REPORTS_DIR, FALLBACK_REPORTS_DIR]
    last_error = None
    for index, target_dir in enumerate(target_dirs):
        try:
            json_path, html_path = write_pair(target_dir)
            if target_dir == FALLBACK_REPORTS_DIR:
                REPORTS_DIR = FALLBACK_REPORTS_DIR
            return {
                "json_name": json_path.name,
                "html_name": html_path.name,
                "json_path": str(json_path),
                "html_path": str(html_path),
            }
        except OSError as exc:
            last_error = exc
            if exc.errno == errno.ENOSPC or 'No space left' in str(exc):
                logger.warning(f"⚠️ 报告写入磁盘不足，尝试清理旧报告缓存: {exc}")
                cleanup_old_report_artifacts(min_free_mb=256)
                if index == 0:
                    continue
            raise
        except Exception:
            raise

    if last_error is not None:
        raise last_error
    raise RuntimeError('报告保存失败')

# ================================================================
# FastAPI 应用
# ================================================================
app = FastAPI(
    title="FlameDetect Pro - YOLOv8 火焰检测 API",
    version="2.0.0",
    description="高性能火焰检测系统，支持多模型管理、批量检测、实时监控"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "null"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================================================================
# 健康检查
# ================================================================
@app.get("/health")
async def health_check():
    """服务健康检查"""
    info = model_manager.get_model_info()
    report_entries = []
    for report_path in sorted(REPORTS_DIR.glob('*'), key=lambda p: p.stat().st_mtime, reverse=True):
        if report_path.is_file():
            report_entries.append({
                "name": report_path.name,
                "size": report_path.stat().st_size,
                "updated_at": datetime.fromtimestamp(report_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            })
    latest_report = report_entries[0] if report_entries else None
    return {
        "status": "ok",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": "2.0.0",
        "model_loaded": model_manager.current_model is not None,
        "model_name": model_manager.current_model_name if model_manager.current_model else None,
        "classes": model_manager.model_classes if model_manager.current_model else None,
        "class_count": len(model_manager.model_classes) if model_manager.current_model else 0,
        "available_models": model_manager.available_models,
        "total_inferences": model_manager.total_inferences,
        "device": model_manager.device,
        "uptime": round(time.time() - model_manager.start_time, 0),
        "memory_usage_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 1),
        "report_count": len(report_entries),
        "latest_report": latest_report,
        "data_dir": str(DATA_DIR)
    }


# ================================================================
# 检测接口
# ================================================================
@app.post("/predict")
async def predict(
        file: UploadFile = File(...),
        conf_threshold: Optional[float] = Form(0.25, ge=0.0, le=1.0),
        return_image: Optional[bool] = Form(False)
):
    """图片/视频帧检测"""
    if model_manager.current_model is None and not ensure_default_model_loaded():
        raise HTTPException(status_code=503, detail="模型未加载，请先上传模型")

    try:
        contents = await read_upload_with_limit(file, MAX_IMAGE_UPLOAD_SIZE)

        # 处理图片
        img = process_image(contents)

        # 保存临时文件
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, img)
            tmp_path = tmp_file.name

        try:
            # 推理
            result = await run_in_threadpool(model_manager.predict, tmp_path, conf_threshold)
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_path)
            except:
                pass

        # 筛选符合置信度阈值的框
        fire_boxes = [b for b in result["boxes"] if b[4] >= conf_threshold]

        response = {
            "boxes": result["boxes"],
            "names": result["names"],
            "total": len(result["boxes"]),
            "fire_count": len(fire_boxes),
            "model": result["model_name"],
            "inference_time_ms": result["inference_time_ms"],
            "conf_threshold": result["conf_threshold"],
            "file_name": file.filename
        }

        # 如果需要返回标注图片
        if return_image:
            annotated = draw_boxes_on_image(img, result["boxes"], result["names"])
            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            response["image"] = base64.b64encode(buffer).decode('utf-8')

        report_image = response.get('image', '')
        if not report_image:
            annotated = draw_boxes_on_image(img, result["boxes"], result["names"])
            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            report_image = base64.b64encode(buffer).decode('utf-8')

        report_results = [{
            "file_name": file.filename,
            "boxes": result["boxes"],
            "names": result["names"],
            "total": len(result["boxes"]),
            "fire_count": len(fire_boxes),
            "model": result["model_name"],
            "inference_time_ms": result["inference_time_ms"],
            "conf_threshold": result["conf_threshold"],
            "image": report_image,
        }]
        report_payload = make_report_summary(report_results, report_name="single_detection")
        report_paths = save_report_files(report_payload, report_name="single_detection")
        response["report"] = {
            "report_name": report_payload['report_name'],
            "json_name": report_paths['json_name'],
            "html_name": report_paths['html_name'],
            "json_url": f"/reports/{report_paths['json_name']}",
            "html_url": f"/reports/{report_paths['html_name']}",
            "fire_images": report_payload['fire_images'],
            "safe_images": report_payload['safe_images'],
            "total_images": report_payload['total_images'],
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 检测错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(
        files: List[UploadFile] = File(...),
        conf_threshold: Optional[float] = Form(0.25, ge=0.0, le=1.0),
        return_images: Optional[bool] = Form(False)
):
    """批量检测多张图片并生成报告"""
    valid_files = []
    seen = set()
    for file in files or []:
        if file is None or not getattr(file, 'filename', None):
            continue
        label = (file.filename or '').strip()
        if not label:
            continue
        key = f"{label}:{getattr(file, 'size', 0)}"
        if key in seen:
            continue
        seen.add(key)
        valid_files.append(file)

    if not valid_files:
        raise HTTPException(status_code=400, detail="请至少上传一张有效图片")
    if len(valid_files) > 20:
        raise HTTPException(status_code=400, detail="批量检测最多支持 20 张图片")
    if model_manager.current_model is None and not ensure_default_model_loaded():
        raise HTTPException(status_code=503, detail="模型未加载，请先上传模型")

    results = []
    for file in valid_files:
        try:
            contents = await read_upload_with_limit(file, MAX_IMAGE_UPLOAD_SIZE)
            img = process_image(contents)
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, img)
                tmp_path = tmp_file.name
            try:
                result = await run_in_threadpool(model_manager.predict, tmp_path, conf_threshold)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            filtered_boxes = [b for b in result["boxes"] if b[4] >= conf_threshold]
            item = {
                "file_name": file.filename,
                "boxes": result["boxes"],
                "names": result["names"],
                "total": len(result["boxes"]),
                "fire_count": len(filtered_boxes),
                "model": result["model_name"],
                "inference_time_ms": result["inference_time_ms"],
                "conf_threshold": result["conf_threshold"]
            }

            if return_images:
                annotated = draw_boxes_on_image(img, result["boxes"], result["names"])
                _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                item["image"] = base64.b64encode(buffer).decode('utf-8')

            results.append(item)
        except Exception as exc:
            logger.error(f"❌ 批量检测失败 ({file.filename}): {exc}")
            results.append({
                "file_name": file.filename,
                "error": str(exc),
                "fire_count": 0,
                "total": 0,
                "model": model_manager.current_model_name,
                "inference_time_ms": 0,
                "conf_threshold": conf_threshold,
            })

    report_payload = make_report_summary(results, report_name="batch_detection")
    report_paths = save_report_files(report_payload, report_name="batch_detection")

    return {
        "status": "success",
        "total_images": len(valid_files),
        "processed_images": len(results),
        "fire_images": sum(1 for item in results if item.get('fire_count', 0) > 0),
        "safe_images": sum(1 for item in results if item.get('fire_count', 0) == 0 and 'error' not in item),
        "results": results,
        "report": {
            "report_name": report_payload['report_name'],
            "json_name": report_paths['json_name'],
            "html_name": report_paths['html_name'],
            "json_url": f"/reports/{report_paths['json_name']}",
            "html_url": f"/reports/{report_paths['html_name']}",
            "fire_images": report_payload['fire_images'],
            "safe_images": report_payload['safe_images'],
            "total_images": report_payload['total_images']
        }
    }


# ================================================================
# 模型管理接口
# ================================================================
@app.get("/models/list")
async def list_models():
    """获取所有可用模型列表"""
    models = model_manager.scan_models()
    return {
        "status": "success",
        "models": models,
        "current_model": model_manager.current_model_name,
        "total": len(models)
    }


@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)):
    """上传并加载模型"""
    try:
        model_filename = safe_model_filename(file.filename)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    try:
        content = await read_upload_with_limit(file, MAX_MODEL_UPLOAD_SIZE)

        # 保存
        file_path = MODELS_DIR / model_filename
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"📁 模型已保存: {file_path}")

        # 刷新列表
        model_manager.scan_models()

        # 尝试加载
        success, message = await run_in_threadpool(model_manager.load_model, str(file_path))

        if success:
            return {
                "status": "success",
                "message": message,
                "model_name": model_manager.current_model_name,
                "classes": model_manager.current_model.names,
                "class_count": len(model_manager.current_model.names),
                "available_models": model_manager.available_models
            }
        else:
            # 加载失败，删除文件
            try:
                os.remove(file_path)
            except:
                pass
            raise HTTPException(status_code=400, detail=message)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 上传失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load_model")
async def load_model_api(model_name: str = Body(..., embed=True)):
    """加载指定的模型"""
    if model_manager.is_loading:
        raise HTTPException(status_code=409, detail="模型正在加载中，请稍后...")

    success, message = await run_in_threadpool(model_manager.load_model_by_name, model_name)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "status": "success",
        "message": message,
        "model_name": model_manager.current_model_name,
        "classes": model_manager.current_model.names,
        "class_count": len(model_manager.current_model.names)
    }


@app.post("/load_default")
async def load_default_model():
    """加载默认模型"""
    if not DEFAULT_MODEL_PATH or not os.path.exists(DEFAULT_MODEL_PATH):
        raise HTTPException(status_code=404, detail="默认模型不存在")

    success, message = await run_in_threadpool(model_manager.load_model, DEFAULT_MODEL_PATH)

    if success:
        return {
            "status": "success",
            "message": message,
            "model_name": model_manager.current_model_name,
            "classes": model_manager.current_model.names,
            "class_count": len(model_manager.current_model.names)
        }
    else:
        raise HTTPException(status_code=500, detail=message)


# ================================================================
# 信息接口
# ================================================================
@app.get("/model_info")
async def get_model_info():
    """获取当前模型信息"""
    return model_manager.get_model_info()


@app.get("/stats")
async def get_stats():
    """获取性能统计"""
    return {
        "status": "success",
        "stats": model_manager.get_stats(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


@app.post("/cache/clear")
async def clear_cache():
    """清空推理缓存"""
    model_manager.cache.clear()
    return {"status": "success", "message": "缓存已清空"}


@app.post("/system/gc")
async def force_garbage_collection():
    """强制垃圾回收"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "status": "success",
        "message": "垃圾回收完成",
        "memory_usage_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
    }


@app.get("/system/info")
async def get_system_info():
    """获取系统信息"""
    return {
        "cpu": {
            "cores": psutil.cpu_count(),
            "usage": psutil.cpu_percent(interval=0.5)
        },
        "memory": {
            "total_gb": round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2),
            "available_gb": round(psutil.virtual_memory().available / 1024 / 1024 / 1024, 2),
            "used_gb": round(psutil.virtual_memory().used / 1024 / 1024 / 1024, 2),
            "percent": psutil.virtual_memory().percent
        },
        "disk": {
            "total_gb": round(psutil.disk_usage('/').total / 1024 / 1024 / 1024, 2),
            "used_gb": round(psutil.disk_usage('/').used / 1024 / 1024 / 1024, 2),
            "free_gb": round(psutil.disk_usage('/').free / 1024 / 1024 / 1024, 2),
            "percent": psutil.disk_usage('/').percent
        }
    }


# ================================================================
# 前端服务
# ================================================================
@app.get("/")
async def serve_index():
    """提供前端页面"""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)


@app.get("/history/{filename}")
async def get_history_image(filename: str):
    """获取历史图片"""
    file_path = HISTORY_DIR / filename
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="图片不存在")


@app.get("/reports")
async def list_reports():
    """列出已生成的检测报告"""
    files = []
    for report_path in sorted(REPORTS_DIR.glob('*'), key=lambda p: p.stat().st_mtime, reverse=True):
        if report_path.is_file():
            files.append({
                "name": report_path.name,
                "size": report_path.stat().st_size,
                "updated_at": datetime.fromtimestamp(report_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            })
    latest_report = files[0] if files else None
    return {"status": "success", "total": len(files), "reports": files, "latest_report": latest_report}


@app.get("/reports/{filename}")
async def get_report_detail(filename: str):
    """读取单个报告内容，支持 JSON 和 HTML 文件。"""
    if not filename or '..' in Path(filename).parts:
        raise HTTPException(status_code=400, detail="非法的报告文件名")

    report_path = (REPORTS_DIR / filename).resolve()
    if not report_path.is_file() or REPORTS_DIR.resolve() not in report_path.parents and report_path != REPORTS_DIR.resolve():
        raise HTTPException(status_code=404, detail="报告不存在")

    if report_path.suffix.lower() == '.json':
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            return JSONResponse(content=payload)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=500, detail=f"报告 JSON 无效: {exc.msg}")

    if report_path.suffix.lower() == '.html':
        return FileResponse(report_path)

    raise HTTPException(status_code=400, detail="不支持的报告类型")


@app.delete("/reports")
async def delete_reports(request: Request):
    """批量删除指定报告文件。"""
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    names = payload.get('names') if isinstance(payload, dict) else payload
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, list):
        raise HTTPException(status_code=400, detail="参数错误：应传入 names 列表")

    deleted = []
    for raw_name in names:
        if not isinstance(raw_name, str) or not raw_name or '..' in Path(raw_name).parts:
            continue

        report_path = (REPORTS_DIR / raw_name).resolve()
        if not report_path.is_file() or (REPORTS_DIR.resolve() not in report_path.parents and report_path != REPORTS_DIR.resolve()):
            continue

        report_path.unlink(missing_ok=True)
        deleted.append(raw_name)

    return {"status": "deleted", "count": len(deleted), "deleted": deleted}


@app.delete("/reports/{filename}")
async def delete_report(filename: str):
    """删除指定报告文件。"""
    if not filename or '..' in Path(filename).parts:
        raise HTTPException(status_code=400, detail="非法的报告文件名")

    report_path = (REPORTS_DIR / filename).resolve()
    if not report_path.is_file() or REPORTS_DIR.resolve() not in report_path.parents and report_path != REPORTS_DIR.resolve():
        raise HTTPException(status_code=404, detail="报告不存在")

    report_path.unlink(missing_ok=True)
    return {"status": "deleted", "filename": filename, "deleted": True}


# 挂载静态文件
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/history", StaticFiles(directory=str(HISTORY_DIR)), name="history")
app.mount("/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")


# ================================================================
# 启动入口
# ================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🔥 FlameDetect Pro - YOLOv8 火焰检测服务 v2.0.0")
    print("=" * 60)
    print(f"📁 数据目录: {DATA_DIR}")
    print(f"📁 模型目录: {MODELS_DIR}")
    print(f"📋 可用模型: {model_manager.available_models}")
    print(f"📡 当前模型: {model_manager.current_model_name if model_manager.current_model else '未加载'}")
    print(f"💻 设备: {model_manager.device}")
    print("=" * 60)
    print("🚀 服务启动中...")
    print("📡 访问地址: http://localhost:8000")
    print("📚 API 文档: http://localhost:8000/docs")
    print("=" * 60)

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=BACKEND_PORT,
        log_level="info",
        workers=1,
        access_log=False
    )