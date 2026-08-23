#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ================================================================
# main.py - FastAPI 后端服务
# 优化: 更好的内存管理、缓存策略、错误处理、日志系统
# ================================================================

from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
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
import atexit

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
DATA_DIR = os.environ.get('FLAME_DETECT_DATA_DIR', './data')
STATIC_DIR = Path(__file__).parent
UPLOAD_DIR = Path(DATA_DIR) / 'uploads'
MODELS_DIR = Path(DATA_DIR) / 'models'
REPORTS_DIR = Path(DATA_DIR) / 'reports'
LOGS_DIR = Path(DATA_DIR) / 'logs'
HISTORY_DIR = Path(DATA_DIR) / 'history'

for d in [UPLOAD_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR, HISTORY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logger.info(f"📁 数据目录: {DATA_DIR}")
logger.info(f"📁 模型目录: {MODELS_DIR}")

# ================================================================
# 配置
# ================================================================
def find_default_model() -> Optional[str]:
    """自动查找可用的默认模型"""
    possible_paths = []

    # 扫描 models 目录
    for ext in ['.pt', '.onnx', '.pth', '.weights']:
        for file_path in MODELS_DIR.glob(f'*{ext}'):
            if file_path.is_file() and file_path.stat().st_size > 1024 * 1024:  # > 1MB
                possible_paths.append(str(file_path))

    # 常见的固定路径
    possible_paths.extend([
        r"C:\Users\hemen\Desktop\训练结果9\fire_detect_finetune\weights\best.pt",
        "./models/best.pt",
        "./models/fire.pt",
        "./models/yolov8n.pt",
        "./models/yolov8s.pt",
    ])

    for path in possible_paths:
        if os.path.exists(path) and os.path.getsize(path) > 1024 * 1024:
            logger.info(f"✅ 找到默认模型: {path}")
            return path

    logger.warning("⚠️ 未找到默认模型")
    return None


DEFAULT_MODEL_PATH = find_default_model()
MAX_IMAGE_SIZE = 4096
MAX_HISTORY_SIZE = 100
SUPPORTED_EXTENSIONS = {'.pt', '.onnx', '.pth', '.weights'}
SUPPORTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

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

        # 加载默认模型 (异步)
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

    def load_model(self, model_path: str) -> tuple:
        """加载指定模型"""
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

            # 清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 加载模型
            model = YOLO(model_path)

            # 预热 (使用小尺寸)
            test_img = np.zeros((320, 320, 3), dtype=np.uint8)
            _ = model(test_img, verbose=False)

            # 更新状态
            self.current_model = model
            self.current_model_path = model_path
            self.current_model_name = os.path.basename(model_path)
            self.model_classes = model.names
            self.load_time = time.time() - start_time

            # 清空缓存
            self.cache.clear()

            logger.info(f"✅ 模型加载成功: {self.current_model_name} ({self.load_time:.2f}s)")
            logger.info(f"💻 设备: {self.device}")

            return True, f"模型加载成功 (耗时: {self.load_time:.2f}s)"

        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            return False, f"模型加载失败: {str(e)}"
        finally:
            self.is_loading = False

    def load_model_by_name(self, model_name: str) -> tuple:
        """通过模型名称加载"""
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
        if self.current_model is None:
            raise ValueError("模型未加载")

        start_time = time.time()

        # 执行推理
        results = self.current_model(image_path, conf=conf_threshold, verbose=False)[0]

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
            "names": self.current_model.names,
            "total": len(boxes),
            "model_name": self.current_model_name,
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
    allow_origins=["*"],
    allow_credentials=True,
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
        "memory_usage_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 1)
    }


# ================================================================
# 检测接口
# ================================================================
@app.post("/predict")
async def predict(
        file: UploadFile = File(...),
        conf_threshold: Optional[float] = Query(0.25, ge=0.0, le=1.0),
        return_image: Optional[bool] = Query(False)
):
    """图片/视频帧检测"""
    if model_manager.current_model is None:
        raise HTTPException(status_code=503, detail="模型未加载，请先上传模型")

    try:
        contents = await file.read()

        # 检查文件大小
        if len(contents) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="文件过大，请上传小于50MB的图片")

        # 处理图片
        img = process_image(contents)

        # 保存临时文件
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, img)
            tmp_path = tmp_file.name

        try:
            # 推理
            result = model_manager.predict(tmp_path, conf_threshold)
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
            "conf_threshold": result["conf_threshold"]
        }

        # 如果需要返回标注图片
        if return_image:
            annotated = draw_boxes_on_image(img, result["boxes"], result["names"])
            _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            response["image"] = base64.b64encode(buffer).decode('utf-8')

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 检测错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    if not any(file.filename.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"只支持 {', '.join(SUPPORTED_EXTENSIONS)} 格式"
        )

    try:
        content = await file.read()
        if len(content) > 500 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="模型文件过大，请上传小于500MB的文件")

        # 保存
        file_path = MODELS_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"📁 模型已保存: {file_path}")

        # 刷新列表
        model_manager.scan_models()

        # 尝试加载
        success, message = model_manager.load_model(str(file_path))

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

    success, message = model_manager.load_model_by_name(model_name)

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

    success, message = model_manager.load_model(DEFAULT_MODEL_PATH)

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


# 挂载静态文件
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/history", StaticFiles(directory=str(HISTORY_DIR)), name="history")


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
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=1,
        access_log=False
    )