# FlameDetect Pro

基于 Electron、FastAPI 和 Ultralytics YOLO 的桌面端火焰检测系统。应用提供图片检测、视频实时监控、检测统计、历史记录和模型管理功能，适合用于本地演示、算法验证和火焰识别场景的快速测试。

## 功能

- **图片检测**：上传 JPG、JPEG、PNG、BMP 或 TIFF 图片，查看检测框、火焰数量、置信度和推理耗时。
- **视频监控**：读取本地视频或摄像头画面，按可调检测间隔进行实时检测，支持暂停、截图和检测趋势查看。
- **历史记录**：保存图片检测结果，支持查看详情、清空记录以及导出 JPG 图片或 JSON 数据。
- **模型管理**：扫描、上传和切换 YOLO 模型，支持 `.pt`、`.onnx`、`.pth` 和 `.weights` 格式。
- **运行状态**：查看当前模型、推理次数、设备类型、系统资源和缓存状态。
- **桌面应用体验**：Electron 负责窗口和媒体处理，应用启动时自动拉起本地 FastAPI 后端。

## 技术栈

- Electron 28
- FastAPI + Uvicorn
- Ultralytics YOLO
- OpenCV、NumPy、PyTorch
- Chart.js
- FFmpeg（由 `ffmpeg-static` 提供）

## 运行环境

- Node.js 18 或更高版本
- Python 3.9 或更高版本（仅从 Python 源码启动后端时需要）
- Windows、macOS 或 Linux
- 若使用 CUDA 推理，需要安装与本机驱动匹配的 PyTorch CUDA 版本

## 快速开始

### 方式一：启动桌面应用

在项目根目录执行：

```bash
npm install
npm start
```

Windows 也可以双击 `start_app.bat` 启动。启动时 Electron 会优先查找项目根目录中的 `backend.exe`，找不到时再尝试使用虚拟环境或系统 Python 启动 `main.py`。

### 方式二：单独启动后端

确保已准备好 Python 依赖后执行：

```bash
python start_backend.py
```

后端默认地址为 <http://localhost:8000>，FastAPI 交互式文档为 <http://localhost:8000/docs>。

> 仓库当前未提供 `requirements.txt`。如果使用源码方式运行后端，请根据 `main.py` 中的导入安装对应依赖，或使用项目已有的 Python 虚拟环境。直接使用已构建的 `backend.exe` 时不需要单独安装 Python。

## 模型使用

1. 将训练好的模型放入 `models/`，或在应用的“模型管理”页面上传。
2. 支持的模型扩展名：`.pt`、`.onnx`、`.pth`、`.weights`。
3. 模型文件需要大于 1 MB 才会被扫描为可用模型。
4. 首次启动如果没有可用模型，检测接口会提示“模型未加载”，请先上传并加载模型。

默认模型会按以下顺序尝试查找：数据目录中的模型、项目 `models/` 目录，以及应用内置的常见模型路径。实际使用时建议将模型放入应用数据目录，或通过界面上传。

## 配置

后端支持以下环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `FLAME_DETECT_DATA_DIR` | `./data` | 上传文件、模型、报告、日志和历史记录的根目录 |
| `FLAME_DETECT_PORT` | `8000` | FastAPI 服务端口 |

例如：

```bash
set FLAME_DETECT_DATA_DIR=D:\\FlameDetectData
set FLAME_DETECT_PORT=8001
python main.py
```

Electron 启动的后端会自动使用当前用户数据目录下的 `flame-detect-data`，不会把运行时数据写入安装包本身。

## 数据目录

开发模式下默认目录结构如下：

```text
data/
├─ history/     # 历史检测图片
├─ logs/        # 日志
├─ models/      # 模型文件
├─ reports/     # 检测报告
└─ uploads/     # 上传文件
```

图片上传大小上限为 50 MB，模型上传大小上限为 500 MB。应用内历史记录还会使用浏览器本地存储保存索引信息。

## 常用 API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查和当前运行状态 |
| `POST` | `/predict` | 上传图片或视频帧并执行检测 |
| `GET` | `/models/list` | 获取可用模型列表 |
| `POST` | `/upload_model` | 上传并加载模型 |
| `POST` | `/load_model` | 加载指定模型 |
| `POST` | `/load_default` | 加载默认模型 |
| `GET` | `/model_info` | 获取当前模型信息 |
| `GET` | `/stats` | 获取推理统计 |
| `POST` | `/cache/clear` | 清空推理缓存 |
| `GET` | `/docs` | 查看完整 OpenAPI 文档 |

`/predict` 支持 `conf_threshold`（0 到 1，默认 0.25）和 `return_image` 参数；后者开启时会返回标注后的图片数据。

## 构建安装包

```bash
npm run build
```

按平台构建：

```bash
npm run build:win
npm run build:mac
npm run build:linux
```

构建产物输出到 `dist/`。Windows 安装包使用 NSIS，应用名称为 **FlameDetect Pro**。

## 项目结构

```text
main.js          # Electron 主进程，负责窗口、后端进程和媒体转换
preload.js       # Electron 安全预加载桥接
index.html       # 前端界面
main.py          # FastAPI 服务和 YOLO 推理逻辑
start_backend.py # 单独启动后端的开发脚本
start_app.bat    # Windows 启动脚本
assets/          # 应用图标等资源
models/          # 开发环境模型目录
data/            # 运行时数据目录
```

## 许可证

本项目使用 MIT License。依赖项的许可证以各自上游项目声明为准。
