// ================================================================
// main.js - Electron 主进程 (完整优化版)
// ================================================================

const { app, BrowserWindow, ipcMain, dialog, shell, Menu } = require('electron');
const path = require('path');
const http = require('http');
const { spawn, exec, execFileSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const ffmpegPath = require('ffmpeg-static');

function getFfmpegPath() {
    if (!ffmpegPath) return null;
    const unpackedPath = ffmpegPath.replace(`${path.sep}app.asar${path.sep}`, `${path.sep}app.asar.unpacked${path.sep}`);
    if (app.isPackaged && fs.existsSync(unpackedPath)) return unpackedPath;
    return ffmpegPath;
}

// ================================================================
// 硬件加速配置
// ================================================================
app.disableHardwareAcceleration = false;
app.commandLine.appendSwitch('enable-gpu-rasterization');
app.commandLine.appendSwitch('enable-accelerated-2d-canvas');
app.commandLine.appendSwitch('ignore-gpu-blocklist');
app.commandLine.appendSwitch('enable-webgl');
app.commandLine.appendSwitch('enable-features', 'Canvas2dMSAA');

// ================================================================
// 配置
// ================================================================
const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;
const isMac = process.platform === 'darwin';
const isWin = process.platform === 'win32';

let mainWindow = null;
let backendProcess = null;
let backendPort = 8000;
let isBackendReady = false;
let backendStartAttempts = 0;
let backendReadyPoller = null;
let isQuitting = false;
let backendStartInProgress = false;
const MAX_BACKEND_RETRIES = 3;

// ================================================================
// 用户数据目录
// ================================================================
const userDataPath = app.getPath('userData');
const dataDir = path.join(userDataPath, 'flame-detect-data');

console.log(`📂 用户数据目录: ${dataDir}`);
console.log(`📂 应用目录: ${__dirname}`);
console.log(`📂 资源目录: ${process.resourcesPath || '未定义'}`);
console.log(`📦 运行环境: ${isDev ? '开发' : '生产'}`);

// ================================================================
// 确保数据目录存在
// ================================================================
function ensureDataDirectories() {
    const dirs = ['models', 'uploads', 'reports', 'logs', 'history'];
    const created = [];
    for (const d of dirs) {
        const dirPath = path.join(dataDir, d);
        try {
            if (!fs.existsSync(dirPath)) {
                fs.mkdirSync(dirPath, { recursive: true });
                created.push(d);
            }
        } catch (err) {
            console.error(`❌ 创建目录失败: ${dirPath} - ${err.message}`);
        }
    }

    const runtimeModelCandidates = [
        path.join(__dirname, 'data', 'models'),
        path.join(__dirname, 'models'),
        path.join(process.resourcesPath || '', 'models'),
        path.join(process.resourcesPath || '', 'app', 'models'),
    ];

    let copiedModel = null;
    for (const modelDir of runtimeModelCandidates) {
        if (!modelDir || !fs.existsSync(modelDir)) continue;

        const modelFiles = fs.readdirSync(modelDir).filter((name) => {
            const ext = path.extname(name).toLowerCase();
            return ['.pt', '.onnx', '.pth', '.weights'].includes(ext);
        }).sort();

        for (const modelFile of modelFiles) {
            const source = path.join(modelDir, modelFile);
            const target = path.join(dataDir, 'models', modelFile);
            const shouldCopy = !fs.existsSync(target) || fs.statSync(source).size !== fs.statSync(target).size;
            if (shouldCopy) {
                try {
                    fs.copyFileSync(source, target);
                    copiedModel = target;
                    console.log(`📦 已同步模型到运行目录: ${source} -> ${target}`);
                } catch (err) {
                    console.warn(`⚠️ 同步模型失败 ${source}: ${err.message}`);
                }
            }
        }

        if (copiedModel) break;
    }

    if (created.length > 0) {
        console.log(`📁 创建目录: ${created.join(', ')}`);
    }
    if (copiedModel) {
        console.log(`📦 可用模型已就绪: ${copiedModel}`);
    }
    return dataDir;
}

// ================================================================
// 路径查找工具
// ================================================================
function findFile(possiblePaths) {
    for (const p of possiblePaths) {
        try {
            if (p && fs.existsSync(p)) {
                return p;
            }
            if (isWin && (p === 'python.exe' || p === 'python')) {
                const resolvedPath = execFileSync('where.exe', [p], {
                    encoding: 'utf8',
                    windowsHide: true,
                }).split(/\r?\n/).map(item => item.trim()).find(Boolean);
                if (resolvedPath && fs.existsSync(resolvedPath)) {
                    return resolvedPath;
                }
            }
        } catch (e) {
            // 忽略
        }
    }
    return null;
}

function getResourcePath() {
    if (app.isPackaged) {
        return process.resourcesPath || path.join(path.dirname(app.getPath('exe')), 'resources');
    }
    return __dirname;
}

function killProcessesOnPort(port) {
    return new Promise((resolve) => {
        if (!isWin) {
            resolve([]);
            return;
        }

        exec(`netstat -ano | findstr :${port} | findstr LISTENING`, (err, stdout) => {
            const pids = [...new Set((stdout || '')
                .split(/\r?\n/)
                .map(line => line.trim())
                .filter(Boolean)
                .map(line => line.split(/\s+/).pop())
                .filter(pid => /^\d+$/.test(pid)))];

            if (!pids.length) {
                resolve([]);
                return;
            }

            let remaining = pids.length;
            const killed = [];

            pids.forEach((pid) => {
                exec(`taskkill /F /PID ${pid} >nul 2>&1`, (killErr) => {
                    if (!killErr) {
                        killed.push(pid);
                        console.log(`🧹 已清理占用端口 ${port} 的进程 PID=${pid}`);
                    } else {
                        console.warn(`⚠️ 清理端口 ${port} 失败 PID=${pid}: ${killErr.message}`);
                    }

                    remaining -= 1;
                    if (remaining === 0) {
                        resolve(killed);
                    }
                });
            });
        });
    });
}

// ================================================================
// 查找后端 (优先 backend.exe，失败时才回退到 Python)
// ================================================================
function findBackend() {
    const resourcePath = getResourcePath();
    const possiblePaths = [];
    const rootBackendExe = path.join(__dirname, 'backend.exe');
    const distBackendExe = path.join(__dirname, 'dist', 'backend.exe');

    console.log('🔍 开始查找后端...');

    // 1. 最优先顺序：project root backend.exe > flamegpu python > dist/backend.exe
    if (isWin) {
        if (isDev || !app.isPackaged) {
            possiblePaths.push(rootBackendExe, distBackendExe);
        }

        if (app.isPackaged) {
            possiblePaths.push(
                path.join(resourcePath, 'backend.exe'),
                path.join(resourcePath, 'app', 'backend.exe'),
                path.join(path.dirname(app.getPath('exe')), 'backend.exe'),
                rootBackendExe,
                distBackendExe
            );
        }
    }

    // 2. 如果没有有效的 backend.exe，则优先直接使用已验证的 flamegpu Python 环境，避免旧 dist CPU 版本回退
    if (isWin) {
        const localPythonCandidates = [
            path.join(__dirname, 'venv_backup', 'Scripts', 'python.exe'),
            path.join(__dirname, 'venv', 'Scripts', 'python.exe'),
            path.join(__dirname, 'venv', 'bin', 'python3'),
            path.join(__dirname, 'venv311', 'Scripts', 'python.exe'),
            path.join(__dirname, '.venv', 'Scripts', 'python.exe'),
        ];

        if (isDev || !app.isPackaged) {
            possiblePaths.push(...localPythonCandidates);
        }

        if (app.isPackaged) {
            possiblePaths.push(
                path.join(resourcePath, 'venv_backup', 'Scripts', 'python.exe'),
                path.join(resourcePath, 'venv', 'Scripts', 'python.exe'),
                path.join(resourcePath, 'app', 'venv', 'Scripts', 'python.exe'),
                ...localPythonCandidates,
            );
        }
    }

    // 3. 系统 Python (兜底)
    if (isWin) {
        // 优先使用当前开发环境中的 Python，避免选中未安装依赖的旧版本。
        possiblePaths.push('python.exe', 'python');
        for (let v of ['313', '312', '311', '310', '39', '38']) {
            possiblePaths.push(`C:\\Python${v}\\python.exe`);
            possiblePaths.push(`C:\\Users\\${process.env.USERNAME}\\AppData\\Local\\Programs\\Python\\Python${v}\\python.exe`);
        }
    } else {
        possiblePaths.push('python3', 'python');
    }

    // 去重
    const uniquePaths = [...new Set(possiblePaths)];

    // 打印所有查找路径（调试）
    console.log('🔍 查找路径:');
    uniquePaths.forEach(p => console.log(`   ${p}`));

    const found = findFile(uniquePaths);

    if (found) {
        console.log(`✅ 找到后端: ${found}`);
        return found;
    }

    console.warn('⚠️ 未找到后端，使用默认命令');
    return isWin ? 'python.exe' : 'python3';
}

// ================================================================
// 获取后端脚本路径
// ================================================================
function getBackendScriptPath() {
    const resourcePath = getResourcePath();
    const possiblePaths = [];

    if (isDev) {
        possiblePaths.push(
            path.join(__dirname, 'main.py'),
            path.join(__dirname, 'start_backend.py'),
        );
    }

    if (app.isPackaged) {
        possiblePaths.push(
            path.join(resourcePath, 'app', 'main.py'),
            path.join(resourcePath, 'app', 'start_backend.py'),
            path.join(resourcePath, 'main.py'),
            path.join(resourcePath, 'start_backend.py'),
        );
    }

    possiblePaths.push(
        path.join(__dirname, 'main.py'),
        path.join(path.dirname(__dirname), 'main.py'),
    );

    const found = findFile(possiblePaths);
    if (found) {
        console.log(`✅ 找到后端脚本: ${found}`);
        return found;
    }

    console.error('❌ 未找到后端脚本!');
    return path.join(__dirname, 'main.py');
}

// ================================================================
// 启动后端
// ================================================================
function startBackend() {
    if (backendStartInProgress || (backendProcess && !backendProcess.killed)) {
        console.log('ℹ️ 后端已启动或正在启动，跳过重复启动');
        return;
    }

    backendStartInProgress = true;
    ensureDataDirectories();

    killProcessesOnPort(backendPort)
        .then(() => {
            return new Promise((resolve) => {
                exec('taskkill /F /IM backend.exe 2>nul', () => resolve());
            });
        })
        .then(() => {
            launchBackend();
        })
        .catch((err) => {
            console.warn(`⚠️ 清理端口占用失败: ${err.message}`);
            launchBackend();
        });
}

function waitForBackendReady() {
    if (backendReadyPoller) {
        clearInterval(backendReadyPoller);
    }

    const startedAt = Date.now();
    backendReadyPoller = setInterval(() => {
        if (isBackendReady) {
            clearInterval(backendReadyPoller);
            backendReadyPoller = null;
            return;
        }

        const req = http.request({
            host: '127.0.0.1',
            port: backendPort,
            path: '/health',
            method: 'GET',
            timeout: 1500,
        }, (res) => {
            if (res.statusCode >= 200 && res.statusCode < 500) {
                isBackendReady = true;
                clearInterval(backendReadyPoller);
                backendReadyPoller = null;
                console.log(`✅ 实际健康检查通过，后端已就绪 on http://127.0.0.1:${backendPort}`);
                if (mainWindow && !mainWindow.isDestroyed()) {
                    mainWindow.webContents.send('backend-ready');
                }
            }
            res.resume();
        });

        req.on('error', () => {
            // 后端尚未完全就绪，继续轮询
        });

        req.on('timeout', () => {
            req.destroy();
        });

        req.end();

        if (Date.now() - startedAt > 45000) {
            clearInterval(backendReadyPoller);
            backendReadyPoller = null;
            console.warn('⚠️ 健康检查超时，后端可能仍在启动中');
        }
    }, 1000);
}

function launchBackend() {
    const backendPath = findBackend();
    const backendFileName = backendPath ? path.basename(backendPath).toLowerCase() : '';
    const isBackendExe = backendFileName === 'backend.exe' && backendPath && fs.existsSync(backendPath);
    const isPythonExecutable = backendFileName === 'python.exe' && backendPath && fs.existsSync(backendPath);
    const backendScriptPath = getBackendScriptPath();

    console.log(`🔧 启动后端: ${backendPath}`);
    console.log(`📂 数据目录: ${dataDir}`);
    waitForBackendReady();

    const env = {
        ...process.env,
        PYTHONUNBUFFERED: '1',
        FLAME_DETECT_DATA_DIR: dataDir,
        FLAME_DETECT_PORT: String(backendPort),
    };

    let spawnCommand = backendPath;
    let spawnArgs = [];
    let spawnCwd = path.dirname(backendPath);

    if (isBackendExe) {
        const rootBackendPath = path.join(__dirname, 'backend.exe');
        const distBackendPath = path.join(__dirname, 'dist', 'backend.exe');
        if (backendPath === distBackendPath && fs.existsSync(rootBackendPath)) {
            console.warn('⚠️ 检测到根目录真实 backend.exe，强制忽略 dist/backend.exe 的旧版本');
            spawnCommand = rootBackendPath;
            spawnArgs = [];
            spawnCwd = path.dirname(rootBackendPath);
        } else {
            console.log('🚀 使用独立后端 (backend.exe)');
            console.log(`📂 后端目录: ${path.dirname(backendPath)}`);
        }
    } else if (isPythonExecutable) {
        console.log('🚀 使用目标电脑的 Python 解释器启动后端');
        spawnCommand = backendPath;
        spawnArgs = [backendScriptPath];
        spawnCwd = path.dirname(backendScriptPath);
    } else {
        console.error('❌ 未找到可用后端：请确认安装包包含 resources\\backend.exe，或目标电脑已安装后端依赖。');
        if (mainWindow && !mainWindow.isDestroyed()) {
            dialog.showErrorBox(
                '后端启动失败',
                    '未找到可用后端。请重新安装完整安装包，或准备目标电脑的 Python 后端环境。'
            );
        }
        return;
    }

    backendProcess = spawn(spawnCommand, spawnArgs, {
        cwd: spawnCwd,
        env: env,
        stdio: ['pipe', 'pipe', 'pipe'],
        windowsHide: true,
    });

    setTimeout(() => {
        if (!isBackendReady && backendProcess) {
            console.log('⏳ backend.exe 正在启动，请稍候...');
        }
    }, 5000);

    // ---- 日志处理 ----
    backendProcess.stdout.on('data', (data) => {
        const output = data.toString();
        console.log(`[Backend] ${output.trim()}`);

        // 检测后端就绪
        if (output.includes('Uvicorn running on') ||
            output.includes('Application startup complete') ||
            output.includes('服务启动中') ||
            output.includes('Started server process') ||
            output.includes('FastAPI')) {
            isBackendReady = true;
            backendStartAttempts = 0;
            if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.webContents.send('backend-ready');
                console.log('✅ 后端已就绪，通知渲染进程');
            }
        }

        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('backend-log', output);
        }
    });

    backendProcess.stderr.on('data', (data) => {
        const output = data.toString();
        console.error(`[Backend Error] ${output.trim()}`);
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('backend-log', `❌ ${output}`);
        }
    });

    backendProcess.on('close', (code) => {
        console.log(`🔚 后端进程退出，退出码: ${code}`);
        isBackendReady = false;
        if (backendReadyPoller) {
            clearInterval(backendReadyPoller);
            backendReadyPoller = null;
        }
        backendStartInProgress = false;

        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('backend-exited', code);
        }

        if (!isQuitting && code !== 0 && code !== null && backendStartAttempts < MAX_BACKEND_RETRIES) {
            backendStartAttempts += 1;
            console.log(`🔄 后端异常退出，${backendStartAttempts}/${MAX_BACKEND_RETRIES} 次重试...`);
            setTimeout(() => {
                if (!backendProcess && !backendStartInProgress) {
                    startBackend();
                }
            }, 3000);
        } else if (code !== 0) {
            console.error(`❌ 后端多次启动失败，请检查日志`);
            if (mainWindow && !mainWindow.isDestroyed()) {
                dialog.showErrorBox(
                    '后端启动失败',
                    `后端服务多次启动失败 (退出码: ${code})\n\n请检查日志文件了解详情。`
                );
            }
        }

        backendProcess = null;
    });

    backendProcess.on('error', (err) => {
        console.error(`❌ 后端启动失败: ${err.message}`);
        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('backend-error', err.message);
        }
    });

    // 超时检测
    setTimeout(() => {
        if (!isBackendReady && backendProcess) {
            console.warn('⚠️ 后端启动超时 (30s)，可能正在加载模型...');
            if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.webContents.send('backend-log', '⏳ 后端正在加载模型，请稍候...');
            }
        }
    }, 30000);
}

// ================================================================
// 停止后端
// ================================================================
function stopBackend() {
    if (!backendProcess) return;

    console.log('🛑 停止后端服务...');

    if (isWin) {
        const pid = backendProcess.pid;
        if (pid) {
            try {
                exec(`taskkill /F /T /PID ${pid}`, (err) => {
                    if (err) {
                        console.log(`⚠️ 停止后端时出错: ${err.message}`);
                        try {
                            backendProcess.kill('SIGKILL');
                        } catch (e) {
                            console.log(`⚠️ 强制 kill 失败: ${e.message}`);
                        }
                    } else {
                        console.log('✅ 后端已停止 (taskkill)');
                    }
                });
            } catch (e) {
                console.log(`⚠️ 停止后端时出错: ${e.message}`);
                try {
                    backendProcess.kill('SIGKILL');
                } catch (e2) {
                    console.log(`⚠️ 强制 kill 失败: ${e2.message}`);
                }
            }
        }
    } else {
        backendProcess.kill('SIGTERM');
        setTimeout(() => {
            if (backendProcess && !backendProcess.killed) {
                backendProcess.kill('SIGKILL');
            }
        }, 3000);
    }

    backendProcess = null;
    isBackendReady = false;
}

// ================================================================
// 创建窗口
// ================================================================
function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 900,
        minHeight: 600,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js'),
            enableWebGL: true,
            hardwareAcceleration: true,
            webgl: true,
        },
        backgroundColor: '#f3f5f9',
        show: false,
        transparent: false,
        frame: true,
        autoHideMenuBar: true,
        titleBarStyle: 'hidden',
        titleBarOverlay: {
            color: '#f3f5f9',
            symbolColor: '#475569',
            height: 36,
        },
    });

    // 加载页面
    loadMainPage();

    mainWindow.once('ready-to-show', () => {
        const showWindow = () => {
            if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.show();
                mainWindow.focus();
            }
        };

        showWindow();
    });

    if (isDev) {
        mainWindow.webContents.openDevTools({ mode: 'detach' });
    }

    const menu = Menu.buildFromTemplate(getMenuTemplate());
    Menu.setApplicationMenu(menu);

    mainWindow.on('closed', () => {
        mainWindow = null;
    });

    mainWindow.webContents.on('did-finish-load', () => {
        if (isBackendReady) {
            mainWindow.webContents.send('backend-ready');
        }
    });

    console.log('🪟 主窗口创建完成');
}

// ================================================================
// 加载主页面
// ================================================================
function loadMainPage() {
    const resourcePath = getResourcePath();
    const possiblePaths = [
        path.join(__dirname, 'index.html'),
        path.join(resourcePath, 'app', 'index.html'),
        path.join(resourcePath, 'index.html'),
    ];

    for (const p of possiblePaths) {
        if (fs.existsSync(p)) {
            console.log(`📄 加载页面: ${p}`);
            mainWindow.loadFile(p);
            return;
        }
    }

    console.warn('⚠️ 未找到 index.html，尝试加载 localhost');
    mainWindow.loadURL(`http://localhost:${backendPort}`).catch((err) => {
        console.error(`❌ 加载页面失败: ${err.message}`);
        mainWindow.loadURL(`data:text/html,<h1>加载失败</h1><p>${err.message}</p>`);
    });
}

// ================================================================
// 菜单模板
// ================================================================
function getMenuTemplate() {
    const template = [
        {
            label: '文件',
            submenu: [
                {
                    label: '重启后端服务',
                    click: () => {
                        if (mainWindow && !mainWindow.isDestroyed()) {
                            mainWindow.webContents.send('restart-backend');
                        }
                        stopBackend();
                        backendStartAttempts = 0;
                        setTimeout(startBackend, 1500);
                    }
                },
                {
                    label: '打开数据目录',
                    click: () => {
                        const target = fs.existsSync(dataDir) ? dataDir : userDataPath;
                        shell.openPath(target).catch(err => {
                            console.error(`❌ 打开目录失败: ${err}`);
                            dialog.showErrorBox('打开失败', `无法打开目录: ${target}`);
                        });
                    }
                },
                { type: 'separator' },
                {
                    label: '退出',
                    accelerator: isMac ? 'Cmd+Q' : 'Ctrl+Q',
                    click: () => { app.quit(); }
                }
            ]
        },
        {
            label: '查看',
            submenu: [
                { role: 'reload' },
                { role: 'forceReload' },
                { role: 'toggleDevTools' },
                { type: 'separator' },
                { role: 'resetZoom' },
                { role: 'zoomIn' },
                { role: 'zoomOut' },
                { type: 'separator' },
                { role: 'togglefullscreen' }
            ]
        },
        {
            label: '窗口',
            submenu: [
                { role: 'minimize' },
                { role: 'zoom' },
                ...(isMac ? [
                    { type: 'separator' },
                    { role: 'front' },
                    { type: 'separator' },
                    { role: 'window' }
                ] : [
                    { role: 'close' }
                ])
            ]
        },
        {
            label: '帮助',
            submenu: [
                {
                    label: '查看日志目录',
                    click: () => {
                        const logDir = path.join(dataDir, 'logs');
                        const target = fs.existsSync(logDir) ? logDir : dataDir;
                        shell.openPath(target).catch(err => {
                            console.error(`❌ 打开目录失败: ${err}`);
                        });
                    }
                },
                {
                    label: '查看模型目录',
                    click: () => {
                        const modelDir = path.join(dataDir, 'models');
                        const target = fs.existsSync(modelDir) ? modelDir : dataDir;
                        shell.openPath(target).catch(err => {
                            console.error(`❌ 打开目录失败: ${err}`);
                        });
                    }
                },
                {
                    label: '查看历史记录目录',
                    click: () => {
                        const historyDir = path.join(dataDir, 'history');
                        if (!fs.existsSync(historyDir)) {
                            fs.mkdirSync(historyDir, { recursive: true });
                        }
                        shell.openPath(historyDir).catch(err => {
                            console.error(`❌ 打开目录失败: ${err}`);
                        });
                    }
                },
                { type: 'separator' },
                {
                    label: '关于',
                    click: () => {
                        dialog.showMessageBox({
                            title: '关于 FlameDetect Pro',
                            message: 'FlameDetect Pro v2.0.0',
                            detail: `基于 YOLOv8 的火焰检测系统\n\n技术栈:\n• Electron ${process.versions.electron}\n• FastAPI\n• YOLOv8\n• PyTorch\n\n数据目录: ${dataDir}`,
                            buttons: ['确定'],
                        }).catch(() => {});
                    }
                }
            ]
        }
    ];

    if (isMac) {
        template.unshift({
            label: app.getName(),
            submenu: [
                { role: 'about' },
                { type: 'separator' },
                { role: 'services' },
                { type: 'separator' },
                { role: 'hide' },
                { role: 'hideothers' },
                { role: 'unhide' },
                { type: 'separator' },
                { role: 'quit' }
            ]
        });
    }

    return template;
}

// ================================================================
// IPC 通信
// ================================================================
ipcMain.handle('get-backend-status', () => {
    return {
        running: backendProcess !== null && !backendProcess.killed,
        ready: isBackendReady,
        port: backendPort,
        pid: backendProcess ? backendProcess.pid : null,
    };
});

ipcMain.handle('set-title-bar-theme', (event, isDark) => {
    if (!mainWindow || mainWindow.isDestroyed() || typeof mainWindow.setTitleBarOverlay !== 'function') {
        return { success: false };
    }

    mainWindow.setTitleBarOverlay({
        color: isDark ? '#0f0f1a' : '#f3f5f9',
        symbolColor: isDark ? '#cbd5e1' : '#475569',
        height: 36,
    });
    return { success: true };
});

ipcMain.handle('restart-backend', () => {
    console.log('🔄 重启后端服务...');
    stopBackend();
    backendStartAttempts = 0;
    setTimeout(startBackend, 1500);
    return { success: true };
});

ipcMain.handle('open-directory', (event, dir) => {
    const dirPath = path.join(dataDir, dir || '');
    if (fs.existsSync(dirPath)) {
        shell.openPath(dirPath).catch(err => {
            console.error(`❌ 打开目录失败: ${err}`);
            return { success: false, error: err.message };
        });
        return { success: true };
    }
    return { success: false, error: '目录不存在' };
});

ipcMain.handle('get-app-path', () => {
    return __dirname;
});

ipcMain.handle('get-data-path', () => {
    return dataDir;
});

ipcMain.handle('convert-video', async (event, videoData) => {
    const converterPath = getFfmpegPath();
    if (!converterPath || !videoData) throw new Error('FFmpeg 或视频数据不可用');

    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'flamedetect-video-'));
    const inputPath = path.join(tempDir, 'input.webm');
    const outputPath = path.join(tempDir, 'output.mp4');
    try {
        fs.writeFileSync(inputPath, Buffer.from(videoData));
        await new Promise((resolve, reject) => {
            const converter = spawn(converterPath, [
                '-y', '-i', inputPath,
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', outputPath,
            ], { windowsHide: true });
            let errorOutput = '';
            converter.stderr.on('data', data => { errorOutput += data.toString(); });
            converter.on('error', reject);
            converter.on('close', code => {
                if (code === 0 && fs.existsSync(outputPath)) resolve();
                else reject(new Error(errorOutput.trim() || `FFmpeg 退出码: ${code}`));
            });
        });
        return fs.readFileSync(outputPath);
    } finally {
        fs.rmSync(tempDir, { recursive: true, force: true });
    }
});

// ================================================================
// 应用生命周期
// ================================================================
app.whenReady().then(() => {
    console.log('🚀 应用启动中...');
    console.log(`📦 运行环境: ${isDev ? '开发' : '生产'}`);
    console.log(`💻 平台: ${process.platform}`);

    ensureDataDirectories();
    startBackend();
    setTimeout(createWindow, 500);
});

app.on('window-all-closed', () => {
    stopBackend();
    if (!isMac) {
        app.quit();
    }
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});

app.on('before-quit', () => {
    isQuitting = true;
    stopBackend();
});

// ================================================================
// 错误处理
// ================================================================
process.on('uncaughtException', (err) => {
    console.error('💥 未捕获异常:', err);
    console.error('📚 堆栈:', err.stack);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('💥 未处理的 Promise 拒绝:', reason);
    if (reason instanceof Error) {
        console.error('📚 堆栈:', reason.stack);
    }
});

console.log('✅ Main process loaded');