// ================================================================
// main.js - Electron 主进程 (完整优化版)
// ================================================================

const { app, BrowserWindow, ipcMain, dialog, shell, Menu } = require('electron');
const path = require('path');
const { spawn, exec } = require('child_process');
const fs = require('fs');
const os = require('os');

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
    if (created.length > 0) {
        console.log(`📁 创建目录: ${created.join(', ')}`);
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

// ================================================================
// 查找后端 (优先 backend.exe)
// ================================================================
function findBackend() {
    const resourcePath = getResourcePath();
    const possiblePaths = [];

    console.log('🔍 开始查找后端...');

    // 1. 优先查找 backend.exe (打包和开发环境)
    if (isWin) {
        // 开发环境：项目根目录
        if (isDev || !app.isPackaged) {
            possiblePaths.push(
                path.join(__dirname, 'backend.exe'),
                path.join(__dirname, 'dist', 'backend.exe'),
            );
        }

        // 打包环境：resources 目录
        if (app.isPackaged) {
            possiblePaths.push(
                path.join(resourcePath, 'backend.exe'),
                path.join(resourcePath, 'app', 'backend.exe'),
                path.join(path.dirname(app.getPath('exe')), 'backend.exe'),
            );
        }
    }

    // 2. 查找 venv (开发环境备用)
    if (isDev) {
        possiblePaths.push(
            path.join(__dirname, 'venv', 'Scripts', 'python.exe'),
            path.join(__dirname, 'venv', 'bin', 'python3'),
            path.join(__dirname, 'venv311', 'Scripts', 'python.exe'),
        );
    }

    // 3. 打包环境 venv (备用)
    if (app.isPackaged) {
        possiblePaths.push(
            path.join(resourcePath, 'venv', 'Scripts', 'python.exe'),
            path.join(resourcePath, 'app', 'venv', 'Scripts', 'python.exe'),
        );
    }

    // 4. 系统 Python (兜底)
    if (isWin) {
        // 常见 Python 安装路径
        for (let v of ['313', '312', '311', '310', '39', '38']) {
            possiblePaths.push(`C:\\Python${v}\\python.exe`);
            possiblePaths.push(`C:\\Users\\${process.env.USERNAME}\\AppData\\Local\\Programs\\Python\\Python${v}\\python.exe`);
        }
        possiblePaths.push('python.exe', 'python');
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
    ensureDataDirectories();

    const backendPath = findBackend();
    const isBackendExe = backendPath && backendPath.endsWith('.exe') && fs.existsSync(backendPath);

    console.log(`🔧 启动后端: ${backendPath}`);
    console.log(`📂 数据目录: ${dataDir}`);

    const env = {
        ...process.env,
        PYTHONUNBUFFERED: '1',
        FLAME_DETECT_DATA_DIR: dataDir,
        FLAME_DETECT_PORT: String(backendPort),
    };

    // 如果是 backend.exe，直接启动
    if (isBackendExe) {
        console.log('🚀 使用独立后端 (backend.exe)');
        console.log(`📂 后端目录: ${path.dirname(backendPath)}`);

        backendProcess = spawn(backendPath, [], {
            cwd: path.dirname(backendPath),
            env: env,
            stdio: ['pipe', 'pipe', 'pipe'],
            windowsHide: true,
        });

        // 设置超时提醒
        setTimeout(() => {
            if (!isBackendReady && backendProcess) {
                console.log('⏳ 后端正在启动，请稍候...');
            }
        }, 5000);

    } else {
        // 使用 Python + 脚本
        const scriptPath = getBackendScriptPath();

        if (!fs.existsSync(scriptPath)) {
            console.error(`❌ 后端脚本不存在: ${scriptPath}`);
            if (mainWindow) {
                dialog.showErrorBox(
                    '启动失败',
                    `无法找到后端脚本:\n${scriptPath}\n\n请确保应用安装完整。`
                );
            }
            return;
        }

        const scriptDir = path.dirname(scriptPath);
        const pythonDir = path.dirname(backendPath);

        if (pythonDir && fs.existsSync(pythonDir)) {
            env.PATH = pythonDir + path.delimiter + (process.env.PATH || '');
        }

        if (isDev) {
            const venvSitePackages = path.join(__dirname, 'venv', 'lib', 'python3.11', 'site-packages');
            if (fs.existsSync(venvSitePackages)) {
                env.PYTHONPATH = venvSitePackages + path.delimiter + (env.PYTHONPATH || '');
            }
        }

        console.log(`🐍 使用 Python: ${backendPath}`);
        console.log(`📄 脚本: ${scriptPath}`);

        backendProcess = spawn(backendPath, [scriptPath], {
            cwd: scriptDir,
            env: env,
            stdio: ['pipe', 'pipe', 'pipe'],
            windowsHide: true,
        });
    }

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

        if (mainWindow && !mainWindow.isDestroyed()) {
            mainWindow.webContents.send('backend-exited', code);
        }

        // 自动重启
        if (code !== 0 && code !== null && backendStartAttempts < MAX_BACKEND_RETRIES) {
            backendStartAttempts++;
            console.log(`🔄 后端异常退出，${backendStartAttempts}/${MAX_BACKEND_RETRIES} 次重试...`);
            setTimeout(startBackend, 3000);
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
        backgroundColor: '#0f0f1a',
        show: false,
        transparent: false,
        frame: true,
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

        if (isBackendReady) {
            showWindow();
        } else {
            let attempts = 0;
            const maxAttempts = 100;
            const checkReady = setInterval(() => {
                attempts++;
                if (isBackendReady || attempts >= maxAttempts) {
                    clearInterval(checkReady);
                    showWindow();
                    if (!isBackendReady) {
                        console.warn('⚠️ 窗口显示但后端未就绪，可能加载较慢');
                    }
                }
            }, 100);
        }
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