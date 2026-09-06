// ================================================================
// preload.js - Electron 预加载脚本
// 安全地暴露 IPC API 给渲染进程
// ================================================================

const { contextBridge, ipcRenderer } = require('electron');

// 有效的 IPC 通道列表（白名单）
const VALID_CHANNELS = [
    'backend-ready',
    'backend-log',
    'backend-exited',
    'backend-error',
    'restart-backend'
];

contextBridge.exposeInMainWorld('electronAPI', {
    // ===== 后端状态 =====
    getBackendStatus: () => ipcRenderer.invoke('get-backend-status'),
    restartBackend: () => ipcRenderer.invoke('restart-backend'),
    openDirectory: (dir) => ipcRenderer.invoke('open-directory', dir),
    getAppPath: () => ipcRenderer.invoke('get-app-path'),
    getDataPath: () => ipcRenderer.invoke('get-data-path'),
    convertVideo: (videoData) => ipcRenderer.invoke('convert-video', videoData),
    setTitleBarTheme: (isDark) => ipcRenderer.invoke('set-title-bar-theme', Boolean(isDark)),
    onBackendReady: (callback) => {
        return ipcRenderer.on('backend-ready', (event, ...args) => callback(...args));
    },
    onBackendLog: (callback) => {
        return ipcRenderer.on('backend-log', (event, ...args) => callback(...args));
    },
    onBackendExited: (callback) => {
        return ipcRenderer.on('backend-exited', (event, ...args) => callback(...args));
    },
    onBackendError: (callback) => {
        return ipcRenderer.on('backend-error', (event, ...args) => callback(...args));
    },
    onRestartBackend: (callback) => {
        return ipcRenderer.on('restart-backend', (event, ...args) => callback(...args));
    },

    // ===== 事件监听（带清理功能） =====
    on: (channel, callback) => {
        if (!VALID_CHANNELS.includes(channel)) {
            throw new Error(`不允许的通道: ${channel}`);
        }
        const wrapper = (event, ...args) => callback(...args);
        ipcRenderer.on(channel, wrapper);
        // 返回清理函数
        return () => ipcRenderer.removeListener(channel, wrapper);
    },

    // ===== 一次性事件监听 =====
    once: (channel, callback) => {
        if (!VALID_CHANNELS.includes(channel)) {
            throw new Error(`不允许的通道: ${channel}`);
        }
        ipcRenderer.once(channel, (event, ...args) => callback(...args));
    },

    // ===== 移除所有监听器 =====
    removeAllListeners: (channel) => {
        if (channel && !VALID_CHANNELS.includes(channel)) {
            throw new Error(`不允许的通道: ${channel}`);
        }
        ipcRenderer.removeAllListeners(channel);
    },

    // ===== 平台信息 =====
    platform: process.platform,
    isPackaged: process.env.NODE_ENV === 'production' || (typeof app !== 'undefined' && app.isPackaged),
});

// 开发模式下的调试工具
if (process.env.NODE_ENV === 'development') {
    console.log('🔧 [Preload] 开发模式已启用');
    console.log('📦 平台:', process.platform);
}

console.log('✅ Preload script loaded successfully');