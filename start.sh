#!/usr/bin/env bash
# ============================================================
# Platform — 前后端同时启动脚本
# 1. 检测 Docker 及必要容器
# 2. 启动后端 API 服务 (FastAPI + uvicorn)
# 3. 启动前端 (PySide6)
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/Platform/frontend"
BACKEND_DIR="$SCRIPT_DIR/Platform/backend"
VENV_DIR="$FRONTEND_DIR/.venv"

# ---- PID 跟踪 ----
API_PID=""
FRONTEND_PID=""

cleanup() {
    echo ""
    echo ">>> 正在关闭服务..."
    if [ -n "$API_PID" ] && kill -0 "$API_PID" 2>/dev/null; then
        kill "$API_PID" 2>/dev/null
        echo ">>> 后端 API 已停止"
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill "$FRONTEND_PID" 2>/dev/null
        taskkill.exe /PID "$FRONTEND_PID" /F 2>/dev/null
        echo ">>> 前端 GUI 已停止"
    fi
    exit 0
}
trap cleanup INT TERM

# ============================================================
# 1. Docker 检测
# ============================================================
echo "============================================"
echo "  Platform — 前后端启动"
echo "============================================"
echo ""

# 检查 Docker 是否可用（容器由 API 按需启动）
echo ">>> 检测 Docker..."
if ! docker ps >/dev/null 2>&1; then
    echo "❌ Docker 未运行！请先启动 Docker Desktop。"
    exit 1
fi
echo "✅ Docker 可用（容器按需启动）"

# ============================================================
# 2. 启动后端 API
# ============================================================
echo ""
echo ">>> 启动后端 API (端口 8000)..."

cd "$BACKEND_DIR"

# 检查后端依赖
python3 -c "import fastapi, uvicorn" 2>/dev/null || {
    echo ">>> 安装后端依赖..."
    pip3 install --user fastapi uvicorn python-multipart aiofiles >/dev/null 2>&1
}

python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 &
API_PID=$!
echo "✅ 后端 API 已启动 (PID: $API_PID)"

# 等待 API 就绪
echo -n ">>> 等待 API 就绪"
for i in $(seq 1 15); do
    if curl -s http://localhost:8000/api/projects >/dev/null 2>&1; then
        echo " ✅"
        break
    fi
    echo -n "."
    sleep 1
done
if ! curl -s http://localhost:8000/api/projects >/dev/null 2>&1; then
    echo ""
    echo "⚠️  API 可能未就绪，请检查日志"
fi

# ============================================================
# 3. 启动前端 (Windows PySide6 GUI)
# ============================================================
echo ""
echo ">>> 启动前端 (Windows 侧)..."

# WSL 中调用 Windows exe 时，exe 路径用 WSL 格式，参数用 Windows 格式
WIN_PYTHON_EXE="/mnt/e/Photogrammetry/Frontend/.venv/Scripts/python.exe"
WIN_MAIN="e:/Photogrammetry/Frontend/main.py"
WIN_FRONTEND_DIR="/mnt/e/Photogrammetry/Frontend"

# 检查 Windows Python 是否可访问
if [ ! -f "$WIN_PYTHON_EXE" ]; then
    echo "❌ Windows Python 未找到: $WIN_PYTHON_EXE"
    echo "   请确保 E:\\Photogrammetry\\Frontend\\.venv 已创建"
    cleanup
    exit 1
fi

# 检查前端文件是否已同步
if [ ! -f "$WIN_FRONTEND_DIR/main.py" ]; then
    echo "⚠️  Windows 端未找到 main.py，正在同步..."
    rsync -av --delete \
        --exclude ".venv" --exclude "__pycache__" --exclude ".git" \
        --exclude "node_modules" --exclude "*.pyc" \
        "$FRONTEND_DIR/" "$WIN_FRONTEND_DIR/"
    echo "✅ 同步完成"
fi

# 启动 Windows 前端（exe 路径用 WSL 格式，参数用 Windows 原生路径）
"$WIN_PYTHON_EXE" "$WIN_MAIN" &
FRONTEND_PID=$!
echo "✅ 前端 GUI 已启动 (WSL PID: $FRONTEND_PID)"

echo ""
echo "============================================"
echo "  后端 API : http://localhost:8000"
echo "  前端     : Windows PySide6 GUI"
echo "  Ctrl+C   : 停止所有服务"
echo "============================================"
echo ""

# 等待前端进程退出
wait "$FRONTEND_PID" 2>/dev/null || true

# 前端退出后清理
cleanup
