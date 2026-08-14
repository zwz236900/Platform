#!/usr/bin/env bash
# ============================================================
# Platform — 关闭后端服务脚本
# 1. 释放 SAM 模型显存（若已加载）
# 2. 优雅停止后端 API 进程（端口检测 → SIGTERM → SIGKILL）
# 3. 可选：停止 sam3 容器
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 可配置项 ----
API_URL="${API_URL:-http://localhost:8000}"
PROJECT="${PROJECT:-项目4}"

echo "============================================"
echo "  Platform — 关闭后端服务"
echo "============================================"
echo ""

# ---- 1. 释放 SAM 模型显存 ----
echo ">>> 释放 SAM 模型显存..."
if curl -sS --connect-timeout 3 -X POST "$API_URL/api/projects/$PROJECT/sam/server-release" \
    -d "" 2>/dev/null; then
    echo ""
    echo "✅ SAM 模型已释放"
else
    echo "⚠️  无法释放 SAM 模型（可能未加载或 API 不可达）"
fi

# ---- 2. 停止后端 API（优先：端口检测）----
echo ""
echo ">>> 停止后端 API..."

# 方法 1：通过监听端口找 PID（最准确）
API_PIDS=$(fuser 8000/tcp 2>/dev/null | tr -d ' ')

# 方法 2：pgrep 兜底（端口检测不到时）
if [ -z "$API_PIDS" ]; then
    API_PIDS=$(pgrep -f "[u]vicorn main" 2>/dev/null || true)
fi

if [ -z "$API_PIDS" ]; then
    echo "⚠️  未找到运行中的后端 API 进程（端口 8000 无监听）"
else
    for pid in $API_PIDS; do
        # 跳过非数字
        case "$pid" in
            ''|*[!0-9]*) continue ;;
        esac
        echo "  停止 PID: $pid (SIGTERM)"
        kill "$pid" 2>/dev/null || true
    done

    # 等待退出（最多 5 秒）
    for i in $(seq 1 5); do
        ALL_GONE=true
        for pid in $API_PIDS; do
            case "$pid" in ''|*[!0-9]*) continue ;; esac
            kill -0 "$pid" 2>/dev/null && ALL_GONE=false
        done
        $ALL_GONE && break
        sleep 1
    done

    # SIGKILL 兜底
    for pid in $API_PIDS; do
        case "$pid" in ''|*[!0-9]*) continue ;; esac
        if kill -0 "$pid" 2>/dev/null; then
            echo "  强制结束 PID: $pid (SIGKILL)"
            kill -9 "$pid" 2>/dev/null || true
        fi
    done

    # 最终验证
    if fuser 8000/tcp >/dev/null 2>&1; then
        echo "❌ 端口 8000 仍被占用！请手动检查"
    else
        echo "✅ 后端 API 已停止（端口 8000 已释放）"
    fi
fi

# ---- 3. 可选：停止 sam3 容器 ----
echo ""
read -p ">>> 是否停止 sam3 Docker 容器？[y/N] " -n 1 -r REPLY 2>/dev/null || REPLY=""
echo
if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    if docker stop sam3 2>/dev/null; then
        echo "✅ sam3 容器已停止"
    else
        echo "⚠️  sam3 容器未在运行"
    fi
else
    echo "   已跳过"
fi

echo ""
echo "============================================"
echo "  所有服务已停止"
echo "============================================"
