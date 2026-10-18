# Platform 数据流梳理

> 平台架构：**Windows PySide6 桌面前端 + WSL FastAPI 后端 + 四个 Docker 计算容器**。
> 本文梳理各模块之间的数据流动，以及前端每次点击最终驱动了哪些后端调用、改写了磁盘上哪些数据。
> 基于 2026-08-13 当前代码整理。

---

## 1. 总体架构

**前端是唯一入口，后端只做编排，重计算全在 Docker 容器里。**

```
Windows PySide6 GUI（frontend/）──HTTP localhost:8000──▶ FastAPI 后端（backend/main.py）
                                                              │ 编排
                                          ┌───────────────────┼───────────────────┐
                                   docker exec            docker exec        docker exec
                                          ▼                    ▼                   ▼
                                   colmap-dev             openmvs              gs3 / sam3
                                   sfm_cli.py             mvs_cli.py           gs_cli.py / sam_worker.py
                                          │                    │                   │
                                          └──────────┬─────────┴──────────┬────────┘
                                                     ▼                     ▼
                                   DATA 共享卷  /home/zwz/Code/DATA  ↔  /data（容器内挂载）
                                   SAM 代码卷  /home/zwz/Code/SAM   ↔  /workspace（sam3 容器内挂载）
```

- **前端**：7 个页面，全部用 `QNetworkAccessManager` 走 HTTP，`API_BASE = "http://localhost:8000"`（`frontend/pages/projects.py:44`）。
- **后端**：FastAPI 不做计算，把表单参数翻译成 `docker exec <容器> python3 …cli.py` 命令，在后台线程 `Popen` 启动。
- **数据交换**：四个容器与两个算法模块通过共享卷 `/home/zwz/Code/DATA`（容器内挂载为 `/data`）读写中间产物；sam3 另把代码卷 `/home/zwz/Code/SAM`（挂载为 `/workspace`）用于 SAM 脚本。

**模块清单**

| 目录 | 内容 |
|---|---|
| `frontend/` | PySide6 GUI：`main.py` → `app.py`（7 页导航）+ `pages/*` + `widgets/sidebar.py` |
| `backend/main.py` | FastAPI 路由 + 全部编排逻辑（约 70 个端点） |
| `backend/modules/` | `task_manager`（项目 CRUD）、`docker_runner` / `mvs_runner` / `gs_runner` / `sam_runner`（容器执行器）、`image_preprocessor`（去元数据/去畸变）、`docker_logger`（容器日志跟随）、`logger` |
| `backend/algorithm/` | `pointcloud_filter`（点云过滤）、`image_mask`（图像掩码）、`common`（COLMAP/PLY 读写）、`SAM_box` / `SAM_point`（SAM 单图脚本） |

---

## 2. 端到端管线数据流

真正的依赖关系不是一条直线，而是**主链 + 两条支线**：

- **主链**：输入图片 → SfM 稀疏重建 → 去畸变 → MVS 稠密重建 → 点云过滤
- **SAM 支线**：输入图片 → SAM 分割 → 掩码分别喂给「点云过滤」与「图像掩码」两个算法
- **3DGS 支线**：只依赖 `sparse/0` + 原始图片，与 MVS/SAM 无关

```
Input/input_image/  ──►  SfM 稀疏重建 ──►  SFM_output/sparse/0 + database.db
    │  (colmap-dev · sfm_cli.py)              │
    │                                        ▼
    │                             去畸变 image_undistorter
    │                                        │
    │                                        ▼
    │                             SFM_output/undistorted/（images + PINHOLE sparse）
    │                                        │
    │                                        ▼
    │                             MVS 稠密重建 ──►  MVS_output/colmap/fused.ply
    │                             (colmap / openmvs)     或 openmvs/scene_dense.ply
    │                                        │
    │                                        ▼
    │                             算法·点云过滤 ──►  algorithm_output/filtered_scene.ply
    │                             (pointcloud_filter)      ▲
    │                                                    │ SAM 掩码
    ├──────────────►  SAM 分割  ──►  SAM_output/mask*.png ┘
    │               (sam3)            + manifest.json     └──► 算法·图像掩码 ──► algorithm_output/input_masked/
    │
    └──────►  3DGS（可选）──►  3DGS_output/（只依赖 sparse/0 + 原始图片）
```

### 各阶段 输入 / 引擎 / 输出

| 阶段 | 输入（读） | 执行者 | 输出（写） |
|---|---|---|---|
| 项目管理 / 上传 | 用户选图（可 zip） | FastAPI 直接落盘 + exiftool 去 IPTC/EXIF | `Input/input_image/` |
| SfM | `input_image/` | colmap-dev · `sfm_cli.py`（extract→match→sfm） | `SFM_output/sparse/0` · `database.db` · `pipeline_result.json` · `pipeline.log` |
| 稀疏点云导出 | `SFM_output/sparse/0`（points3D） | 进程内 `common.ensure_sparse_ply`（下载时触发） | `SFM_output/sparse/0/points3D.ply` |
| 去畸变（可选） | `input_image/` + `sparse/0` | colmap `image_undistorter` | `SFM_output/undistorted/`（images + PINHOLE sparse） |
| MVS 稠密 | `sparse/0`（colmap）或 `SFM_output`（openmvs） | colmap patch_match/fuse 或 openmvs `mvs_cli.py` | `MVS_output/colmap/fused.ply` 或 `openmvs/scene_dense.ply` |
| SAM 分割 | `input_image/` | sam3 · 批量 `SAM_CLI.py` / 交互常驻 `sam_worker.py` | `SAM_output/`：`mask*.png` · `manifest.json` · `box_results/` · `point_results/` · `point_marks/` |
| 点云过滤 | MVS 稠密 PLY + `undistorted/sparse`（cameras.bin/images.bin）+ SAM 掩码 | 进程内 `pointcloud_filter.py` | `algorithm_output/filtered_scene.ply` |
| 图像掩码 | 去畸变图 + SAM 掩码 | 进程内 `image_mask.py` | `algorithm_output/input_masked/`（RGBA PNG） |
| 3DGS（可选） | `input_image/` + `sparse/0` | gs3 · `gs_cli.py`（prepare→train→render） | `3DGS_output/`：`train/ours_*/renders/*.png` · `point_cloud/` |

---

## 3. 后端模块数据流

### 3.1 任务状态机（`task.json` 是唯一真源）

```
前端 POST /run → main.py 写 task.json(status=running) → run_pipeline_async 起子进程
   → 进程退出 → _check_process_completion 检测
   → sync_project_from_result 读 SFM_output/pipeline_result.json
   → 回灌 task.json（steps / elapsed_seconds / status）
```

- `backend/modules/task_manager.py` 提供项目 CRUD 与路径助手，所有状态落在 `DATA_DIR/<项目名>/task.json`。
- 各 runner 的 `_check_*_completion`（SfM/MVS/3DGS/SAM 各自一份）负责「进程退出 → 回写状态 → 从注册表移除」。

### 3.2 执行模型（所有 runner 同构）

`docker_runner.py` / `mvs_runner.py` / `gs_runner.py` / `sam_runner.py` 共享同一种模式：

1. 后端把参数翻译成 `docker exec <容器> python3 …cli.py …` 命令。
2. 后台守护线程 `subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT, bufsize=1)`，进程存入全局注册表 `dict[项目名]`。
3. stdout 逐行实时写入**两个文件**：
   - 项目日志（`SFM_output/pipeline.log` / `MVS_output/openmvs/mvs_pipeline.log` / `3DGS_output/gs_pipeline.log` / `SAM_output/sam.log`）
   - 容器总日志（`log/docker_containers/<容器>.log`）
4. 前端轮询 GET 日志端点读文件尾；进程 `poll()` 非空时判定成败并回写状态。

### 3.3 两种执行方式

| 方式 | 使用场景 | 实现 |
|---|---|---|
| 异步 docker exec | SfM / MVS / 3DGS / SAM 批量 | `run_*_async` 后台线程，前端轮询日志 |
| 同步 subprocess | 去元数据（exiftool）、去畸变 | `image_preprocessor.py` 同步执行，路由里用 `asyncio.to_thread` 包裹 |
| 进程内直跑 | 点云过滤、图像掩码 | `algorithm/pointcloud_filter.py` / `image_mask.py`，`asyncio.to_thread` 包裹 |
| 常驻 worker | SAM 交互式 box/point | 见下 |

### 3.4 SAM 常驻 worker（交互路径）

模型加载昂贵，交互式 box/point 走一个**长驻进程**而非每次 docker exec：

- `_ensure_sam3_worker` 用 `docker exec -i sam3 python3 /workspace/sam_worker.py --serve` 启动，等待 JSON ready 信号后模型常驻 GPU。
- 所有命令经 `_send_worker_command`（`_worker_lock` 串行）以「stdin 一行 JSON 命令 / stdout 一行 JSON 响应」通信：
  `set_image`（同图免重载）→ `box` / `point`。
- 关机 `stop.sh` 调 `/sam/server-release` 释放显存。

---

## 4. 前端 → 后端调用数据流

前端所有数据都来自后端 API，靠「一次性请求 + 定时轮询」两条腿：POST 表单触发任务，GET 轮询刷新状态与日志。

### 4.1 公共 HTTP 层

| 客户端 | 定义 | 用途 |
|---|---|---|
| `ApiClient`（单例） | `frontend/pages/projects.py:60-173` | projects / sfm / mvs / filter 共用 |
| `SamApiClient`（单例） | `frontend/pages/sam.py:41-117` | SAM 页专用，含 `fetch_image()` 取回结果图 |
| 页面私有 `QNetworkAccessManager` | `mvs.py:152` / `filter_page.py:59` | 产物文件下载（刻意绕过 ApiClient 以免破坏 JSON 解析） |

共同机制：

- 内部单个 `QNetworkAccessManager`；POST 一律 `application/x-www-form-urlencoded`（参数全转字符串），上传走 `QHttpMultiPart`。
- 统一完成信号 `finished(method, path, payload)`；每个请求带 `tag`，页面在 `_on_response` 里按 tag 分发。
- 响应注入 `_tag`；失败解析 FastAPI 的 `detail` 后广播 `{"_error": …, "_tag": …}`。

### 4.2 各页面调用的端点

**项目管理 `projects.py`**

| 方法 | 路径 | 时机 |
|---|---|---|
| GET | `/api/projects` | 初始化、刷新、每 5s 轮询 |
| POST | `/api/projects` | 创建项目骨架 |
| POST | `/api/projects/{p}/images` | 上传（zip 自动解包）；后端随后跑 exiftool |
| GET | `/api/projects/{p}` | 选中卡片 / 创建成功 / 刷新 → 详情 |
| DELETE | `/api/projects/{p}` | 删除（运行中拒绝） |

**SfM 重建 `sfm.py`**

| 方法 | 路径 | 时机 |
|---|---|---|
| GET | `/api/projects` | 填充下拉框 |
| POST | `/api/projects/{p}/run` | 全量配置表单（steps/matcher/camera_model/use_gpu/…） |
| POST | `/api/projects/{p}/stop` | 终止子进程 + 状态置 failed |
| GET | `/api/projects/{p}` | 轮询中每第 5 次（≈15s）查状态 |
| GET | `/api/projects/{p}/log?lines=40` | 每 3s 整体替换日志框（切页保持轮询） |
| GET | `/api/projects/{p}/sfm/files` | 切项目 / pipeline 完成 / 点刷新；填充下载下拉框 |
| GET | `/api/projects/{p}/sfm/download/{file}` | 私有下载管理器落盘（稀疏点云 PLY 由后端按需导出） |

**MVS 重建 `mvs.py`**

| 方法 | 路径 | 时机 |
|---|---|---|
| GET | `/api/projects` | 填充下拉框 |
| POST | `/api/projects/{p}/mvs/run` | mvs_type + command + 分辨率/视图/导出格式全量表单 |
| GET | `/api/projects/{p}/mvs/log?lines=60&mvs_type=` | 每 3s；识别 `=== MVS/PIPELINE END … exit_code: 0` 后停 |
| GET | `/api/projects/{p}/mvs/result?mvs_type=` | 切项目 / 切引擎 / 每 3s tick；返回 `run_status` 判定完成 |
| GET | `/api/projects/{p}/mvs/download/{file}` | 私有下载管理器落盘 |

**SAM 分割 `sam.py`（框选 Box / 点选 Point 两个面板）**

| 方法 | 路径 | 时机 |
|---|---|---|
| GET | `/api/projects` · `/{p}/images/list` | 项目 / 图片下拉 |
| GET | `/api/projects/{p}/images/view/{f}` | `fetch_image` 拉原图到左画布 |
| GET | `/api/projects/{p}/sam/box-result/{f}` · `point-result/{f}` | `fetch_image` 拉结果轮廓图到右画布 |
| POST | `/api/projects/{p}/sam/warmup` · `server-release` | 加载 / 释放模型 |
| POST | `/api/projects/{p}/sam/box-segment` | filename + `x1,y1,x2,y2`（图片像素坐标） |
| POST | `/api/projects/{p}/sam/point-segment` | filename + `points`=JSON `[[x,y,label]…]` |
| POST | `/api/projects/{p}/sam/points/save` | 标点变化即持久化 |
| GET | `/api/projects/{p}/sam/points/load/{stem}` | 切图后恢复标点 |

**点云过滤 `filter_page.py`**

| 方法 | 路径 | 时机 |
|---|---|---|
| GET | `/api/projects` · `/{p}/filter/result` | 列表 + 切项目时检查已有结果 |
| POST | `/api/projects/{p}/preprocess/filter-pointcloud` | threshold + prompt_slug；一次性同步返回保留点数 |
| GET | `/api/projects/{p}/filter/download/filtered_scene.ply` | 私有下载管理器 |

> `preprocess.py` 与 `mask.py` 目前是**占位页**：不接 API、不轮询；`mask.py` 的“运行”按钮甚至未连任何槽。

### 4.3 交互式 SAM 的前后端数据流

框选 / 点选两条路径结构一致（共五步）：

```
① 前端画布采集「图片原始像素坐标」（_widget_to_image 抵消画布缩放留边）
        │ POST box-segment / point-segment
② FastAPI → 常驻 sam3 worker（stdin JSON 命令）
        │ set_image → box/point 推理
③ worker 把结果写盘 SAM_output/box_results|point_results/（mask/overlay/contour PNG）
        │ 返回 JSON：mask_path / contour_path / score
④ 前端再 fetch_image GET box-result|point-result/{f} 拉图
        │
⑤ 显示在右画布并缓存（_result_cache）
```

关键点：

- 前后端传的坐标全是**图片原始像素坐标**（前端做缩放/留边补偿）。
- 结果以「后端生成的 PNG 文件 URL」回传，前端**再拉一次 GET**，而不是 JSON 内嵌字节。
- 点选面板有「继承」机制：切到无标点的图时自动沿用最近一次手动点，并用 `points/load` 覆盖恢复。

### 4.4 轮询节奏

| 页面 | 定时器 | 行为 | 切页时 |
|---|---|---|---|
| projects | 5s | 刷项目列表 | hideEvent 停 |
| sfm | 3s + 每 5 tick 查状态 | 拉日志（40 行）+ 查状态 | hideEvent 不停 |
| mvs | 3s（每 tick 拉日志 + 查后端状态） | 拉日志 + 查 `/mvs/result` 的 `run_status`，非 running 停表 | — |
| filter / sam | 无 | 一次性请求 | — |

---

## 5. 数据落盘布局

`DATA_DIR = /home/zwz/Code/DATA`，每个项目一个目录；`task.json` 记录状态与配置，其余是各阶段产物。容器内同一路径挂载为 `/data`。

```
/home/zwz/Code/DATA/<项目名>/
├── task.json                    ← 状态机真源（status/config/steps）
├── Input/
│   ├── input_image/             ← 上传图片（exiftool 已去元数据）
│   └── input_video/
├── SFM_output/                  ← SfM · colmap-dev
│   ├── database.db
│   ├── sparse/0/                ← 稀疏模型 cameras/images/points3D
│   │   └── points3D.ply         ← 稀疏点云（首次下载时由后端就地导出）
│   ├── undistorted/             ← 去畸变 images/ + sparse（PINHOLE）
│   ├── dense/                   ← colmap 稠密旧布局（fused.ply 等）
│   ├── pipeline_result.json
│   └── pipeline.log
├── MVS_output/
│   ├── colmap/                  ← stereo/ · fused.ply
│   └── openmvs/                 ← scene.mvs · scene_dense.ply · mvs_pipeline.log
├── SAM_output/                  ← sam3
│   ├── box_results/ · point_results/ · point_marks/
│   ├── <stem>_<prompt>_mask*.png · *_p*.json
│   ├── manifest.json
│   └── sam.log
├── algorithm_output/
│   ├── filtered_scene.ply       ← 点云过滤产物
│   └── input_masked/            ← 图像掩码（RGBA PNG）
├── 3DGS_output/                 ← gs3 · train/ours_*/renders/ · point_cloud/
└── PoinTR_output/               ← 预留
```

> 另外 `/home/zwz/Code/SAM/`（容器内挂载为 `/workspace`）存有 `sam_box.py`、`sam_point.py`、`sam_worker.py`（`_setup_sam_script` 从 `backend/algorithm/` 与同目录 `SAM_CLI.py` 同步而来，供 sam3 容器使用）。

---

## 6. 值得注意的机制

1. **跨页共享「当前项目」**：`frontend/state.py` 的 `ProjectState` 单例（与 `ApiClient` 同款 `_instance` + 信号模式）持有全局选中项目。任一面板选中项目（点卡片 / 改下拉框）即 `set_current()` 广播 `current_changed`，其余页面同步下拉框 / 高亮；重填下拉框走 `fill_project_combo`（`blockSignals` 屏蔽信号，选中「全局当前项目」，无则留空不自动选第一个）。运行中的页面（sfm/mvs/filter 的 `_running`、SAM 点选的 `_batch_mode`）用守卫拒绝跟随，避免后台轮询目标被切页选择改掉。
2. **轮询差异**：projects 页 hideEvent 会停轮询，sfm 页刻意不停（重建中途切走也能刷回）。
3. **完成判定靠后端状态，不解析日志**：`mvs_runner` 在进程退出时把运行状态落定为 `ok`/`failed`（`_mvs_run_state`，幂等）；`/mvs/result` 据此返回 `run_status`（openmvs 看 runner 状态，colmap 复用 SfM 的 task.json 状态机）。mvs 页每 3s tick 查 `run_status`，非 `running` 即停表——不再解析 `=== MVS/PIPELINE END` 日志关键字。
4. **后台上传即去元数据**：上传接口和每次跑管线前都 `strip_metadata`（exiftool），防止 OpenImageIO 崩溃。
5. **算法模块进程内同步跑**：过滤/掩码用 `asyncio.to_thread`；大点云按 `CHUNK_SIZE=50000` 分块计分，结果查询只读 PLY 头计数。
6. **部署方式**：`start.sh` 在 WSL 起 FastAPI（:8000），同时调 Windows 侧 Python 起 PySide6 GUI；`sync_frontend.py` 用 watchdog + rsync 把 `frontend/` 单向同步到 `/mnt/e/Photogrammetry/Frontend`。
