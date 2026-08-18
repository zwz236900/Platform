# Docker 镜像备份 / 恢复配置文件

> 用途：记录已导出的镜像压缩包位置、恢复方法、以及重建各容器的完整命令。
> 创建时间：2026-08-14

---

## 1. 压缩包清单（保存在 C 盘）

| 压缩包 | 对应镜像 | 原始大小 | 压缩后 | 完整性校验 |
|--------|---------|---------|--------|-----------|
| `C:\Users\lenovo\Downloads\platform-gs-20260813.tar.zst` | `platform/gs:latest` | 25GB | 8.7GB | ✅ 已验证 (zstd -t) |
| `C:\Users\lenovo\Downloads\platform-colmap-20260813.tar.zst` | `platform/colmap:latest` | 42.6GB | 14GB | ✅ 已验证 (zstd -t) |
| `C:\Users\lenovo\Downloads\platform-sam-20260813.tar.zst` | `platform/sam:latest` | 43.1GB | 16GB | ✅ 已验证 (zstd -t) |
| `C:\Users\lenovo\Downloads\platform-pointr-20260814.tar.zst` | `platform/pointr:latest` | 30.6GB | 9.2GB | ✅ 已验证 (zstd -t) |

**C盘合计：约 48GB**（压缩参数：`zstd -12 -T4`，WSL 内原生 zstd v1.4.8）

> PoinTr 镜像已**烘焙代码**（`COPY . /workspace/`，排除 `data`/`.git`/`.local`/`.nv`），
> 30.6GB = 基础镜像(29GB) + 代码 1.6GB。`data` 数据集（26GB）不包含在镜像内，运行时通过挂载访问。

### G 盘备份（2026-08-14，目录 `G:\docker-backups\`）

| 压缩包                       | 内容               | 原始大小 | 压缩后 |
| ---------------------------- | ------------------ | -------- | ------ |
| pointr-data-20260814.tar.zst | PoinTr data 数据集 | 26GB     | 9.6GB  |
| openmvs-20260814.tar.zst     | openmvs 镜像       | 2.48GB   | 588MB  |

> ✅ 两者均已通过 zstd -t 完整性校验
>
> 为什么 data 单独存：PoinTr 镜像只烘焙代码，26GB 数据集按用户要求**不打包进镜像**，
> 而是单独压缩保存到 G 盘。恢复时解压回 `/home/zwz/Code/PoinTr/data` 即可。
>
> 为什么 openmvs 直接打包：MVS 宿主文件夹仅一个 68KB 的 CLI 脚本，无需烘焙，
> 镜像直接 `docker save` 压缩即可。

---

## 2. 恢复镜像（从压缩包）

在 WSL 中执行（每个镜像一条）：

```bash
# gs
zstd -d < /mnt/c/Users/lenovo/Downloads/platform-gs-20260813.tar.zst | docker load

# colmap
zstd -d < /mnt/c/Users/lenovo/Downloads/platform-colmap-20260813.tar.zst | docker load

# sam
zstd -d < /mnt/c/Users/lenovo/Downloads/platform-sam-20260813.tar.zst | docker load

# pointr
zstd -d < /mnt/c/Users/lenovo/Downloads/platform-pointr-20260814.tar.zst | docker load

# openmvs
zstd -d < /mnt/g/docker-backups/openmvs-20260814.tar.zst | docker load
```

恢复后镜像 tag 自动还原为 `platform/gs:latest`、`platform/colmap:latest`、`platform/sam:latest`、`platform/pointr:latest`、`openmvs/openmvs-ubuntu:latest`。

### 恢复 PoinTr 数据集（解压回 data 目录）

```bash
mkdir -p /home/zwz/Code/PoinTr
zstd -d < /mnt/g/docker-backups/pointr-data-20260814.tar.zst | tar -C /home/zwz/Code/PoinTr -xf -
# 解压后应生成 /home/zwz/Code/PoinTr/data/
```

---

## 3. 重建容器命令

> ⚠️ 这些镜像**已内置打包了代码**（通过 Dockerfile COPY 烘焙，非挂载），恢复后可直接按下面命令重建容器。DATA 数据仍通过挂载访问，不入镜像。

### gs3（平台/gs:latest）
```bash
docker run -d --name gs3 --gpus all -m 8g --shm-size=4g \
  -v /home/zwz/Code/DATA:/data \
  -w /workspace/3DGS \
  --entrypoint bash platform/gs:latest -c "while true; do sleep 3600; done"
```

### sam3（platform/sam:latest）
```bash
docker run -d --name sam3 --gpus all --shm-size=8g \
  -v /home/zwz/Code/DATA:/data \
  -w /workspace \
  --entrypoint /opt/nvidia/nvidia_entrypoint.sh platform/sam:latest \
  bash -c "while true; do sleep 3600; done"
```

### colmap-dev（platform/colmap:latest）
```bash
docker run -d --name colmap-dev --gpus all --shm-size=8g \
  -v /home/zwz/Code/DATA:/data \
  -w /colmap \
  --entrypoint /opt/nvidia/nvidia_entrypoint.sh platform/colmap:latest \
  bash -c "while true; do sleep 3600; done"
```

### PoinTr（platform/pointr:latest）— 已导出

```bash
docker run -d --name PoinTr --gpus all --shm-size=8g \
  -v /home/zwz/Code/DATA:/data \
  -w /workspace \
  platform/pointr:latest bash -c "while true; do sleep 3600; done"
```

> 注：镜像内已烘焙代码（`/workspace`，不含 `data`）。挂载宿主 `PoinTr` 文件夹会覆盖镜像内代码，
> 且宿主的 `data` 通过挂载可访问——挂载是首选方式，烘焙镜像是备份。

### openmvs（openmvs/openmvs-ubuntu:latest）— 已导出

```bash
docker run -d --name openmvs --gpus all --shm-size=64m \
  -v /home/zwz/Code/DATA:/data \
  openmvs/openmvs-ubuntu:latest tail -f /dev/null
```

---


---

## 5. 注意事项

1. **D 盘空间**：删除镜像后 D 盘仍显示 100% 满（3.1GB）是正常的——Docker 的
   `D:\Docker\DockerDesktopWSL\disk\docker_data.vhdx` 文件不会自动缩小。
   需在 Docker Desktop → 设置 → Resources → Disk 里执行 **Compact**（或等它自动压缩）
   才能真正释放 D 盘空间。

2. **镜像内容说明**：gs / sam / colmap 镜像已把宿主代码文件夹烘焙进镜像内部
   （如 `/workspace/3DGS`、`/workspace`、`/colmap`），这是通过
   `FROM <旧镜像>` + `COPY . <目标>` 重新构建实现的，而非 `docker commit`
   （commit 会排除挂载点）。重建容器时挂载相同路径即为覆盖外部代码版本。

3. **打包用的 zstd**：WSL 内原生 zstd v1.4.8（`/usr/bin/zstd`），8GB 内存限制下
   使用 `-T4` 线程安全（`-T0` 全线程会 OOM）。如需 Windows 侧解压可用
   `C:\Program Files\7-Zip` 或 MSYS2 的 zstd.exe。

4. **重新导出**（如需重新压缩）：
   ```bash
   docker save <镜像> | zstd -12 -T4 -o /mnt/c/Users/lenovo/Downloads/<名字>.tar.zst
   ```
