# 数据集下载加速经验

## 问题

ultralytics 内置的自动下载（`check_det_dataset(autodownload=True)`）走 `ultralytics.com/assets/`，国内访问极慢，约 **14KB/s**。VisDrone 三个文件共 ~1.8GB，预计需要 **~14 小时**。

## 排查过程

1. **初试**：使用 `check_det_dataset` 自动下载，速度 14KB/s，频繁 connection timeout
2. **换个源**：`ultralytics.com` 实际重定向到 `release-assets.githubusercontent.com`（GitHub Release CDN），仍然慢
3. **多线程 `aria2c`**：遇到 416 Range Not Satisfiable 错误，gh-proxy 代理不支持 range 请求
4. **最终方案**：`gh-proxy.org` + 换 `yolov5` release 源

## 根因

源代码里下载走的是 Python `urllib`，不受 git 代理配置影响。服务器已配置 `url.https://gh-proxy.org/https://github.com/.insteadof=https://github.com/`，但那只对 git 操作有效。

## 最终方案

- **下载工具**：`wget`（不用 `aria2c`，避免 range 请求兼容问题）
- **代理**：`gh-proxy.org` 前缀
- **源地址**：`https://github.com/ultralytics/yolov5/releases/download/v1.0/...`（比 `ultralytics/assets` 更快）

## 实测速度对比

| 方案 | 速度 | VisDrone 耗时 |
|------|------|-------------|
| `check_det_dataset` 直连 | ~14KB/s | **~14 小时** |
| `gh-proxy.org` + `ultralytics/assets` | ~195KB/s | ~1 小时 |
| **`gh-proxy.org` + `yolov5/releases`** | **~2.8MB/s** | **~10 分钟** |

## 命令

```bash
# VisDrone（2.3GB）
wget -P /root/workspace/datasets/VisDrone \
  "https://gh-proxy.org/https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip" \
  "https://gh-proxy.org/https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip" \
  "https://gh-proxy.org/https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-test-dev.zip"

# 下载完成后用 ultralytics 自动完成解压 + 标注转换
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('VisDrone.yaml', autodownload=True)"
```

## 通用模式

其他 GitHub release 下载场景可复用此模式：

```bash
wget "https://gh-proxy.org/https://github.com/<org>/<repo>/releases/download/<tag>/<file>"
```

> 注意：如果文件源不在 GitHub（如 SKU-110K 的 S3），需另寻加速方案。
