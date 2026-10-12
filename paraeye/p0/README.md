# ParaEye P0 验证工程

手机端实时 2D→平行眼 3D 转换（ParaEye）的 P0 离线验证脚本。
目标：验证深度估算 + 视差合成 + SBS 输出的效果与符号约定，产出默认参数。详见 `../dev-v2.md` 第四节。

## 环境

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

首次运行会自动从 HuggingFace 下载 Depth Anything V2 Small（约 100MB）。

## 步骤

### 1. 符号约定验证（0.5 天任务）

```bash
python make_synthetic.py -o synthetic.mp4
python pipeline.py synthetic.mp4 -o sbs_check.mp4 --max-disp 0.01
```

用平行眼裸眼观看 `sbs_check.mp4`：
- 预期：红球（近）**出屏**，蓝黄球（远）入屏；左上角 "L" 水印在左图。
- 若近球进屏 → 符号颠倒，运行时加 `--swap`，并在 `synthesize.py` 头部注释中更新结论。

### 2. 真实视频验证（P0 主实验）

```bash
python pipeline.py T1.mp4 -o sbs_T1.mp4            # 默认 max-disp 1%（P0 实测定档）
```

**P0 结论（2026-10-12 定档）**：默认 `--max-disp 0.01`（1%）、`--alpha 0.7`。
实测：720×1280 竖屏 / 30fps 视频，CPU 单帧整帧耗时约 1.38s（离线验证用，与移动端无关）。

## 文件说明

| 文件 | 作用 |
|------|------|
| `depth.py` | DA V2 Small 推理封装（输出 [0,1]，1=近） |
| `postprocess.py` | 引导滤波（模型分辨率）→ EMA → 低频百分位重映射 |
| `synthesize.py` | 视差反向映射 + 空洞 inpaint + Full-SBS 拼接 |
| `pipeline.py` | 主 CLI |
| `make_synthetic.py` | 合成场景（渐变背景 + 三球），符号验证用 |
