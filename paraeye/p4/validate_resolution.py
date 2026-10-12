# ParaEye P4 - 输入降分辨率质量闸门
# 目的：导出更小输入的 tflite 之前，先用本地 torch 验证深度结构是否保持。
# 口径与 Android 侧一致：bilinear  resize 到 (W,H) + ImageNet 归一化 + NCHW，原始输出不做归一化。
# 参考基线 = 686x518（当前线上模型的输入，横宽高比，会把 720x1280 竖屏视频拉伸 2.35 倍）。

import cv2
import numpy as np
import torch
from transformers import AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
VIDEO = "D:/project/3D/781d6c67089968bf3d3ee0fa4437b249.mp4"  # 720x1280 竖屏
OUT_DIR = "D:/project/3D/paraeye/p4"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# (W, H) —— W 必须被 14 整除（ViT patch）
# 686x518 = 当前线上（横宽高比，竖屏视频被竖挤 2.35 倍）
# 518x924 = 原生宽高比高分辨率参考（近似 720x1280，当作质量真值）
CANDIDATES = [(686, 518), (518, 924), (518, 518), (392, 518), (280, 518)]
REF = (686, 518)          # 当前线上
REF_NATIVE = (518, 924)   # 原生宽高比真值
N_FRAMES = 8


def preprocess(frame_bgr, w, h):
    """frame_bgr: HxWx3 uint8 -> (1,3,h,w) float32，与 Android ImagePreprocessor 同口径。"""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    x = small.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1)[None]).contiguous()


@torch.inference_mode()
def infer_depth(model, frame_bgr, w, h):
    x = preprocess(frame_bgr, w, h)
    out = model(pixel_values=x).predicted_depth
    return out.squeeze().float().cpu().numpy()


def minmax(d):
    lo, hi = d.min(), d.max()
    return (d - lo) / (hi - lo + 1e-8)


def compare(d_ref, d_new):
    """把两张深度图放到共同画布上比结构：Pearson 相关 + 相对 MAE。"""
    h = max(d_ref.shape[0], d_new.shape[0])
    w = max(d_ref.shape[1], d_new.shape[1])
    a = cv2.resize(d_ref, (w, h), interpolation=cv2.INTER_LINEAR)
    b = cv2.resize(d_new, (w, h), interpolation=cv2.INTER_LINEAR)
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    mae = float(np.abs(a - b).mean() / (a.max() - a.min() + 1e-8))
    return corr, mae


def pseudo_color(d):
    """近=红 远=蓝，与 Android DepthVisualizer 色带一致（hue=240*(1-v)）。"""
    v = minmax(d)
    hsv = np.stack([240.0 * (1.0 - v), np.full_like(v, 0.85), np.ones_like(v)], axis=-1)
    hsv = hsv.astype(np.float32)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return (bgr * 255).astype(np.uint8)


def main():
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).eval()
    print(f"model: {MODEL_ID}")

    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 2, N_FRAMES).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    print(f"video: {VIDEO} {total} frames, sampled {len(frames)}")

    # 跑全部组合
    depths = {(w, h): [] for (w, h) in CANDIDATES}
    for (w, h) in CANDIDATES:
        for fr in frames:
            depths[(w, h)].append(infer_depth(model, fr, w, h))
        print(f"  done {w}x{h}")

    # 指标：每个候选 vs 两个参考（当前线上 686x518 / 原生宽高比 518x924）
    print(f"\n{'size':>10} {'tokens':>7} | {'vs686':>6} {'mae':>6} | {'vs924':>6} {'mae':>6}")
    results = {}
    for (w, h) in CANDIDATES:
        tokens = (h // 14) * (w // 14)
        c1, m1, c2, m2 = [], [], [], []
        for i in range(len(frames)):
            if (w, h) != REF:
                c, m = compare(depths[REF][i], depths[(w, h)][i])
                c1.append(c); m1.append(m)
            if (w, h) != REF_NATIVE:
                c, m = compare(depths[REF_NATIVE][i], depths[(w, h)][i])
                c2.append(c); m2.append(m)
        s1 = f"{np.mean(c1):>6.3f} {np.mean(m1):>6.3f}" if c1 else "      -       -"
        s2 = f"{np.mean(c2):>6.3f} {np.mean(m2):>6.3f}" if c2 else "      -       -"
        results[(w, h)] = (float(np.mean(c1)) if c1 else None, float(np.mean(c2)) if c2 else None, tokens)
        print(f"{w}x{h:>6} {tokens:>7} | {s1} | {s2}")
    print("（vs686=与当前线上失真输入比；vs924=与原生宽高比真值比，越小越该换）")

    # 目检网格：第 1/4/8 帧 × 各尺寸伪彩深度（统一显示高度，拉伸差异肉眼可见）
    picks = [0, len(frames) // 3, 2 * len(frames) // 3, len(frames) - 1]
    DISP_H = 360

    def tile(d):
        img = pseudo_color(d)
        scale = DISP_H / img.shape[0]
        return cv2.resize(img, (max(1, int(img.shape[1] * scale)), DISP_H))

    rows = []
    for fi in picks:
        row = [tile(depths[REF][fi])]
        for (w, h) in CANDIDATES:
            if (w, h) == REF:
                continue
            row.append(tile(depths[(w, h)][fi]))
        rows.append(cv2.hconcat(row))
    grid = cv2.vconcat(rows)
    out = f"{OUT_DIR}/depth_quality_grid.png"
    cv2.imwrite(out, grid)
    print(f"\nsaved: {out}")
    print("列顺序: 686x518(线上) | 518x924(真值) | 518x518 | 392x518 | 280x518")


if __name__ == "__main__":
    main()
