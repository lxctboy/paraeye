# ParaEye P0 - 合成场景生成器（符号约定验证用，dev-v2.md 4.3）
#
# 场景设计：
#   背景: 水平渐变色的平滑墙面（远景，深度 0.1）
#   球体: 三个已知位置的球，深度分别 0.9 / 0.6 / 0.3（近/中/远）
#   画面左上角写 "L" 水印，用于肉眼确认 SBS 的左右顺序
#
# 预期结果（平行眼、L/R 顺序）：
#   深度 0.9 的球明显"出屏"（负视差），0.3 的球在屏后，渐变背景几乎无视差。
#   若看反（近球进屏），说明符号约定颠倒 → 运行 pipeline 时加 --swap。

import argparse
import cv2
import numpy as np


def make_frame(w: int, h: int, t: float) -> np.ndarray:
    """t: [0,1) 归一化时间，控制球体轻微移动以形成动态视频。"""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # 渐变背景（蓝→橙），模拟平滑远景
    gx = np.linspace(0, 1, w, dtype=np.float32)
    frame[:, :, 0] = (60 + 120 * gx).astype(np.uint8)          # B
    frame[:, :, 2] = (200 - 120 * gx).astype(np.uint8)         # R
    frame[:, :, 1] = 110                                        # G

    # 三个球：半径 60~90px，深度 0.9(近) 0.6(中) 0.3(远)
    balls = [
        (0.30 + 0.04 * np.sin(2 * np.pi * t), 0.45, 80, 0.9, (60, 60, 230)),   # 近·红
        (0.50 + 0.04 * np.sin(4 * np.pi * t), 0.55, 70, 0.6, (80, 200, 80)),   # 中·绿
        (0.72 + 0.04 * np.sin(3 * np.pi * t), 0.50, 60, 0.3, (200, 180, 60)),  # 远·蓝黄
    ]
    for cx, cy, r, _d, color in balls:
        center = (int(cx * w), int(cy * h))
        cv2.circle(frame, center, r, color, -1)
        # 高光增强立体感
        cv2.circle(frame, (center[0] - r // 3, center[1] - r // 3), r // 4,
                   (255, 255, 255), -1)

    # L 水印（左上角），确认拼接顺序
    cv2.putText(frame, "L", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.2,
                (255, 255, 255), 5)
    return frame


def make_depth(w: int, h: int, t: float) -> np.ndarray:
    """合成场景的 ground-truth 深度图（1=近），用于对照模型输出。"""
    d = np.full((h, w), 0.1, dtype=np.float32)
    balls = [
        (0.30 + 0.04 * np.sin(2 * np.pi * t), 0.45, 80, 0.9),
        (0.50 + 0.04 * np.sin(4 * np.pi * t), 0.55, 70, 0.6),
        (0.72 + 0.04 * np.sin(3 * np.pi * t), 0.50, 60, 0.3),
    ]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for cx, cy, r, depth in balls:
        cx, cy = cx * w, cy * h
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        d[dist <= r] = depth
    return d


def main():
    ap = argparse.ArgumentParser(description="生成 ParaEye 符号验证合成视频")
    ap.add_argument("-o", "--output", default="synthetic.mp4")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()

    n = int(args.seconds * args.fps)
    vw = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (args.width, args.height))
    for i in range(n):
        vw.write(make_frame(args.width, args.height, i / n))
    vw.release()
    print(f"OK: {args.output} ({n} frames, {args.width}x{args.height}@{args.fps})")
    print("预期（平行眼/LR）: 红球(近)出屏、蓝黄球(远)入屏；看反则 pipeline 加 --swap")


if __name__ == "__main__":
    main()
