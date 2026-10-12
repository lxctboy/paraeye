# ParaEye P0 - 主流水线 CLI
#
# 用法示例:
#   python pipeline.py synthetic.mp4 -o sbs.mp4 --max-disp 0.0075 --alpha 0.7
#   python pipeline.py input.mp4 -o sbs.mp4 --swap            # 交叉眼顺序
#   python pipeline.py input.mp4 -o sbs.mp4 --no-inpaint      # 关闭空洞 inpaint
#
# 参数:
#   --max-disp   总视差占图像宽度比例，默认 0.01 (1%)，P0 实测定档（dev-v2.md）
#   --alpha      EMA 时域平滑系数，0.6~0.8；1.0 = 不平滑（默认 0.7）
#   --swap       最终 SBS 用 R/L 顺序（交叉眼）
#   --no-guided  关闭引导滤波
#   --no-inpaint 关闭空洞 inpaint
#   --remap      开启低频百分位深度重映射（默认关）

import argparse
import time

import cv2

from depth import DepthEstimator
from postprocess import DepthPostProcessor
from synthesize import concat_sbs, inpaint_holes, synthesize_view


def main():
    ap = argparse.ArgumentParser(description="ParaEye P0: 2D -> Full-SBS 平行眼")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default="output_sbs.mp4")
    ap.add_argument("--max-disp", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--no-guided", action="store_true")
    ap.add_argument("--no-inpaint", action="store_true")
    ap.add_argument("--remap", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"无法打开输入视频: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_disp_px = int(round(args.max_disp * w))
    print(f"输入: {w}x{h}@{fps:.1f}  max_disp={max_disp_px}px ({args.max_disp*100:.2f}%)")

    estimator = DepthEstimator()
    post = DepthPostProcessor(alpha=args.alpha, use_guided=not args.no_guided,
                              use_remap=args.remap)
    model_h, model_w = estimator.processor.size["height"], estimator.processor.size["width"]

    vw = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (2 * w, h))

    t_infer = t_total = 0.0
    n = 0
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (model_w, model_h), interpolation=cv2.INTER_AREA)
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        t1 = time.perf_counter()
        d_raw = estimator.infer(frame)
        t2 = time.perf_counter()
        t_infer += t2 - t1

        # 处理器按宽高比缩放（如 518x924），引导图跟随深度图实际尺寸
        if gray_small.shape != d_raw.shape:
            gray_small = cv2.resize(gray_small, (d_raw.shape[1], d_raw.shape[0]),
                                    interpolation=cv2.INTER_AREA)
        d = post(d_raw, gray_small)
        depth_full = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)

        left = synthesize_view(frame, depth_full, "left", max_disp_px)
        right = synthesize_view(frame, depth_full, "right", max_disp_px)
        if not args.no_inpaint:
            left, right = inpaint_holes(left, right, depth_full, max_disp_px, frame)
        sbs = concat_sbs(left, right, swap=args.swap)
        vw.write(sbs)

        t_total += time.perf_counter() - t0
        n += 1
        if n % 30 == 0:
            print(f"  frame {n}: 平均推理 {t_infer/n*1000:.1f}ms, 平均整帧 {t_total/n*1000:.1f}ms")

    cap.release()
    vw.release()
    if n:
        print(f"完成: {n} 帧 → {args.output}")
        print(f"  平均推理 {t_infer/n*1000:.1f}ms/帧 | 整帧 {t_total/n*1000:.1f}ms/帧 | 等效 {n/t_total:.1f} FPS")


if __name__ == "__main__":
    main()
