# ParaEye P0 - 符号约定数值验证（v2：模板匹配精确测位移）
#
# 用原帧中球体的图像块做模板，在左/右眼图中搜索匹配位置，得到精确的水平位移。
# 预期（L/R 平行眼）：近球位移 = x_left - x_right > 0（左眼内容偏右 = 负视差 = 出屏）。
#
# 附带对比：开启百分位重映射（--remap 思路）能否拉开合成场景被压缩的深度范围。

import numpy as np
import cv2

from make_synthetic import make_frame
from depth import DepthEstimator
from postprocess import DepthPostProcessor
from synthesize import synthesize_view, concat_sbs

W, H = 1280, 720
MAX_DISP = int(0.01 * W)

BALLS = [  # (cx, cy, r, gt_depth, 名称)
    (0.30, 0.45, 80, 0.9, "红球(近)"),
    (0.72, 0.50, 60, 0.3, "蓝黄球(远)"),
]


def locate(base: np.ndarray, patch: np.ndarray, cx: int, cy: int, search: int) -> float:
    """在 base 的窗口内模板匹配 patch，返回匹配中心 x。"""
    y0, y1 = cy - search, cy + search
    x0, x1 = cx - search, cx + search
    y0, x0 = max(0, y0), max(0, x0)
    region = base[y0:y1, x0:x1]
    res = cv2.matchTemplate(region, patch, cv2.TM_CCOEFF_NORMED)
    _, _, _, peak = cv2.minMaxLoc(res)
    return x0 + peak[0] + patch.shape[1] / 2


def measure(frame, depth_full, label):
    left = synthesize_view(frame, depth_full, "left", MAX_DISP)
    right = synthesize_view(frame, depth_full, "right", MAX_DISP)
    print(f"\n[{label}]  max_disp={MAX_DISP}px")
    all_ok = True
    for cx, cy, r, gt, name in BALLS:
        cxi, cyi = int(cx * W), int(cy * H)
        patch = frame[cyi - r:cyi + r, cxi - r:cxi + r]
        xl = locate(left, patch, cxi, cyi, r + MAX_DISP + 16)
        xr = locate(right, patch, cxi, cyi, r + MAX_DISP + 16)
        shift = xl - xr
        expect = gt * MAX_DISP
        ok = shift > 0.5 * expect if gt > 0.5 else abs(shift) < expect + 2
        all_ok &= ok
        print(f"  {name}: x_left={xl:.1f} x_right={xr:.1f} 位移={shift:+.1f}px "
              f"(GT 预期≈+{expect:.0f}px) {'✅' if ok else '❌'}")
    return all_ok, left, right


def main():
    frame = make_frame(W, H, 0.0)
    est = DepthEstimator()

    d_raw = est.infer(frame)
    gray = cv2.cvtColor(cv2.resize(frame, (d_raw.shape[1], d_raw.shape[0])),
                        cv2.COLOR_BGR2GRAY)

    results = {}
    for use_remap in (False, True):
        post = DepthPostProcessor(alpha=0.7, use_guided=True, use_remap=use_remap)
        d = post(d_raw, gray)
        depth_full = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
        ok, left, right = measure(frame, depth_full,
                                  "百分位重映射" if use_remap else "原始范围")
        results[use_remap] = (ok, left, right)

    cv2.imwrite("sbs_preview.png", concat_sbs(results[False][1], results[False][2]))
    cv2.imwrite("sbs_preview_remap.png", concat_sbs(results[True][1], results[True][2]))
    print("\n预览帧: sbs_preview.png / sbs_preview_remap.png（左半=左眼，右半=右眼）")


if __name__ == "__main__":
    main()
