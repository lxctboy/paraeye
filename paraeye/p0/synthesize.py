# ParaEye P0 - 视图合成 + SBS 拼接模块
#
# 符号约定（2026-10-12 已用合成场景 + 模板匹配数值验证通过，verify_sign.py）：
#   深度 d: 1=近, 0=远；disparity = d * max_disp（总视差，像素）
#   左眼反向采样: I_L(u) = I(u - disparity/2)   → 内容右移
#   右眼反向采样: I_R(u) = I(u + disparity/2)   → 内容左移
#   ⇒ 近物左眼图在右、右眼图在左 = 负视差（交叉）= 出屏 ✅ 实测确认
#   若目标设备为交叉眼，用 --swap 交换拼接顺序即可。
#
# 空洞填充 P0 策略：反向映射 + 双线性为主；边缘采样用 BORDER_REPLICATE，
# 大空洞区域用 cv2.inpaint 兜底（可关）。

import cv2
import numpy as np


def synthesize_view(frame: np.ndarray, depth: np.ndarray, side: str,
                    max_disp: int) -> np.ndarray:
    """frame: (H,W,3) uint8；depth: (H,W) float32 [0,1]，1=近。
    side: 'left' | 'right'。返回与 frame 同尺寸的眼图。"""
    h, w = frame.shape[:2]
    disparity = depth * float(max_disp)  # (H,W)
    half = disparity * 0.5

    xs = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
    if side == "left":
        map_x = xs - half      # I_L(u) = I(u - dis/2)，内容右移
    elif side == "right":
        map_x = xs + half
    else:
        raise ValueError(side)
    map_y = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)

    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def inpaint_holes(left: np.ndarray, right: np.ndarray, depth: np.ndarray,
                  max_disp: int, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """粗略空洞兜底：位移大于阈值的近景边缘区域用 inpaint 平滑填充。
    P0 简化实现——仅对深度梯度大的窄带做处理，避免全图 inpaint 的开销。"""
    dy, dx = np.gradient(depth)
    edge_mask = (np.abs(dx) * max_disp > 2.0).astype(np.uint8) * 255
    edge_mask = cv2.dilate(edge_mask, np.ones((3, 3), np.uint8), iterations=2)
    if edge_mask.any():
        left = cv2.inpaint(left, edge_mask, 3, cv2.INPAINT_TELEA)
        right = cv2.inpaint(right, edge_mask, 3, cv2.INPAINT_TELEA)
    return left, right


def concat_sbs(left: np.ndarray, right: np.ndarray, swap: bool = False) -> np.ndarray:
    """Full-SBS 拼接。swap=True 时输出 R/L 顺序（交叉眼）。"""
    a, b = (right, left) if swap else (left, right)
    return np.hstack([a, b])
