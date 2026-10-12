# ParaEye P0 - 深度后处理模块
# 1) 引导滤波（在模型分辨率做，边缘对齐）
# 2) EMA 时域平滑（P0 不做光流，见 dev-v2.md 模块 3 修正）
# 3) 可选：按低频更新的百分位深度重映射（禁用逐帧直方图均衡）

import cv2
import numpy as np


def _box(img: np.ndarray, r: int) -> np.ndarray:
    return cv2.boxFilter(img, -1, (2 * r + 1, 2 * r + 1), borderType=cv2.BORDER_REFLECT)


def guided_filter(guide_gray: np.ndarray, src: np.ndarray, r: int = 8, eps: float = 1e-3) -> np.ndarray:
    """灰度引导滤波。guide_gray/src: (h, w) float32（内部统一 2D 运算，
    规避 cv2.boxFilter 对 (h,w,1) 输入返回 2D 的广播陷阱）。"""
    I = np.asarray(guide_gray, np.float32)
    p = np.asarray(src, np.float32)
    if I.ndim == 3:
        I = I[..., 0]
    if p.ndim == 3:
        p = p[..., 0]

    mean_I = _box(I, r)
    mean_p = _box(p, r)
    var_I = _box(I * I, r) - mean_I * mean_I
    cov_Ip = _box(I * p, r) - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = _box(a, r)
    mean_b = _box(b, r)
    return mean_a * I + mean_b


class TemporalSmoother:
    """EMA: D_t_smooth = a*D_t + (1-a)*D_{t-1}_smooth。alpha 越大响应越快。"""

    def __init__(self, alpha: float = 0.7):
        assert 0.0 < alpha <= 1.0
        self.alpha = alpha
        self.prev: np.ndarray | None = None

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        if self.prev is None:
            self.prev = depth.copy()
            return depth
        self.prev = self.alpha * depth + (1.0 - self.alpha) * self.prev
        return self.prev

    def reset(self):
        self.prev = None


class DepthRemapper:
    """低频更新的百分位重映射：避免逐帧均衡化导致的闪烁。
    每 update_every 帧用 [lo, hi] 百分位重算一次映射，场景内保持恒定。"""

    def __init__(self, lo: float = 2.0, hi: float = 98.0, update_every: int = 45):
        self.lo, self.hi = lo, hi
        self.update_every = update_every
        self._frame_idx = 0
        self._vmin = 0.0
        self._vmax = 1.0

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        if self._frame_idx % self.update_every == 0:
            self._vmin, self._vmax = np.percentile(depth, [self.lo, self.hi])
        self._frame_idx += 1
        if self._vmax - self._vmin < 1e-6:
            return depth
        return np.clip((depth - self._vmin) / (self._vmax - self._vmin), 0.0, 1.0).astype(np.float32)


class DepthPostProcessor:
    """组合：引导滤波（模型分辨率）→ EMA → 可选重映射。"""

    def __init__(self, alpha: float = 0.7, use_guided: bool = True,
                 use_remap: bool = False, guided_radius: int = 8):
        self.use_guided = use_guided
        self.guided_radius = guided_radius
        self.smoother = TemporalSmoother(alpha)
        self.remapper = DepthRemapper() if use_remap else None

    def __call__(self, depth_model_res: np.ndarray, frame_small_gray: np.ndarray) -> np.ndarray:
        """输入均为模型分辨率。输出模型分辨率深度图。"""
        d = depth_model_res
        if self.use_guided:
            guide = frame_small_gray.astype(np.float32) / 255.0
            d = guided_filter(guide, d, r=self.guided_radius)
        d = self.smoother(d)
        if self.remapper is not None:
            d = self.remapper(d)
        return d.astype(np.float32)
