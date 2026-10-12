# ParaEye P0 - 深度估算模块
# 模型: Depth Anything V2 Small (24.8M)
# 输出约定: 深度值域 [0,1]，1=最近，0=最远（inverse depth，DA 系列原生即高值=近）

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator:
    """封装 DA V2 Small 推理。infer() 返回模型分辨率下的原始深度图 [0,1]，1=近。"""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        use_fp16 = self.device == "cuda"
        dtype = torch.float16 if use_fp16 else torch.float32

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(self.device).eval()
        self.dtype = dtype

    @torch.inference_mode()
    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        """frame_bgr: 原始 BGR 帧。返回 (h_m, w_m) float32 深度图，1=近。"""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device, self.dtype if v.dtype.is_floating_point else None)
                  for k, v in inputs.items()}

        outputs = self.model(**inputs)
        depth = outputs.predicted_depth  # [1, h_m, w_m] 或 [1,1,h_m,w_m]
        depth = depth.squeeze().float().cpu().numpy()

        # 归一化到 [0,1]，1=近（DA 输出本质是 inverse depth，高值即近）
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min < 1e-6:
            return np.zeros_like(depth, dtype=np.float32)
        return ((depth - d_min) / (d_max - d_min)).astype(np.float32)
