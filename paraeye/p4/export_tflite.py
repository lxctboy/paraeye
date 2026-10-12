# ParaEye P4 - DA V2 Small 重导出：更小输入分辨率（降 token 数 = 降 CPU 推理耗时）
#
# 背景：当前线上模型输入 686x518（横宽高比），而内容是 720x1280 竖屏视频——
# 降采样 FBO 把画面竖挤了 2.35 倍。torch 级质量闸门（validate_resolution.py）显示：
#   280x518 竖屏输入 vs 原生宽高比真值：corr 0.977（当前 686x518 仅 0.766）
# 即换小尺寸同时拿到「消除拉伸 + token 1813→740（0.41 倍）」两个收益。
#
# 量化方案（v2，2026-10-12 修正）：
#   v1 用 litert-torch 的 pt2e（prepare/convert_pt2e + 又传 quant_config 给 convert）
#   导出的模型是坏的——输入无关、输出恒为常数 0.0887（重复量化）。
#   社区 litert-community 模型 README 明确写的方案是：
#     先转 FP32 tflite，再用 AI Edge Quantizer 的 dynamic_wi8_afp32 配方量化
#     （int8 权重 + fp32 激活/接口，无需校准数据），与线上模型同 scheme，
#     Android 侧 float 进 float 出，零改动。
#   本脚本照此实现，并在导出后立刻用 ai-edge-litert 跑一遍与 torch 对比自验证，
#   corr < 0.95 或输出方差为 0（常数折叠的典型症状）则直接非零退出，
#   避免坏的 tflite 被发布到 release。
#
# 依赖 litert-torch / ai-edge-quantizer（均无 Windows wheel）——由
# .github/workflows/export-tflite.yml 在 Linux runner 上调用。
#
# 用法: python export_tflite.py --sizes 686x518,280x518 --out-dir out

import argparse
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForDepthEstimation

import litert_torch

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DepthOnly(torch.nn.Module):
    """剥掉 HF 外壳，只留 pixel_values -> predicted_depth，便于导出。"""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m(pixel_values=pixel_values).predicted_depth


def synth_frame(seed: int, w: int, h: int) -> np.ndarray:
    """合成一张有结构的 '类自然' 图：地面渐变 + 若干块状物体。

    目的：CI 里没有真实测试视频，用合成图自验证足以抓出「常数折叠 /
    量化炸掉」这类硬错误（真机上 corr 0.998 级别的细精度由本地
    validate_tflite.py 用真实视频复核）。
    """
    rng = np.random.default_rng(seed)
    hh = max(h // 2, 64)
    ww = max(w // 2, 64)
    img = np.zeros((hh, ww, 3), dtype=np.float32)
    # 背景：上暗下亮的垂直渐变（模拟天花板->地面）
    grad = np.linspace(0.25, 0.85, hh, dtype=np.float32)[:, None]
    img += grad[..., None]
    # 几个不同亮度的块（模拟近处物体）
    for _ in range(4):
        y0 = int(rng.integers(0, hh - hh // 5))
        x0 = int(rng.integers(0, ww - ww // 5))
        y1 = y0 + int(rng.integers(hh // 8, hh // 3))
        x1 = x0 + int(rng.integers(ww // 8, ww // 3))
        img[y0:y1, x0:x1] = rng.uniform(0.05, 0.95, size=3).astype(np.float32)
    return (img * 255.0).clip(0, 255).astype(np.uint8)


def preprocess(bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    rgb = bgr[..., ::-1] if bgr.shape[2] == 3 else bgr
    import cv2

    small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    x = small.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None].astype(np.float32)


@torch.inference_mode()
def torch_depth(model, frame: np.ndarray, w: int, h: int) -> np.ndarray:
    x = torch.from_numpy(preprocess(frame, w, h))
    return model(pixel_values=x).predicted_depth.squeeze().float().cpu().numpy()


def tflite_depth(path: str, frame: np.ndarray, w: int, h: int) -> np.ndarray:
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(model_path=path, num_threads=4)
    interp.allocate_tensors()
    i = interp.get_input_details()[0]["index"]
    o = interp.get_output_details()[0]["index"]
    interp.set_tensor(i, preprocess(frame, w, h))
    interp.invoke()
    return interp.get_tensor(o).squeeze().astype(np.float32)


def self_validate(model, tflite_path: str, w: int, h: int) -> bool:
    """导出后立刻自检验：tflite vs torch，抓常数折叠 / 量化炸掉。"""
    print(f"  self-validate @ {w}x{h}:")
    ok = True
    for seed in (0, 7):
        fr = synth_frame(seed, 686, 518)  # 合成固定尺寸，再按目标尺寸预处理
        d_t = torch_depth(model, fr, w, h)
        d_l = tflite_depth(tflite_path, fr, w, h)
        if d_l.shape != d_t.shape:
            print(f"    shape mismatch: tflite {d_l.shape} vs torch {d_t.shape}")
            ok = False
            continue
        std = float(d_l.std())
        corr = float(np.corrcoef(d_t.ravel(), d_l.ravel())[0, 1])
        rel = float(np.abs(d_t - d_l).mean() / (d_t.max() - d_t.min() + 1e-8))
        print(f"    seed{seed}: corr={corr:.4f} relMAE={rel:.4f} std={std:.4f}")
        if std < 1e-3:
            print("    ❌ 输出为常数（多半是常量折叠 / 重复量化）")
            ok = False
        if not corr > 0.95:
            print("    ❌ 与 torch 相关性过低")
            ok = False
    print("    ✅ 通过" if ok else "    ❌ 未通过")
    return ok


def export_size(model, w: int, h: int, out_path: str) -> bool:
    wrapper = DepthOnly(model).eval()
    sample = (torch.randn(1, 3, h, w),)

    t0 = time.time()
    ep = torch.export.export(wrapper, sample)
    print(f"  torch.export ok ({time.time() - t0:.0f}s)")

    t0 = time.time()
    # FP32 中间产物单独放 fp32/ 子目录，免得被 artifact / release 一起上传（99MB 级）
    fp32_dir = os.path.join(os.path.dirname(out_path) or ".", "fp32")
    os.makedirs(fp32_dir, exist_ok=True)
    fp32_path = os.path.join(fp32_dir, os.path.basename(out_path).replace(
        ".tflite", "_fp32.tflite"))
    litert_torch.convert(ep, sample).export(fp32_path)
    print(f"  fp32 tflite: {os.path.getsize(fp32_path) / 1e6:.1f} MB "
          f"({time.time() - t0:.0f}s)")

    t0 = time.time()
    from ai_edge_quantizer import quantizer as aeq_quantizer
    from ai_edge_quantizer import recipe as aeq_recipe

    qt = aeq_quantizer.Quantizer(fp32_path)
    qt.load_quantization_recipe(aeq_recipe.dynamic_wi8_afp32())
    qt.quantize().export_model(out_path)
    print(f"  quantized: {os.path.getsize(out_path) / 1e6:.1f} MB "
          f"({time.time() - t0:.0f}s)")

    # I/O 契约必须保持 float32 进 float32 出（Android 侧零改动的前提）
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=out_path)
    it.allocate_tensors()
    id0, od0 = it.get_input_details()[0], it.get_output_details()[0]
    print(f"  io: in {id0['shape']} {id0['dtype'].__name__} -> "
          f"out {od0['shape']} {od0['dtype'].__name__}")
    if id0["dtype"] != np.float32 or od0["dtype"] != np.float32:
        print("  ❌ 接口不是 float32，Android 侧无法直接替换")
        return False
    del it

    return self_validate(model, out_path, w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="280x518,686x518",
                    help="WxH 列表，W/H 都必须被 14 整除（ViT patch）")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).eval()
    print(f"model: {MODEL_ID}")

    failed = []
    for spec in args.sizes.split(","):
        w, h = (int(x) for x in spec.lower().split("x"))
        assert w % 14 == 0 and h % 14 == 0, f"{spec}: 宽高必须被 14 整除"
        out = os.path.join(args.out_dir, f"dav2s_wi8_{w}x{h}.tflite")
        print(f"\n=== {w}x{h} ({(h // 14) * (w // 14)} tokens) ===")
        ok = export_size(model, w, h, out)
        if not ok and not args.no_validate:
            failed.append(spec)

    if failed:
        print(f"\n❌ 以下尺寸自检验未通过: {', '.join(failed)}")
        sys.exit(1)
    print("\nall done.")


if __name__ == "__main__":
    main()
