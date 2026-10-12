# ParaEye P4 - DA V2 Small 重导出：更小输入分辨率（降 token 数 = 降 CPU 推理耗时）
#
# 背景：当前线上模型输入 686x518（横宽高比），而内容是 720x1280 竖屏视频——
# 降采样 FBO 把画面竖挤了 2.35 倍。torch 级质量闸门（validate_resolution.py）显示：
#   280x518 竖屏输入 vs 原生宽高比真值：corr 0.977（当前 686x518 仅 0.766）
# 即换小尺寸同时拿到「消除拉伸 + token 1813→740（0.41 倍）」两个收益。
#
# 量化方案：pt2e 对称动态 INT8（per-channel 权重 + DRQ 激活，无需校准数据），
# 与社区 wi8_afp32 模型同 scheme；本地验证基线 corr 0.9983 / relMAE 0.0104。
#
# 依赖 litert-torch（pip 包名 ai-edge-torch 已更名），仅 Linux/macOS 可跑
# （litert-converter 无 Windows wheel）——由 .github/workflows/export-tflite.yml 调用。
#
# 用法: python export_tflite.py --sizes 686x518,392x518,280x518 --out-dir out

import argparse
import os
import time

import torch
from transformers import AutoModelForDepthEstimation

import litert_torch
from litert_torch.quantize import pt2e_quantizer
from litert_torch.quantize import quant_config

try:  # torch>=2.14 起 quantize_pt2e 搬进 torchao；CI 的 torch<2.14 可能仍在 torch.ao
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
except ImportError:
    from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthOnly(torch.nn.Module):
    """剥掉 HF 外壳，只留 pixel_values -> predicted_depth，便于 trace/导出。"""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m(pixel_values=pixel_values).predicted_depth


def _prepare(ep, quantizer):
    """prepare_pt2e 的入参随 torch 版本漂移：老版吃 ExportedProgram，新版要 GraphModule。"""
    candidates = [ep]
    if hasattr(ep, "module"):
        candidates.append(ep.module())
    last = None
    for c in candidates:
        try:
            return prepare_pt2e(c, quantizer)
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"prepare_pt2e failed: {last}")


def export_size(model, w: int, h: int, out_path: str) -> None:
    wrapper = DepthOnly(model).eval()
    sample = (torch.randn(1, 3, h, w),)

    quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
        pt2e_quantizer.get_symmetric_quantization_config(
            is_per_channel=True, is_dynamic=True
        )
    )

    # 动态量化（DRQ）：int8 per-channel 权重 + 运行时动态激活量化，float 进 float 出
    # （与社区 wi8_afp32 模型同 scheme，Android 侧零改动）。
    # 注意：pt2e 动态量化图在 eager 模式下跑不出正确数值（DRQ 语义靠转换器 lowering），
    # 数值验证必须等 tflite 产出后用 validate_tflite.py 做。
    t0 = time.time()
    try:
        ep = torch.export.export(wrapper, sample)
    except Exception as e:  # torch 版本差异的兜底
        print(f"  torch.export 失败({e})，尝试 capture_pre_autograd_graph")
        ep = torch._export.capture_pre_autograd_graph(wrapper, sample)
    print(f"  export ok ({time.time() - t0:.0f}s)")

    t0 = time.time()
    ep_q = _prepare(ep, quantizer)
    ep_q = convert_pt2e(ep_q, fold_quantize=False)
    print(f"  pt2e quantize ok ({time.time() - t0:.0f}s)")

    t0 = time.time()
    edge = litert_torch.convert(
        ep_q,
        sample,
        quant_config=quant_config.QuantConfig(pt2e_quantizer=quantizer),
    )
    edge.export(out_path)
    print(f"  tflite exported: {out_path} "
          f"({os.path.getsize(out_path) / 1e6:.1f} MB, {time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="686x518,392x518,280x518",
                    help="WxH 列表，W/H 都必须被 14 整除（ViT patch）")
    ap.add_argument("--out-dir", default="out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).eval()
    print(f"model: {MODEL_ID}")

    for spec in args.sizes.split(","):
        w, h = (int(x) for x in spec.lower().split("x"))
        assert w % 14 == 0 and h % 14 == 0, f"{spec}: 宽高必须被 14 整除"
        out = os.path.join(args.out_dir, f"dav2s_wi8_{w}x{h}.tflite")
        print(f"\n=== {w}x{h} ({(h // 14) * (w // 14)} tokens) ===")
        export_size(model, w, h, out)

    print("\nall done.")


if __name__ == "__main__":
    main()
