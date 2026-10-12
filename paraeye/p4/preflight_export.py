# P4 本地预检：不依赖 litert-converter（无 Windows wheel），
# 只验证 CI 导出脚本的前半段：torch.export + pt2e 动态 INT8 量化图。
#
# quantizer 源码从 litert-torch wheel 拷出、改内部 import 后按文件加载：
#   pip download litert-torch==0.9.4 --no-deps -d /tmp/ltp
#   unzip /tmp/ltp/*.whl -d /tmp/ltp/pkg
#   mkdir -p "$TEMP/pt2e_local"
#   cp /tmp/ltp/pkg/litert_torch/quantize/pt2e_quantizer*.py "$TEMP/pt2e_local/"
#   sed -i 's/from litert_torch.quantize.pt2e_quantizer_utils import/from pt2e_quantizer_utils import/' \
#       "$TEMP/pt2e_local/pt2e_quantizer.py"
#
# 注意：pt2e 动态量化图在 eager 模式下跑不出正确数值（DRQ 语义靠转换器 lowering
# 成 TFLite DRQ kernel），本脚本只验证「export + 量化图构建」不炸，
# 数值正确性必须等 tflite 产出后用 validate_tflite.py 验。

import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForDepthEstimation

PT2E_DIR = os.environ.get(
    "PT2E_DIR", "C:/Users/ADMINI~1/AppData/Local/Temp/pt2e_local"
)
sys.path.insert(0, PT2E_DIR)
from pt2e_quantizer import PT2EQuantizer, get_symmetric_quantization_config  # noqa: E402
try:  # torch>=2.14 搬去了 torchao；CI 的 torch<2.14 仍在 torch.ao
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e  # noqa: E402
except ImportError:
    from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e  # noqa: E402

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
W, H = 280, 518


class DepthOnly(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m(pixel_values=pixel_values).predicted_depth


@torch.inference_mode()
def run(model, x):
    return model(x)


def main():
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).eval()
    wrapper = DepthOnly(model).eval()
    sample = (torch.randn(1, 3, H, W),)

    # 参考输出（fp32）
    ref = run(wrapper, sample[0]).numpy()

    t0 = time.time()
    ep = torch.export.export(wrapper, sample)
    print(f"1. torch.export OK ({time.time() - t0:.1f}s), graph nodes: {len(ep.graph.nodes)}")

    quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
    )
    t0 = time.time()
    ep_q = prepare_pt2e(ep.module(), quantizer)  # 注意：要 GraphModule 不是 ExportedProgram
    ep_q = convert_pt2e(ep_q, fold_quantize=False)
    print(f"2. pt2e prepare+convert OK ({time.time() - t0:.1f}s)")

    # 量化图前向（动态量化：运行时算激活 scale，无需校准）
    t0 = time.time()
    out_q = ep_q(sample[0]).detach().numpy()
    print(f"3. quantized forward OK ({time.time() - t0:.1f}s)")

    corr = float(np.corrcoef(ref.ravel(), out_q.ravel())[0, 1])
    rel = float(np.abs(ref - out_q).mean() / (ref.max() - ref.min()))
    print(f"\nquantized vs fp32 @ {W}x{H}: corr={corr:.4f} relMAE={rel:.4f}")
    print("（eager 数值无意义是预期的——DRQ 语义要转换器 lowering；"
          "此脚本只验「export+量化图构建」不炸）")
    print("链路预演通过：CI 只剩 litert_torch.convert 一步，"
          "数值闸门用 validate_tflite.py 在 tflite 产出后验")


if __name__ == "__main__":
    main()
