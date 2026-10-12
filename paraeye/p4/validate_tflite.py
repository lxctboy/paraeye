# ParaEye P4 - tflite 与 torch 输出一致性验证
# 用法: python validate_tflite.py <model.tflite> <W> <H> [--frames 6]
# 作用：CI 导出的 tflite 下载回来后，用本脚本与本地 torch 模型逐帧对比，
#       确认「转换 + INT8 量化」没有破坏深度输出。本地 x86 CPU 速度不代表手机，
#       只验证数值一致性（结构相关 + 相对误差）。

import argparse
import cv2
import numpy as np
import torch
from transformers import AutoModelForDepthEstimation
from ai_edge_litert import interpreter

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
VIDEO = "D:/project/3D/781d6c67089968bf3d3ee0fa4437b249.mp4"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(frame_bgr, w, h):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    x = small.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None].astype(np.float32)


@torch.inference_mode()
def torch_depth(model, frame_bgr, w, h):
    x = torch.from_numpy(preprocess(frame_bgr, w, h))
    return model(pixel_values=x).predicted_depth.squeeze().float().cpu().numpy()


def tflite_depth(interp, in_idx, out_idx, frame_bgr, w, h):
    x = preprocess(frame_bgr, w, h)
    interp.set_tensor(in_idx, x)
    interp.invoke()
    return interp.get_tensor(out_idx).squeeze()


def compare(a, b):
    h = max(a.shape[0], b.shape[0])
    w = max(a.shape[1], b.shape[1])
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    rel = float(np.abs(a - b).mean() / (a.max() - a.min() + 1e-8))
    return corr, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("w", type=int)
    ap.add_argument("h", type=int)
    ap.add_argument("--frames", type=int, default=6)
    args = ap.parse_args()

    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).eval()
    interp = interpreter.Interpreter(model_path=args.model, num_threads=4)
    interp.allocate_tensors()
    in_idx = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]
    print(f"tflite: {args.model}")
    print(f"  input: {interp.get_input_details()[0]['shape']} {interp.get_input_details()[0]['dtype']}")
    print(f"  output: {interp.get_output_details()[0]['shape']} {interp.get_output_details()[0]['dtype']}")

    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 2, args.frames).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()

    corrs, rels = [], []
    for fr in frames:
        d_t = torch_depth(model, fr, args.w, args.h)
        d_l = tflite_depth(interp, in_idx, out_idx, fr, args.w, args.h)
        c, r = compare(d_t, d_l)
        corrs.append(c)
        rels.append(r)
    print(f"\ntflite vs torch @ {args.w}x{args.h}（{args.frames} 帧）:")
    print(f"  corr   = {np.mean(corrs):.4f}  (min {min(corrs):.4f})")
    print(f"  relMAE = {np.mean(rels):.4f}  (max {max(rels):.4f})")
    if np.mean(corrs) > 0.98:
        print("  ✅ 转换一致性良好")
    elif np.mean(corrs) > 0.95:
        print("  ⚠️ 有可见偏差，建议目检")
    else:
        print("  ❌ 转换可能有问题")


if __name__ == "__main__":
    main()
