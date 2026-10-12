#!/bin/bash
# 跑真实视频两个视差档位 + 提取预览帧
set -e
cd "D:/project/3D/paraeye/p0"
export HF_ENDPOINT=https://hf-mirror.com
PY="C:/Users/Administrator/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
IN="D:/project/3D/781d6c67089968bf3d3ee0fa4437b249.mp4"

"$PY" pipeline.py "$IN" -o sbs_real_075.mp4 --max-disp 0.0075 --alpha 0.7
"$PY" pipeline.py "$IN" -o sbs_real_150.mp4 --max-disp 0.015 --alpha 0.7

"$PY" - <<'EOF'
import cv2
for tag in ['075', '150']:
    cap = cv2.VideoCapture(f'sbs_real_{tag}.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 260)
    ok, f = cap.read()
    if ok:
        cv2.imwrite(f'sbs_real_{tag}_preview.png', f)
    cap.release()
print('previews saved')
EOF
echo "ALL DONE"
