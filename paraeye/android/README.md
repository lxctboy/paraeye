# ParaEye Android（P1 + P2 + P3）

阶段目标：

- **P1**：LiteRT GPU/CPU 上跑通 Depth Anything V2 Small 单帧推理，输出深度图与耗时基准，为模型选型提供实测数据 ✅
- **P2**：GPU 视图合成——深度驱动左右眼 warp，输出 Full-SBS，可在手机上直接观看 3D ✅
- **P3**：本地视频实时流水线——MediaCodec 硬解 → GPU 归一化/降采样 → 异步推理 → SBS 实时渲染 🚧（编译通过，待真机验证）

## 技术栈

- Kotlin 2.4.20 / AGP 8.7.3 / Gradle 8.11.1，compileSdk 35，minSdk 26
- 推理：**LiteRT 2.3.0 V2 CompiledModel API**（`com.google.ai.edge.litert:litert` + `litert-gpu`）
- 渲染：**OpenGL ES 2.0**（兼容性优先，未用 GLES3 浮点纹理）
- 解码：**MediaExtractor + MediaCodec**（输出 Surface → SurfaceTexture，全程无 CPU 拷贝）
- 模型：HuggingFace `litert-community/depth-anything-v2-small`（官方 LiteRT 导出）
  - INT8（dynamic wi8 / afp32，27.7MB）—— 移动端首选
  - FP32（99.5MB）—— 精度基准
  - 输入契约：`[1, 3, 518, 686]` NCHW，RGB，/255 + ImageNet mean/std，输出 `[1, 518, 686]` 相对深度

> ⚠️ **降级红线**：LiteRT 2.3.0 的 Kotlin API 元数据为 2.4.0，**Kotlin 插件不可低于 2.4**，否则 `compileDebugKotlin` 全量报 “incompatible Kotlin metadata”。

## 模型下载

模型文件**不进 Git**。`app/build.gradle.kts` 注册了 `downloadDepthModels` 任务并挂到 `preBuild`：首次构建自动从 HuggingFace 下载两个 tflite 到 `app/src/main/assets/models/`。

国内网络访问 HuggingFace 失败时，先设镜像再构建：

```bash
export HF_ENDPOINT=https://hf-mirror.com   # Git Bash
```

## 构建与运行

**本地（Android Studio）**：直接 Open 本目录，Sync 后 Run。首次同步会自动生成 Gradle Wrapper 并下载模型。

**命令行**（本机已有 Gradle 8.11+ 和 Android SDK 时）：

```bash
gradle assembleDebug          # 输出 app/build/outputs/apk/debug/app-debug.apk
gradle testDebugUnitTest      # JVM 单测（预处理契约 + ImageNet 归一化 + NCHW 布局 + EMA）
```

**CI**：push 到 `main` 或打 `v*` tag 时，GitHub Actions（`.github/workflows/android.yml`）自动构建 debug APK 并上传 artifact（无需本地 JDK/SDK）。tag 触发的 Release 签名待 P5 配置 keystore secret。

## 使用

### 静态图（P1/P2 单帧验证）

1. 「选择图片」从相册选一张（自动降采样到 2048 长边防 OOM）
2. 选加速器 CPU/GPU × 模型 INT8/FP32
3. 「生成 3D」：推理一次 + 跑基准，随即由 GL 渲染到下方画布
4. **视差滑块**：实时调 0~3%（默认 1%，P0 定档），改完立刻重绘，用于复现 P0 参数扫描

### 视频（P3 实时流水线）

1. 「选择视频」从相册选一段本地视频（自动循环播放）
2. 选 CPU/GPU × INT8/FP32 —— **实时场景建议先试 INT8**
3. 画面直接以 SBS 实时渲染；状态区显示视频帧率 / 深度更新帧率 / 单帧推理耗时
4. 画面上下颠倒时勾选「视频上下翻转」（部分机型 OES 变换矩阵方向相反，见下方已知边界）
5. 「暂停播放」后画面停在当前帧，适合静态观察某一帧的立体感

## P3 流水线结构

```
MediaExtractor ──▶ MediaCodec(硬解) ──▶ Surface(OES 纹理)
                                            │  [归一化 pass: texMtx + 可选翻转]
                                            ▼
                                       displayTex (2D, row0 = 图像顶部)
                                            │  [降采样 pass → FBO 686×518]
                                            ▼
                                     glReadPixels ──▶ RGBA 字节
                                            │  （GL 线程上 copyOf 后交给推理线程）
                                            ▼
                          ImagePreprocessor.rgbaToNchw → LiteRT inferNchw
                                            │  normalize → EMA(α=0.7) → depthToR8
                                            ▼
                                      depthTex (GL_LUMINANCE)
                                            │
                                            ▼
                              单 program + 两次 glViewport → Full-SBS
```

关键设计：

- **推理与渲染解耦**：推理跑在单线程 executor 上，`busy` 时直接丢帧，显示帧率不受推理耗时拖累（代价是深度更新率低于视频帧率）
- **TensorBuffer 复用**：P1 每次推理都 create/close，`inferNchw` 改为复用输入/输出缓冲，消除实时路径下的 GC 压力
- **EMA 与 P0 同口径**：`D_t = 0.7·cur + 0.3·prev_smoothed`，作用在归一化后的深度上（帧间深度尺度会漂移，平滑原始深度会产生呼吸感）
- **EGL 上下文保留**：`setPreserveEGLContextOnPause(true)`，否则 onPause/onResume 重建 GL 资源会让解码器持有的 SurfaceTexture 悬空

## 视图合成约定（与 P0 完全一致）

来自 `paraeye/p0/synthesize.py`，已用合成场景 + 模板匹配数值验证：

| 项 | 约定 |
|----|------|
| 深度 | `d ∈ [0,1]`，1=近 |
| 总视差 | `disparity = d × maxDispRatio`（以图像宽度比例表达，转 UV 后与分辨率无关） |
| 左眼 | `I_L(u) = I(u − dis/2)` → 内容右移 |
| 右眼 | `I_R(u) = I(u + dis/2)` → 内容左移 |
| 结果 | 近景在左图中偏右、右图中偏左 ⇒ **平行眼下近物出屏** |
| 边缘 | 纹理 `CLAMP_TO_EDGE`，等价 P0 的 `BORDER_REPLICATE` |

**纹理坐标总约定**：`displayTex` / `depthTex` 的 **row0 = 图像顶部**（与 `GLUtils.texImage2D` 上传 Bitmap 的布局一致）。GL 帧缓冲原点在左下，所以所有「渲染到 FBO」的 pass 都要用 `aPos.y*0.5+0.5` 再翻一次——这是 P3 最容易踩的坑。

深度以单通道 8bit（`GL_LUMINANCE`）上传：视差仅 1% 宽度，8bit 量化误差约 0.2px（1080p），可忽略。上传前须 `glPixelStorei(GL_UNPACK_ALIGNMENT, 1)`，因为 686 不是 4 的倍数。

观看 SBS 请用**平行眼（Parallel View）**：放松双眼让视线“看向远处”，使左眼盯左半图、右眼盯右半图，中间会出现第三个融合画面。

## 验收清单（真机执行）

### P1/P2

- [x] APK 编译 + 单测通过（CI，commit `9b1beae`）
- [ ] GPU + INT8 首次加载时间与稳定性（GPU 编译约需数秒）
- [ ] CPU/GPU × INT8/FP32 四组合基准数据，填入 dev-v2.md 5.1.1（GPU×FP32 已填：avg 176.8ms）
- [ ] SBS 在手机上用平行眼能否融合出立体感（对比 P0 Python 输出）
- [ ] 视差滑块 0.5%~3% 主观舒适度：眼胀/重影边界在哪
- [ ] GPU 兼容性：中低端机 GPU 失败时是否正常回退 CPU（`ensureLoaded` 返回 false 的提示路径）

### P3

- [ ] 视频能正常播放且方向正确（若上下颠倒，验证「上下翻转」开关是否修正）
- [ ] 视频帧率是否达到源片源帧率（30fps）
- [ ] 深度更新帧率与单帧推理耗时（INT8 预期显著优于 FP32）
- [ ] glReadPixels 的实际开销：对比「有/无降采样 pass」的帧率差
- [ ] 播放流畅度主观评价：深度滞后（帧丢弃策略）是否可接受

## 已知边界

- 预处理仍是 CPU 循环（`ImagePreprocessor.rgbaToNchw`，686×518×3 ≈ 106 万元素），未计入 GL
- 模型固定 686×518 拉伸输入；GL 渲染按源图宽高比 letterbox，Depth 采样跟随图像 UV，因此不会因为拉伸错位
- 空洞内插（P0 的 inpaint）尚未移植到 GL；当前靠 CLAMP_TO_EDGE 拉伸，近景边缘可能有轻微涂抹
- **OES 方向**：按 Grafika 约定推导（原始 `t=1` 对应视频顶部），部分机型/片源可能相反，故保留「上下翻转」开关而非硬编码
- 视频旋转元数据（`KEY_ROTATION`）未处理：竖屏录制但带 90° 旋转标记的视频可能显示不正确，先用翻转开关兜底
- 摄像头流在 P4；视频保存（≤1920×1080 + 3D 元数据）在 P5
