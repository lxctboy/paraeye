# ParaEye Android（P1 + P2 + P3 + P3.5）

阶段目标：

- **P1**：LiteRT GPU/CPU 上跑通 Depth Anything V2 Small 单帧推理，输出深度图与耗时基准，为模型选型提供实测数据 ✅
- **P2**：GPU 视图合成——深度驱动左右眼 warp，输出 Full-SBS，可在手机上直接观看 3D ✅（真机验证通过）
- **P3**：本地视频实时流水线——MediaCodec 硬解 → GPU 归一化/降采样 → 异步推理 → SBS 实时渲染 ✅（编译通过，真机跑通）
- **P3.5**：帧率优化——预处理 LUT/预分配、GL↔推理零拷贝、渲染线程逐帧深度过渡（治 6FPS 跳变）、GPU FP16 选项 ✅

## 技术栈

- Kotlin 2.4.20 / AGP 8.7.3 / Gradle 8.11.1，compileSdk 35，minSdk 26
- 推理：**LiteRT 2.3.0 V2 CompiledModel API**（`com.google.ai.edge.litert:litert` + `litert-gpu`）
- 渲染：**OpenGL ES 2.0**（兼容性优先，未用 GLES3 浮点纹理）
- 解码：**MediaExtractor + MediaCodec**（输出 Surface → SurfaceTexture，全程无 CPU 拷贝）
- 模型：**P4 起改用自导出的 280×518 INT8**（社区原版只有 686×518，会把竖屏视频竖挤 2.35 倍）
  - 由 `.github/workflows/export-tflite.yml` 重导出，发布到滚动 release「models」，构建时下载
  - 量化与社区原版同款：AI Edge Quantizer `dynamic_wi8_afp32`（int8 权重 + fp32 接口，27.6MB）
  - 输入契约：`[1, 3, 518, 280]` NCHW，RGB，/255 + ImageNet mean/std，输出 `[1, 518, 280]` 相对深度
  - 换 280×518 的收益：深度质量 corr 0.766 → 0.977（vs 原生宽高比真值）、token 数 0.41 倍；
    tflite vs torch 一致性 corr 0.9998 / relMAE 0.0072

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
2. 选 CPU/GPU × INT8/FP32 —— **实时场景建议先试 INT8**；GPU 上 INT8 不会更快（GPU delegate 权重按 FP32 执行，见下）
3. 画面直接以 SBS 实时渲染；状态区显示视频帧率 / 深度更新帧率 / 单帧推理耗时
4. 画面上下颠倒时勾选「视频上下翻转」（部分机型 OES 变换矩阵方向相反，见下方已知边界）
5. 「GPU FP16」复选框：仅 GPU 有效，通常快 1.3~2 倍、精度略降，可自行对比深度图质量
6. 「暂停播放」后画面停在当前帧，适合静态观察某一帧的立体感

> **GPU × INT8 为何不更快**：LiteRT GPU delegate 对 wi8/afp32 模型按 FP32 精度执行（`GpuOptions.Precision.FP32` 是 DA 解码器动态范围的保险默认），所以 GPU×INT8 与 GPU×FP32 耗时几乎相同（实测 177ms vs 176.8ms）。GPU 上要更快只有两条路：勾 FP16，或换更强的 GPU。

## P3 流水线结构

```
MediaExtractor ──▶ MediaCodec(硬解) ──▶ Surface(OES 纹理)
                                            │  [归一化 pass: texMtx + 可选翻转]
                                            ▼
                                       displayTex (2D, row0 = 图像顶部)
                                            │  [降采样 pass → FBO 280×518]
                                            ▼
                                     glReadPixels ──▶ RGBA 字节
                                            │  （双缓冲池零拷贝交推理线程，用完 recycle 归还）
                                            ▼
                          ImagePreprocessor.rgbaToNchwInto → LiteRT inferNchw
                                            │  normalize（目标深度，发布给渲染器）
                                            ▼
                             渲染线程：EMA advance(τ≈138ms) → depthToR8 → depthTex
                                            │
                                            ▼
                              单 program + 两次 glViewport → Full-SBS
```

关键设计：

- **推理与渲染解耦**：推理跑在单线程 executor 上，`busy` 时直接丢帧，显示帧率不受推理耗时拖累（代价是深度更新率低于视频帧率）
- **P3.5 逐帧深度过渡（治卡核心）**：推理只有 ~6FPS 时，离散 EMA 也只在深度到来时生效，中间渲染帧深度不变，看起来一跳一跳。`TemporalSmoother.advance()` 把同一个时间常数 τ≈138ms（与 P0 定档 α=0.7@6FPS **数学等效**，有单测）展开到每个渲染帧，视觉上把 6FPS 深度插值成渲染帧率
- **零分配热路径**：NCHW 输入、rgba 双缓冲池、深度上传 buffer、R8 输出全部预分配复用；归一化走 256×3 LUT（省 106 万次除法/帧）
- **TensorBuffer 复用**：P1 每次推理都 create/close，`inferNchw` 改为复用输入/输出缓冲，消除实时路径下的 GC 压力
- **EMA 与 P0 同口径**：作用在归一化后的深度上（帧间深度尺度会漂移，平滑原始深度会产生呼吸感）
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

深度以单通道 8bit（`GL_LUMINANCE`）上传：视差仅 1% 宽度，8bit 量化误差约 0.2px（1080p），可忽略。上传前设 `glPixelStorei(GL_UNPACK_ALIGNMENT, 1)`（宽度非 4 的倍数时必需；当前 280 满足，但改尺寸容易踩回这个坑）。

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

- [x] 视频能正常播放且方向正确（真机确认：默认方向正确，无需勾翻转，2026-10-12）
- [x] 视频帧率达到源片源帧率（真机 27.9 FPS @30fps 片源）
- [x] 深度更新帧率与单帧推理耗时（真机：深度 6.0 FPS / 推理 177.4ms，GPU×INT8）
- [x] SBS 立体效果真机可见（静态图路径用户确认）
- [ ] 逐帧过渡后的主观流畅度（P3.5 核心目标，待用户复测）
- [ ] GPU FP16 耗时与深度质量（待测）
- [ ] CPU×INT8 / CPU×FP32 耗时（待测，可能比弱 GPU 更快）
- [ ] glReadPixels 的实际开销：对比「有/无降采样 pass」的帧率差
- [ ] 播放流畅度主观评价：深度滞后（帧丢弃策略）是否可接受

## 已知边界

- **GPU×INT8 不加速**：GPU delegate 按 FP32 执行权重（保险默认），INT8 模型在 GPU 上与 FP32 耗时几乎相同；GPU 上提速靠 FP16 复选框
- **自导出模型 GPU 加载失败**：litert-torch 转换引入 GATHER_ND/SELECT_V2 等 GPU delegate 不支持的算子，选 GPU 会自动回退 CPU（统计行显示 GPU→CPU）
- 推理耗时：CPU×INT8 4 线程下 280×518 实测 **165.8ms**（686×518 为 485.9ms）；发热控制见「深度帧率上限」单选（2~5FPS，不限时占空比 ~90%）
- 预处理 LUT + 预分配已做，但仍是 CPU 循环（280×518×3 ≈ 43.5 万元素）
- 模型固定 280×518 拉伸输入；GL 渲染按源图宽高比 letterbox，Depth 采样跟随图像 UV，因此不会因为拉伸错位
- 空洞内插（P0 的 inpaint）尚未移植到 GL；当前靠 CLAMP_TO_EDGE 拉伸，近景边缘可能有轻微涂抹
- **OES 方向**：按 Grafika 约定推导（原始 `t=1` 对应视频顶部），部分机型/片源可能相反，故保留「上下翻转」开关而非硬编码
- 视频旋转元数据（`KEY_ROTATION`）未处理：竖屏录制但带 90° 旋转标记的视频可能显示不正确，先用翻转开关兜底
- 全屏观看：GL 铺满全屏 + 控制面板半透明覆盖层；进全屏自动转横屏（SBS 为 1440×1280 横宽高比，竖屏 letterbox 黑边大），`configChanges` 保证旋转不中断播放
- 摄像头实时流：**已跳过（2026-10-13 用户决定）**，不再做；视频保存（≤1920×1080 + 3D 元数据）在 P5
