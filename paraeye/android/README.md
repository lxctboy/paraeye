# ParaEye Android（P1 + P2）

阶段目标：

- **P1**：LiteRT GPU/CPU 上跑通 Depth Anything V2 Small 单帧推理，输出深度图与耗时基准，为模型选型提供实测数据 ✅
- **P2**：GPU 视图合成——深度驱动左右眼 warp，输出 Full-SBS，可在手机上直接观看 3D ✅

## 技术栈

- Kotlin 2.4.20 / AGP 8.7.3 / Gradle 8.11.1，compileSdk 35，minSdk 26
- 推理：**LiteRT 2.3.0 V2 CompiledModel API**（`com.google.ai.edge.litert:litert` + `litert-gpu`）
- 渲染：**OpenGL ES 2.0**（兼容性优先，未用 GLES3 浮点纹理）
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
gradle testDebugUnitTest      # JVM 单测（预处理契约 + ImageNet 归一化 + NCHW 布局）
```

**CI**：push 到 `main` 或打 `v*` tag 时，GitHub Actions（`.github/workflows/android.yml`）自动构建 debug APK 并上传 artifact（无需本地 JDK/SDK）。tag 触发的 Release 签名待 P5 配置 keystore secret。

## 使用

1. 「选择图片」从相册选一张（自动降采样到 2048 长边防 OOM）
2. 选加速器 CPU/GPU × 模型 INT8/FP32
3. 「生成 3D」：推理一次 + 跑基准，随即由 GL 渲染到下方画布
4. **视差滑块**：实时调 0~3%（默认 1%，P0 定档），改完立刻重绘，用于复现 P0 参数扫描
5. **显示模式**：深度图（红=近，蓝=远）/ SBS 3D
6. **左右交换**：交叉眼观看时勾选；平行眼（默认）不勾

观看 SBS 请用**平行眼（Parallel View）**：放松双眼让视线“看向远处”，使左眼盯左半图、右眼盯右半图，中间会出现第三个融合画面。初练可用 Virti/平行眼 3D 图片练习；也可用 VR 盒子+支持 SBS 的播放器。

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

深度以单通道 8bit（`GL_LUMINANCE`）上传：视差仅 1% 宽度，8bit 量化误差约 0.2px（1080p），可忽略。上传前须 `glPixelStorei(GL_UNPACK_ALIGNMENT, 1)`，因为 686 不是 4 的倍数。

## P1/P2 验收清单（真机执行）

- [x] APK 编译 + 单测通过（CI，commit `9b1beae`）
- [ ] GPU + INT8 首次加载时间与稳定性（GPU 编译约需数秒）
- [ ] CPU/GPU × INT8/FP32 四组合基准数据，填入 dev-v2.md 5.1.1（GPU×FP32 已填：avg 176.8ms）
- [ ] SBS 在手机上用平行眼能否融合出立体感（对比 P0 Python 输出）
- [ ] 视差滑块 0.5%~3% 主观舒适度：眼胀/重影边界在哪
- [ ] GPU 兼容性：中低端机 GPU 失败时是否正常回退 CPU（`ensureLoaded` 返回 false 的提示路径）

## 已知边界

- 预处理仍是 CPU 循环（`ImagePreprocessor.toNchwFloat`，686×518×3 ≈ 106 万元素），未计入 GL
- 模型固定 686×518 拉伸输入；GL 渲染按源图宽高比 letterbox， Depth 采样跟随图像 UV，因此不会因为拉伸错位
- 空洞内插（P0 的 inpaint）尚未移植到 GL；当前靠 CLAMP_TO_EDGE 拉伸，近景边缘可能有轻微涂抹
- 每帧都 `createInputBuffers/createOutputBuffers`，存在固定开销，可做预分配优化
- 摄像头流 / 视频解码在 P3
