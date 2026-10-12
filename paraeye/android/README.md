# ParaEye Android（P1）

P1 目标：**Android 单帧推理跑通**——LiteRT GPU/CPU 上运行 Depth Anything V2 Small，输出深度图与耗时基准，为 P1 的模型选型（V2 Small vs DA3 Small）提供实测数据。

## 技术栈

- Kotlin 2.4.20 / AGP 8.7.3 / Gradle 8.11.1，compileSdk 35，minSdk 26
- 推理：**LiteRT 2.3.0 V2 CompiledModel API**（`com.google.ai.edge.litert:litert` + `litert-gpu`）
- 模型：HuggingFace `litert-community/depth-anything-v2-small`（官方 LiteRT 导出）
  - INT8（dynamic wi8 / afp32，27.7MB）—— 移动端首选
  - FP32（99.5MB）—— 精度基准
  - 输入契约：`[1, 3, 518, 686]` NCHW，RGB，/255 + ImageNet mean/std，输出 `[1, 518, 686]` 相对深度

## 模型下载

模型文件**不进 Git**。`app/build.gradle.kts` 注册了 `downloadDepthModels` 任务并挂到 `preBuild`：首次构建自动从 HuggingFace 下载两个 tflite 到 `app/src/main/assets/models/`。

国内网络访问 HuggingFace 失败时，先设镜像再构建：

```bash
export HF_ENDPOINT=https://hf-mirror.com   # Git Bash
# 或在 gradle.properties 临时加：systemProp.https.proxyHost=...
```

## 构建与运行

**本地（Android Studio）**：直接 Open 本目录，Sync 后 Run。首次同步会自动生成 Gradle Wrapper 并下载模型。

**命令行**（本机已有 Gradle 8.11+ 和 Android SDK 时）：

```bash
gradle assembleDebug          # 输出 app/build/outputs/apk/debug/app-debug.apk
gradle testDebugUnitTest      # JVM 单测（预处理契约 + ImageNet 归一化）
```

**CI**：push 到 `main` 或打 `v*` tag 时，GitHub Actions（`.github/workflows/android.yml`）自动构建 debug APK 并上传 artifact（无需本地 JDK/SDK）。tag 触发的 Release 签名待 P5 配置 keystore secret。

## 使用

1. 「选择图片」从相册选一张（自动降采样到 2048 长边防 OOM）
2. 选加速器 CPU/GPU × 模型 INT8/FP32
3. 「运行推理基准」：预热 3 次 + 20 次迭代，显示深度图（红=近，蓝=远）与 avg/p50/min/max 耗时

## P1 验收清单（真机执行）

- [ ] GPU + INT8 首次加载时间与稳定性（GPU 编译约需数秒）
- [ ] CPU/GPU × INT8/FP32 四组合的基准数据，填入 dev-v2.md 5.1 对比表
- [ ] 深度图与 P0 Python 版输出的定性对比（抽 2~3 张同一图片）
- [ ] GPU 兼容性：中低端机 GPU 失败时是否正常回退 CPU（`ensureLoaded` 返回 false 的提示路径）

## 已知边界

- 单帧版本：预处理是 CPU 循环（约 30~60ms），P2 迁移到 GPU/GL
- 模型固定 686×518 拉伸输入，竖屏长图内容会横向压缩；P2 评估是否改为 padding 方案
- 摄像头流、SBS 合成不在 P1 范围
