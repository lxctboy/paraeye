# ParaEye（平行眼）· 手机端实时 2D→平行眼 3D 转换 · 开发计划 v2（修正版）

> 基于 v1 方案（dev.md）+ 12 条疑问核查后的修正版。  
> 修正日期：2026-10-12

**项目命名**：**ParaEye（平行眼）**（2026-10-12 确认）
- GitHub 仓库名：`paraeye`
- Android 包名：`com.paraeye.app`

---

## 一、关键决策（已确定项）

| 项目       | 决策                                                                                       |
| -------- | ---------------------------------------------------------------------------------------- |
| 深度模型（主选） | **Depth Anything V2 Small（24.8M）**，社区验证充分，Core ML 有官方转换版                                 |
| 深度模型（备选） | **Depth Anything 3 Small**（字节 Seed，2025），LiteRT 社区有 GPU 版，FP16 约 55MB；参数量/移动端实测数据待 P1 核实 |
| 已移除      | ~~ZipDepth（6.1M）~~ —— 未找到公开发表来源                                                          |
| SBS 顺序   | 默认 **L/R**，App 内提供 L/R 切换开关                                                              |
| 目标播放设备   | **裸眼平行眼（已确认）**：默认 L/R 顺序；视差按裸眼舒适区取保守值，L/R 切换开关保留                                  |
| 视差幅度     | **默认 1%（总视差，P0 实测定档）**，用户可调；负视差（出屏）上限 1%，正视差（入屏）上限 2%                        |
| 平台优先级    | **Android 单人开发（已确认）**，CI/CD 用 **GitHub Actions 自动打包 APK**；iOS 暂不做                                  |
| 模型终选机制   | P1 阶段用 LiteRT GPU 版实测 DA3 Small 推理耗时，与 V2 Small 对比后再决定是否替换 |
| 摄像头流     | P3 只做**本地文件**，摄像头路径推迟到 P4/后续版本                                                           |

---

## 二、修正后的流水线与帧预算

目标 30 FPS（每帧 33ms），**端到端延迟 <100ms**（解码→显示）。

```
视频帧输入（硬件解码）
    ↓ [5~10ms] 预处理：缩放至模型输入 + 归一化（GPU）
    ↓ [15~20ms] 深度推理：DA V2 Small，INT8/FP16 + GPU Delegate
    ↓ [3~5ms] 深度后处理：低分辨率引导滤波 → 上采样 → EMA 时域平滑
    ↓ [3~5ms] 视图合成 + SBS 拼接（GPU 着色器）
    ↓ [2~3ms] 渲染输出
```

合计约 28~43ms，超预算时按优先级降级：先降推理输入分辨率 → 再开帧率自适应（30→24 FPS）。

---

## 三、各模块修正摘要

### 模块 2：深度估算

- 模型换为 DA V2 Small / DA3 Small；FP16 为主，INT8 需验证深度精度无损后再启用。

### 模块 3：深度后处理

- **引导滤波在模型分辨率（518×518）做，再上采样到全分辨率**，不做全分辨率引导滤波。
- 时域平滑：P0 只用 EMA（α=0.6~0.8）。若实测拖影明显，P1/P2 评估轻量光流对齐（PWC-Net 轻量版类）。
- 深度范围重映射：**按场景更新**（场景切换时重算），禁止逐帧直方图均衡；备选低频更新（每 30~60 帧）。

### 模块 4：视图合成

- 符号约定 P0 必须验证：近景负视差（出屏）、远景正视差（入屏），左右眼位移方向不能颠倒。
- 空洞填充：反向映射 + 双线性为主，邻域填充兜底。

### 模块 5：拼接

- Full-SBS，默认 L/R，可切换。

### 模块 6：渲染与输出

- **保存路径限制 SBS 输出 ≤1920×1080**（单眼 960×1080），避免触发编码器上限；分辨率需为 16 的倍数。
- 保存文件写入 SBS 3D 元数据（如 Matroska StereoMode / MP4 对应字段）。

---

## 四、P0 实验设计（详细）

### 4.1 环境

- PC 端 Python：PyTorch + Depth Anything V2 Small + OpenCV，离线生成 SBS 视频输出。

### 4.2 测试视频集（共 6 段，每段 10~15s，1080p）

| 编号 | 类型   | 内容要求                |
| -- | ---- | ------------------- |
| T1 | 静态场景 | 室内/风景固定机位，前景背景分明    |
| T2 | 静态场景 | 人像特写（人脸深度平滑很重要）     |
| T3 | 运动场景 | 相机平移（横摇），考验时域拖影     |
| T4 | 运动场景 | 物体快速移动（行人/车流）       |
| T5 | 遮挡场景 | 前景物体遮挡大面积背景（考验空洞填充） |
| T6 | 混合   | 手持拍摄，含轻微抖动          |

### 4.3 符号约定验证（先行，0.5 天）

- 用合成素材（已知深度的渐变背景 + 已知位置球体）验证：
  - 近景应"出屏"（负视差）、远景"入屏"（正视差）；
  - L/R 顺序下左眼图、右眼图方向是否正确；
  - 若颠倒，翻转位移符号即可，但要记录最终公式。

### 4.4 视差参数扫描

- max_disparity ∈ {0.5%, 0.75%, 1%, 1.5%, 2%, 3%} × 图像宽度（总视差）。
- 每个 T1~T6 × 每档生成一段 SBS 输出。

### 4.5 时域参数扫描

- EMA α ∈ {0.6, 0.7, 0.8, 1.0（不平滑）}，只对 T3/T4/T6 做。
- 记录拖影程度，决定是否需要引入光流。

### 4.6 评审表格模板（每位评审 × 每段视频一份）

| 评分项                | 记录 |
| ------------------ | -- |
| 立体融合时间（秒，多次取均值）    |    |
| 立体感强度（1~5，5 最强）    |    |
| 眼胀/头晕（是/否）         |    |
| 深度闪烁（1~5，5 无闪烁）    |    |
| 边缘伪影/空洞（1~5，5 无伪影） |    |
| 拖影（仅运动类，1~5）       |    |

评审人数 ≥3，评审者需会用平行眼/有 3D 观看经验。

### 4.7 P0 验收标准

- **融合通过率 ≥80%**：评审者 10 秒内成功融合出立体感；
- **无持续性眼胀/头晕反馈**；
- 从参数扫描中**选出默认视差值与 EMA α**（最佳舒适度-立体感折中）；
- 明确 T3/T4/T6 的拖影是否可接受 → 决定光流是否纳入 P1/P2。
- ~~确认目标播放设备~~ → 已确认：裸眼平行眼，默认 L/R。

### 4.9 P0 执行结果（2026-10-12 完成 ✅）

| 项目 | 结果 |
|------|------|
| 验证视频 | `781d6c67089968bf3d3ee0fa4437b249.mp4`（720×1280 竖屏 / 30fps / 523 帧 / 17.4s） |
| 符号约定 | 合成场景模板匹配验证通过：近球左眼在右（负视差=出屏），方向无需翻转 ✅ |
| 参数扫描 | 0.75% / 1.0% / 1.5% 三档均跑通（523 帧全量） |
| **定档参数** | **max_disp = 1%（图像宽度），alpha = 0.7（EMA）** |
| CPU 参考耗时 | 单帧整帧约 1.30~1.47s（PC 离线，仅作验证；移动端走 GPU/NPU） |
| 代码默认值 | `pipeline.py` 默认已改为 `--max-disp 0.01` |
| 待补实验 | 运动类视频拖影评估（决定光流是否纳入 P1/P2）尚未单独记录 |

### 4.8 P0 产出

1. Python 验证脚本仓库（含参数配置）；
2. 参数扫描结果矩阵 + 评审汇总表；
3. 默认参数定稿；
4. Go/No-Go 结论：效果达标 → 进 P1（Android 推理）；不达标 → 回到模型/后处理方案调整。

---

## 五、里程碑 v2

| 阶段 | 目标                | 关键变更                                         |
| -- | ----------------- | -------------------------------------------- |
| P0 | 离线验证 + 参数定稿 | **✅ 已完成（2026-10-12）**：符号验证通过，定档 max_disp=1% / alpha=0.7，见 4.9 |
| P1 | Android 单帧推理跑通    | **🟡 CI 已全绿（2026-10-12）**：APK 编译 + 单测通过；待真机回填 5.1 表（V2 Small vs DA3 Small 实测后终选模型） |
| P2 | GPU 渲染：视图合成 + SBS | **✅ 真机验证通过（2026-10-12）**：静态图路径 SBS 3D 效果确认可见（平行眼），符号约定无需翻转；GL 视图合成 + Full-SBS + 视差滑块（默认 1%）；低分辨率引导滤波仍在 P0 Python 侧，未移植 |
| P3 | 实时流水线（本地文件）       | **🟡 CI 已全绿（2026-10-12，commit `9eecbbe`）**：MediaCodec 硬解 → OES → 降采样 FBO → readPixels → 异步推理 → SBS，**待真机验证**；摄像头流移出本阶段 |
| P4 | 性能优化 + 摄像头流       | 帧预算达标、功耗控制、评估光流                              |
| P5 | 产品化               | L/R 开关、视差滑块、保存（≤1920×1080 + 3D 元数据）、兼容性      |

### 5.1 CI/CD：GitHub Actions 打包 APK（单人工作流）

仓库结构建议：`main` 分支开发，打 tag（如 `v0.1.0`）触发发布构建。

**P1 阶段先搭 debug 构建**，workflow 草案：

```yaml
# .github/workflows/android.yml
name: Android APK
on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-java@v4
        with: { distribution: temurin, java-version: '17' }
      - uses: gradle/actions/setup-gradle@v4
      - name: Build APK
        run: |
          if [[ "$GITHUB_REF" == refs/tags/v* ]]; then
            ./gradlew assembleRelease
          else
            ./gradlew assembleDebug
          fi
      - uses: actions/upload-artifact@v4
        with:
          name: app-apk
          path: app/build/outputs/apk/**/*.apk
```

要点：
- **模型文件（tflite，约 25~55MB）不进 Git**：用 Git LFS 或在 CI 中从 release URL 下载，避免仓库膨胀和 Actions 慢；
- tag 触发 Release 构建需配置签名：`keystore` 存为 repo secret，CI 中还原后 `assembleRelease`；
- 真机性能测试无法在 CI 做，APK 产出后手动装到测试机跑基准（P1 验收即为此流程）。

### P1 模型对比实测清单（LiteRT，同一测试机）

判定规则：若 DA3 Small 推理耗时 ≤ V2 Small 且精度不差 → 换 DA3；否则保持 V2 Small。

#### 5.1.1 真机基准数据（口径：单帧端到端 = 预处理 + 缓冲区 + 纯推理，预热 3 + 20 迭代）

| 组合 | avg | p50 | min | max | FPS | 备注 |
|------|-----|-----|-----|-----|-----|------|
| GPU × FP32 | **176.8 ms** | 177.8 | 168.4 | 181.3 | 5.7 | 首个真机数据（2026-10-12），设备型号待补 |
| GPU × INT8 | 待测 | | | | | |
| CPU × FP32 | 待测 | | | | | |
| CPU × INT8 | 待测 | | | | | |

深度图验证（GPU×FP32，餐饮场景照片）：碗沿最近（红）→ 桌面中间调 → 背景最远（蓝），分层连续正确 ✅。色带约定：**红=近，蓝=远**（近暖远冷）。

**立体效果验证（2026-10-12）**：静态图路径 SBS 输出，用户平行眼观看**确认可见 3D 效果** ✅ —— GL 渲染符号约定（近物出屏）在真机成立，`--swap`/左右交换开关仅留给交叉眼场景。**待确认**：视频路径（P3）的画面方向与三个实时数字。

距 30FPS 帧预算（推理 ≤15~20ms）尚差约一个数量级，候选优化按 dev-v2 降级顺序：
1. **输入降分辨率**（686×518 → 518×392 或更小，ViT 计算量随 token 数平方下降，收益最大）；
2. **隔帧推理 + 深度复用**（每 N 帧推一次，中间帧复用上帧深度 + 光流/平移修正——本来就排在 P4 评估）；
3. ~~预分配输入输出缓冲区~~ → P3 已做（TensorBuffer 复用 + `inferNchw()` 直传，跳过 Bitmap 转换）；
4. INT8 量化（等 5.1.1 数据）；
5. NPU 路径（Vendor 加速器，P4 再评估）。

### 5.2 P1 工程状态（2026-10-12）

工程位置：`paraeye/android/`（Kotlin + AGP 8.7.3 / Gradle 8.11.1 / minSdk 26）

| 组件 | 文件 | 状态 |
|------|------|------|
| LiteRT 推理封装 | `depth/DepthEstimator.kt` | ✅ CompiledModel V2 API，CPU/GPU × INT8/FP32 四组合 |
| 预处理 | `depth/ImagePreprocessor.kt` | ✅ 契约：686×518 NCHW，ImageNet mean/std（JVM 单测覆盖） |
| 可视化 | `depth/DepthVisualizer.kt` | ✅ HSV 色带（蓝=近，红=远） |
| 基准 | `depth/DepthBenchmark.kt` | ✅ 预热 3 + 20 迭代，avg/p50/min/max |
| UI | `MainActivity.kt` + `activity_main.xml` | ✅ 选图 → 四组合切换 → 基准 |
| 模型下载 | `app/build.gradle.kts` `downloadDepthModels` | ✅ HF `litert-community/depth-anything-v2-small`，构建时下载，不进 Git |
| CI | `.github/workflows/android.yml` | ✅ main push / v* tag → debug APK artifact + 单测 |
| DA3 Small tflite | — | ⏳ 待 LiteRT 社区提供导出后加入对比（当前仅有 V2 Small） |

**本机无 JDK/Android SDK**，编译验证走 CI；API 均已按官方 codelab/示例核对（CompiledModel.create(assets, name, Options(accelerator), env) + GpuOptions FP32）。

**P1 剩余（需真机）**：四组合基准数据回填 5.1 表；GPU 失败回退 CPU 路径验证；与 P0 Python 输出定性对比。

#### 5.2.1 CI 首次构建结果（2026-10-12，commit `9b1beae`）

**✅ assembleDebug + testDebugUnitTest 全绿**，产物 `paraeye-debug-apk`（128 MB，内嵌 FP32+INT8 两个 tflite）。

修了三轮，全部为依赖/测试问题，**实现代码无误**：

| # | 报错 | 根因 | 解法 |
|---|------|------|------|
| 1 | `Class 'litert.CompiledModel' was compiled with an incompatible version of Kotlin. Actual metadata 2.4.0, compiler 2.0.0 reads up to 2.1.0` | LiteRT 2.3.0 的 Kotlin API 用 Kotlin 2.4 编译；Kotlin 编译器只能读高一个版本的元数据 | Kotlin 插件 **2.0.21 → 2.4.20**（AGP 8.7.3 / Gradle 8.11.1 无需连带升级） |
| 2 | `MainActivity.kt:56:70 Unresolved reference 'Request'` | `PickVisualMediaRequest` 是 `androidx.activity.result` 的**顶层类**，不是 `PickVisualMedia` 的嵌套类 | `PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)` |
| 3 | 2 个单测 AssertionError | 测试自身写错：张量元素数魔术数字 1065684（实际 `3×518×686 = 1,066,044`）；断言中灰归一化 `< 0`（实际 `(128/255-0.485)/0.229 = +0.074`） | 修正预期值；并把 `argbToNchw` 抽成脱离 Bitmap 的纯函数，补 NCHW 三平面布局测试 |

**⚠️ 降级红线**：Kotlin 插件不可降到 2.4 以下，否则 Litert API 元数据读不了，会整编译失败。

**技术债**：CI 有 `Gradle 8.11.1 已废弃（Kotlin 2.5 起最低要求 8.14.4）` 警告，当前不阻塞；将来升 Kotlin 时需同步升 Gradle。APK 128 MB 偏大，P5 可改为模型按需下载或 ABI 分包。

| 事项 | 结论 | 确认时间 |
|------|------|----------|
| 开发规模 | 单人开发 | 2026-10-12 |
| 平台 | Android-only，GitHub Actions 打包 APK | 2026-10-12 |
| 目标播放设备 | 裸眼平行眼，默认 L/R | 2026-10-12 |
| 模型终选 | P1 实测 DA3 Small（LiteRT GPU）后决定是否替换 V2 Small | 2026-10-12 |
