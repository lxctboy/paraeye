package com.paraeye.app.depth

import android.content.Context
import android.graphics.Bitmap
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

/**
 * LiteRT 2.3.0 V2 CompiledModel 封装的 DA V2 Small 深度估算器（P3：支持实时帧流）。
 *
 * 模型：litert-community/depth-anything-v2-small
 * - INT8(dynamic wi8, afp32)：27.7MB，移动端首选
 * - FP32：99.5MB，精度基准
 *
 * 注意：GPU 路径使用 CompiledModel API（LiteRT V2 中 Interpreter + GpuDelegate 已不可用）。
 *
 * P3 变更：
 * - 输入/输出 TensorBuffer 复用（原实现每次推理都 create/close，实时帧流下会造成明显 GC 压力）；
 * - 新增 inferNchw()，直接吃 glReadPixels 转换出的 NCHW 数组，绕过 Bitmap。
 */
class DepthEstimator(private val context: Context) {

    enum class ModelType(val assetName: String, val label: String) {
        INT8("models/depth_anything_v2_small_wi8_afp32.tflite", "INT8 (27.7MB)"),
        FP32("models/depth_anything_v2_small.tflite", "FP32 (99.5MB)"),
    }

    data class DepthResult(
        val depth: FloatArray,     // [MODEL_H * MODEL_W]，相对深度，越大越近（需 min-max 归一化后使用）
        val width: Int,
        val height: Int,
    ) {
        override fun equals(other: Any?): Boolean =
            other is DepthResult && width == other.width && height == other.height &&
                depth.contentEquals(other.depth)

        override fun hashCode(): Int = (width * 31 + height) * 31 + depth.contentHashCode()
    }

    private var model: CompiledModel? = null
    private var loadedKey: String? = null
    // 注意：LiteRT V2 的 createInputBuffers()/createOutputBuffers() 返回的是 List 而非 Array
    private var inputBuffers: List<TensorBuffer>? = null
    private var outputBuffers: List<TensorBuffer>? = null

    val isLoaded: Boolean get() = model != null

    /** 加载（或复用）模型；参数变化时重载。返回是否加载成功。 */
    fun ensureLoaded(modelType: ModelType, useGpu: Boolean, useFp16: Boolean = false): Boolean {
        val key = "${modelType.name}:$useGpu:$useFp16"
        if (key == loadedKey && model != null) return true
        close()
        return try {
            val accelerator = if (useGpu) Accelerator.GPU else Accelerator.CPU
            val options = CompiledModel.Options(accelerator).apply {
                if (useGpu) {
                    // FP32 是保险默认（DA 解码器输出动态范围大）；FP16 在多数 GPU 上
                    // 快 1.3~2 倍，精度略降，由用户在 UI 上显式选择（P3.5）
                    gpuOptions = CompiledModel.GpuOptions(
                        precision = if (useFp16) {
                            CompiledModel.GpuOptions.Precision.FP16
                        } else {
                            CompiledModel.GpuOptions.Precision.FP32
                        },
                    )
                } else {
                    // CPU：显式指定 XNNPACK 推理线程数。缺省值偏保守（真机实测
                    // ~973ms/帧，深度只有 ~1FPS）；手机 SoC 上 4 线程是常见甜点，
                    // 小核机型至少保 2 线程。仅 CPU 路径有效。
                    cpuOptions = CompiledModel.CpuOptions(
                        numThreads = Runtime.getRuntime().availableProcessors().coerceIn(2, 4),
                    )
                }
            }
            model = CompiledModel.create(context.assets, modelType.assetName, options, null)
            loadedKey = key
            true
        } catch (e: Exception) {
            model = null
            loadedKey = null
            false
        }
    }

    /** 单帧推理：bitmap 任意尺寸，内部拉伸到模型输入。 */
    fun infer(bitmap: Bitmap): DepthResult {
        val input = ImagePreprocessor.resize(bitmap)
        return try {
            inferNchw(ImagePreprocessor.toNchwFloat(input))
        } finally {
            if (input != bitmap) input.recycle()
        }
    }

    /**
     * 单帧推理：直接吃 NCHW float（长度 3×518×686）。实时流水线主入口。
     * 非线程安全 —— 调用方必须固定在同一线程（P3 用单线程 executor 保证）。
     */
    fun inferNchw(nchw: FloatArray): DepthResult {
        val compiled = model ?: error("model not loaded")
        val inputs = inputBuffers ?: compiled.createInputBuffers().also { inputBuffers = it }
        val outputs = outputBuffers ?: compiled.createOutputBuffers().also { outputBuffers = it }
        inputs[0].writeFloat(nchw)
        compiled.run(inputs, outputs)
        val depth = outputs[0].readFloat()
        return DepthResult(
            depth = depth,
            width = ImagePreprocessor.MODEL_W,
            height = ImagePreprocessor.MODEL_H,
        )
    }

    fun close() {
        inputBuffers?.forEach { runCatching { it.close() } }
        inputBuffers = null
        outputBuffers?.forEach { runCatching { it.close() } }
        outputBuffers = null
        model?.close()
        model = null
        loadedKey = null
    }
}
