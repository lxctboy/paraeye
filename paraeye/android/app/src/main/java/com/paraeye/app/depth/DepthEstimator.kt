package com.paraeye.app.depth

import android.content.Context
import android.graphics.Bitmap
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel

/**
 * LiteRT 2.3.0 V2 CompiledModel 封装的 DA V2 Small 深度估算器（P1 单帧版）。
 *
 * 模型：litert-community/depth-anything-v2-small
 * - INT8(dynamic wi8, afp32)：27.7MB，移动端首选
 * - FP32：99.5MB，精度基准
 *
 * 注意：GPU 路径使用 CompiledModel API（LiteRT V2 中 Interpreter + GpuDelegate 已不可用）。
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

    val isLoaded: Boolean get() = model != null

    /** 加载（或复用）模型；参数变化时重载。返回是否加载成功。 */
    fun ensureLoaded(modelType: ModelType, useGpu: Boolean): Boolean {
        val key = "${modelType.name}:$useGpu"
        if (key == loadedKey && model != null) return true
        close()
        return try {
            val accelerator = if (useGpu) Accelerator.GPU else Accelerator.CPU
            val options = CompiledModel.Options(accelerator).apply {
                if (useGpu) {
                    // DA 解码器输出动态范围大，GPU 侧保持 FP32 精度
                    gpuOptions = CompiledModel.GpuOptions(
                        precision = CompiledModel.GpuOptions.Precision.FP32,
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
        val compiled = model ?: error("model not loaded")
        val input = ImagePreprocessor.resize(bitmap)
        val inputBuffers = compiled.createInputBuffers()
        val outputBuffers = compiled.createOutputBuffers()
        try {
            inputBuffers[0].writeFloat(ImagePreprocessor.toNchwFloat(input))
            compiled.run(inputBuffers, outputBuffers)
            val depth = outputBuffers[0].readFloat()
            return DepthResult(
                depth = depth,
                width = ImagePreprocessor.MODEL_W,
                height = ImagePreprocessor.MODEL_H,
            )
        } finally {
            inputBuffers.forEach { it.close() }
            outputBuffers.forEach { it.close() }
            if (input != bitmap) input.recycle()
        }
    }

    fun close() {
        model?.close()
        model = null
        loadedKey = null
    }
}
