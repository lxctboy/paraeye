package com.paraeye.app.depth

import android.graphics.Bitmap

/**
 * DA V2 Small LiteRT 预处理（契约来自 litert-community/depth-anything-v2-small README）：
 * - 输入固定 [1, 3, 518, 686]，NCHW，RGB
 * - resize 到 686x518（直接拉伸，不保持宽高比，与官方导出一致）
 * - 像素 /255 后按 ImageNet mean/std 归一化
 */
object ImagePreprocessor {

    const val MODEL_W = 686
    const val MODEL_H = 518
    val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    /** 拉伸缩放到模型输入尺寸 */
    fun resize(bitmap: Bitmap): Bitmap =
        Bitmap.createScaledBitmap(bitmap, MODEL_W, MODEL_H, true)

    /** Bitmap(ARGB_8888, 686x518) -> NCHW float 数组 */
    fun toNchwFloat(resized: Bitmap): FloatArray {
        require(resized.width == MODEL_W && resized.height == MODEL_H)
        val pixels = IntArray(MODEL_W * MODEL_H)
        resized.getPixels(pixels, 0, MODEL_W, 0, 0, MODEL_W, MODEL_H)
        return argbToNchw(pixels)
    }

    /**
     * 纯函数：ARGB 像素数组 -> NCHW float（长度 3×H×W）。
     * 平面布局：plane0=R, plane1=G, plane2=B；每平面按行优先逐像素排列。
     * 抽出来是为了能在 JVM 单测里脱离 Bitmap 验证通道顺序。
     */
    fun argbToNchw(pixels: IntArray): FloatArray {
        val out = FloatArray(3 * pixels.size)
        val plane = pixels.size
        for (i in pixels.indices) {
            val p = pixels[i]
            out[i] = normalize((p shr 16) and 0xFF, 0)
            out[plane + i] = normalize((p shr 8) and 0xFF, 1)
            out[2 * plane + i] = normalize(p and 0xFF, 2)
        }
        return out
    }

    /** 纯函数：单通道归一化（JVM 单测直接覆盖） */
    fun normalize(value8bit: Int, channel: Int): Float =
        (value8bit / 255f - MEAN[channel]) / STD[channel]
}
