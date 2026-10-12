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

    /**
     * 纯函数：glReadPixels 得到的 RGBA8 字节流 -> NCHW float。
     * 用于 P3 视频流水线：GPU 已把帧降采样到 686x518 并写入 FBO，
     * 这里跳过 Bitmap，直接从像素缓冲构造模型输入（省掉一次 IntArray 转换与 setPixels）。
     *
     * 前置条件：rgba 的行 0 必须是**图像顶部**（GL readPixels 默认行 0 是 FBO 底部，
     * 由 SbsRenderer 的降采样 pass 用 aPos.y*0.5+0.5 翻转修正）。
     */
    fun rgbaToNchw(rgba: ByteArray, width: Int, height: Int): FloatArray {
        val n = width * height
        require(rgba.size >= n * 4) { "rgba 缓冲区过小：${rgba.size} < ${n * 4}" }
        val out = FloatArray(3 * n)
        var p = 0
        for (i in 0 until n) {
            val r = rgba[p].toInt() and 0xFF
            val g = rgba[p + 1].toInt() and 0xFF
            val b = rgba[p + 2].toInt() and 0xFF
            p += 4
            out[i] = normalize(r, 0)
            out[n + i] = normalize(g, 1)
            out[2 * n + i] = normalize(b, 2)
        }
        return out
    }

    /** 纯函数：单通道归一化（JVM 单测直接覆盖） */
    fun normalize(value8bit: Int, channel: Int): Float =
        (value8bit / 255f - MEAN[channel]) / STD[channel]
}
