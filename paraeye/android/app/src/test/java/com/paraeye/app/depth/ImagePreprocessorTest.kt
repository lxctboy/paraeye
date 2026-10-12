package com.paraeye.app.depth

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** ImagePreprocessor 的纯函数部分（不依赖 Android 框架，可在 JVM 直接跑） */
class ImagePreprocessorTest {

    @Test
    fun modelInputContract() {
        // 契约：P4 自导出模型的固定输入 [1, 3, 518, 280] NCHW（竖屏视频友好）
        assertEquals(280, ImagePreprocessor.MODEL_W)
        assertEquals(518, ImagePreprocessor.MODEL_H)
        // 张量元素数 = 3 通道 × H × W（用算式而非写死的魔术数字）
        assertEquals(3 * ImagePreprocessor.MODEL_H * ImagePreprocessor.MODEL_W, 435_120)
    }

    @Test
    fun normalizationMatchesImageNet() {
        // (1 - 0.485) / 0.229 ≈ 2.2489
        assertEquals(2.2489f, ImagePreprocessor.normalize(255, 0), 1e-3f)
        // (0 - 0.485) / 0.229 ≈ -2.1179
        assertEquals(-2.1179f, ImagePreprocessor.normalize(0, 0), 1e-3f)
        // 通道1: (1 - 0.456) / 0.224 ≈ 2.4286
        assertEquals(2.4286f, ImagePreprocessor.normalize(255, 1), 1e-3f)
        // 通道2: (1 - 0.406) / 0.225 ≈ 2.6400
        assertEquals(2.6400f, ImagePreprocessor.normalize(255, 2), 1e-3f)
    }

    @Test
    fun midGrayIsNearZero() {
        // 中灰 128 略高于 ImageNet 均值（mean[0]×255 ≈ 123.7），故结果略为正：
        // (128/255 - 0.485) / 0.229 ≈ +0.074
        val g = ImagePreprocessor.normalize(128, 0)
        assertEquals(0.0741f, g, 1e-3f)
        assertTrue("中灰应落在零附近 ±0.1 内，实际 $g", g in -0.1f..0.1f)
    }

    @Test
    fun nchwPlaneLayoutIsChannelFirst() {
        // 4 个像素：纯红 / 纯绿 / 纯蓝 / 白
        val pixels = intArrayOf(0xFFFF0000.toInt(), 0xFF00FF00.toInt(), 0xFF0000FF.toInt(), 0xFFFFFFFF.toInt())
        val out = ImagePreprocessor.argbToNchw(pixels)

        assertEquals(3 * 4, out.size)
        val plane = 4
        val n = ImagePreprocessor::normalize

        // plane0 = R：①红 ✓ ②绿 ✗ ③蓝 ✗ ④白 ✓
        assertEquals(n(255, 0), out[0], 1e-5f)
        assertEquals(n(0, 0), out[1], 1e-5f)
        assertEquals(n(0, 0), out[2], 1e-5f)
        assertEquals(n(255, 0), out[3], 1e-5f)

        // plane1 = G：①红 ✗ ②绿 ✓ ③蓝 ✗ ④白 ✓
        assertEquals(n(0, 1), out[plane + 0], 1e-5f)
        assertEquals(n(255, 1), out[plane + 1], 1e-5f)
        assertEquals(n(0, 1), out[plane + 2], 1e-5f)
        assertEquals(n(255, 1), out[plane + 3], 1e-5f)

        // plane2 = B：①红 ✗ ②绿 ✗ ③蓝 ✓ ④白 ✓
        assertEquals(n(0, 2), out[2 * plane + 0], 1e-5f)
        assertEquals(n(0, 2), out[2 * plane + 1], 1e-5f)
        assertEquals(n(255, 2), out[2 * plane + 2], 1e-5f)
        assertEquals(n(255, 2), out[2 * plane + 3], 1e-5f)

        // 白色像素在三个平面上的归一化值应当各不相同（ImageNet 均值/方差逐通道不同）
        assertTrue(n(255, 0) != n(255, 1))
        assertTrue(n(255, 1) != n(255, 2))
    }

    @Test
    fun rgbaToNchwSkipsAlphaAndKeepsChannelOrder() {
        // P3 视频路径：glReadPixels 出来的 RGBA8 字节流（alpha 通道必须被忽略）
        // 2 个像素：纯红 / 纯绿，alpha 分别为 0xFF 与 0x00（不应影响结果）
        val rgba = byteArrayOf(
            255.toByte(), 0, 0, 255.toByte(),
            0, 255.toByte(), 0, 0,
        )
        val out = ImagePreprocessor.rgbaToNchw(rgba, 2, 1)
        val n = ImagePreprocessor::normalize

        assertEquals(3 * 2, out.size)
        // plane0 = R
        assertEquals(n(255, 0), out[0], 1e-5f)
        assertEquals(n(0, 0), out[1], 1e-5f)
        // plane1 = G
        assertEquals(n(0, 1), out[2], 1e-5f)
        assertEquals(n(255, 1), out[3], 1e-5f)
        // plane2 = B：两个像素的 B 都是 0
        assertEquals(n(0, 2), out[4], 1e-5f)
        assertEquals(n(0, 2), out[5], 1e-5f)
    }

    @Test
    fun rgbaToNchwMatchesArgbPath() {
        // 同一组颜色走两条路径（Bitmap ARGB / GL readPixels RGBA）必须完全一致，
        // 否则视频流水线的深度会和 P2 静态图路径对不上。
        val w = 4
        val h = 2
        val n = w * h
        val rgba = ByteArray(n * 4)
        val argb = IntArray(n)
        for (i in 0 until n) {
            val r = (i * 37) % 256
            val g = (i * 91) % 256
            val b = (i * 13) % 256
            rgba[i * 4] = r.toByte()
            rgba[i * 4 + 1] = g.toByte()
            rgba[i * 4 + 2] = b.toByte()
            rgba[i * 4 + 3] = 255.toByte()
            argb[i] = (255 shl 24) or (r shl 16) or (g shl 8) or b
        }
        val fromRgba = ImagePreprocessor.rgbaToNchw(rgba, w, h)
        val fromArgb = ImagePreprocessor.argbToNchw(argb)
        assertEquals(fromArgb.size, fromRgba.size)
        for (i in fromRgba.indices) {
            assertEquals("索引 $i 处两条路径不一致", fromArgb[i], fromRgba[i], 1e-6f)
        }
    }

    @Test
    fun rgbaToNchwRejectsShortBuffer() {
        var threw = false
        try {
            ImagePreprocessor.rgbaToNchw(ByteArray(10), 4, 4)
        } catch (e: IllegalArgumentException) {
            threw = true
        }
        assertTrue("缓冲区过短时应抛 IllegalArgumentException", threw)
    }
}
