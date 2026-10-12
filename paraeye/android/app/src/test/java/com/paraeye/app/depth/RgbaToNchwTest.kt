package com.paraeye.app.depth

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * rgbaToNchw 的 LUT 零分配版本必须与纯计算版本逐元素一致
 * （查表只是 normalize 的预计算形式，任何偏差都会让深度图整体偏移）。
 */
class RgbaToNchwTest {

    @Test
    fun lutVersionMatchesPureVersion() {
        val w = 8
        val h = 4
        val rgba = ByteArray(w * h * 4)
        // 伪随机但确定的填充：覆盖 0/255 边界与中间值
        var s = 12345
        for (i in rgba.indices) {
            s = (s * 1103515245 + 12345) and 0x7fffffff
            rgba[i] = (s shr 16 and 0xFF).toByte()
        }
        val pure = ImagePreprocessor.rgbaToNchw(rgba, w, h)
        val into = FloatArray(3 * w * h)
        ImagePreprocessor.rgbaToNchwInto(rgba, w, h, into)
        assertArrayEquals(pure, into, 1e-6f)
    }

    @Test
    fun planeOrderIsRgb() {
        // 单像素 R=255,G=0,B=0：三个平面必须分别是 R/G/B 通道的归一化值
        // （平面顺序写错会让深度图整体错乱，故用精确值断言而非阈值）
        val rgba = byteArrayOf(255.toByte(), 0, 0, 255.toByte())
        val out = FloatArray(3)
        ImagePreprocessor.rgbaToNchwInto(rgba, 1, 1, out)
        assertEquals(2.2489f, out[0], 1e-3f)    // n(255, 0)
        assertEquals(-2.0357f, out[1], 1e-3f)   // n(0, 1)
        assertEquals(-1.8044f, out[2], 1e-3f)   // n(0, 2)
    }
}
