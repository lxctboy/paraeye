package com.paraeye.app.depth

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** ImagePreprocessor 的纯函数部分（不依赖 Android 框架，可在 JVM 直接跑） */
class ImagePreprocessorTest {

    @Test
    fun modelInputContract() {
        // 契约：litert-community/depth-anything-v2-small 的固定输入 686x518 NCHW
        assertEquals(686, ImagePreprocessor.MODEL_W)
        assertEquals(518, ImagePreprocessor.MODEL_H)
        assertEquals(3 * 518 * 686, 1065684)
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
        // 中间灰
        assertTrue(ImagePreprocessor.normalize(128, 0) < 0f)
        assertTrue(ImagePreprocessor.normalize(128, 0) > -0.1f)
    }
}
