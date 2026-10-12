package com.paraeye.app.depth

import org.junit.Assert.assertEquals
import org.junit.Test

/** EMA 时域平滑：语义必须与 P0 postprocess.py::TemporalSmoother 一致 */
class TemporalSmootherTest {

    @Test
    fun firstFramePassesThrough() {
        val s = TemporalSmoother(alpha = 0.7f)
        val out = s.apply(floatArrayOf(0.4f, 0.8f))
        assertEquals(0.4f, out[0], 1e-6f)
        assertEquals(0.8f, out[1], 1e-6f)
    }

    @Test
    fun emaFollowsP0Formula() {
        // D_t = 0.7 * cur + 0.3 * prev_smoothed（P0: alpha * depth + (1-alpha) * prev）
        val s = TemporalSmoother(alpha = 0.7f)
        s.apply(floatArrayOf(0f))
        assertEquals(0.7f, s.apply(floatArrayOf(1f))[0], 1e-5f)
        assertEquals(0.91f, s.apply(floatArrayOf(1f))[0], 1e-5f)
        // 输入回到 0：0.7*0 + 0.3*0.91 = 0.273
        assertEquals(0.273f, s.apply(floatArrayOf(0f))[0], 1e-5f)
    }

    @Test
    fun resetDropsHistory() {
        val s = TemporalSmoother(alpha = 0.7f)
        s.apply(floatArrayOf(0f))
        s.reset()
        // reset 后首帧直通，不应残留上一场景的深度
        assertEquals(1f, s.apply(floatArrayOf(1f))[0], 1e-6f)
    }

    @Test
    fun sizeChangeRestarts() {
        val s = TemporalSmoother(alpha = 0.7f)
        s.apply(FloatArray(4) { 0f })
        // 长度变化时历史失效，直通返回
        val out = s.apply(FloatArray(6) { 1f })
        assertEquals(1f, out[5], 1e-6f)
    }
}
