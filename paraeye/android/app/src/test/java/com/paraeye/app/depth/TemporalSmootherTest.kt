package com.paraeye.app.depth

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
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

    // ---- advance()：帧率无关连续 EMA（P3.5 渲染线程逐帧过渡） ----

    @Test
    fun advanceFirstCallSnapsToTarget() {
        val s = TemporalSmoother(alpha = 0.7f)
        val out = s.advance(floatArrayOf(0.3f, 0.9f), 16f)
        assertEquals(0.3f, out[0], 1e-6f)
        assertEquals(0.9f, out[1], 1e-6f)
    }

    @Test
    fun advanceZeroDtIsNoop() {
        val s = TemporalSmoother(alpha = 0.7f)
        s.advance(floatArrayOf(0f), 16f)
        val out = s.advance(floatArrayOf(1f), 0f)
        assertEquals(0f, out[0], 1e-6f)
    }

    @Test
    fun advanceStepMatchesDiscreteEmaAtRefInterval() {
        // 核心口径验证：dt = 166.7ms（P0 的 6FPS 深度间隔）一步的系数
        // 必须恰为 alpha = 0.7，与 apply() 严格一致（τ 的定义使然）
        val s = TemporalSmoother(alpha = 0.7f)
        s.advance(floatArrayOf(0f), 16f)
        val out = s.advance(floatArrayOf(1f), 1000f / 6f)
        assertEquals(0.7f, out[0], 1e-3f)
    }

    @Test
    fun advanceConvergesToTarget() {
        val s = TemporalSmoother(alpha = 0.7f)
        s.advance(floatArrayOf(0f), 16f)
        var out = FloatArray(1)
        // 30FPS × 2 秒 = 60 步，应收敛到 1
        repeat(60) { out = s.advance(floatArrayOf(1f), 33f) }
        assertEquals(1f, out[0], 1e-3f)
    }

    @Test
    fun advanceTracksMovingTargetWithoutOvershoot() {
        val s = TemporalSmoother(alpha = 0.7f)
        s.advance(floatArrayOf(0f), 16f)
        var prev = 0f
        repeat(30) {
            val target = it / 30f
            val out = s.advance(floatArrayOf(target), 33f)[0]
            assertTrue("不得超出目标（无过冲）: $out > $target", out <= target + 1e-6f)
            assertTrue("应持续靠近目标: $out < $prev + 1e-6f || $out >= $prev", out >= prev - 1e-6f)
            prev = out
        }
    }

    @Test
    fun advanceResetClearsCur() {
        val s = TemporalSmoother(alpha = 0.7f)
        s.advance(floatArrayOf(0.2f), 33f)
        s.reset()
        // reset 后首帧直通目标，不残留上一场景
        val out = s.advance(floatArrayOf(0.8f), 33f)
        assertEquals(0.8f, out[0], 1e-6f)
    }
}
