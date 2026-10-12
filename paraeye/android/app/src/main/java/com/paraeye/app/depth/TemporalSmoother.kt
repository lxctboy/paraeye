package com.paraeye.app.depth

/**
 * EMA 时域平滑，语义与 P0 `postprocess.py::TemporalSmoother` 完全一致：
 *
 *   D_t_smooth = alpha * D_t + (1 - alpha) * D_{t-1}_smooth
 *
 * alpha 越大响应越快（越接近不平滑）。P0 定档 alpha = 0.7。
 * 作用于**归一化后**的深度：视频帧间深度尺度会漂移，直接平滑原始深度会引入呼吸感。
 */
class TemporalSmoother(private val alpha: Float = 0.7f) {

    private var prev: FloatArray? = null

    /** 切换视频 / 换模型 / 长间隔后必须调用，否则会把无关场景的深度混进来 */
    fun reset() {
        prev = null
    }

    /** 就地改写并返回 input（调用方需保证 input 可写且之后不再复用旧值） */
    fun apply(current: FloatArray): FloatArray {
        val p = prev
        if (p == null || p.size != current.size) {
            prev = current.copyOf()
            return current
        }
        val a = alpha
        for (i in current.indices) {
            current[i] = a * current[i] + (1f - a) * p[i]
        }
        // 调用方传入的是 normalize() 生成的新数组，故 p 与 current 不可能是同一对象
        prev = current
        return current
    }
}
