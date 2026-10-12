package com.paraeye.app.depth

import kotlin.math.exp
import kotlin.math.ln

/**
 * EMA 时域平滑，语义与 P0 `postprocess.py::TemporalSmoother` 完全一致：
 *
 *   D_t_smooth = alpha * D_t + (1 - alpha) * D_{t-1}_smooth
 *
 * alpha 越大响应越快（越接近不平滑）。P0 定档 alpha = 0.7。
 * 作用于**归一化后**的深度：视频帧间深度尺度会漂移，直接平滑原始深度会引入呼吸感。
 *
 * 视频实时路径（P3.5）不用 apply() 而用 advance()：
 * 推理只有 ~6FPS 时，离散 EMA 也只在深度到来时生效，中间渲染帧深度不变，
 * 看起来就是一跳一跳。advance() 把同一个时间常数 τ 展开到每个渲染帧，
 * 视觉上把 6FPS 的深度更新插值成渲染帧率。
 */
class TemporalSmoother(private val alpha: Float = 0.7f) {

    /** 与离散 EMA(alpha @ refFps) 等效的连续时间常数（毫秒） */
    private val tauMs: Float = -(1000f / REF_FPS) / ln((1f - alpha).toDouble()).toFloat()

    private var prev: FloatArray? = null

    /** 渲染线程维护的当前（过渡中）深度 */
    private var cur: FloatArray? = null

    /** 切换视频 / 换模型 / 长间隔后必须调用，否则会把无关场景的深度混进来 */
    fun reset() {
        prev = null
        cur = null
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

    /**
     * 帧率无关的连续 EMA（渲染线程每帧调用）：
     *   cur += (1 - e^(-dt/tau)) * (target - cur)
     *
     * dt = tau 时一步走到 63.2%；dt = 166ms（≈P0 的 6FPS 深度间隔）时
     * 系数恰为 alpha = 0.7，与 apply() 口径严格一致（数学上自洽，有单测）。
     *
     * @return 内部维护的 cur 数组（只读，勿修改）
     */
    fun advance(target: FloatArray, dtMs: Float): FloatArray {
        val c = cur
        if (c == null || c.size != target.size) {
            cur = target.copyOf()
            return cur!!
        }
        val k = if (dtMs <= 0f) 0f else 1f - exp(-dtMs / tauMs)
        for (i in c.indices) {
            c[i] += k * (target[i] - c[i])
        }
        return c
    }

    private companion object {
        /** P0 定档 alpha=0.7 时对应的深度更新间隔参考帧率（6FPS ≈ 166ms） */
        const val REF_FPS = 6f
    }
}
