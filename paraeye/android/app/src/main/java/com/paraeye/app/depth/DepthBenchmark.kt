package com.paraeye.app.depth

import android.graphics.Bitmap

/** 单帧推理基准：预热 + N 次迭代，输出耗时统计（P1 用于 V2 Small vs DA3 Small 对比） */
object DepthBenchmark {

    data class Result(
        val iterations: Int,
        val avgMs: Double,
        val minMs: Double,
        val maxMs: Double,
        val p50Ms: Double,
    ) {
        val fps: Double get() = 1000.0 / avgMs
    }

    fun run(
        estimator: DepthEstimator,
        bitmap: Bitmap,
        warmup: Int = 3,
        iterations: Int = 20,
    ): Result {
        repeat(warmup) { estimator.infer(bitmap) }
        val times = DoubleArray(iterations)
        repeat(iterations) { i ->
            val t0 = System.nanoTime()
            estimator.infer(bitmap)
            times[i] = (System.nanoTime() - t0) / 1_000_000.0
        }
        times.sort()
        return Result(
            iterations = iterations,
            avgMs = times.average(),
            minMs = times.first(),
            maxMs = times.last(),
            p50Ms = times[iterations / 2],
        )
    }
}
