package com.paraeye.app.depth

import android.graphics.Bitmap
import android.graphics.Color

/** 深度图可视化：min-max 归一化 + HSV 色带（蓝=近，红=远） */
object DepthVisualizer {

    fun normalize(depth: FloatArray): FloatArray {
        var min = Float.MAX_VALUE
        var max = -Float.MAX_VALUE
        for (v in depth) {
            if (v < min) min = v
            if (v > max) max = v
        }
        if (max - min < 1e-6f) return FloatArray(depth.size)
        val out = FloatArray(depth.size)
        val inv = 1f / (max - min)
        for (i in depth.indices) out[i] = (depth[i] - min) * inv
        return out
    }

    fun toBitmap(depth: FloatArray, width: Int, height: Int): Bitmap {
        val norm = normalize(depth)
        val colors = IntArray(norm.size)
        val hsv = FloatArray(3)
        for (i in norm.indices) {
            val v = norm[i].coerceIn(0f, 1f)
            hsv[0] = 240f * (1f - v)   // 近(1)=蓝 240°，远(0)=红 0°
            hsv[1] = 0.85f
            hsv[2] = 1f
            colors[i] = Color.HSVToColor(hsv)
        }
        return Bitmap.createBitmap(colors, width, height, Bitmap.Config.ARGB_8888)
    }
}
