package com.paraeye.app.ui

/**
 * 播放器时间格式化（纯函数，便于单测）。
 *
 * 规则：
 * - < 1 小时：`mm:ss`（分钟补零到两位，如 01:05）
 * - ≥ 1 小时：`h:mm:ss`（小时不补零）
 * - 负值按 0 处理；四舍五入到秒
 */
object TimeFormat {

    fun clock(ms: Long): String {
        val totalSec = (ms.coerceAtLeast(0L) + 500L) / 1000L
        val sec = totalSec % 60
        val min = (totalSec / 60) % 60
        val hour = totalSec / 3600
        return if (hour > 0) {
            "%d:%02d:%02d".format(hour, min, sec)
        } else {
            "%02d:%02d".format(min, sec)
        }
    }

    /** 进度条两端的「当前 / 总长」文本 */
    fun position(positionMs: Long, durationMs: Long): String =
        "${clock(positionMs)} / ${clock(durationMs)}"
}
