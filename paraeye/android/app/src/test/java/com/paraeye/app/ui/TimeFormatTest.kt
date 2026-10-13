package com.paraeye.app.ui

import org.junit.Assert.assertEquals
import org.junit.Test

/** 播放器时间文本格式 */
class TimeFormatTest {

    @Test
    fun underOneHourUsesMmSs() {
        assertEquals("00:00", TimeFormat.clock(0))
        assertEquals("00:05", TimeFormat.clock(5_000))
        assertEquals("01:05", TimeFormat.clock(65_000))
        assertEquals("12:34", TimeFormat.clock(754_000))
        // 59.6s 四舍五入进位到下一分钟
        assertEquals("01:00", TimeFormat.clock(59_600))
    }

    @Test
    fun oneHourPlusUsesHms() {
        assertEquals("1:00:00", TimeFormat.clock(3_600_000))
        assertEquals("2:03:04", TimeFormat.clock(7_384_000))
    }

    @Test
    fun negativeIsClampedToZero() {
        assertEquals("00:00", TimeFormat.clock(-1))
        assertEquals("00:00", TimeFormat.clock(-999_999))
    }

    @Test
    fun positionJoinsBothEnds() {
        assertEquals("00:00 / 01:00", TimeFormat.position(0, 60_000))
        assertEquals("01:05 / 1:02:03", TimeFormat.position(65_000, 3_723_000))
    }
}
