package com.paraeye.app.video

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * A/V 同步判据单测。这段阈值逻辑是「声画不分离」的唯一正确性判据，
 * 必须在 CI 里锁住，不能只靠真机听感验证。
 */
class AvSyncTest {

    @Test
    fun driftIsNullWhenEitherPositionUnknown() {
        assertNull(AvSync.driftMs(0L, 5_000_000L))
        assertNull(AvSync.driftMs(5_000_000L, 0L))
        assertNull(AvSync.driftMs(-1L, 5_000_000L))
    }

    @Test
    fun positiveDriftMeansAudioAheadOfVideo() {
        // 音频 10.5s，视频 10.0s -> 音频超前 500ms
        assertEquals(500L, AvSync.driftMs(10_500_000L, 10_000_000L)!!)
        // 音频落后 500ms
        assertEquals(-500L, AvSync.driftMs(10_000_000L, 10_500_000L)!!)
    }

    @Test
    fun smallDriftWithinLimitDoesNotResync() {
        assertFalse(AvSync.needsResync(10_200_000L, 10_000_000L, 10_000L))
        assertFalse(AvSync.needsResync(10_000_000L, 10_200_000L, 10_000L))
    }

    @Test
    fun driftBeyondLimitTriggersResyncInBothDirections() {
        assertTrue(AvSync.needsResync(10_500_000L, 10_000_000L, 10_000L))
        assertTrue(AvSync.needsResync(10_000_000L, 10_500_000L, 10_000L))
    }

    @Test
    fun cooldownSuppressesRepeatedResync() {
        // 漂移很大，但距上次检查不足冷却期 -> 不动，避免咔哒声
        assertFalse(AvSync.needsResync(20_000_000L, 10_000_000L, AvSync.DRIFT_COOLDOWN_MS - 1))
        assertTrue(AvSync.needsResync(20_000_000L, 10_000_000L, AvSync.DRIFT_COOLDOWN_MS))
    }

    @Test
    fun resyncRequiresBothPositionsKnown() {
        assertFalse(AvSync.needsResync(0L, 10_000_000L, 10_000L))
        assertFalse(AvSync.needsResync(10_000_000L, 0L, 10_000L))
    }

    @Test
    fun seekSkipWindowBacksOff30msToDropStaleSamples() {
        assertEquals(970_000L, AvSync.skipUntilUsForSeek(1_000_000L))
        // seek 到 0 时容差为负，表示不丢弃任何样本
        assertEquals(-30_000L, AvSync.skipUntilUsForSeek(0L))
    }

    @Test
    fun framePositionConversionUsesBaseOffsetAfterSeek() {
        // 44100Hz，44100 帧 = 1 秒
        assertEquals(1_000_000L, AvSync.positionUsFromFrames(44_100L, 44_100, 0L))
        // seek 到 5s 后播放头归零，已播 1s -> 绝对位置 6s
        assertEquals(6_000_000L, AvSync.positionUsFromFrames(44_100L, 44_100, 5_000_000L))
        // 采样率非法时退化为基准位置，不产生跳变
        assertEquals(5_000_000L, AvSync.positionUsFromFrames(44_100L, 0, 5_000_000L))
    }
}