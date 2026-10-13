package com.paraeye.app.video

/**
 * A/V 漂移判定（纯函数，供单测覆盖）。
 *
 * 抽出来是因为这段阈值逻辑是「音画同步」的唯一正确性判据，不该埋在
 * MediaCodec 循环里靠真机听感验证。
 */
object AvSync {

    /** 超过该差值（毫秒）才认为音画分离，需要硬同步 */
    const val DRIFT_LIMIT_MS = 400L

    /** 两次硬同步的最小间隔（毫秒），避免反复重锚产生咔哒声 */
    const val DRIFT_COOLDOWN_MS = 8_000L

    /**
     * 音频播放位置与视频位置的差（毫秒）。正值 = 音频超前。
     * 任一参考位置未知（<= 0）时返回 null，表示此刻不做判断。
     */
    fun driftMs(audioPositionUs: Long, videoPositionUs: Long): Long? {
        if (audioPositionUs <= 0L || videoPositionUs <= 0L) return null
        return (audioPositionUs - videoPositionUs) / 1000L
    }

    /** 是否需要硬同步：差值超过阈值才算，且冷却期已过 */
    fun needsResync(
        audioPositionUs: Long,
        videoPositionUs: Long,
        msSinceLastCheck: Long,
    ): Boolean {
        if (msSinceLastCheck < DRIFT_COOLDOWN_MS) return false
        val drift = driftMs(audioPositionUs, videoPositionUs) ?: return false
        return drift > DRIFT_LIMIT_MS || drift < -DRIFT_LIMIT_MS
    }

    /**
     * seek 后需要丢弃的样本截止 pts（微秒）。
     * 提前 30ms 作为容差：解码器落到最近同步点之前，多丢一点避免回放上一段声音。
     */
    fun skipUntilUsForSeek(seekUs: Long): Long = seekUs - 30_000L

    /** 把已播帧数换算成播放位置（微秒），供漂移比较 */
    fun positionUsFromFrames(playedFrames: Long, sampleRate: Int, baseUs: Long): Long {
        if (sampleRate <= 0) return baseUs
        return baseUs + playedFrames * 1_000_000L / sampleRate
    }
}