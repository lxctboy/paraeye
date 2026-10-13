package com.paraeye.app.video

import android.content.Context
import android.graphics.SurfaceTexture
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface

/**
 * 本地视频解码器：MediaExtractor + MediaCodec（硬件解码）→ SurfaceTexture。
 *
 * 设计要点：
 * - 输出走 Surface（OES 纹理），全程不经过 CPU 拷贝，这是移动端能把解码压到 5~10ms 的前提；
 * - 解码循环跑在独立 HandlerThread 上，与 GL 线程解耦；SurfaceTexture 由 GL 线程持有；
 * - 按 presentationTimeUs 做节奏控制（相对首帧 PTS 对齐墙钟），避免"一口气解完"；
 * - 到末尾自动 seek 回 0 循环播放；pause 时不 release 输出缓冲，画面停在最后一帧；
 * - 对外暴露 positionUs 与 seekToMs()，供播放器进度条使用（GOP 内对齐，误差一个 GOP）。
 */
class VideoDecoder(
    private val context: Context,
    private val uri: Uri,
    private val surfaceTexture: SurfaceTexture,
    private val onFrame: () -> Unit,
    private val onError: (String) -> Unit,
) {

    @Volatile var running = false
        private set
    @Volatile var paused = false

    @Volatile var videoWidth = 0
        private set
    @Volatile var videoHeight = 0
        private set
    @Volatile var durationUs = 0L
        private set

    /** 当前播放位置（微秒，播放器进度条用；seek 后为近似值，误差一个 GOP） */
    @Volatile var positionUs = 0L
        private set

    /** UI 线程写入的 seek 请求（微秒，-1 = 无）；解码循环每轮消费 */
    @Volatile private var pendingSeekUs = -1L

    private var thread: HandlerThread? = null
    private var extractor: MediaExtractor? = null
    private var codec: MediaCodec? = null

    fun start() {
        if (running) return
        running = true
        thread = HandlerThread("paraeye-decode").apply { start() }
        Handler(thread!!.looper).post { decodeLoop() }
    }

    /**
     * 跳转到指定位置（毫秒）。线程安全：仅置位请求，实际 flush/seek 由解码线程执行。
     * 落在最近的关键帧之前（GOP 内对齐），恢复后按新位置重锚墙钟节奏。
     */
    fun seekToMs(ms: Long) {
        pendingSeekUs = ms.coerceAtLeast(0L) * 1000L
    }

    fun release() {
        running = false
        try {
            thread?.quitSafely()
            thread?.join(1000)
        } catch (ignored: Exception) {
        }
        thread = null
        try {
            codec?.stop()
        } catch (ignored: Exception) {
        }
        try {
            codec?.release()
        } catch (ignored: Exception) {
        }
        codec = null
        try {
            extractor?.release()
        } catch (ignored: Exception) {
        }
        extractor = null
    }

    private fun decodeLoop() {
        val ex = MediaExtractor()
        try {
            ex.setDataSource(context, uri, null)
        } catch (e: Exception) {
            onError("无法打开视频：${e.message}")
            running = false
            return
        }
        extractor = ex

        var trackIndex = -1
        var mime = ""
        for (i in 0 until ex.trackCount) {
            val format = ex.getTrackFormat(i)
            val m = format.getString(MediaFormat.KEY_MIME) ?: continue
            if (m.startsWith("video/")) {
                trackIndex = i
                mime = m
                break
            }
        }
        if (trackIndex < 0) {
            onError("视频中没有找到视频轨")
            running = false
            return
        }
        ex.selectTrack(trackIndex)
        val format = ex.getTrackFormat(trackIndex)
        // 旋转元数据（KEY_ROTATION）：MediaCodec 按**编码尺寸**出帧，旋转由
        // SurfaceTexture 的 transform matrix 施加（OES pass 已应用 uTexMtx）。
        // 早年忽略它导致：竖拍视频（很多机型/App 存成 1920×1080 + rot90）被当成
        // 横屏——内容拉伸进横屏纹理、喂深度模型的帧二次畸变（深度废掉）、SBS 整对
        // 几何也全错（真机反馈「1920×1080 / 1080×1920 看不到 3D」的根因之一）。
        // 对外暴露**显示尺寸**；setDefaultBufferSize 仍用编码尺寸（解码器真实输出）。
        val rotation = if (format.containsKey(MediaFormat.KEY_ROTATION)) {
            format.getInteger(MediaFormat.KEY_ROTATION)
        } else {
            0
        }
        val codedW = format.getInteger(MediaFormat.KEY_WIDTH)
        val codedH = format.getInteger(MediaFormat.KEY_HEIGHT)
        videoWidth = if (rotation == 90 || rotation == 270) codedH else codedW
        videoHeight = if (rotation == 90 || rotation == 270) codedW else codedH
        if (format.containsKey(MediaFormat.KEY_DURATION)) {
            durationUs = format.getLong(MediaFormat.KEY_DURATION)
        }
        // 让 SurfaceTexture 的缓冲尺寸与视频一致，避免驱动额外缩放（编码尺寸）
        surfaceTexture.setDefaultBufferSize(codedW, codedH)

        val decoder = try {
            MediaCodec.createDecoderByType(mime)
        } catch (e: Exception) {
            onError("创建解码器失败：${e.message}")
            running = false
            return
        }
        codec = decoder
        try {
            decoder.configure(format, Surface(surfaceTexture), null, 0)
            decoder.start()
        } catch (e: Exception) {
            onError("解码器配置失败：${e.message}")
            running = false
            return
        }

        val info = MediaCodec.BufferInfo()
        var inputDone = false
        var startPts = -1L
        var startNano = 0L
        // 位置基准：loop 回 0 或 seek 后重锚，positionUs = (pts - startPts) + baseUs
        var baseUs = 0L

        while (running) {
            if (paused) {
                Thread.sleep(20)
                continue
            }

            // UI 发来的 seek：先刷掉解码器内部状态，再回到目标位置前最近的关键帧
            val seekUs = pendingSeekUs
            if (seekUs >= 0) {
                pendingSeekUs = -1
                decoder.flush()
                inputDone = false
                ex.seekTo(seekUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
                startPts = -1L
                baseUs = seekUs
                continue
            }

            if (!inputDone) {
                val inIndex = decoder.dequeueInputBuffer(10_000)
                if (inIndex >= 0) {
                    val buffer = decoder.getInputBuffer(inIndex) ?: error("input buffer null")
                    val size = ex.readSampleData(buffer, 0)
                    if (size < 0) {
                        decoder.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        inputDone = true
                    } else {
                        decoder.queueInputBuffer(inIndex, 0, size, ex.sampleTime, ex.sampleFlags)
                        ex.advance()
                    }
                }
            }

            val outIndex = decoder.dequeueOutputBuffer(info, 10_000)
            if (outIndex == MediaCodec.INFO_TRY_AGAIN_LATER ||
                outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED ||
                outIndex == MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED
            ) {
                continue
            }
            if (outIndex < 0) continue

            if (info.size > 0) {
                val pts = info.presentationTimeUs
                if (startPts < 0L) {
                    startPts = pts
                    startNano = System.nanoTime()
                }
                positionUs = (pts - startPts + baseUs).coerceIn(0L, maxOf(durationUs, pts))
                val targetMs = (pts - startPts) / 1000L
                val elapsedMs = (System.nanoTime() - startNano) / 1_000_000L
                val delay = targetMs - elapsedMs
                if (delay in 1L..500L) Thread.sleep(delay)
            }

            val render = info.size > 0 && !paused
            decoder.releaseOutputBuffer(outIndex, render)
            if (render) onFrame()

            if ((info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                // 循环播放：flush 解码器 + 回到首个关键帧
                decoder.flush()
                ex.seekTo(0L, MediaExtractor.SEEK_TO_CLOSEST_SYNC)
                inputDone = false
                startPts = -1L
                baseUs = 0L
            }
        }
    }
}
