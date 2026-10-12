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
 * - 到末尾自动 seek 回 0 循环播放；pause 时不 release 输出缓冲，画面停在最后一帧。
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

    private var thread: HandlerThread? = null
    private var extractor: MediaExtractor? = null
    private var codec: MediaCodec? = null

    fun start() {
        if (running) return
        running = true
        thread = HandlerThread("paraeye-decode").apply { start() }
        Handler(thread!!.looper).post { decodeLoop() }
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
        videoWidth = format.getInteger(MediaFormat.KEY_WIDTH)
        videoHeight = format.getInteger(MediaFormat.KEY_HEIGHT)
        if (format.containsKey(MediaFormat.KEY_DURATION)) {
            durationUs = format.getLong(MediaFormat.KEY_DURATION)
        }
        // 让 SurfaceTexture 的缓冲尺寸与视频一致，避免驱动额外缩放
        surfaceTexture.setDefaultBufferSize(videoWidth, videoHeight)

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

        while (running) {
            if (paused) {
                Thread.sleep(20)
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
            }
        }
    }
}
