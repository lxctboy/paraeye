package com.paraeye.app.video

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.util.Log

private const val TAG = "AudioPlayer"

/**
 * 伴音轨播放器：MediaExtractor 抽 audio 轨 → MediaCodec 解码 PCM → AudioTrack。
 *
 * 为什么不复用 [VideoDecoder]：MediaExtractor 同时 select 两条轨后 readSampleData
 * 会在两轨间交错返回，无法把视频样本稳定地喂给视频解码器。所以开第二个
 * extractor 独立抽音轨，两路各自解码、共享同一个 Uri。
 *
 * 同步策略：AudioTrack(MODE_STREAM) 的硬件时钟本身就是播放节奏，阻塞写天然限速，
 * 因此不需要像视频那样 sleep 对齐墙钟；只在 A/V 位置差超过 [DRIFT_LIMIT_MS] 时
 * 做一次硬同步（seek + 重置播放头），避免长时间播放后音画分离。
 */
class AudioPlayer(
    private val context: Context,
    private val uri: Uri,
    /** 取视频当前位置（微秒）作为同步参考；返回 0 表示暂无视频参考 */
    private val referencePositionUs: () -> Long = { 0L },
    /** 被其他应用抢走音频焦点时回调，UI 需同步把播放按钮置为暂停 */
    private val onFocusPause: () -> Unit = {},
) {

    @Volatile var running = false
        private set

    /** 播放器的暂停态与视频共用语义（true = 暂停） */
    @Volatile var paused = true

    /** 媒体是否含音轨；false 时整个类是空转的（不报错，静默） */
    @Volatile var hasAudio = false
        private set

    /** UI 写入的 seek 请求（微秒，-1 = 无） */
    @Volatile private var pendingSeekUs = -1L

    /**
     * 播放头时间基准：重置 AudioTrack 播放头时，已播帧数从 0 重新计数，
     * 用它换算出绝对播放位置（微秒）。
     */
    @Volatile private var headBaseUs = 0L

    @Volatile private var volume = 1f
    @Volatile private var muted = false

    private var thread: HandlerThread? = null
    private var audioTrack: AudioTrack? = null
    private var codec: MediaCodec? = null
    private var extractor: MediaExtractor? = null
    private var focusRequest: AudioFocusRequest? = null

    fun start() {
        if (running) return
        running = true
        paused = false
        requestAudioFocus()
        thread = HandlerThread("paraeye-audio").apply { start() }
        Handler(thread!!.looper).post { decodeLoop() }
    }

    fun setMuted(m: Boolean) {
        muted = m
        applyVolume()
    }

    fun setVolume(v: Float) {
        volume = v.coerceIn(0f, 1f)
        applyVolume()
    }

    /** 跳转（毫秒）；线程安全，仅置位请求 */
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
        abandonAudioFocus()
        runCatching { codec?.stop() }
        runCatching { codec?.release() }
        codec = null
        runCatching { extractor?.release() }
        extractor = null
        runCatching {
            audioTrack?.pause()
            audioTrack?.flush()
            audioTrack?.release()
        }
        audioTrack = null
    }

    private fun applyVolume() {
        val v = if (muted) 0f else volume
        runCatching { audioTrack?.setVolume(v) }
    }

    // ---------------- 音频焦点 ----------------

    private fun requestAudioFocus() {
        if (focusRequest != null) return
        val am = context.getSystemService(Context.AUDIO_SERVICE) as? AudioManager ?: return
        val attrs = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_MOVIE)
            .build()
        val listener = AudioManager.OnAudioFocusChangeListener { change ->
            when (change) {
                AudioManager.AUDIOFOCUS_LOSS,
                AudioManager.AUDIOFOCUS_LOSS_TRANSIENT,
                -> {
                    if (!paused) {
                        paused = true
                        onFocusPause()
                    }
                }
                AudioManager.AUDIOFOCUS_GAIN -> Unit
            }
        }
        runCatching {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                val req = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN)
                    .setAudioAttributes(attrs)
                    .setOnAudioFocusChangeListener(listener, Handler(Looper.getMainLooper()))
                    .build()
                focusRequest = req
                am.requestAudioFocus(req)
            }
        }
    }

    private fun abandonAudioFocus() {
        val req = focusRequest ?: return
        focusRequest = null
        val am = context.getSystemService(Context.AUDIO_SERVICE) as? AudioManager ?: return
        runCatching { am.abandonAudioFocusRequest(req) }
    }

    // ---------------- 解码与播放 ----------------

    private fun decodeLoop() {
        val ex = MediaExtractor()
        val opened = runCatching { ex.setDataSource(context, uri, null) }
        if (opened.isFailure) {
            Log.w(TAG, "无法打开音频源，静默降级为无声播放", opened.exceptionOrNull())
            hasAudio = false
            running = false
            return
        }

        var trackIndex = -1
        var mime = ""
        for (i in 0 until ex.trackCount) {
            val m = ex.getTrackFormat(i).getString(MediaFormat.KEY_MIME) ?: continue
            if (m.startsWith("audio/")) {
                trackIndex = i
                mime = m
                break
            }
        }
        if (trackIndex < 0) {
            // 纯无声视频：不是错误，直接退出，UI 仍可看 3D
            hasAudio = false
            runCatching { ex.release() }
            running = false
            return
        }
        hasAudio = true
        ex.selectTrack(trackIndex)
        extractor = ex

        val audioCodec = try {
            MediaCodec.createDecoderByType(mime)
        } catch (e: Exception) {
            Log.w(TAG, "创建音频解码器失败，转为无声播放", e)
            hasAudio = false
            running = false
            return
        }
        codec = audioCodec
        try {
            audioCodec.configure(ex.getTrackFormat(trackIndex), null, null, 0)
            audioCodec.start()
        } catch (e: Exception) {
            Log.w(TAG, "音频解码器配置失败，转为无声播放", e)
            hasAudio = false
            running = false
            return
        }

        val info = MediaCodec.BufferInfo()
        var inputDone = false
        var trackPaused = false
        var sampleRate = 44100
        // seek 后丢弃 pts 早于目标的样本，直到追平，避免回退一大段声音
        var skipUntilUs = Long.MIN_VALUE
        var lastDriftCheckAt = 0L

        while (running) {
            if (paused) {
                if (!trackPaused) {
                    runCatching { audioTrack?.pause() }
                    trackPaused = true
                }
                Thread.sleep(20)
                continue
            }
            if (trackPaused) {
                runCatching { audioTrack?.play() }
                trackPaused = false
            }

            // ---- seek / 漂移硬同步 ----
            val seekUs = pendingSeekUs
            if (seekUs >= 0) {
                pendingSeekUs = -1
                headBaseUs = seekUs
                skipUntilUs = AvSync.skipUntilUsForSeek(seekUs)
                runCatching { audioCodec.flush() }
                ex.seekTo(seekUs, MediaExtractor.SEEK_TO_CLOSEST_SYNC)
                inputDone = false
                resetTrackHead()
                // 从现在开始重新计冷却，避免刚重锚完立刻又触发一次漂移判断
                lastDriftCheckAt = System.currentTimeMillis()
                continue
            }

            val now = System.currentTimeMillis()
            val sinceCheck = now - lastDriftCheckAt
            lastDriftCheckAt = now
            // 采样一次视频位置，避免重复调用拿到不同时刻的值
            val videoUs = referencePositionUs()
            val audioUs = audioPositionUs(sampleRate)
            if (AvSync.needsResync(audioUs, videoUs, sinceCheck)) {
                Log.i(TAG, "A/V 漂移 ${AvSync.driftMsOrZero(audioUs, videoUs)}ms，重新对齐")
                pendingSeekUs = videoUs
                continue
            }

            // ---- 喂输入 ----
            if (!inputDone) {
                val inIndex = audioCodec.dequeueInputBuffer(10_000)
                if (inIndex >= 0) {
                    val buffer = audioCodec.getInputBuffer(inIndex)
                    if (buffer != null) {
                        val size = ex.readSampleData(buffer, 0)
                        if (size < 0) {
                            audioCodec.queueInputBuffer(
                                inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM
                            )
                            inputDone = true
                        } else {
                            audioCodec.queueInputBuffer(
                                inIndex, 0, size, ex.sampleTime, ex.sampleFlags
                            )
                            ex.advance()
                        }
                    }
                    // buffer 为 null 时不投喂空包，等下一轮 dequeue
                }
            }

            // ---- 取输出 ----
            val outIndex = audioCodec.dequeueOutputBuffer(info, 10_000)
            if (outIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                val outFormat = audioCodec.outputFormat
                sampleRate = outFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                buildTrack(outFormat, sampleRate)?.let { track ->
                    audioTrack = track
                    applyVolume()
                    runCatching { track.play() }
                    trackPaused = false
                }
                continue
            }
            if (outIndex < 0) continue

            val pts = info.presentationTimeUs
            val track = audioTrack
            if (info.size > 0 && track != null) {
                if (pts < skipUntilUs) {
                    audioCodec.releaseOutputBuffer(outIndex, false)
                    continue
                }
                skipUntilUs = Long.MIN_VALUE
                val buf = audioCodec.getOutputBuffer(outIndex)
                if (buf != null) {
                    buf.position(info.offset)
                    buf.limit(info.offset + info.size)
                    // 阻塞写：AudioTrack 缓冲满时在这里等待，硬件时钟就是播放节奏
                    track.write(buf, info.size, AudioTrack.WRITE_BLOCKING)
                }
            }
            audioCodec.releaseOutputBuffer(outIndex, false)

            if ((info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                // 循环播放：与视频一样回到 0，音频播放头同步归零
                runCatching { audioCodec.flush() }
                ex.seekTo(0L, MediaExtractor.SEEK_TO_CLOSEST_SYNC)
                inputDone = false
                headBaseUs = 0L
                skipUntilUs = Long.MIN_VALUE
                resetTrackHead()
            }
        }
    }

    /**
     * 丢弃 AudioTrack 内部缓冲并把播放头归零（seek / 循环时调用）。
     * 暂停态下只归位不恢复播放，避免把暂停中的视频意外出声。
     */
    private fun resetTrackHead() {
        val t = audioTrack ?: return
        runCatching {
            t.pause()
            t.flush()
            t.setPlaybackHeadPosition(0L)
            if (!paused) t.play()
        }
    }

    /** 音频当前实际播放位置（微秒），由已播帧数换算 */
    private fun audioPositionUs(sampleRate: Int): Long {
        val t = audioTrack ?: return 0L
        val frames = runCatching { t.playbackHeadPosition.toLong() and 0xFFFFFFFFL }.getOrElse { return 0L }
        return AvSync.positionUsFromFrames(frames, sampleRate, headBaseUs)
    }

    private fun buildTrack(format: MediaFormat, sampleRate: Int): AudioTrack? = runCatching {
        val channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        val encoding = if (format.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
            format.getInteger(MediaFormat.KEY_PCM_ENCODING)
        } else {
            AudioFormat.ENCODING_PCM_16BIT
        }
        val channelMask = if (channels >= 2) {
            AudioFormat.CHANNEL_IN_STEREO
        } else {
            AudioFormat.CHANNEL_IN_MONO
        }
        // getMinBufferSize 失败时返回负值错误码，不能直接参与 maxOf，否则缓冲可能算成负数
        val minBytes = AudioTrack.getMinBufferSize(sampleRate, channelMask, encoding)
            .let { if (it > 0) it else 0 }
        val bytesPerSample = if (encoding == AudioFormat.ENCODING_PCM_FLOAT) 4 else 2
        // 50ms 一块的缓冲：够小则延迟低，够大则不会频繁阻塞
        val chunkBytes = sampleRate / 20 * channels * bytesPerSample
        val bufferBytes = maxOf(minBytes, chunkBytes)

        val attrs = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_MOVIE)
            .build()
        val fmt = AudioFormat.Builder()
            .setEncoding(encoding)
            .setSampleRate(sampleRate)
            .setChannelMask(channelMask)
            .build()
        AudioTrack.Builder()
            .setAudioAttributes(attrs)
            .setAudioFormat(fmt)
            .setBufferSizeInBytes(bufferBytes)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
    }.onFailure {
        Log.w(TAG, "创建 AudioTrack 失败，转为无声播放", it)
    }.getOrNull()
}