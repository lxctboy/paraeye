package com.paraeye.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.CheckBox
import android.widget.RadioGroup
import android.widget.SeekBar
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.lifecycleScope
import com.paraeye.app.depth.DepthBenchmark
import com.paraeye.app.depth.DepthEstimator
import com.paraeye.app.depth.DepthVisualizer
import com.paraeye.app.depth.ImagePreprocessor
import com.paraeye.app.gl.FrameDepthConsumer
import com.paraeye.app.gl.SbsRenderer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

/**
 * ParaEye P1/P2/P3 验证壳：
 *  - P1：单帧 LiteRT 推理 + 基准计时
 *  - P2：GL 视图合成（Full-SBS / 伪彩深度）+ 视差滑块
 *  - P3：本地视频实时流水线（MediaCodec 硬解 → GPU 归一化/降采样 → 异步推理 → SBS）
 */
class MainActivity : ComponentActivity() {

    private lateinit var statusText: TextView
    private lateinit var statsText: TextView
    private lateinit var dispLabel: TextView
    private lateinit var glView: GLSurfaceView
    private lateinit var renderer: SbsRenderer
    private lateinit var acceleratorGroup: RadioGroup
    private lateinit var modelGroup: RadioGroup
    private lateinit var runButton: Button
    private lateinit var playButton: Button

    private var estimator: DepthEstimator? = null
    private var currentBitmap: Bitmap? = null

    /** 推理固定跑在单线程上（LiteRT CompiledModel 非线程安全，且这样天然串行） */
    private val inferExecutor = Executors.newSingleThreadExecutor()

    /** 预分配 NCHW 输入（实时路径每帧复用，避免 4.3MB/帧 的分配） */
    private val nchwBuf = FloatArray(3 * ImagePreprocessor.MODEL_W * ImagePreprocessor.MODEL_H)

    @Volatile private var inflight = false
    @Volatile private var lastInferMs = 0f
    @Volatile private var useGpu = false
    @Volatile private var useFp16 = false
    @Volatile private var modelType = DepthEstimator.ModelType.INT8
    @Volatile private var videoMode = false
    @Volatile private var playing = false

    private val statsHandler = Handler(Looper.getMainLooper())
    private val statsTask = object : Runnable {
        override fun run() {
            if (videoMode) {
                val acc = buildString {
                    append(if (useGpu) "GPU" else "CPU")
                    if (useGpu && useFp16) append("/FP16")
                }
                statsText.text = buildString {
                    appendLine("实时流水线 (${modelType.label} / $acc)")
                    appendLine("视频帧率: %.1f FPS".format(renderer.videoFps))
                    appendLine("深度更新: %.1f FPS".format(renderer.depthFps))
                    appendLine("单帧推理: %.1f ms".format(lastInferMs))
                }
            }
            statsHandler.postDelayed(this, 500)
        }
    }

    private val pickImage = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri: Uri? ->
        uri?.let {
            stopVideo()
            loadBitmapFromUri(it)
        }
    }

    private val pickVideo = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri: Uri? ->
        uri?.let { startVideo(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.statusText)
        statsText = findViewById(R.id.statsText)
        dispLabel = findViewById(R.id.dispLabel)
        acceleratorGroup = findViewById(R.id.acceleratorGroup)
        modelGroup = findViewById(R.id.modelGroup)
        runButton = findViewById(R.id.runButton)
        playButton = findViewById(R.id.playButton)
        glView = findViewById(R.id.glView)

        renderer = SbsRenderer()
        glView.setEGLContextClientVersion(2)
        // 保留 EGL 上下文：否则 onPause/onResume 会重建 GL 资源，
        // 导致解码器持有的 SurfaceTexture 变成悬空引用（P3 视频路径必需）
        glView.setPreserveEGLContextOnPause(true)
        glView.setRenderer(renderer)
        glView.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY
        renderer.requestRender = { glView.requestRender() }
        renderer.depthConsumer = frameConsumer
        renderer.onVideoError = { statusText.text = "视频失败：$it" }
        renderer.onVideoStarted = { w, h, dur ->
            statusText.text = "视频 ${w}x${h}，%.1f 秒，实时 3D 中".format(dur / 1_000_000f)
        }

        findViewById<Button>(R.id.pickButton).setOnClickListener {
            pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        findViewById<Button>(R.id.pickVideoButton).setOnClickListener {
            pickVideo.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly))
        }
        runButton.setOnClickListener { runSingleFrame() }

        playButton.setOnClickListener {
            playing = !playing
            renderer.setPaused(!playing)
            playButton.text = if (playing) "暂停播放" else "继续播放"
        }

        acceleratorGroup.setOnCheckedChangeListener { _, id ->
            useGpu = (id == R.id.radioGpu)
            onConfigChanged()
        }
        modelGroup.setOnCheckedChangeListener { _, id ->
            modelType = if (id == R.id.radioInt8) {
                DepthEstimator.ModelType.INT8
            } else {
                DepthEstimator.ModelType.FP32
            }
            onConfigChanged()
        }

        val seek = findViewById<SeekBar>(R.id.dispSeek)
        seek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                val ratio = progress / 1000f
                renderer.maxDispRatio = ratio
                dispLabel.text = "当前：%.1f%%%s".format(
                    ratio * 100f,
                    if (progress == 10) "（P0 定档，默认）" else ""
                )
                glView.requestRender()
            }

            override fun onStartTrackingTouch(sb: SeekBar?) = Unit
            override fun onStopTrackingTouch(sb: SeekBar?) = Unit
        })

        findViewById<RadioGroup>(R.id.viewModeGroup)
            .setOnCheckedChangeListener { _, checkedId ->
                renderer.mode = if (checkedId == R.id.radioModeSbs) {
                    SbsRenderer.MODE_SBS
                } else {
                    SbsRenderer.MODE_DEPTH
                }
                glView.requestRender()
            }

        findViewById<CheckBox>(R.id.swapCheck)
            .setOnCheckedChangeListener { _, isChecked ->
                renderer.swapLr = isChecked
                glView.requestRender()
            }

        findViewById<CheckBox>(R.id.fp16Check)
            .setOnCheckedChangeListener { _, isChecked ->
                useFp16 = isChecked
                onConfigChanged()
            }

        findViewById<CheckBox>(R.id.flipCheck)
            .setOnCheckedChangeListener { _, isChecked ->
                renderer.videoFlipY = isChecked
            }

        statusText.text = "选图片做单帧验证，或选视频跑实时流水线。SBS 请用平行眼观看"
    }

    override fun onResume() {
        super.onResume()
        glView.onResume()
        if (videoMode) renderer.setPaused(!playing)
        statsHandler.post(statsTask)
    }

    override fun onPause() {
        statsHandler.removeCallbacks(statsTask)
        renderer.setPaused(true)
        glView.onPause()
        super.onPause()
    }

    override fun onDestroy() {
        glView.queueEvent { renderer.releaseVideo() }
        inferExecutor.shutdown()
        estimator?.close()
        estimator = null
        super.onDestroy()
    }

    private fun onConfigChanged() {
        // 换模型/加速器后时域平滑的历史深度失效
        renderer.resetDepthSmoother()
    }

    private val frameConsumer = object : FrameDepthConsumer {
        override val busy: Boolean get() = inflight

        override fun onFrameForDepth(rgba: ByteArray) {
            // P3.5：rgba 来自渲染器双缓冲池，无需拷贝；用完 recycle 归还
            inflight = true
            inferExecutor.execute {
                try {
                    val est = ensureEstimator()
                    val t0 = System.nanoTime()
                    ImagePreprocessor.rgbaToNchwInto(
                        rgba, ImagePreprocessor.MODEL_W, ImagePreprocessor.MODEL_H, nchwBuf
                    )
                    val raw = est.inferNchw(nchwBuf)
                    lastInferMs = (System.nanoTime() - t0) / 1_000_000f
                    // 归一化后的目标深度交给渲染器，由其逐帧 EMA 过渡（治 6FPS 跳变）
                    renderer.submitDepthTarget(DepthVisualizer.normalize(raw.depth))
                } catch (e: Exception) {
                    runOnUiThread { statusText.text = "推理失败：${e.message}" }
                } finally {
                    inflight = false
                    recycle(rgba)
                }
            }
        }

        override fun recycle(rgba: ByteArray) {
            renderer.recycleRgba(rgba)
        }
    }

    private fun ensureEstimator(): DepthEstimator {
        val est = estimator ?: DepthEstimator(applicationContext).also { estimator = it }
        if (!est.ensureLoaded(modelType, useGpu, useFp16)) {
            error("模型加载失败（GPU 不受支持时请改用 CPU）")
        }
        return est
    }

    // ---------------- 视频路径 ----------------

    private fun startVideo(uri: Uri) {
        currentBitmap = null
        videoMode = true
        playing = true
        renderer.resetDepthSmoother()
        playButton.text = "暂停播放"
        playButton.isEnabled = true
        glView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        renderer.startVideo(this, uri)
        statusText.text = "视频加载中..."
    }

    private fun stopVideo() {
        if (!videoMode) return
        videoMode = false
        playing = false
        renderer.stopVideo()
        glView.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY
        playButton.text = "播放 / 暂停"
        playButton.isEnabled = false
    }

    // ---------------- 静态图路径 ----------------

    private fun loadBitmapFromUri(uri: Uri) {
        try {
            val raw = contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
            if (raw == null) {
                statusText.text = "图片解码失败"
                return
            }
            // 大图降采样到 2048 长边，避免 OOM
            currentBitmap = downscale(raw, 2048)
            statusText.text = "已加载图片 ${currentBitmap!!.width}x${currentBitmap!!.height}，点「生成 3D」"
        } catch (e: Exception) {
            statusText.text = "加载失败: ${e.message}"
        }
    }

    private fun downscale(bitmap: Bitmap, maxSide: Int): Bitmap {
        val w = bitmap.width
        val h = bitmap.height
        val max = maxOf(w, h)
        if (max <= maxSide) return bitmap
        val scale = maxSide.toFloat() / max
        return Bitmap.createScaledBitmap(bitmap, (w * scale).toInt(), (h * scale).toInt(), true)
    }

    private fun runSingleFrame() {
        if (videoMode) {
            statusText.text = "实时模式占用推理线程，请先选一张图片切回单帧模式"
            return
        }
        val bitmap = currentBitmap ?: run {
            statusText.text = "请先选择图片（实时流请选视频）"
            return
        }
        lifecycleScope.launch {
            statusText.text = "推理中（${modelType.label} / ${if (useGpu) "GPU" else "CPU"}）..."
            try {
                val (result, bench) = withContext(Dispatchers.Default) {
                    val est = ensureEstimator()
                    val depth = est.infer(bitmap)
                    val b = DepthBenchmark.run(est, bitmap)
                    depth to b
                }
                val r8 = SbsRenderer.depthToR8(DepthVisualizer.normalize(result.depth))
                renderer.submitFrame(bitmap, r8)
                glView.requestRender()

                statsText.text = buildString {
                    appendLine("模型: ${modelType.label}  |  加速: ${if (useGpu) "GPU" else "CPU"}")
                    appendLine("迭代 ${bench.iterations} 次（含预热 3 次）")
                    appendLine("平均: %.1f ms (%.1f FPS)".format(bench.avgMs, bench.fps))
                    appendLine("min: %.1f ms | p50: %.1f ms | max: %.1f ms".format(bench.minMs, bench.p50Ms, bench.maxMs))
                }
                statusText.text = "完成 · SBS 已渲染，滑块可实时调视差"
            } catch (e: Exception) {
                statusText.text = "失败: ${e.message}"
            }
        }
    }
}
