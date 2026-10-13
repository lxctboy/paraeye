package com.paraeye.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.View
import android.widget.CheckBox
import android.widget.ImageButton
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
import com.paraeye.app.ui.TimeFormat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

/**
 * ParaEye 播放器壳（v1.1.0 播放器化重构）：
 *  - 画面层：GLSurfaceView 常驻全屏，上面浮三层 chrome（标题/控制条/性能浮层）
 *  - 播放层：进度条 + 拖动 seek（VideoDecoder GOP 内对齐）、大播放键、循环播放
 *  - 设置层：底部抽屉（视差/显示内容/观看方式/深度帧率上限/推理后端）
 *  - 来源：本地视频走实时流水线（MediaCodec→深度→SBS），本地图片走单帧推理
 */
class MainActivity : ComponentActivity() {

    private lateinit var topBar: View
    private lateinit var bottomBar: View
    private lateinit var statsChip: TextView
    private lateinit var statusText: TextView
    private lateinit var statsText: TextView
    private lateinit var modelText: TextView
    private lateinit var dispLabel: TextView
    private lateinit var glView: GLSurfaceView
    private lateinit var renderer: SbsRenderer
    private lateinit var centerStage: View
    private lateinit var centerHint: TextView
    private lateinit var bigPlayButton: ImageButton
    private lateinit var playButton: ImageButton
    private lateinit var fullscreenButton: ImageButton
    private lateinit var settingsButton: ImageButton
    private lateinit var settingsPanel: View
    private lateinit var positionSeek: SeekBar
    private lateinit var timeText: TextView
    private lateinit var acceleratorGroup: RadioGroup

    private var estimator: DepthEstimator? = null
    private var currentBitmap: Bitmap? = null

    /** 推理固定跑在单线程上（LiteRT CompiledModel 非线程安全，且这样天然串行） */
    private val inferExecutor = Executors.newSingleThreadExecutor()

    /** 预分配 NCHW 输入（实时路径每帧复用，280×518 下约 1.7MB/帧） */
    private val nchwBuf = FloatArray(3 * ImagePreprocessor.MODEL_W * ImagePreprocessor.MODEL_H)

    @Volatile private var inflight = false
    @Volatile private var lastInferMs = 0f
    @Volatile private var useGpu = false
    @Volatile private var useFp16 = false
    @Volatile private var modelType = DepthEstimator.ModelType.INT8
    @Volatile private var videoMode = false
    @Volatile private var playing = false

    /** chrome（标题栏 + 控制条 + 性能浮层）是否可见；点画面切换 */
    private var chromeVisible = true

    /** 沉浸式全屏（隐藏系统栏）；方向仍跟随传感器——竖屏片源裸眼观看本就是竖屏布局 */
    private var fullscreen = false

    /** 设置抽屉是否展开 */
    private var settingsOpen = false

    /** 进度条拖动中：暂停用解码位置覆盖 UI */
    private var seeking = false

    /**
     * 深度帧率上限（FPS，0 = 不限）。发热控制的关键旋钮：
     * 推理占空比 ≈ 深度FPS × 单帧耗时。280×518 下 165.8ms/帧，
     * 不限≈5.5FPS → 90% 占空比（发热主因）；3FPS → 50%；2FPS → 33%。
     * 低于目标间隔时 consumer.busy 返回 true，渲染器跳过本帧派发
     * （不取缓冲、无需 recycle），CPU 自然空闲，EMA 逐帧插值兜底平滑。
     */
    @Volatile private var depthFpsCap = 0

    /** 上一次派发推理的时刻（GL 线程写、GL 线程读） */
    @Volatile private var lastInferAtMs = 0L

    /** 单帧模式的基准结果（图片路径跑完后显示） */
    private var lastBench: DepthBenchmark.Result? = null

    private val versionName: String by lazy {
        runCatching { packageManager.getPackageInfo(packageName, 0).versionName }.getOrNull() ?: "?"
    }

    private val handler = Handler(Looper.getMainLooper())

    private val hideChromeTask = Runnable { hideChrome() }

    /** 250ms 心跳：推进进度条 + 刷新性能浮层 */
    private val tickTask = object : Runnable {
        override fun run() {
            updateProgress()
            updateStatsChip()
            handler.postDelayed(this, 250)
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
        bindViews()

        renderer = SbsRenderer()
        glView.setEGLContextClientVersion(2)
        // 保留 EGL 上下文：否则 onPause/onResume 会重建 GL 资源，
        // 导致解码器持有的 SurfaceTexture 变成悬空引用（视频路径必需）
        glView.setPreserveEGLContextOnPause(true)
        glView.setRenderer(renderer)
        glView.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY
        renderer.requestRender = { glView.requestRender() }
        renderer.depthConsumer = frameConsumer
        renderer.onVideoError = { msg -> setStatus("视频失败：$msg") }
        renderer.onVideoStarted = { w, h, dur ->
            setStatus("已加载 ${w}×${h} · %.1f 秒 · 实时 3D 中".format(dur / 1_000_000f))
        }
        // 音频焦点被抢占：音频侧已自行暂停，这里把 UI 也切回暂停态，避免按钮显示"正在播放"
        renderer.onAudioFocusLost = {
            if (playing) {
                playing = false
                syncPlayIcons()
                refreshIdleHints()
                showChrome(autoHide = false)
            }
        }

        wirePlayerControls()
        wireSettings()
        wireGesture()

        modelText.text = "模型 ${modelType.label} · ${ImagePreprocessor.MODEL_W}×${ImagePreprocessor.MODEL_H} · v$versionName"
        setStatus(getString(R.string.hint_pick_source))
        refreshIdleHints()
    }

    private fun bindViews() {
        topBar = findViewById(R.id.topBar)
        bottomBar = findViewById(R.id.bottomBar)
        statsChip = findViewById(R.id.statsChip)
        statusText = findViewById(R.id.statusText)
        statsText = findViewById(R.id.statsText)
        modelText = findViewById(R.id.modelText)
        dispLabel = findViewById(R.id.dispLabel)
        glView = findViewById(R.id.glView)
        centerStage = findViewById(R.id.centerStage)
        centerHint = findViewById(R.id.centerHint)
        bigPlayButton = findViewById(R.id.bigPlayButton)
        playButton = findViewById(R.id.playButton)
        fullscreenButton = findViewById(R.id.fullscreenButton)
        settingsButton = findViewById(R.id.settingsButton)
        settingsPanel = findViewById(R.id.settingsPanel)
        positionSeek = findViewById(R.id.positionSeek)
        timeText = findViewById(R.id.timeText)
        acceleratorGroup = findViewById(R.id.acceleratorGroup)
    }

    // ---------------- 播放控制 ----------------

    private fun wirePlayerControls() {
        findViewById<TextView>(R.id.pickVideoButton).setOnClickListener {
            pickVideo.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly))
        }
        findViewById<TextView>(R.id.pickButton).setOnClickListener {
            pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        findViewById<TextView>(R.id.runButton).setOnClickListener { runSingleFrame() }

        val playToggle = View.OnClickListener { togglePlay() }
        playButton.setOnClickListener(playToggle)
        bigPlayButton.setOnClickListener(playToggle)

        fullscreenButton.setOnClickListener { setFullscreen(!fullscreen) }
        settingsButton.setOnClickListener { toggleSettings() }

        positionSeek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                if (!fromUser) return
                val durMs = renderer.videoDurationUs / 1000
                if (durMs > 0) {
                    timeText.text = TimeFormat.position(durMs * progress / 1000, durMs)
                }
            }

            override fun onStartTrackingTouch(sb: SeekBar?) {
                seeking = true
                handler.removeCallbacks(hideChromeTask)
            }

            override fun onStopTrackingTouch(sb: SeekBar?) {
                seeking = false
                val durMs = renderer.videoDurationUs / 1000
                if (durMs > 0) sb?.let { renderer.seekToMs(durMs * it.progress / 1000) }
            }
        })
    }

    private fun togglePlay() {
        if (!videoMode) {
            pickVideo.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.VideoOnly))
            return
        }
        playing = !playing
        renderer.setPaused(!playing)
        syncPlayIcons()
        refreshIdleHints()
        if (playing) scheduleAutoHide() else showChrome(autoHide = false)
    }

    /** 播放/暂停图标与可用态；暂停时按钮要一直留在屏幕上 */
    private fun syncPlayIcons() {
        val icon = if (playing) R.drawable.ic_pause else R.drawable.ic_play
        playButton.setImageResource(icon)
        playButton.isEnabled = videoMode
        playButton.alpha = if (videoMode) 1f else 0.4f
    }

    /** 空态提示与大播放键的可见性 */
    private fun refreshIdleHints() {
        centerHint.visibility = if (videoMode) View.GONE else View.VISIBLE
        bigPlayButton.visibility = if (videoMode && !playing) View.VISIBLE else View.GONE
    }

    // ---------------- chrome 显示策略 ----------------

    /** 单击画面：显示/隐藏控制条（播放器标准手势） */
    private fun toggleChrome() {
        if (chromeVisible) hideChrome() else showChrome()
    }

    private fun showChrome(autoHide: Boolean = true) {
        chromeVisible = true
        topBar.visibility = View.VISIBLE
        bottomBar.visibility = View.VISIBLE
        if (videoMode) statsChip.visibility = View.VISIBLE
        fullscreenButton.setImageResource(
            if (fullscreen) R.drawable.ic_fullscreen_exit else R.drawable.ic_fullscreen
        )
        if (autoHide) scheduleAutoHide() else handler.removeCallbacks(hideChromeTask)
    }

    private fun hideChrome() {
        chromeVisible = false
        topBar.visibility = View.GONE
        bottomBar.visibility = View.GONE
        statsChip.visibility = View.GONE
    }

    /** 只在「正在播放视频」时自动隐藏，暂停/空态/抽屉打开时保持可见 */
    private fun scheduleAutoHide() {
        handler.removeCallbacks(hideChromeTask)
        if (videoMode && playing && !settingsOpen) {
            handler.postDelayed(hideChromeTask, 3_500)
        }
    }

    private fun toggleSettings() {
        settingsOpen = !settingsOpen
        settingsPanel.visibility = if (settingsOpen) View.VISIBLE else View.GONE
        if (settingsOpen) {
            showChrome(autoHide = false)
            settingsButton.alpha = 0.6f
        } else {
            settingsButton.alpha = 1f
            scheduleAutoHide()
        }
    }

    private fun setFullscreen(on: Boolean) {
        fullscreen = on
        applyImmersive(on)
        if (on) {
            // 进全屏先收起来，点画面唤出（与播放器一致）
            hideChrome()
        } else {
            showChrome(autoHide = false)
        }
    }

    @Suppress("DEPRECATION")
    private fun applyImmersive(on: Boolean) {
        window.decorView.systemUiVisibility = if (on) {
            View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
                View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                View.SYSTEM_UI_FLAG_FULLSCREEN or
                View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION or
                View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN or
                View.SYSTEM_UI_FLAG_LAYOUT_STABLE
        } else {
            View.SYSTEM_UI_FLAG_LAYOUT_STABLE
        }
    }

    private fun wireGesture() {
        val detector = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onDown(e: MotionEvent): Boolean = true

            override fun onSingleTapConfirmed(e: MotionEvent): Boolean {
                toggleChrome()
                return true
            }

            override fun onDoubleTap(e: MotionEvent): Boolean {
                setFullscreen(!fullscreen)
                return true
            }
        })
        glView.setOnTouchListener { _, event -> detector.onTouchEvent(event) }
    }

    // ---------------- 设置抽屉 ----------------

    private fun wireSettings() {
        val dispSeek = findViewById<SeekBar>(R.id.dispSeek)
        dispSeek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                val ratio = progress / 1000f
                renderer.maxDispRatio = ratio
                dispLabel.text = "当前 %.1f%%%s".format(
                    ratio * 100f,
                    if (progress == 10) " · P0 定档默认" else ""
                )
                glView.requestRender()
            }

            override fun onStartTrackingTouch(sb: SeekBar?) = Unit
            override fun onStopTrackingTouch(sb: SeekBar?) = Unit
        })
        dispLabel.text = "当前 1.0% · P0 定档默认"

        findViewById<RadioGroup>(R.id.viewModeGroup).setOnCheckedChangeListener { _, checkedId ->
            renderer.mode = if (checkedId == R.id.radioModeSbs) {
                SbsRenderer.MODE_SBS
            } else {
                SbsRenderer.MODE_DEPTH
            }
            glView.requestRender()
        }

        findViewById<CheckBox>(R.id.swapCheck).setOnCheckedChangeListener { _, checked ->
            renderer.swapLr = checked
            glView.requestRender()
        }

        findViewById<CheckBox>(R.id.flipCheck).setOnCheckedChangeListener { _, checked ->
            renderer.videoFlipY = checked
        }

        findViewById<CheckBox>(R.id.muteCheck).setOnCheckedChangeListener { _, checked ->
            renderer.setMuted(checked)
        }

        findViewById<CheckBox>(R.id.fp16Check).setOnCheckedChangeListener { _, checked ->
            useFp16 = checked
            onConfigChanged()
        }

        findViewById<RadioGroup>(R.id.depthCapGroup).setOnCheckedChangeListener { _, checkedId ->
            depthFpsCap = when (checkedId) {
                R.id.radioCap5 -> 5
                R.id.radioCap4 -> 4
                R.id.radioCap3 -> 3
                R.id.radioCap2 -> 2
                else -> 0 // 不限
            }
        }

        acceleratorGroup.setOnCheckedChangeListener { _, id ->
            useGpu = (id == R.id.radioGpu)
            onConfigChanged()
        }
    }

    private fun onConfigChanged() {
        // 换模型/加速器后时域平滑的历史深度失效
        renderer.resetDepthSmoother()
        // 允许再次提示 GPU 回退（用户可能切走了又切回 GPU）
        gpuFallbackNotified = false
        // 图片模式下换后端要重新推理才生效
        if (!videoMode && currentBitmap != null) runSingleFrame()
    }

    // ---------------- 进度与性能浮层 ----------------

    private fun updateProgress() {
        if (!videoMode) {
            positionSeek.progress = 0
            timeText.text = "图片 · 单帧"
            return
        }
        val durUs = renderer.videoDurationUs
        val posUs = renderer.videoPositionUs
        if (durUs <= 0) {
            timeText.text = "--:-- / --:--"
            return
        }
        if (!seeking) {
            positionSeek.progress = ((posUs * 1000) / durUs).toInt().coerceIn(0, 1000)
            timeText.text = TimeFormat.position(posUs / 1000, durUs / 1000)
        }
    }

    private fun updateStatsChip() {
        if (!videoMode) {
            // 图片模式：单帧基准（跑完才有数据）
            val bench = lastBench
            if (bench != null && chromeVisible) {
                statsChip.text = "单帧\n%.0f ms\n%s".format(bench.avgMs, modelType.label)
                statsChip.visibility = View.VISIBLE
            } else if (statsChip.visibility != View.GONE) {
                statsChip.visibility = View.GONE
            }
            return
        }
        val acc = when {
            useGpu && estimator?.gpuFallback == true -> "CPU"
            useGpu -> "GPU"
            else -> "CPU"
        }
        statsChip.text = "%.1f FPS\n深度 %.1f\n%.0f ms".format(
            renderer.videoFps, renderer.depthFps, lastInferMs
        ) + "\n$acc" + (if (depthFpsCap > 0) " ·≤$depthFpsCap" else "")
        if (chromeVisible && statsChip.visibility != View.VISIBLE) {
            statsChip.visibility = View.VISIBLE
        } else if (!chromeVisible && statsChip.visibility != View.GONE) {
            statsChip.visibility = View.GONE
        }
    }

    private fun setStatus(text: String) {
        statusText.text = text
    }

    // ---------------- 生命周期 ----------------

    override fun onResume() {
        super.onResume()
        glView.onResume()
        // 系统栏可能因回桌面/通知栏重现，全屏状态下每次回前台补一次沉浸式
        if (fullscreen) applyImmersive(true)
        if (videoMode) renderer.setPaused(!playing)
        handler.post(tickTask)
        if (videoMode && playing) scheduleAutoHide()
    }

    override fun onPause() {
        handler.removeCallbacks(tickTask)
        handler.removeCallbacks(hideChromeTask)
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

    // ---------------- 深度推理 ----------------

    private val frameConsumer = object : FrameDepthConsumer {
        override val busy: Boolean
            get() = inflight ||
                // 深度帧率上限：距上次派发不足目标间隔时假装忙，
                // 渲染器便跳过本帧（缓冲留在池中，无需 recycle）
                (depthFpsCap > 0 &&
                    System.currentTimeMillis() - lastInferAtMs < 1000L / depthFpsCap)

        override fun onFrameForDepth(rgba: ByteArray) {
            // rgba 来自渲染器双缓冲池，无需拷贝；用完 recycle 归还
            lastInferAtMs = System.currentTimeMillis()
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
                    // 归一化后的目标深度交给渲染器，由其逐帧 EMA 过渡（治低帧率跳变）
                    renderer.submitDepthTarget(DepthVisualizer.normalize(raw.depth))
                } catch (e: Exception) {
                    runOnUiThread { setStatus("推理失败：${e.message}") }
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

    @Volatile private var gpuFallbackNotified = false

    private fun ensureEstimator(): DepthEstimator {
        val est = estimator ?: DepthEstimator(applicationContext).also { estimator = it }
        if (!est.ensureLoaded(modelType, useGpu, useFp16)) {
            error("模型加载失败（GPU 与 CPU 均无法加载模型）")
        }
        if (est.gpuFallback && !gpuFallbackNotified) {
            gpuFallbackNotified = true
            runOnUiThread {
                setStatus("GPU 不受支持（自导出模型含 GPU delegate 不支持的算子），已回退 CPU")
            }
        }
        return est
    }

    // ---------------- 视频路径 ----------------

private fun startVideo(uri: Uri) {
    currentBitmap = null
        lastBench = null
        videoMode = true
        playing = true
        renderer.resetDepthSmoother()
        syncPlayIcons()
        refreshIdleHints()
        showChrome()
        glView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        renderer.startVideo(this, uri)
        setStatus("视频加载中…")
        statsText.text = ""
    }

    private fun stopVideo() {
        if (!videoMode) return
        videoMode = false
        playing = false
        renderer.stopVideo()
        glView.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY
        syncPlayIcons()
        refreshIdleHints()
        updateProgress()
    }

    // ---------------- 静态图路径 ----------------

    private fun loadBitmapFromUri(uri: Uri) {
        try {
            val raw = contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
            if (raw == null) {
                setStatus("图片解码失败")
                return
            }
            // 大图降采样到 2048 长边，避免 OOM
            currentBitmap = downscale(raw, 2048)
            showChrome(autoHide = false)
            // 播放器化后不再让用户多点一次，选完图直接出 3D
            runSingleFrame()
        } catch (e: Exception) {
            setStatus("加载失败：${e.message}")
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
            setStatus("实时模式占用推理线程，请先选一张图片切回单帧模式")
            return
        }
        val bitmap = currentBitmap ?: run {
            setStatus("请先选择图片（实时流请选视频）")
            return
        }
        val acc = if (useGpu) "GPU" else "CPU"
        lifecycleScope.launch {
            setStatus("推理中（${modelType.label} / $acc）…")
            try {
                val (result, bench) = withContext(Dispatchers.Default) {
                    val est = ensureEstimator()
                    val depth = est.infer(bitmap)
                    val b = DepthBenchmark.run(est, bitmap)
                    depth to b
                }
                lastBench = bench
                val r8 = SbsRenderer.depthToR8(DepthVisualizer.normalize(result.depth))
                renderer.submitFrame(bitmap, r8)
                glView.requestRender()

                statsText.text = buildString {
                    appendLine("单帧 ${bitmap.width}×${bitmap.height} · ${modelType.label} · $acc")
                    appendLine("迭代 ${bench.iterations} 次（含预热 3 次）")
                    appendLine("平均 %.1f ms（%.1f FPS）".format(bench.avgMs, bench.fps))
                    appendLine("min %.1f · p50 %.1f · max %.1f ms".format(bench.minMs, bench.p50Ms, bench.maxMs))
                }
                setStatus("已生成 3D · 视差可在设置里调")
            } catch (e: Exception) {
                setStatus("失败：${e.message}")
            }
        }
    }
}
