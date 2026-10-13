package com.paraeye.app

import android.Manifest
import android.content.pm.PackageManager
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
import android.widget.Button
import android.widget.CheckBox
import android.widget.RadioGroup
import android.widget.SeekBar
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.paraeye.app.camera.CameraController
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
    private lateinit var controlPanel: View
    private lateinit var fullscreenButton: Button
    private lateinit var cameraButton: Button
    private lateinit var cameraSwitchButton: Button

    private var estimator: DepthEstimator? = null
    private var currentBitmap: Bitmap? = null

    /** P4：CameraX 绑定层（摄像头实时 3D），null = 未启动 */
    private var cameraController: CameraController? = null

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

    /** 全屏观看（隐藏控制面板 + 沉浸式系统栏；方向跟随传感器） */
    private var fullscreen = false

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

    private val statsHandler = Handler(Looper.getMainLooper())
    private val statsTask = object : Runnable {
        override fun run() {
            if (videoMode) {
                val acc = buildString {
                    if (useGpu && estimator?.gpuFallback == true) {
                        // GPU delegate 编译不了自导出模型，已回退 CPU（见 DepthEstimator.gpuFallback）
                        append("GPU→CPU（不受支持）")
                    } else {
                        append(if (useGpu) "GPU" else "CPU")
                        if (useGpu && useFp16) append("/FP16")
                    }
                }
                statsText.text = buildString {
                    appendLine("实时流水线 (${modelType.label} / $acc" +
                        if (depthFpsCap > 0) " · 深度≤${depthFpsCap}FPS" else "" + ")")
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
            stopCamera()
            stopVideo()
            loadBitmapFromUri(it)
        }
    }

    private val pickVideo = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri: Uri? ->
        uri?.let {
            stopCamera()
            startVideo(it)
        }
    }

    /** P4：相机运行时权限（CAMERA 是危险权限，清单声明不够） */
    private val requestCameraPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else statusText.text = "需要相机权限才能看实时 3D（可在系统设置里补授权）"
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
        controlPanel = findViewById(R.id.controlPanel)
        fullscreenButton = findViewById(R.id.fullscreenButton)
        cameraButton = findViewById(R.id.cameraButton)
        cameraSwitchButton = findViewById(R.id.cameraSwitchButton)
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
        renderer.onCameraStarted = { w, h ->
            val lens = if (cameraController?.lensFacing == CameraSelector.LENS_FACING_FRONT) "前置" else "后置"
            statusText.text = "摄像头 ${w}x${h} · 实时 3D 中（$lens）"
        }

        cameraButton.setOnClickListener {
            if (cameraController?.isActive == true) stopCamera() else ensureCameraPermissionAndStart()
        }
        cameraSwitchButton.setOnClickListener {
            cameraController?.switchLens()
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
        modelGroup.setOnCheckedChangeListener { _, _ ->
            modelType = DepthEstimator.ModelType.INT8
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

        findViewById<RadioGroup>(R.id.depthCapGroup)
            .setOnCheckedChangeListener { _, checkedId ->
                depthFpsCap = when (checkedId) {
                    R.id.radioCap5 -> 5
                    R.id.radioCap4 -> 4
                    R.id.radioCap3 -> 3
                    R.id.radioCap2 -> 2
                    else -> 0 // 不限
                }
            }

        fullscreenButton.setOnClickListener { setFullscreen(!fullscreen) }

        // 双击画面切换全屏（控制面板遮住的地方用按钮，露出的画面区域用双击）
        val tapDetector = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onDown(e: MotionEvent): Boolean = true
            override fun onDoubleTap(e: MotionEvent): Boolean {
                setFullscreen(!fullscreen)
                return true
            }
        })
        glView.setOnTouchListener { _, event -> tapDetector.onTouchEvent(event) }

        statusText.text = "选图片做单帧验证，或选视频跑实时流水线。SBS 请用平行眼观看"
    }

    override fun onResume() {
        super.onResume()
        glView.onResume()
        // 系统栏可能因回桌面/通知栏重现，全屏状态下每次回前台补一次沉浸式
        if (fullscreen) applyImmersive(true)
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
        // 允许再次提示 GPU 回退（用户可能切走了又切回 GPU）
        gpuFallbackNotified = false
    }

    // ---------------- 全屏观看 ----------------

    /**
     * 切换全屏：隐藏控制面板 + 沉浸式系统栏。
     * 不强制 orientation：SBS 片源以竖屏为主，平行眼裸眼观看就是竖屏布局；
     * P3.6 真机反馈——强制横屏后两眼之间被撑出黑带、无法融合出 3D。
     * 方向跟随传感器，横竖屏下渲染器都会把左右眼贴成紧邻的一对。
     */
    private fun setFullscreen(on: Boolean) {
        fullscreen = on
        controlPanel.visibility = if (on) View.GONE else View.VISIBLE
        fullscreenButton.text = if (on) "退出全屏" else "3. 全屏观看（双击画面退出）"
        applyImmersive(on)
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

    private val frameConsumer = object : FrameDepthConsumer {
        override val busy: Boolean
            get() = inflight ||
                // 深度帧率上限：距上次派发不足目标间隔时假装忙，
                // 渲染器便跳过本帧（缓冲留在池中，无需 recycle）
                (depthFpsCap > 0 &&
                    System.currentTimeMillis() - lastInferAtMs < 1000L / depthFpsCap)

        override fun onFrameForDepth(rgba: ByteArray) {
            // P3.5：rgba 来自渲染器双缓冲池，无需拷贝；用完 recycle 归还
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

    @Volatile private var gpuFallbackNotified = false

    private fun ensureEstimator(): DepthEstimator {
        val est = estimator ?: DepthEstimator(applicationContext).also { estimator = it }
        if (!est.ensureLoaded(modelType, useGpu, useFp16)) {
            error("模型加载失败（GPU 与 CPU 均无法加载模型）")
        }
        if (est.gpuFallback && !gpuFallbackNotified) {
            gpuFallbackNotified = true
            runOnUiThread {
                statusText.text = "GPU 不受支持（自导出模型含 GPU delegate 不支持的算子），已回退 CPU"
            }
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

    // ---------------- 摄像头路径（P4） ----------------

    private fun ensureCameraPermissionAndStart() {
        val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        if (granted) startCamera() else requestCameraPermission.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        // 摄像头与视频/图片互斥；videoST 在 GL 线程创建，就绪后再绑 CameraX
        stopVideo()
        videoMode = true
        playing = true
        playButton.text = "暂停播放"
        playButton.isEnabled = true
        glView.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        renderer.resetDepthSmoother()
        cameraButton.text = "1c. 关闭摄像头"
        cameraSwitchButton.isEnabled = true
        statusText.text = "摄像头启动中..."
        val controller = cameraController ?: CameraController(
            activity = this,
            renderer = renderer,
            onStatus = { msg -> statusText.text = msg },
        ).also { cameraController = it }
        glView.queueEvent {
            val st = renderer.cameraSurfaceTexture
            if (st == null) {
                runOnUiThread { statusText.text = "摄像头启动失败：GL 表面未就绪，稍后再试" }
                return@queueEvent
            }
            runOnUiThread { controller.start(st) }
        }
    }

    private fun stopCamera() {
        cameraController?.stop()
        if (videoMode) {
            videoMode = false
            playing = false
            glView.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY
        }
        playButton.text = "播放 / 暂停"
        playButton.isEnabled = false
        cameraButton.text = "1c. 摄像头（实时 3D）"
        cameraSwitchButton.isEnabled = false
        statusText.text = "摄像头已关闭"
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
