package com.paraeye.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.opengl.GLSurfaceView
import android.os.Bundle
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
import com.paraeye.app.gl.SbsRenderer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * ParaEye P1 + P2 验证：选图 → LiteRT 深度推理 → GL 视图合成（Full-SBS / 伪彩深度）。
 * 视差滑块可实时调节，用于复现 P0 的参数扫描；真机动作在这儿完成。
 * P3 起把「选静态图」替换为「相机/视频帧流」。
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

    private var estimator: DepthEstimator? = null
    private var currentBitmap: Bitmap? = null
    private var busy = false

    private val pickImage = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri: Uri? ->
        uri?.let { loadBitmapFromUri(it) }
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
        glView = findViewById(R.id.glView)

        renderer = SbsRenderer()
        glView.setEGLContextClientVersion(2)
        glView.setRenderer(renderer)
        glView.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY

        findViewById<Button>(R.id.pickButton).setOnClickListener {
            pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        runButton.setOnClickListener { runGeneration() }

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

        statusText.text = "选择一张图片 → 生成 3D。SBS 请用平行眼观看（或用 VR 盒子）"
    }

    override fun onResume() {
        super.onResume()
        glView.onResume()
    }

    override fun onPause() {
        glView.onPause()
        super.onPause()
    }

    override fun onDestroy() {
        estimator?.close()
        estimator = null
        super.onDestroy()
    }

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

    private fun runGeneration() {
        val bitmap = currentBitmap ?: run {
            statusText.text = "请先选择图片"
            return
        }
        if (busy) return
        val useGpu = acceleratorGroup.checkedRadioButtonId == R.id.radioGpu
        val modelType = if (modelGroup.checkedRadioButtonId == R.id.radioInt8) {
            DepthEstimator.ModelType.INT8
        } else {
            DepthEstimator.ModelType.FP32
        }

        busy = true
        runButton.isEnabled = false
        statusText.text = "推理中（${modelType.label} / ${if (useGpu) "GPU" else "CPU"}）..."

        lifecycleScope.launch {
            try {
                val (result, bench) = withContext(Dispatchers.Default) {
                    val est = estimator ?: DepthEstimator(this@MainActivity).also { estimator = it }
                    if (!est.ensureLoaded(modelType, useGpu)) {
                        error("模型加载失败（GPU 不受支持时请改用 CPU）")
                    }
                    val depth = est.infer(bitmap)
                    val b = DepthBenchmark.run(est, bitmap)
                    depth to b
                }
                // 深度归一化 → 单通道 8bit → 交给 GL 渲染 SBS / 伪彩深度
                val r8 = SbsRenderer.depthToR8(DepthVisualizer.normalize(result.depth))
                renderer.submitFrame(bitmap, r8, result.width, result.height)
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
            } finally {
                busy = false
                runButton.isEnabled = true
            }
        }
    }
}
