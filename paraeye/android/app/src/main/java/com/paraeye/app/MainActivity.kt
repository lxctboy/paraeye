package com.paraeye.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.RadioGroup
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.lifecycle.lifecycleScope
import com.paraeye.app.depth.DepthBenchmark
import com.paraeye.app.depth.DepthEstimator
import com.paraeye.app.depth.DepthVisualizer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * ParaEye P1 单帧推理验证：选图 → LiteRT 推理 → 深度图 + 耗时基准。
 * P2 起替换为 GL 视图合成流水线。
 */
class MainActivity : ComponentActivity() {

    private lateinit var statusText: TextView
    private lateinit var statsText: TextView
    private lateinit var sourceImage: ImageView
    private lateinit var depthImage: ImageView
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
        sourceImage = findViewById(R.id.sourceImage)
        depthImage = findViewById(R.id.depthImage)
        acceleratorGroup = findViewById(R.id.acceleratorGroup)
        modelGroup = findViewById(R.id.modelGroup)
        runButton = findViewById(R.id.runButton)

        findViewById<Button>(R.id.pickButton).setOnClickListener {
            pickImage.launch(ActivityResultContracts.PickVisualMedia.Request("image/*"))
        }
        runButton.setOnClickListener { runBenchmark() }

        statusText.text = "选择一张图片，然后运行基准（首次加载模型约需数秒）"
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
            sourceImage.setImageBitmap(currentBitmap)
            statusText.text = "已加载图片 ${currentBitmap!!.width}x${currentBitmap!!.height}"
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

    private fun runBenchmark() {
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
        statusText.text = "加载模型 ${modelType.label} (${if (useGpu) "GPU" else "CPU"}) ..."

        lifecycleScope.launch {
            try {
                val (depth, bench) = withContext(Dispatchers.Default) {
                    val est = estimator ?: DepthEstimator(this@MainActivity).also { estimator = it }
                    val ok = est.ensureLoaded(modelType, useGpu)
                    if (!ok) error("模型加载失败（GPU 不受支持时请改用 CPU）")
                    val result = est.infer(bitmap)
                    val b = DepthBenchmark.run(est, bitmap)
                    result to b
                }
                val bmp = DepthVisualizer.toBitmap(depth.depth, depth.width, depth.height)
                depthImage.setImageBitmap(bmp)
                statsText.text = buildString {
                    appendLine("模型: ${modelType.label}  |  加速: ${if (useGpu) "GPU" else "CPU"}")
                    appendLine("迭代 ${bench.iterations} 次（含预热 3 次）")
                    appendLine("平均: %.1f ms (%.1f FPS)".format(bench.avgMs, bench.fps))
                    appendLine("min: %.1f ms | p50: %.1f ms | max: %.1f ms".format(bench.minMs, bench.p50Ms, bench.maxMs))
                }
                statusText.text = "完成"
            } catch (e: Exception) {
                statusText.text = "失败: ${e.message}"
            } finally {
                busy = false
                runButton.isEnabled = true
            }
        }
    }
}
