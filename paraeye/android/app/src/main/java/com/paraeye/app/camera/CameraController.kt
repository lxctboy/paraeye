package com.paraeye.app.camera

import android.graphics.SurfaceTexture
import android.util.Log
import android.view.Surface
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.core.ResolutionSelector
import androidx.camera.core.AspectRatioStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.paraeye.app.gl.SbsRenderer

/**
 * ParaEye P4：摄像头实时 3D 的相机绑定层。
 *
 * 设计要点：
 *  - **零拷贝复用视频路径**：CameraX Preview 直接写渲染器的 OES SurfaceTexture，
 *    之后的归一化 / 降采样 / 推理 / SBS 合成与视频路径完全同一条 GL 管线；
 *  - **方向**：CameraX 对 SurfaceTexture 只保证变换矩阵把内容旋到「设备自然方向」
 *    （竖屏 upright），显示方向由消费方负责（见 CameraX Preview 文档）。
 *    ParaEye 固定竖屏观看，故渲染器 FBO 恒取竖屏内容尺寸（SbsRenderer.cameraContentSize），
 *    横竖屏手持都由 SBS 整对 letterbox 适配；
 *  - **16:9 预览**：内容宽高比 9:16 ≈ 模型输入 280×518（0.5409），深度几乎不形变；
 *  - **生命周期**：bindToLifecycle 绑 Activity，息屏/退后台自动停，回来自动恢复；
 *    前后置切换 = unbindAll + 重新 bind。
 */
class CameraController(
    private val activity: ComponentActivity,
    private val renderer: SbsRenderer,
    private val onStatus: (String) -> Unit,
) {
    private var provider: ProcessCameraProvider? = null

    /** 当前 LensFacing（切换后重新 bind） */
    var lensFacing: Int = CameraSelector.LENS_FACING_BACK
        private set

    /** 上一次 bind 使用的 SurfaceTexture（切前后置时复用） */
    private var surfaceTexture: SurfaceTexture? = null

    val isActive: Boolean get() = provider != null

    /**
     * 绑定 CameraX 预览到渲染器的 OES SurfaceTexture。
     * 调用方需已切好 UI 状态（videoMode/播放中/连续渲染）。
     */
    fun start(st: SurfaceTexture) {
        surfaceTexture = st
        renderer.startCamera()
        val future = ProcessCameraProvider.getInstance(activity)
        future.addListener({
            try {
                val p = future.get()
                p.unbindAll()
                val preview = Preview.Builder()
                    .setResolutionSelector(
                        ResolutionSelector.Builder()
                            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO)
                            .build()
                    )
                    .setTargetRotation(currentDisplayRotation())
                    .build()
                preview.setSurfaceProvider { request ->
                    val res = request.resolution
                    renderer.setCameraBufferSize(res.width, res.height)
                    val surface = Surface(st)
                    request.provideSurface(surface, ContextCompat.getMainExecutor(activity)) {
                        surface.release()
                    }
                }
                provider = p
                p.bindToLifecycle(
                    activity,
                    CameraSelector.Builder().requireLensFacing(lensFacing).build(),
                    preview,
                )
            } catch (e: Exception) {
                Log.e(TAG, "bindToLifecycle failed", e)
                renderer.stopCamera()
                provider = null
                onStatus("摄像头启动失败：${e.message}")
            }
        }, ContextCompat.getMainExecutor(activity))
    }

    /** 解绑并退出相机模式 */
    fun stop() {
        provider?.unbindAll()
        provider = null
        renderer.stopCamera()
    }

    /** 切换前/后置（仅在已激活时有意义） */
    fun switchLens() {
        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_BACK) {
            CameraSelector.LENS_FACING_FRONT
        } else {
            CameraSelector.LENS_FACING_BACK
        }
        surfaceTexture?.let { start(it) }
    }

    @Suppress("DEPRECATION")
    private fun currentDisplayRotation(): Int =
        when (activity.windowManager.defaultDisplay.rotation) {
            Surface.ROTATION_90 -> Surface.ROTATION_90
            Surface.ROTATION_180 -> Surface.ROTATION_180
            Surface.ROTATION_270 -> Surface.ROTATION_270
            else -> Surface.ROTATION_0
        }

    private companion object {
        const val TAG = "ParaEyeCamera"
    }
}
