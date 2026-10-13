package com.paraeye.app.gl

import android.content.Context
import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.net.Uri
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.GLUtils
import android.os.Handler
import android.os.Looper
import com.paraeye.app.depth.ImagePreprocessor
import com.paraeye.app.depth.TemporalSmoother
import com.paraeye.app.video.VideoDecoder
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * GL 帧深度消费者：由上层注入。
 *
 * onFrameForDepth 在 **GL 线程** 被调用，rgba 来自渲染器的双缓冲池
 * （P3.5 起不再需要调用方拷贝）；使用完毕后**必须**调用 recycle 归还，
 * 否则池耗尽后续帧会被丢弃。推理在实现方自有线程完成，结果经
 * submitDepthTarget 交回渲染器。
 */
interface FrameDepthConsumer {
    val busy: Boolean
    fun onFrameForDepth(rgba: ByteArray)
    fun recycle(rgba: ByteArray)
}

/**
 * ParaEye P2 + P3：GPU 视图合成渲染器（OpenGL ES 2.0，兼容性优先）。
 *
 * 数据源（二选一）：
 *  - 静态 Bitmap（P2 验证路径）
 *  - 本地视频（P3）：MediaCodec 硬解 → SurfaceTexture(OES) → [归一化 pass] → displayTex
 *
 * 输出：
 *  - MODE_DEPTH：伪彩深度图（红=近，蓝=远）
 *  - MODE_SBS ：Full-SBS（左半=左眼，右半=右眼）
 *
 * 符号约定与 P0 `paraeye/p0/synthesize.py` **保持一致**（已数值验证）：
 *   深度 d ∈ [0,1]，1=近；总视差 = d × maxDispRatio（以图像宽度的比例计）
 *   左眼反向采样 I_L(u) = I(u - dis/2)，右眼 I_R(u) = I(u + dis/2)
 *   ⇒ 近景左眼图在右、右眼图在左 = 出屏（平行眼观看时近物浮现）
 * 位移以「图像宽度比例」表达，转 UV 后与分辨率无关，故 Shader 内只需一个标量 shift。
 *
 * 纹理坐标总约定：**displayTex / depthTex 的行 0 = 图像顶部**（与 GLUtils.texImage2D
 * 上传 Bitmap 的布局一致）。GL 帧缓冲原点在左下，
 * 故所有「渲染到 FBO」的 pass 都要用 aPos.y*0.5+0.5 再翻一次（详见 DOWN_VERT 注释）。
 */
class SbsRenderer : GLSurfaceView.Renderer {

    companion object {
        const val MODE_DEPTH = 0
        const val MODE_SBS = 1

        /** min-max 归一化后的深度打包成单通道 8bit（手机 GPU 上比 float 纹理更稳） */
        fun depthToR8(normalized: FloatArray): ByteArray {
            val out = ByteArray(normalized.size)
            depthToR8Into(normalized, out)
            return out
        }

        /**
         * 相机内容尺寸（P4）：CameraX 的 buffer 是 Sensor 方向（几乎总是横），
         * 而 SurfaceTexture 变换矩阵只把内容旋到「设备自然方向」（竖屏 upright，
         * 见 CameraX Preview 文档：SurfaceTexture 场景用 getTransformMatrix 到
         * natural orientation，显示方向由消费方负责）。ParaEye 的相机观看固定
         * 竖屏内容，故 FBO 恒取竖屏尺寸；横竖屏手持都由 SBS 整对的 letterbox 适配。
         */
        fun cameraContentSize(bufferW: Int, bufferH: Int): Pair<Int, Int> = when {
            bufferW <= 0 || bufferH <= 0 -> 1 to 1
            bufferW > bufferH -> bufferH to bufferW
            else -> bufferW to bufferH
        }

        /** depthToR8 的零分配版本（渲染线程每帧复用） */
        fun depthToR8Into(normalized: FloatArray, out: ByteArray) {
            for (i in normalized.indices) {
                val v = (normalized[i].coerceIn(0f, 1f) * 255f + 0.5f).toInt().coerceIn(0, 255)
                out[i] = v.toByte()
            }
        }

        // ---- 主 pass：视图合成 / 伪彩深度 ----
        // y 取反：GL 纹理 row0 对应图像顶部，而 NDC y=-1 在屏幕底部
        private const val VERT = """
attribute vec2 aPos;
varying vec2 vUv;
void main() {
    vUv = vec2(aPos.x * 0.5 + 0.5, 0.5 - aPos.y * 0.5);
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

        private const val FRAG = """
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
varying vec2 vUv;
uniform sampler2D uColor;
uniform sampler2D uDepth;
uniform vec2 uFit;    // letterbox 适配：视图 UV -> 图像 UV
uniform float uShift; // 单眼位移（图像 UV 单位）= maxDispRatio / 2
uniform float uSide;  // -1.0 = 左眼, +1.0 = 右眼
uniform float uMode;  // 0 = 伪彩深度, 1 = SBS

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec2 uv = (vUv - vec2(0.5)) * uFit + vec2(0.5);
    // letterbox 黑边区域直接涂黑
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    float d = texture2D(uDepth, uv).r;
    if (uMode > 0.5) {
        // 与 P0 synthesize_view 一致：左眼 -dis/2，右眼 +dis/2
        float off = uSide * uShift * d;
        vec3 c = texture2D(uColor, clamp(vec2(uv.x + off, uv.y), 0.0, 1.0)).rgb;
        gl_FragColor = vec4(c, 1.0);
    } else {
        // 伪彩：近(1)=红 hue0°，远(0)=蓝 hue240°
        float hue = 240.0 * (1.0 - d);
        gl_FragColor = vec4(hsv2rgb(vec3(hue / 360.0, 0.85, 1.0)), 1.0);
    }
}
"""

        // ---- 归一化 pass：OES(视频) -> displayTex ----
        // 依据 Grafika 约定：原始 t=1 对应视频顶部（t = aPos.y*0.5+0.5 时画面正立）。
        // 我们要让 displayTex 的 row0（= FBO y=0 = NDC y=-1）是视频顶部，
        // 故 t 取 0.5 - aPos.y*0.5；uFlip=1 时切换为另一方向（部分机型/片源方向相反）。
        private const val OES_VERT = """
attribute vec2 aPos;
uniform mat4 uTexMtx;
uniform float uFlip;
varying vec2 vUv;
void main() {
    float t = mix(0.5 - aPos.y * 0.5, aPos.y * 0.5 + 0.5, uFlip);
    vUv = (uTexMtx * vec4(aPos.x * 0.5 + 0.5, t, 0.0, 1.0)).xy;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

        private const val OES_FRAG = """
#extension GL_OES_EGL_image_external : require
precision mediump float;
uniform samplerExternalOES uSrc;
varying vec2 vUv;
void main() {
    gl_FragColor = vec4(texture2D(uSrc, vUv).rgb, 1.0);
}
"""

        // ---- 降采样 pass：displayTex -> 模型分辨率 FBO(280x518) ----
        // displayTex 的 row0 已是图像顶部（t=0）。glReadPixels 的 row0 是 FBO 底部
        // （NDC y=-1），所以要让 NDC y=-1 采到 t=0 → vUv.y = aPos.y*0.5 + 0.5。
        private const val DOWN_VERT = """
attribute vec2 aPos;
varying vec2 vUv;
void main() {
    vUv = vec2(aPos.x * 0.5 + 0.5, aPos.y * 0.5 + 0.5);
    gl_Position = vec4(aPos, 0.0, 1.0);
}
"""

        private const val DOWN_FRAG = """
precision mediump float;
uniform sampler2D uSrc;
varying vec2 vUv;
void main() {
    gl_FragColor = vec4(texture2D(uSrc, vUv).rgb, 1.0);
}
"""
    }

    // ---- 以下字段由 UI 线程写入，GL 线程读取 ----
    @Volatile var maxDispRatio = 0.01f   // P0 定档：总视差 1% 图像宽度
    @Volatile var mode: Int = MODE_SBS
    @Volatile var swapLr = false
    @Volatile var videoFlipY = false

    /** 深度消费者（Activity 注入） */
    @Volatile var depthConsumer: FrameDepthConsumer? = null

    /** 由 Activity 设置，用于在异步推理完成后请求重绘 */
    var requestRender: (() -> Unit)? = null
    var onVideoError: ((String) -> Unit)? = null
    var onVideoStarted: ((Int, Int, Long) -> Unit)? = null

    /** 相机预览就绪（内容尺寸已定，P4）：用于状态栏提示 */
    var onCameraStarted: ((Int, Int) -> Unit)? = null

    // ---- 相机路径（P4）----
    /** 相机模式：videoST 直接收 CameraX 预览帧，绕过 MediaCodec/VideoDecoder */
    @Volatile var cameraMode = false

    /** 相机 buffer 尺寸（Sensor 方向，来自 SurfaceRequest.resolution），UI 线程写、GL 线程读 */
    @Volatile private var camBufW = 0
    @Volatile private var camBufH = 0

    /** GL 线程在 onSurfaceCreated 发布；相机绑定时从 UI 线程读取 */
    @Volatile var cameraSurfaceTexture: SurfaceTexture? = null
        private set

    /** 暂停（相机路径：停止消费帧，预览停在最后一帧；视频路径同时透传给解码器） */
    @Volatile private var paused = false

    // ---- 统计（UI 线程轮询）----
    @Volatile var videoFps = 0f
    @Volatile var depthFps = 0f

    @Volatile private var pendingBitmap: Bitmap? = null
    @Volatile private var pendingDepth: ByteArray? = null

    /** 推理线程发布的最新归一化深度（引用先写内容后发布，volatile 保证可见性） */
    @Volatile private var depthTarget: FloatArray? = null

    private var surfaceW = 1
    private var surfaceH = 1
    private var srcW = 1
    private var srcH = 1

    // ---- GL 资源 ----
    private var program = 0
    private var oesProgram = 0
    private var downProgram = 0

    private var displayTex = 0
    private var displayFbo = 0
    private var depthTex = 0
    private var modelTex = 0
    private var modelFbo = 0
    private var videoTex = 0
    private var videoST: SurfaceTexture? = null
    private var videoDecoder: VideoDecoder? = null
    private val texMtx = FloatArray(16)
    private val frameAvailable = AtomicBoolean(false)

    private var hasColor = false
    private var hasDepth = false

    private val readBuf: ByteBuffer = ByteBuffer
        .allocateDirect(ImagePreprocessor.MODEL_W * ImagePreprocessor.MODEL_H * 4)
        .order(ByteOrder.nativeOrder())

    /** rgba 双缓冲池：GL 线程取出即填，推理线程用完 recycle 归还 */
    private val rgbaPool = ArrayBlockingQueue<ByteArray>(2).apply {
        repeat(2) { add(ByteArray(ImagePreprocessor.MODEL_W * ImagePreprocessor.MODEL_H * 4)) }
    }

    /** 深度纹理上传：预分配 direct buffer + R8 字节（原每帧分配，P3.5 复用） */
    private val depthUploadBuf: ByteBuffer = ByteBuffer
        .allocateDirect(ImagePreprocessor.MODEL_W * ImagePreprocessor.MODEL_H)
        .order(ByteOrder.nativeOrder())
    private val depthR8 = ByteArray(ImagePreprocessor.MODEL_W * ImagePreprocessor.MODEL_H)

    /** 渲染线程的时域平滑（与 P0 定档 alpha=0.7 等效的时间常数 τ≈138ms） */
    private val smoother = TemporalSmoother(alpha = 0.7f)
    private var lastDrawNs = 0L

    // ---- 统计窗口 ----
    // 窗口取 2s：CPU 推理慢（~1s/帧）时 500ms 窗口会整轮落空，
    // 深度帧率显示在 0.0 / 2.0 之间跳变（截图里看到的 0.0 即此假象）
    private var windowStartMs = 0L
    private var videoFrames = 0
    /** 推理线程自增、GL 线程读取并清零，必须原子操作 */
    private val depthFrames = AtomicInteger(0)
    @Volatile private var videoInfoSent = false

    @Volatile private var pendingStart: Pair<Context, Uri>? = null
    @Volatile private var pendingStop = false

    private var uColor = 0
    private var uDepth = 0
    private var uFit = 0
    private var uShift = 0
    private var uSide = 0
    private var uMode = 0
    private var aPos = 0

    private var aOesPos = 0
    private var uOesSrc = 0
    private var uOesMtx = 0
    private var uOesFlip = 0

    private var aDownPos = 0
    private var uDownSrc = 0

    // ---------------- 对外 API ----------------

    /** 静态图路径（P2）：提交源图 + 深度 */
    fun submitFrame(bitmap: Bitmap, depthR8: ByteArray) {
        pendingBitmap = bitmap
        pendingDepth = depthR8
        depthTarget = null
        smoother.reset()
    }

    /**
     * 视频路径（P3）：推理线程提交归一化深度（[0,1]，尺寸恒为 MODEL_W x MODEL_H）。
     * 线程安全；渲染线程随后以帧率无关的 EMA 逐帧过渡到该目标。
     */
    fun submitDepthTarget(normalized: FloatArray) {
        depthTarget = normalized
        depthFrames.incrementAndGet()
        requestRender?.invoke()
    }

    /** 归还 rgba 缓冲到池（推理线程用完必须调用） */
    fun recycleRgba(buf: ByteArray) {
        rgbaPool.offer(buf)
    }

    fun startVideo(context: Context, uri: Uri) {
        pendingStop = false
        pendingStart = context.applicationContext to uri
    }

    fun stopVideo() {
        pendingStart = null
        pendingStop = true
    }

    // ---- 相机路径（P4）----

    /** 开启相机模式：videoST 直接收 CameraX 预览帧（CameraController 绑定成功后调用） */
    fun startCamera() {
        pendingStart = null
        pendingStop = false
        cameraMode = true
        paused = false
        videoInfoSent = false
        hasColor = false
        hasDepth = false
        depthTarget = null
        smoother.reset()
    }

    /** 关闭相机模式（解绑 CameraX 时调用；复用 pendingStop 清场） */
    fun stopCamera() {
        cameraMode = false
        pendingStop = true
    }

    /** 相机 buffer 尺寸（SurfaceRequest 回调里调用，任意线程安全） */
    fun setCameraBufferSize(w: Int, h: Int) {
        camBufW = w
        camBufH = h
    }

    fun setPaused(paused: Boolean) {
        this.paused = paused
        videoDecoder?.paused = paused
    }

    /** 切换视频 / 换模型时重置过渡状态，避免把上一个场景的深度混进来 */
    fun resetDepthSmoother() {
        smoother.reset()
    }

    val isVideoActive: Boolean get() = videoDecoder != null

    /** GL 线程：上传 UI 线程提交的数据 */
    private fun uploadPending() {
        pendingBitmap?.let { bmp ->
            srcW = bmp.width
            srcH = bmp.height
            ensureDisplayTex(srcW, srcH)
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, displayTex)
            GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bmp, 0)
            hasColor = true
            pendingBitmap = null
        }
        pendingDepth?.let { buf ->
            if (buf.size == depthR8.size) {
                System.arraycopy(buf, 0, depthR8, 0, buf.size)
                uploadDepthR8()
            }
            pendingDepth = null
        }
    }

    /** 上传当前 depthR8 到深度纹理（P3.5：direct buffer 预分配复用） */
    private fun uploadDepthR8() {
        val w = ImagePreprocessor.MODEL_W
        val h = ImagePreprocessor.MODEL_H
        depthUploadBuf.clear()
        depthUploadBuf.put(depthR8)
        depthUploadBuf.position(0)
        // 保险：宽度非 4 的倍数时必须设为 1（280 已是 4 的倍数，但改尺寸时容易踩回这个坑）
        GLES20.glPixelStorei(GLES20.GL_UNPACK_ALIGNMENT, 1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, depthTex)
        GLES20.glTexImage2D(
            GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE,
            w, h, 0, GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, depthUploadBuf
        )
        hasDepth = true
    }

    // ---------------- Renderer ----------------

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)

        program = GlUtil.buildProgram(VERT, FRAG)
        aPos = GLES20.glGetAttribLocation(program, "aPos")
        uColor = GLES20.glGetUniformLocation(program, "uColor")
        uDepth = GLES20.glGetUniformLocation(program, "uDepth")
        uFit = GLES20.glGetUniformLocation(program, "uFit")
        uShift = GLES20.glGetUniformLocation(program, "uShift")
        uSide = GLES20.glGetUniformLocation(program, "uSide")
        uMode = GLES20.glGetUniformLocation(program, "uMode")

        oesProgram = GlUtil.buildProgram(OES_VERT, OES_FRAG)
        aOesPos = GLES20.glGetAttribLocation(oesProgram, "aPos")
        uOesSrc = GLES20.glGetUniformLocation(oesProgram, "uSrc")
        uOesMtx = GLES20.glGetUniformLocation(oesProgram, "uTexMtx")
        uOesFlip = GLES20.glGetUniformLocation(oesProgram, "uFlip")

        downProgram = GlUtil.buildProgram(DOWN_VERT, DOWN_FRAG)
        aDownPos = GLES20.glGetAttribLocation(downProgram, "aPos")
        uDownSrc = GLES20.glGetUniformLocation(downProgram, "uSrc")

        displayTex = GlUtil.createTexture2d()
        depthTex = GlUtil.createTexture2d()

        // 模型分辨率 FBO：承接降采样结果，供 readPixels 读取
        modelTex = GlUtil.allocTexture2d(ImagePreprocessor.MODEL_W, ImagePreprocessor.MODEL_H)
        modelFbo = GlUtil.createFramebuffer(modelTex)

        // OES 纹理 + SurfaceTexture：视频/相机的统一入口
        videoTex = GlUtil.createOesTexture()
        videoST = SurfaceTexture(videoTex).apply {
            setOnFrameAvailableListener({ frameAvailable.set(true) }, Handler(Looper.getMainLooper()))
        }
        // 发布给相机路径：CameraX Preview 直接写这个 SurfaceTexture（P4）
        cameraSurfaceTexture = videoST

        hasColor = false
        hasDepth = false
        smoother.reset()
        lastDrawNs = 0L
        windowStartMs = System.currentTimeMillis()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        surfaceW = width.coerceAtLeast(1)
        surfaceH = height.coerceAtLeast(1)
    }

    override fun onDrawFrame(gl: GL10?) {
        val nowNs = System.nanoTime()
        val dtMs = if (lastDrawNs == 0L) 0f else (nowNs - lastDrawNs) / 1_000_000f
        lastDrawNs = nowNs

        handleVideoCommands()
        val newVideoFrame = consumeVideoFrame()
        uploadPending()

        if (newVideoFrame && hasColor && modelFbo != 0) {
            val consumer = depthConsumer
            if (consumer != null && !consumer.busy) {
                dispatchDepthFrame(consumer)
            }
        }

        // 深度过渡（P3.5 治卡的关键）：推理只有 ~6FPS，若直接把推理结果贴到纹理，
        // 中间渲染帧深度不变，看起来一跳一跳。这里用与 P0 定档 alpha=0.7 等效的
        // 连续时间常数 τ≈138ms，把低帧率目标插值到每个渲染帧。
        val target = depthTarget
        if (target != null) {
            val cur = smoother.advance(target, dtMs.coerceAtMost(100f))
            depthToR8Into(cur, depthR8)
            uploadDepthR8()
        }

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        GLES20.glViewport(0, 0, surfaceW, surfaceH)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        if (!hasColor || !hasDepth) {
            updateStats(false)
            return
        }

        GLES20.glUseProgram(program)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, displayTex)
        GLES20.glUniform1i(uColor, 0)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, depthTex)
        GLES20.glUniform1i(uDepth, 1)

        GLES20.glUniform1f(uShift, maxDispRatio * 0.5f)
        GLES20.glEnableVertexAttribArray(aPos)
        GLES20.glVertexAttribPointer(aPos, 2, GLES20.GL_FLOAT, false, 0, GlUtil.QUAD)

        if (mode == MODE_SBS) {
            // SBS 双眼必须紧邻成「一对」：把 (2×srcW × srcH) 的整对做 contain 适配并居中，
            // 左半给左眼、右半给右眼。绝不能把每只眼各自贴进屏幕半区——
            // 横屏时半区宽高比(~1.1)远大于视频(0.5625)，两眼各自水平居中，
            // 中间会被撑出约 600px 黑带，平行眼融合直接失效
            // （P3.6 真机反馈：全屏中间黑边、看不到 3D 效果）。
            val pairAspect = 2f * srcW / srcH
            val surfAspect = surfaceW.toFloat() / surfaceH.toFloat()
            val pairW: Int
            val pairH: Int
            if (surfAspect > pairAspect) {
                // 屏幕更宽（如横屏看竖屏视频）→ 高度撑满，宽度居中
                pairH = surfaceH
                pairW = (surfaceH * pairAspect).toInt().coerceAtMost(surfaceW)
            } else {
                // 屏幕更高（如竖屏）→ 宽度撑满，高度居中
                pairW = surfaceW
                pairH = (surfaceW / pairAspect).toInt().coerceAtMost(surfaceH)
            }
            val x0 = (surfaceW - pairW) / 2
            val y0 = (surfaceH - pairH) / 2
            val viewW = maxOf(1, pairW / 2)
            // swapLr：交叉眼观看时交换左右顺序
            val leftX = if (swapLr) x0 + viewW else x0
            val rightX = if (swapLr) x0 else x0 + viewW
            drawView(leftX, y0, viewW, pairH, -1f, MODE_SBS)
            drawView(rightX, y0, viewW, pairH, 1f, MODE_SBS)
        } else {
            drawView(0, 0, surfaceW, surfaceH, -1f, MODE_DEPTH)
        }

        GLES20.glDisableVertexAttribArray(aPos)
        updateStats(newVideoFrame)
    }

    // ---------------- 视频路径 ----------------

    private fun handleVideoCommands() {
        if (pendingStop) {
            pendingStop = false
            videoDecoder?.release()
            videoDecoder = null
            videoInfoSent = false
            hasColor = false
            hasDepth = false
            depthTarget = null
            smoother.reset()
        }
        pendingStart?.let { (ctx, uri) ->
            pendingStart = null
            videoDecoder?.release()
            paused = false
            val st = videoST ?: return
            videoInfoSent = false
            videoDecoder = VideoDecoder(
                context = ctx,
                uri = uri,
                surfaceTexture = st,
                onFrame = { frameAvailable.set(true) },
                onError = { msg -> mainPost { onVideoError?.invoke(msg) } },
            ).also { it.start() }
        }
    }

    /** 有新视频帧则归一化到 displayTex，返回是否是新帧 */
    private fun consumeVideoFrame(): Boolean {
        val st = videoST ?: return false
        // 相机暂停：不消费帧，frameAvailable 保持 true，恢复后取最新一帧
        if (paused) return false
        if (!frameAvailable.compareAndSet(true, false)) return false

        runCatching { st.updateTexImage() }.getOrElse { return false }
        st.getTransformMatrix(texMtx)

        val decoder = videoDecoder
        if (decoder != null) {
            val w = decoder.videoWidth
            val h = decoder.videoHeight
            if (w <= 0 || h <= 0) return false
            if (w != srcW || h != srcH || displayFbo == 0) {
                srcW = w
                srcH = h
                ensureDisplayTex(w, h)
            }
        } else if (cameraMode) {
            if (camBufW <= 0 || camBufH <= 0) return false
            // 相机路径：FBO 恒取竖屏内容尺寸（变换矩阵只负责旋到自然方向，
            // 见 cameraContentSize 注释）；内容宽高比 = 预览宽高比，后续管线不变
            val (cw, ch) = cameraContentSize(camBufW, camBufH)
            if (cw != srcW || ch != srcH || displayFbo == 0) {
                srcW = cw
                srcH = ch
                ensureDisplayTex(cw, ch)
            }
        } else {
            return false
        }
        if (displayFbo == 0) return false   // FBO 创建失败（驱动不支持），放弃视频路径
        if (!videoInfoSent) {
            videoInfoSent = true
            if (decoder != null) {
                val dur = decoder.durationUs
                mainPost { onVideoStarted?.invoke(srcW, srcH, dur) }
            } else {
                mainPost { onCameraStarted?.invoke(srcW, srcH) }
            }
        }

        // OES -> displayTex
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, displayFbo)
        GLES20.glViewport(0, 0, srcW, srcH)
        GLES20.glUseProgram(oesProgram)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, videoTex)
        GLES20.glUniform1i(uOesSrc, 0)
        GLES20.glUniformMatrix4fv(uOesMtx, 1, false, texMtx, 0)
        GLES20.glUniform1f(uOesFlip, if (videoFlipY) 1f else 0f)
        GLES20.glEnableVertexAttribArray(aOesPos)
        GLES20.glVertexAttribPointer(aOesPos, 2, GLES20.GL_FLOAT, false, 0, GlUtil.QUAD)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(aOesPos)
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)

        hasColor = true
        return true
    }

    /** displayTex -> 模型分辨率 FBO -> readPixels -> 交给消费者（P3.5：池化零拷贝） */
    private fun dispatchDepthFrame(consumer: FrameDepthConsumer) {
        val buf = rgbaPool.poll() ?: return   // 池空（上帧推理未完成）→ 丢帧，不阻塞 GL
        val w = ImagePreprocessor.MODEL_W
        val h = ImagePreprocessor.MODEL_H
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, modelFbo)
        GLES20.glViewport(0, 0, w, h)
        GLES20.glUseProgram(downProgram)
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, displayTex)
        GLES20.glUniform1i(uDownSrc, 0)
        GLES20.glEnableVertexAttribArray(aDownPos)
        GLES20.glVertexAttribPointer(aDownPos, 2, GLES20.GL_FLOAT, false, 0, GlUtil.QUAD)
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
        GLES20.glDisableVertexAttribArray(aDownPos)

        readBuf.position(0)
        GLES20.glReadPixels(0, 0, w, h, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, readBuf)
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        readBuf.position(0)
        readBuf.get(buf)
        consumer.onFrameForDepth(buf)
    }

    private fun ensureDisplayTex(w: Int, h: Int) {
        if (displayFbo != 0) {
            GLES20.glDeleteFramebuffers(1, intArrayOf(displayFbo), 0)
            displayFbo = 0
        }
        if (displayTex != 0) {
            GLES20.glDeleteTextures(1, intArrayOf(displayTex), 0)
        }
        displayTex = GlUtil.allocTexture2d(w, h)
        displayFbo = GlUtil.createFramebuffer(displayTex)
    }

    // ---------------- 绘制与统计 ----------------

    private fun drawView(x: Int, y: Int, w: Int, h: Int, side: Float, modeNow: Int) {
        if (w <= 0 || h <= 0) return
        GLES20.glViewport(x, y, w, h)
        val (fx, fy) = fitScale(w, h)
        GLES20.glUniform2f(uFit, fx, fy)
        GLES20.glUniform1f(uSide, side)
        GLES20.glUniform1f(uMode, modeNow.toFloat())
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
    }

    /** contain 模式：保持源图宽高比，黑边填充 */
    private fun fitScale(viewW: Int, viewH: Int): Pair<Float, Float> {
        val viewAspect = viewW.toFloat() / viewH.toFloat()
        val imgAspect = srcW.toFloat() / srcH.toFloat()
        return if (viewAspect > imgAspect) {
            Pair(viewAspect / imgAspect, 1f)   // 视口更宽 → 高度撑满
        } else {
            Pair(1f, imgAspect / viewAspect)   // 视口更高 → 宽度撑满
        }
    }

    private fun updateStats(newVideoFrame: Boolean) {
        if (newVideoFrame) videoFrames++
        val now = System.currentTimeMillis()
        val elapsed = now - windowStartMs
        if (elapsed >= 2000) {
            videoFps = videoFrames * 1000f / elapsed
            depthFps = depthFrames.getAndSet(0) * 1000f / elapsed
            windowStartMs = now
            videoFrames = 0
        }
    }

    private fun mainPost(block: () -> Unit) {
        Handler(Looper.getMainLooper()).post(block)
    }

    /** 释放视频相关资源（需在 GL 线程调用，Activity 通过 queueEvent 触发） */
    fun releaseVideo() {
        videoDecoder?.release()
        videoDecoder = null
        videoST?.release()
        videoST = null
    }
}
