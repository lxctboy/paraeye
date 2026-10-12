package com.paraeye.app.gl

import android.graphics.Bitmap
import android.opengl.GLES20
import android.opengl.GLUtils
import android.opengl.GLSurfaceView
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * ParaEye P2：GPU 视图合成渲染器（OpenGL ES 2.0，兼容性优先）。
 *
 * 输入：源 RGB 纹理 + 深度纹理（单通道 8bit，行优先，row0 = 图像顶部）
 * 输出：
 *  - MODE_DEPTH：伪彩深度图（红=近，蓝=远，与 DepthVisualizer 一致）
 *  - MODE_SBS ：Full-SBS（左半=左眼，右半=右眼）
 *
 * 符号约定与 P0 `paraeye/p0/synthesize.py` **保持一致**（已数值验证）：
 *   深度 d ∈ [0,1]，1=近；总视差 = d × maxDispRatio（以图像宽度的比例计）
 *   左眼反向采样 I_L(u) = I(u - dis/2)  → 内容右移
 *   右眼反向采样 I_R(u) = I(u + dis/2)  → 内容左移
 *   ⇒ 近景左眼图在右、右眼图在左 = 出屏（平行眼观看时近物浮现）
 * 由于位移以「图像宽度比例」表达，转 UV 后与分辨率无关，故 Shader 内只需一个标量 shift。
 */
class SbsRenderer : GLSurfaceView.Renderer {

    companion object {
        const val MODE_DEPTH = 0
        const val MODE_SBS = 1

        /** min-max 归一化后的深度打包成单通道 8bit（手机 GPU 上比 float 纹理更稳） */
        fun depthToR8(normalized: FloatArray): ByteArray {
            val out = ByteArray(normalized.size)
            for (i in normalized.indices) {
                val v = (normalized[i].coerceIn(0f, 1f) * 255f + 0.5f).toInt().coerceIn(0, 255)
                out[i] = v.toByte()
            }
            return out
        }

        private const val VERT = """
attribute vec2 aPos;
varying vec2 vUv;
void main() {
    // y 取反：GL 纹理 row0 对应图像顶部，而 NDC y=-1 在屏幕底部
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
    }

    // ---- 以下字段由 UI 线程写入，GL 线程读取 ----
    @Volatile var maxDispRatio = 0.01f   // P0 定档：总视差 1% 图像宽度
    @Volatile var mode: Int = MODE_SBS
    @Volatile var swapLr = false

    @Volatile private var pendingBitmap: Bitmap? = null
    @Volatile private var pendingDepth: ByteArray? = null
    @Volatile private var pendingDepthW = 0
    @Volatile private var pendingDepthH = 0

    private var surfaceW = 1
    private var surfaceH = 1
    private var imgW = 1
    private var imgH = 1

    private var program = 0
    private var colorTex = 0
    private var depthTex = 0
    private var hasColor = false
    private var hasDepth = false

    private val quadBuffer: FloatBuffer = ByteBuffer.allocateDirect(4 * 2 * 4)
        .order(ByteOrder.nativeOrder()).asFloatBuffer().apply {
            put(floatArrayOf(-1f, -1f, 1f, -1f, -1f, 1f, 1f, 1f))
            position(0)
        }

    private var uColor = 0
    private var uDepth = 0
    private var uFit = 0
    private var uShift = 0
    private var uSide = 0
    private var uMode = 0
    private var aPos = 0

    /** 提交一帧结果（占位像素 r8 深度 + 源图），线程安全 */
    fun submitFrame(bitmap: Bitmap, depthR8: ByteArray, depthW: Int, depthH: Int) {
        pendingBitmap = bitmap
        pendingDepth = depthR8
        pendingDepthW = depthW
        pendingDepthH = depthH
    }

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        GLES20.glDisable(GLES20.GL_DEPTH_TEST)
        program = buildProgram(VERT, FRAG)
        aPos = GLES20.glGetAttribLocation(program, "aPos")
        uColor = GLES20.glGetUniformLocation(program, "uColor")
        uDepth = GLES20.glGetUniformLocation(program, "uDepth")
        uFit = GLES20.glGetUniformLocation(program, "uFit")
        uShift = GLES20.glGetUniformLocation(program, "uShift")
        uSide = GLES20.glGetUniformLocation(program, "uSide")
        uMode = GLES20.glGetUniformLocation(program, "uMode")
        colorTex = createTexture()
        depthTex = createTexture()
        hasColor = false
        hasDepth = false
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        surfaceW = width.coerceAtLeast(1)
        surfaceH = height.coerceAtLeast(1)
    }

    override fun onDrawFrame(gl: GL10?) {
        uploadPending()
        GLES20.glViewport(0, 0, surfaceW, surfaceH)
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT)
        if (!hasColor || !hasDepth) return

        GLES20.glUseProgram(program)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, colorTex)
        GLES20.glUniform1i(uColor, 0)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE1)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, depthTex)
        GLES20.glUniform1i(uDepth, 1)

        GLES20.glUniform1f(uShift, maxDispRatio * 0.5f)
        GLES20.glEnableVertexAttribArray(aPos)
        GLES20.glVertexAttribPointer(aPos, 2, GLES20.GL_FLOAT, false, 0, quadBuffer)

        if (mode == MODE_SBS) {
            val half = surfaceW / 2
            // swapLr：交叉眼观看时交换左右顺序
            val leftX = if (swapLr) half else 0
            val rightX = if (swapLr) 0 else half
            drawView(leftX, 0, half, surfaceH, -1f, MODE_SBS)
            drawView(rightX, 0, half, surfaceH, 1f, MODE_SBS)
        } else {
            drawView(0, 0, surfaceW, surfaceH, -1f, MODE_DEPTH)
        }

        GLES20.glDisableVertexAttribArray(aPos)
    }

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
        val imgAspect = imgW.toFloat() / imgH.toFloat()
        return if (viewAspect > imgAspect) {
            Pair(viewAspect / imgAspect, 1f)   // 视口更宽 → 高度撑满
        } else {
            Pair(1f, imgAspect / viewAspect)   // 视口更高 → 宽度撑满
        }
    }

    private fun uploadPending() {
        pendingBitmap?.let { bmp ->
            imgW = bmp.width
            imgH = bmp.height
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, colorTex)
            GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bmp, 0)
            hasColor = true
            pendingBitmap = null
        }
        pendingDepth?.let { buf ->
            val w = pendingDepthW
            val h = pendingDepthH
            val data = ByteBuffer.allocateDirect(buf.size).order(ByteOrder.nativeOrder()).apply {
                put(buf); position(0)
            }
            GLES20.glPixelStorei(GLES20.GL_UNPACK_ALIGNMENT, 1) // 宽度非 4 倍数时必须设为 1
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, depthTex)
            GLES20.glTexImage2D(
                GLES20.GL_TEXTURE_2D, 0, GLES20.GL_LUMINANCE,
                w, h, 0, GLES20.GL_LUMINANCE, GLES20.GL_UNSIGNED_BYTE, data
            )
            hasDepth = true
            pendingDepth = null
        }
    }

    private fun createTexture(): Int {
        val ids = IntArray(1)
        GLES20.glGenTextures(1, ids, 0)
        val id = ids[0]
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, id)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        // 边缘采样等价 P0 的 BORDER_REPLICATE
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        return id
    }

    private fun compile(type: Int, src: String): Int {
        val shader = GLES20.glCreateShader(type)
        GLES20.glShaderSource(shader, src)
        GLES20.glCompileShader(shader)
        val status = IntArray(1)
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, status, 0)
        if (status[0] == 0) {
            val log = GLES20.glGetShaderInfoLog(shader)
            GLES20.glDeleteShader(shader)
            error("shader compile failed: $log")
        }
        return shader
    }

    private fun buildProgram(vs: String, fs: String): Int {
        val v = compile(GLES20.GL_VERTEX_SHADER, vs)
        val f = compile(GLES20.GL_FRAGMENT_SHADER, fs)
        val p = GLES20.glCreateProgram()
        GLES20.glAttachShader(p, v)
        GLES20.glAttachShader(p, f)
        GLES20.glLinkProgram(p)
        val status = IntArray(1)
        GLES20.glGetProgramiv(p, GLES20.GL_LINK_STATUS, status, 0)
        if (status[0] == 0) {
            val log = GLES20.glGetProgramInfoLog(p)
            error("program link failed: $log")
        }
        GLES20.glDeleteShader(v)
        GLES20.glDeleteShader(f)
        return p
    }
}
