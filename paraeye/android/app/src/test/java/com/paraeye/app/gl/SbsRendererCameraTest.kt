package com.paraeye.app.gl

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * P4 相机路径：内容尺寸恒取竖屏（CameraX 变换矩阵只旋到「设备自然方向」，
 * 显示方向由消费方负责；ParaEye 固定竖屏观看）。
 */
class SbsRendererCameraTest {

    @Test
    fun cameraContentSize_landscapeBufferBecomesPortrait() {
        // 典型 16:9 预览（Sensor 方向横屏）→ 竖屏内容 1080x1920
        assertEquals(1080 to 1920, SbsRenderer.cameraContentSize(1920, 1080))
        assertEquals(720 to 1280, SbsRenderer.cameraContentSize(1280, 720))
    }

    @Test
    fun cameraContentSize_portraitBufferStays() {
        // 极端机型 buffer 本身就是竖屏 → 原样
        assertEquals(1080 to 1920, SbsRenderer.cameraContentSize(1080, 1920))
    }

    @Test
    fun cameraContentSize_invalidFallsBackToOne() {
        assertEquals(1 to 1, SbsRenderer.cameraContentSize(0, 1080))
        assertEquals(1 to 1, SbsRenderer.cameraContentSize(1920, -1))
    }
}
