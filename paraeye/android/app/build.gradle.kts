import java.net.URI

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

/**
 * DA V2 Small LiteRT 模型下载。
 *
 * P4（2026-10-13）起改用**自导出**的 280×518 模型，来源是本仓库 export-tflite
 * workflow 发布的滚动 release「models」：社区原版只有 686×518（会把竖屏视频
 * 竖挤 2.35 倍），且自导出版量化方案与社区一致（AEQ dynamic_wi8_afp32）。
 * 模型 27.6MB，不进 Git，构建时按需下载到 assets/models/。
 */
val downloadDepthModels by tasks.registering {
    description = "Download ParaEye's self-exported DA V2 Small LiteRT model"
    val assetsDir = file("src/main/assets/models")
    val models = mapOf(
        "dav2s_wi8_280x518.tflite" to
            "https://github.com/lxctboy/paraeye/releases/download/models/dav2s_wi8_280x518.tflite",
    )
    outputs.dir(assetsDir)
    doLast {
        assetsDir.mkdirs()
        models.forEach { (name, url) ->
            val target = File(assetsDir, name)
            if (!target.exists()) {
                logger.lifecycle("Downloading $name from $url ...")
                URI(url).toURL().openStream().use { input ->
                    target.outputStream().use { output -> input.copyTo(output) }
                }
                logger.lifecycle("Saved ${target.length() / 1024 / 1024} MB -> $target")
            }
        }
    }
}

android {
    namespace = "com.paraeye.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.paraeye.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    // tflite 走 mmap 加载，禁止 aapt 压缩
    androidResources {
        noCompress += "tflite"
    }

    testOptions {
        unitTests.isReturnDefaultValues = true
    }
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.litert)
    implementation(libs.litert.gpu)
    testImplementation(libs.junit)
}

// 构建前确保模型就位（首次 assemble 会自动下载）
tasks.named("preBuild") {
    dependsOn(downloadDepthModels)
}
