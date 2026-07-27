package com.exposurewatch.app.ir

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.exposurewatch.app.databinding.ActivityIrDetectorBinding
import java.util.concurrent.Executors
import kotlin.math.sqrt

/**
 * Points the camera at a suspected sensor in low light and looks for the
 * signature of an active IR emitter: a small bright bloom in an otherwise dark
 * frame, optionally pulsing.
 */
class IrDetectorActivity : AppCompatActivity() {

    private lateinit var b: ActivityIrDetectorBinding
    private val analysisExecutor = Executors.newSingleThreadExecutor()

    private val hotHistory = ArrayDeque<Float>()
    private val HISTORY = 16

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityIrDetectorBinding.inflate(layoutInflater)
        setContentView(b.root)
        b.btnBack.setOnClickListener { finish() }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 9)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 9 && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            b.txtVerdict.text = "Camera permission needed for IR detection"
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(b.previewView.surfaceProvider)
            }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build().also {
                    it.setAnalyzer(analysisExecutor) { proxy -> analyze(proxy) }
                }
            runCatching {
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze(proxy: ImageProxy) {
        try {
            val plane = proxy.planes[0]
            val buffer = plane.buffer
            val rowStride = plane.rowStride
            val pixelStride = plane.pixelStride
            val w = proxy.width
            val h = proxy.height

            var sum = 0L
            var samples = 0
            var hot = 0
            val step = 4
            var y = 0
            while (y < h) {
                var x = 0
                val rowBase = y * rowStride
                while (x < w) {
                    val idx = rowBase + x * pixelStride
                    if (idx < buffer.limit()) {
                        val v = buffer.get(idx).toInt() and 0xFF
                        sum += v
                        samples++
                        if (v >= HOT_LUMA) hot++
                    }
                    x += step
                }
                y += step
            }
            if (samples == 0) return
            val meanLuma = (sum / samples).toInt()
            val hotRatio = hot.toFloat() / samples

            synchronized(hotHistory) {
                hotHistory.addLast(hotRatio)
                while (hotHistory.size > HISTORY) hotHistory.removeFirst()
            }
            val flicker = flickerVariance()
            val score = irScore(meanLuma, hotRatio, flicker)
            val dark = meanLuma < DARK_MEAN

            runOnUiThread {
                b.gauge.setValue(score)
                b.txtMean.text = "Scene brightness: $meanLuma / 255"
                b.txtHot.text = "IR-bright pixels: ${"%.3f".format(hotRatio * 100)}%"
                b.txtFlicker.text = "Flicker index: ${"%.3f".format(flicker)}"
                b.txtVerdict.text = verdict(dark, hotRatio, flicker, score)
            }
        } catch (_: Throwable) {
            // frame format edge cases - skip this frame
        } finally {
            proxy.close()
        }
    }

    private fun flickerVariance(): Float {
        synchronized(hotHistory) {
            if (hotHistory.size < 4) return 0f
            val mean = hotHistory.average().toFloat()
            if (mean <= 0f) return 0f
            var acc = 0f
            for (v in hotHistory) { val d = v - mean; acc += d * d }
            val sd = sqrt(acc / hotHistory.size)
            return (sd / (mean + 1e-6f)).coerceIn(0f, 3f)
        }
    }

    private fun irScore(meanLuma: Int, hotRatio: Float, flicker: Float): Int {
        if (meanLuma >= BRIGHT_SCENE) return 0
        val darkness = ((DARK_MEAN - meanLuma).coerceAtLeast(0)).toFloat() / DARK_MEAN
        val bloom = (hotRatio * 900f).coerceAtMost(60f)
        val flick = (flicker * 25f).coerceAtMost(25f)
        val base = darkness * 15f
        return (bloom + flick + base).toInt().coerceIn(0, 100)
    }

    private fun verdict(dark: Boolean, hotRatio: Float, flicker: Float, score: Int): String {
        if (!dark) return "Scene too bright to isolate IR - try in low light"
        if (hotRatio < 0.0004f) return "No IR-like bloom detected"
        val sb = StringBuilder("IR-like light detected")
        if (flicker > 0.35f) sb.append(" - pulsing/flicker pattern")
        if (score >= 70) sb.append("\nStrong active IR source - likely illuminator or night-vision LED")
        else if (score >= 45) sb.append("\nPossible IR emitter near the camera in view")
        return sb.toString()
    }

    override fun onDestroy() {
        analysisExecutor.shutdown()
        super.onDestroy()
    }

    companion object {
        private const val HOT_LUMA = 235
        private const val DARK_MEAN = 60
        private const val BRIGHT_SCENE = 110
    }
}
