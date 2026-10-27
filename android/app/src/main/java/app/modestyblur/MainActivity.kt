package app.modestyblur

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ImageButton
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

/** Clean full-screen view: just the blurred camera feed. All text lives in Settings / the debug overlay. */
class MainActivity : AppCompatActivity() {
    private lateinit var imageView: BlurView
    private lateinit var spinner: View
    private lateinit var controls: View
    private lateinit var banner: View
    private lateinit var bannerText: TextView

    private val executor = Executors.newSingleThreadExecutor()
    @Volatile private var pipeline: Pipeline? = null
    @Volatile private var lens = CameraSelector.LENS_FACING_BACK
    private var runningPreset = -1
    private var bannerDismissed = false

    private val permission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera()
            else Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        imageView = findViewById(R.id.view)
        spinner = findViewById(R.id.spinner)
        controls = findViewById(R.id.controls)
        banner = findViewById(R.id.banner)
        bannerText = findViewById(R.id.bannerText)

        imageView.setOnClickListener { flashControls() }
        findViewById<ImageButton>(R.id.btnSettings).setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }
        findViewById<ImageButton>(R.id.btnFlip).setOnClickListener {
            lens = if (lens == CameraSelector.LENS_FACING_BACK)
                CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK
            if (pipeline != null) startCamera()
            flashControls()
        }
        findViewById<Button>(R.id.btnInstall).setOnClickListener {
            Updater.pending(this)?.let { r ->
                Updater.downloadAndInstall(this, r) { msg -> bannerText.text = msg }
            }
        }
        findViewById<Button>(R.id.btnLater).setOnClickListener {
            bannerDismissed = true       // stays available in Settings
            banner.visibility = View.GONE
        }
        flashControls()

        // Load the models off the main thread, then start the camera.
        Thread {
            try {
                pipeline = PipelineHolder.get(applicationContext)
                runOnUiThread { ensureCameraPermission() }
            } catch (e: Throwable) {
                runOnUiThread {
                    Toast.makeText(this, "Model load failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }.start()

        maybeCheckForUpdates()
    }

    override fun onResume() {
        super.onResume()
        showBannerIfPending()
        // preset changed in Settings -> rebind the camera at the new resolution
        if (pipeline != null && runningPreset != -1 && runningPreset != Prefs.preset(this)) startCamera()
    }

    private fun flashControls() {
        controls.animate().cancel()
        controls.alpha = 1f
        controls.visibility = View.VISIBLE
        controls.animate().alpha(0f).setStartDelay(3000).setDuration(400)
            .withEndAction { controls.visibility = View.INVISIBLE }.start()
    }

    private fun maybeCheckForUpdates() {
        val stale = System.currentTimeMillis() - Prefs.lastCheck(this) > 60 * 60 * 1000L
        if (!Prefs.autoUpdate(this) || !stale) return
        Thread {
            Updater.check(applicationContext)
            runOnUiThread { showBannerIfPending() }
        }.start()
    }

    private fun showBannerIfPending() {
        val r = Updater.pending(this)
        if (r != null && !bannerDismissed) {
            bannerText.text = "Update ${r.name} available"
            banner.visibility = View.VISIBLE
        } else {
            banner.visibility = View.GONE
        }
    }

    private fun ensureCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) startCamera() else permission.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        val preset = Prefs.preset(this)
        runningPreset = preset
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            // always 16:9 (shown in portrait), at the size the chosen preset asks for
            val selector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(
                    ResolutionStrategy(
                        Presets.camSize(preset),
                        ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                    )
                )
                .build()
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(selector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(executor) { proxy ->
                val p = pipeline
                val raw = proxy.toBitmap()
                val rot = proxy.imageInfo.rotationDegrees
                proxy.close()
                if (p != null) {
                    val m = Matrix()
                    m.postRotate(rot.toFloat())
                    if (lens == CameraSelector.LENS_FACING_FRONT) m.postScale(-1f, 1f)
                    val upright = Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
                    val out = p.live(upright)          // returns at once; detection runs on its own thread
                    imageView.show(out)
                    if (spinner.visibility == View.VISIBLE) runOnUiThread { spinner.visibility = View.GONE }
                }
            }
            provider.unbindAll()
            provider.bindToLifecycle(
                this, CameraSelector.Builder().requireLensFacing(lens).build(), analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }
}
