package app.modestyblur

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Size
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var imageView: ImageView
    private lateinit var status: TextView
    private lateinit var banner: View
    private lateinit var bannerText: TextView

    private val executor = Executors.newSingleThreadExecutor()
    @Volatile private var pipeline: Pipeline? = null
    private var lens = CameraSelector.LENS_FACING_BACK
    private var bannerDismissed = false

    private val permission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) startCamera() else status.text = "Camera permission is required"
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        imageView = findViewById(R.id.view)
        status = findViewById(R.id.status)
        banner = findViewById(R.id.banner)
        bannerText = findViewById(R.id.bannerText)

        findViewById<Button>(R.id.btnSettings).setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }
        findViewById<Button>(R.id.btnFlip).setOnClickListener {
            lens = if (lens == CameraSelector.LENS_FACING_BACK)
                CameraSelector.LENS_FACING_FRONT else CameraSelector.LENS_FACING_BACK
            if (pipeline != null) startCamera()
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

        // Load the models off the main thread, then start the camera.
        Thread {
            try {
                pipeline = Pipeline(applicationContext)
                runOnUiThread { ensureCameraPermission() }
            } catch (e: Throwable) {
                runOnUiThread { status.text = "Model load failed: ${e.message}" }
            }
        }.start()

        maybeCheckForUpdates()
    }

    override fun onResume() {
        super.onResume()
        showBannerIfPending()
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
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analysis.setAnalyzer(executor) { proxy ->
                val p = pipeline
                val bmp = proxy.toBitmap()
                val rot = proxy.imageInfo.rotationDegrees
                proxy.close()
                if (p != null) {
                    val m = Matrix()
                    m.postRotate(rot.toFloat())
                    if (lens == CameraSelector.LENS_FACING_FRONT) m.postScale(-1f, 1f)
                    val upright = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
                    val out = p.process(upright)
                    runOnUiThread {
                        status.visibility = View.GONE
                        imageView.setImageBitmap(out)
                    }
                }
            }
            provider.unbindAll()
            provider.bindToLifecycle(
                this, CameraSelector.Builder().requireLensFacing(lens).build(), analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }
}
