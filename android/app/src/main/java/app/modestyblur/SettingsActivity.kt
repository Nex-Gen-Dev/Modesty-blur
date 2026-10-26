package app.modestyblur

import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.Switch
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class SettingsActivity : AppCompatActivity() {
    private lateinit var updateStatus: TextView
    private lateinit var installBtn: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val pad = (16 * resources.displayMetrics.density).toInt()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
        }
        fun label(t: String) = TextView(this).apply { text = t; textSize = 16f; setPadding(0, pad / 2, 0, pad / 4) }

        root.addView(label("Blur sensitivity (lower = safer, blurs more)"))
        val thrText = TextView(this)
        val seek = SeekBar(this).apply {
            max = 55                                            // 0.05 .. 0.60
            progress = ((Prefs.blurThreshold(this@SettingsActivity) - 0.05f) * 100).toInt()
        }
        thrText.text = "threshold %.2f".format(Prefs.blurThreshold(this))
        seek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar?, p: Int, fromUser: Boolean) {
                val v = 0.05f + p / 100f
                Prefs.setBlurThreshold(this@SettingsActivity, v)
                thrText.text = "threshold %.2f".format(v)
            }
            override fun onStartTrackingTouch(s: SeekBar?) {}
            override fun onStopTrackingTouch(s: SeekBar?) {}
        })
        root.addView(thrText)
        root.addView(seek)

        val debug = Switch(this).apply {
            text = "Debug overlay (boxes, probabilities, fps)"
            isChecked = Prefs.debug(this@SettingsActivity)
            setOnCheckedChangeListener { _, v -> Prefs.setDebug(this@SettingsActivity, v) }
        }
        root.addView(debug)

        root.addView(label("Updates"))
        root.addView(TextView(this).apply { text = "Installed version: ${BuildConfig.BUILD_NUMBER} (0.1.${BuildConfig.BUILD_NUMBER})" })
        val auto = Switch(this).apply {
            text = "Check for updates automatically"
            isChecked = Prefs.autoUpdate(this@SettingsActivity)
            setOnCheckedChangeListener { _, v -> Prefs.setAutoUpdate(this@SettingsActivity, v) }
        }
        root.addView(auto)

        updateStatus = TextView(this).apply { setPadding(0, pad / 2, 0, pad / 2) }
        root.addView(updateStatus)

        val checkBtn = Button(this).apply { text = "Check for updates now" }
        installBtn = Button(this).apply { text = "Install update"; visibility = View.GONE }
        root.addView(checkBtn)
        root.addView(installBtn)

        checkBtn.setOnClickListener {
            updateStatus.text = "Checking..."
            Thread {
                val r = Updater.check(applicationContext)
                runOnUiThread {
                    updateStatus.text = if (r != null) "" else "You are up to date (or no release found)."
                    refreshPending()
                }
            }.start()
        }
        installBtn.setOnClickListener {
            Updater.pending(this)?.let { r ->
                Updater.downloadAndInstall(this, r) { msg -> updateStatus.text = msg }
            }
        }
        setContentView(root)
        refreshPending()
    }

    private fun refreshPending() {
        val r = Updater.pending(this)
        if (r != null) {
            updateStatus.text = "Update ${r.name} is available."
            installBtn.visibility = View.VISIBLE
        } else {
            installBtn.visibility = View.GONE
        }
    }
}
