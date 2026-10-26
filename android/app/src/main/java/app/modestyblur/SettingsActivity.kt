package app.modestyblur

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.Switch
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class SettingsActivity : AppCompatActivity() {
    private lateinit var updateStatus: TextView
    private lateinit var installBtn: Button
    private lateinit var familySummary: TextView

    private val pad get() = (16 * resources.displayMetrics.density).toInt()

    private fun heading(t: String) = TextView(this).apply {
        text = t
        textSize = 18f
        setTypeface(typeface, android.graphics.Typeface.BOLD)
        setPadding(0, pad, 0, pad / 4)
    }

    private fun note(t: String) = TextView(this).apply {
        text = t
        textSize = 13f
        alpha = 0.7f
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad * 2)
        }

        // ---- speed / quality ----
        root.addView(heading("Speed and quality"))
        val group = RadioGroup(this)
        val current = Prefs.preset(this)
        for (i in Presets.names.indices) {
            group.addView(RadioButton(this).apply {
                id = View.generateViewId()
                text = "${Presets.names[i]}  -  ${Presets.details[i]}"
                isChecked = i == current
                setOnClickListener { Prefs.setPreset(this@SettingsActivity, i) }
            })
        }
        root.addView(group)
        root.addView(note("Takes effect when you go back to the camera. If it feels choppy, pick Fast."))

        // ---- blur sensitivity ----
        root.addView(heading("Blur sensitivity"))
        val thrText = TextView(this)
        thrText.text = "Blur when the woman-probability is at least %.2f".format(Prefs.blurThreshold(this))
        val seek = SeekBar(this).apply {
            max = 55                                            // 0.05 .. 0.60
            progress = ((Prefs.blurThreshold(this@SettingsActivity) - 0.05f) * 100).toInt()
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(s: SeekBar?, p: Int, fromUser: Boolean) {
                    val v = 0.05f + p / 100f
                    Prefs.setBlurThreshold(this@SettingsActivity, v)
                    thrText.text = "Blur when the woman-probability is at least %.2f".format(v)
                }
                override fun onStartTrackingTouch(s: SeekBar?) {}
                override fun onStopTrackingTouch(s: SeekBar?) {}
            })
        }
        root.addView(thrText)
        root.addView(seek)
        root.addView(note("Lower is safer: more people are blurred, including some men."))

        // ---- detection sensitivity ----
        root.addView(heading("Person detection"))
        val detText = TextView(this)
        detText.text = "Treat something as a person at confidence %.2f".format(Prefs.detectConf(this))
        val detSeek = SeekBar(this).apply {
            max = 50                                            // 0.20 .. 0.70
            progress = ((Prefs.detectConf(this@SettingsActivity) - 0.20f) * 100).toInt()
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(s: SeekBar?, p: Int, fromUser: Boolean) {
                    val v = 0.20f + p / 100f
                    Prefs.setDetectConf(this@SettingsActivity, v)
                    detText.text = "Treat something as a person at confidence %.2f".format(v)
                }
                override fun onStartTrackingTouch(s: SeekBar?) {}
                override fun onStopTrackingTouch(s: SeekBar?) {}
            })
        }
        root.addView(detText)
        root.addView(detSeek)
        root.addView(note("Raise it if feet or floor patches get blurred. Lower it if someone partly hidden is missed."))

        // ---- test lab ----
        root.addView(heading("Test lab"))
        root.addView(note("Run photos through the same pipeline and see exactly what it decides, with a miss-rate summary."))
        root.addView(Button(this).apply {
            text = "Open test lab"
            setOnClickListener { startActivity(Intent(this@SettingsActivity, TestActivity::class.java)) }
        })

        // ---- family ----
        root.addView(heading("Family faces"))
        familySummary = TextView(this)
        root.addView(familySummary)
        root.addView(Button(this).apply {
            text = "Manage family faces"
            setOnClickListener { startActivity(Intent(this@SettingsActivity, FamilyActivity::class.java)) }
        })
        val famText = TextView(this)
        famText.text = "Match strictness %.2f".format(Prefs.familyThreshold(this))
        val famSeek = SeekBar(this).apply {
            max = 40                                            // 0.30 .. 0.70
            progress = ((Prefs.familyThreshold(this@SettingsActivity) - 0.30f) * 100).toInt()
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(s: SeekBar?, p: Int, fromUser: Boolean) {
                    val v = 0.30f + p / 100f
                    Prefs.setFamilyThreshold(this@SettingsActivity, v)
                    famText.text = "Match strictness %.2f".format(v)
                }
                override fun onStartTrackingTouch(s: SeekBar?) {}
                override fun onStopTrackingTouch(s: SeekBar?) {}
            })
        }
        root.addView(famText)
        root.addView(famSeek)
        root.addView(note("Higher is stricter: fewer mistaken matches, but family may stay blurred more often."))

        // ---- debug ----
        root.addView(heading("Debug"))
        root.addView(Switch(this).apply {
            text = "Show debug overlay"
            isChecked = Prefs.debug(this@SettingsActivity)
            setOnCheckedChangeListener { _, v -> Prefs.setDebug(this@SettingsActivity, v) }
        })
        root.addView(note("Labels every person with what the app decided and what it is doing now, plus timings."))

        // ---- updates ----
        root.addView(heading("Updates"))
        root.addView(TextView(this).apply { text = "Installed version 0.1.${BuildConfig.BUILD_NUMBER}" })
        root.addView(Switch(this).apply {
            text = "Check for updates automatically"
            isChecked = Prefs.autoUpdate(this@SettingsActivity)
            setOnCheckedChangeListener { _, v -> Prefs.setAutoUpdate(this@SettingsActivity, v) }
        })
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

        setContentView(ScrollView(this).apply { addView(root) })
        refreshPending()
    }

    override fun onResume() {
        super.onResume()
        FamilyStore.load(this)
        val list = FamilyStore.list()
        familySummary.text = if (list.isEmpty()) "Nobody enrolled yet."
        else "Enrolled: " + list.joinToString(", ") { it.first }
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
