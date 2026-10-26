package app.modestyblur

import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity

/**
 * Test lab: pick photos from the gallery, run them through the real pipeline and see what it decides.
 * Tell it what the photos show (women or men) and it reports how many people were blurred or cleared,
 * i.e. your miss rate, without needing anyone to walk in front of the camera.
 */
class TestActivity : AppCompatActivity() {
    private lateinit var results: LinearLayout
    private lateinit var summary: TextView
    private lateinit var pick: Button
    private lateinit var group: RadioGroup
    private var expectId = 0     // 0 women (expect blur), 1 men (expect clear), 2 just look

    private val pad get() = (16 * resources.displayMetrics.density).toInt()

    private val picker = registerForActivityResult(ActivityResultContracts.GetMultipleContents()) { uris ->
        if (uris.isNotEmpty()) runTests(uris)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad * 2)
        }
        root.addView(TextView(this).apply {
            text = "Test lab"
            textSize = 22f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
        })
        root.addView(TextView(this).apply {
            text = "Choose photos, tell the app what they show, and read the result. Photos stay on this phone " +
                "and are not saved. Tip: use photos of one kind at a time so the miss rate means something."
            textSize = 13f
            alpha = 0.75f
            setPadding(0, pad / 2, 0, pad / 2)
        })

        group = RadioGroup(this)
        val labels = listOf(
            "These show women  (they should be BLURRED)",
            "These show men  (they should be CLEAR)",
            "Just show me what it decides",
        )
        val ids = ArrayList<Int>()
        for ((i, l) in labels.withIndex()) {
            val rb = RadioButton(this).apply {
                id = View.generateViewId()
                text = l
                isChecked = i == 0
            }
            ids.add(rb.id)
            group.addView(rb)
        }
        group.setOnCheckedChangeListener { _, checked -> expectId = ids.indexOf(checked).coerceAtLeast(0) }
        root.addView(group)

        pick = Button(this).apply {
            text = "Choose photos"
            setOnClickListener { picker.launch("image/*") }
        }
        root.addView(pick)

        summary = TextView(this).apply {
            textSize = 16f
            setPadding(0, pad / 2, 0, pad / 2)
        }
        root.addView(summary)
        results = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        root.addView(results)
        setContentView(ScrollView(this).apply { addView(root) })
    }

    private fun runTests(uris: List<Uri>) {
        pick.isEnabled = false
        results.removeAllViews()
        summary.text = "Loading models..."
        val expect = expectId
        Thread {
            val pipeline = try {
                PipelineHolder.get(applicationContext)
            } catch (e: Throwable) {
                runOnUiThread { summary.text = "Model load failed: ${e.message}"; pick.isEnabled = true }
                return@Thread
            }
            var people = 0
            var blurred = 0
            var cleared = 0
            var family = 0
            var noPerson = 0
            for ((i, uri) in uris.withIndex()) {
                runOnUiThread { summary.text = "Testing photo ${i + 1} of ${uris.size}..." }
                val photo = ImageUtil.decode(applicationContext, uri, 1280)
                if (photo == null) {
                    runOnUiThread { addResult(null, "Could not read photo ${i + 1}") }
                    continue
                }
                val out = pipeline.runStill(photo)
                val report = pipeline.lastReport
                if (report.isEmpty()) noPerson++
                for (r in report) {
                    people++
                    when {
                        r.family.isNotEmpty() -> family++
                        r.blurred -> blurred++
                        else -> cleared++
                    }
                }
                val lines = if (report.isEmpty()) "Photo ${i + 1}: no person found (nothing blurred)"
                else "Photo ${i + 1}: " + report.joinToString("   |   ") {
                    "#${it.id} ${it.decision}  pF=%.2f".format(it.pFemale) +
                        if (it.facePFemale >= 0f) "  faceF=%.2f".format(it.facePFemale) else ""
                }
                runOnUiThread { addResult(out, lines) }
                photo.recycle()
            }
            val text = buildString {
                append("$people people found in ${uris.size} photo(s).  Blurred $blurred, cleared $cleared")
                if (family > 0) append(", family $family")
                append(".")
                if (noPerson > 0) append("  $noPerson photo(s) had no person detected.")
                val judged = blurred + cleared
                when (expect) {
                    0 -> if (judged > 0) append("\nMISSED women (shown unblurred): $cleared of $judged = %.0f%%  (aim for 0%%)".format(100.0 * cleared / judged))
                    1 -> if (judged > 0) append("\nMen wrongly blurred: $blurred of $judged = %.0f%%  (this is the price of staying safe)".format(100.0 * blurred / judged))
                    else -> {}
                }
            }
            runOnUiThread {
                summary.text = text
                pick.isEnabled = true
            }
        }.start()
    }

    private fun addResult(img: Bitmap?, caption: String) {
        if (img != null) {
            results.addView(ImageView(this).apply {
                setImageBitmap(img)
                adjustViewBounds = true
                setPadding(0, pad / 2, 0, 0)
            })
        }
        results.addView(TextView(this).apply {
            text = caption
            textSize = 13f
            setPadding(0, pad / 4, 0, pad / 2)
        })
    }
}
