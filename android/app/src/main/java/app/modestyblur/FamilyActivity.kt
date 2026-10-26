package app.modestyblur

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.text.InputType
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity

/**
 * Enroll family members from photos or from a short video of them turning their head.
 * Only the numeric face data is kept on this phone; the photos and video are never stored or uploaded.
 * Only add people who agreed to it.
 *
 * Recognition needs a good spread of samples, not perfect photos: the faces do not need to smile or look
 * at the camera. Each photo is stored four ways (as is, mirrored, tilted left, tilted right), and a short
 * video adds many different angles at once.
 */
class FamilyActivity : AppCompatActivity() {
    private lateinit var list: LinearLayout
    private lateinit var status: TextView
    private lateinit var photoBtn: Button
    private lateinit var videoBtn: Button
    private var engine: FaceEngine? = null
    private var pendingName = ""

    private val pad get() = (16 * resources.displayMetrics.density).toInt()

    private val photoPicker = registerForActivityResult(ActivityResultContracts.GetMultipleContents()) { uris ->
        if (uris.isNotEmpty() && pendingName.isNotBlank()) enrollPhotos(pendingName, uris)
    }
    private val videoPicker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null && pendingName.isNotBlank()) enrollVideo(pendingName, uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        FamilyStore.load(this)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad * 2)
        }
        root.addView(TextView(this).apply {
            text = "Family faces"
            textSize = 22f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
        })
        root.addView(TextView(this).apply {
            text = "People added here are shown normally instead of blurred, but only when their face is visible " +
                "and recognized. Best: record a 10 second video of the person slowly turning their head " +
                "left and right (smiling or not, it does not matter), or pick 4 or more photos from " +
                "different angles. Photos and video are not stored, only numeric face data on this phone. " +
                "Only add people who agreed."
            textSize = 13f
            alpha = 0.75f
            setPadding(0, pad / 2, 0, pad)
        })
        videoBtn = Button(this).apply {
            text = "Add from a short video"
            setOnClickListener { askName(true) }
        }
        photoBtn = Button(this).apply {
            text = "Add from photos"
            setOnClickListener { askName(false) }
        }
        root.addView(videoBtn)
        root.addView(photoBtn)
        status = TextView(this).apply { setPadding(0, pad / 2, 0, pad / 2) }
        root.addView(status)
        list = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        root.addView(list)
        setContentView(ScrollView(this).apply { addView(root) })

        // load the face models once, off the main thread
        setButtons(false)
        status.text = "Loading face models..."
        Thread {
            val e = FaceEngine.create(applicationContext)
            runOnUiThread {
                engine = e
                if (e == null) {
                    status.text = "Face models are not included in this build."
                } else {
                    status.text = ""
                    setButtons(true)
                }
            }
        }.start()
        refreshList()
    }

    private fun setButtons(on: Boolean) {
        photoBtn.isEnabled = on
        videoBtn.isEnabled = on
    }

    private fun refreshList() {
        list.removeAllViews()
        val people = FamilyStore.list()
        if (people.isEmpty()) {
            list.addView(TextView(this).apply { text = "Nobody enrolled yet."; setPadding(0, pad, 0, 0) })
            return
        }
        for ((name, count) in people) {
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                setPadding(0, pad / 2, 0, pad / 2)
            }
            row.addView(TextView(this).apply {
                text = "$name  ($count face sample${if (count == 1) "" else "s"})"
                textSize = 16f
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            })
            row.addView(Button(this).apply {
                text = "Remove"
                setOnClickListener {
                    AlertDialog.Builder(this@FamilyActivity)
                        .setTitle("Remove $name?")
                        .setPositiveButton("Remove") { _, _ ->
                            FamilyStore.remove(this@FamilyActivity, name)
                            refreshList()
                        }
                        .setNegativeButton("Cancel", null)
                        .show()
                }
            })
            list.addView(row)
        }
    }

    private fun askName(video: Boolean) {
        val input = EditText(this).apply {
            hint = "Name (for example Dad)"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_WORDS
        }
        AlertDialog.Builder(this)
            .setTitle("Who is this?")
            .setMessage("Use the same name again to add more samples for the same person.")
            .setView(input)
            .setPositiveButton(if (video) "Choose video" else "Choose photos") { _, _ ->
                val n = input.text.toString().trim()
                if (n.isNotEmpty()) {
                    pendingName = n
                    if (video) videoPicker.launch("video/*") else photoPicker.launch("image/*")
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun finish(name: String, embs: List<FloatArray>, note: String) {
        if (embs.isNotEmpty()) FamilyStore.add(applicationContext, name, embs)
        runOnUiThread {
            status.text = if (embs.isEmpty()) "No usable face found. Try clearer footage with the face in view."
            else "Added ${embs.size} face samples for $name. $note"
            setButtons(true)
            refreshList()
        }
    }

    private fun enrollPhotos(name: String, uris: List<Uri>) {
        val fe = engine ?: return
        setButtons(false)
        Thread {
            val embs = ArrayList<FloatArray>()
            var skipped = 0
            for ((i, uri) in uris.withIndex()) {
                runOnUiThread { status.text = "Reading photo ${i + 1} of ${uris.size}..." }
                val bmp = ImageUtil.decode(applicationContext, uri, 1800)
                if (bmp == null) { skipped++; continue }
                val base = try { fe.embedLargest(bmp, 640, 30f) } catch (e: Throwable) { null }
                if (base == null) {
                    skipped++
                } else {
                    embs.add(base)
                    // mirrored and slightly tilted copies make matching more forgiving of pose
                    for (v in listOf(ImageUtil.flipH(bmp), ImageUtil.rotate(bmp, -10f), ImageUtil.rotate(bmp, 10f))) {
                        try { fe.embedLargest(v, 640, 30f)?.let { embs.add(it) } } catch (e: Throwable) {}
                        v.recycle()
                    }
                }
                bmp.recycle()
            }
            finish(name, embs, if (skipped > 0) "$skipped photo(s) had no clear face and were skipped." else "")
        }.start()
    }

    private fun enrollVideo(name: String, uri: Uri) {
        val fe = engine ?: return
        setButtons(false)
        Thread {
            val kept = ArrayList<FloatArray>()
            val mmr = MediaMetadataRetriever()
            try {
                mmr.setDataSource(applicationContext, uri)
                val durMs = mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
                val frames = 30
                for (i in 0 until frames) {
                    runOnUiThread { status.text = "Reading video frame ${i + 1} of $frames..." }
                    val tUs = durMs * 1000L * i / frames
                    val raw = mmr.getFrameAtTime(tUs, MediaMetadataRetriever.OPTION_CLOSEST) ?: continue
                    val bmp = ImageUtil.limit(raw, 1280)
                    val emb = try { fe.embedLargest(bmp, 640, 40f) } catch (e: Throwable) { null }
                    if (emb != null && kept.size < 16 && kept.none { dot(it, emb) > 0.93f }) {
                        kept.add(emb)      // keep only samples that add a new angle or expression
                    }
                    if (bmp !== raw) bmp.recycle()
                    raw.recycle()
                }
            } catch (e: Exception) {
                runOnUiThread { status.text = "Could not read that video." }
            } finally {
                try { mmr.release() } catch (e: Exception) {}
            }
            finish(name, kept, "Kept only samples that showed a different angle.")
        }.start()
    }

    private fun dot(a: FloatArray, b: FloatArray): Float {
        var s = 0f
        for (i in a.indices) s += a[i] * b[i]
        return s
    }
}
