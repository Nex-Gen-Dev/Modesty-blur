package app.modestyblur

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.text.InputType
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.exifinterface.media.ExifInterface
import kotlin.math.max

/**
 * Enroll family members from photos. Only the numeric face data is kept on this phone; the photos
 * themselves are never stored or uploaded. Only add people who agreed to it.
 */
class FamilyActivity : AppCompatActivity() {
    private lateinit var list: LinearLayout
    private lateinit var status: TextView
    private lateinit var addBtn: Button
    private var engine: FaceEngine? = null
    private var pendingName = ""

    private val pad get() = (16 * resources.displayMetrics.density).toInt()

    private val picker = registerForActivityResult(ActivityResultContracts.GetMultipleContents()) { uris ->
        if (uris.isNotEmpty() && pendingName.isNotBlank()) enroll(pendingName, uris)
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
            text = "People added here are shown normally instead of blurred, but only when their face is " +
                "clearly visible and recognized. Pick 3-5 clear photos per person (front-facing, good light, " +
                "ideally different angles). Photos are not stored, only numeric face data on this phone. " +
                "Only add people who agreed."
            textSize = 13f
            alpha = 0.75f
            setPadding(0, pad / 2, 0, pad)
        })
        addBtn = Button(this).apply {
            text = "Add person or photos"
            setOnClickListener { askName() }
        }
        root.addView(addBtn)
        status = TextView(this).apply { setPadding(0, pad / 2, 0, pad / 2) }
        root.addView(status)
        list = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        root.addView(list)
        setContentView(ScrollView(this).apply { addView(root) })

        // load the face models once, off the main thread
        addBtn.isEnabled = false
        status.text = "Loading face models..."
        Thread {
            val e = FaceEngine.create(applicationContext)
            runOnUiThread {
                engine = e
                if (e == null) {
                    status.text = "Face models are not included in this build."
                } else {
                    status.text = ""
                    addBtn.isEnabled = true
                }
            }
        }.start()
        refreshList()
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
                text = "$name  ($count photo${if (count == 1) "" else "s"})"
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

    private fun askName() {
        val input = EditText(this).apply {
            hint = "Name (for example Dad)"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_WORDS
        }
        AlertDialog.Builder(this)
            .setTitle("Who is in the photos?")
            .setMessage("Use the same name again to add more photos for the same person.")
            .setView(input)
            .setPositiveButton("Choose photos") { _, _ ->
                val n = input.text.toString().trim()
                if (n.isNotEmpty()) {
                    pendingName = n
                    picker.launch("image/*")
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun enroll(name: String, uris: List<Uri>) {
        val fe = engine ?: return
        addBtn.isEnabled = false
        Thread {
            val embs = ArrayList<FloatArray>()
            var noFace = 0
            for ((i, uri) in uris.withIndex()) {
                runOnUiThread { status.text = "Reading photo ${i + 1} of ${uris.size}..." }
                val bmp = decode(uri)
                if (bmp == null) { noFace++; continue }
                val emb = try { fe.embedLargest(bmp, 640, 40f) } catch (e: Throwable) { null }
                bmp.recycle()
                if (emb == null) noFace++ else embs.add(emb)
            }
            if (embs.isNotEmpty()) FamilyStore.add(applicationContext, name, embs)
            runOnUiThread {
                status.text = when {
                    embs.isEmpty() -> "No usable face found in those photos. Try clearer, front-facing ones."
                    noFace > 0 -> "Added ${embs.size} photo(s) for $name. $noFace had no clear face and were skipped."
                    else -> "Added ${embs.size} photo(s) for $name."
                }
                addBtn.isEnabled = true
                refreshList()
            }
        }.start()
    }

    /** Decode a picked photo at a sane size and apply its EXIF rotation. */
    private fun decode(uri: Uri): Bitmap? {
        return try {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
            var sample = 1
            while (max(bounds.outWidth, bounds.outHeight) / sample > 1800) sample *= 2
            val opts = BitmapFactory.Options().apply { inSampleSize = sample }
            val raw = contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) }
                ?: return null
            val orientation = contentResolver.openInputStream(uri)?.use {
                ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
            } ?: ExifInterface.ORIENTATION_NORMAL
            val m = Matrix()
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
                ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
                ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
                ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
                ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
                else -> return raw
            }
            Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
        } catch (e: Exception) {
            null
        }
    }
}
