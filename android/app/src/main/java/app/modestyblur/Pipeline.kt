package app.modestyblur

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log

/** segment -> track -> classify -> blur, with the same fail-closed policy as the desktop version. */
class Pipeline(private val ctx: Context) {
    private val yolo = YoloSeg(ctx)
    private val clip = ClipClassifier(ctx)
    private val tracker = Tracker()
    private val engine = DecisionEngine()
    private var frame = 0
    private var fps = 0f
    private var lastT = System.nanoTime()

    private val holdFrames = 10
    private val strength = 0.14f
    private val dbg = Paint().apply {
        color = Color.YELLOW; textSize = 26f; style = Paint.Style.FILL
    }

    fun process(src: Bitmap): Bitmap {
        return try {
            run(src)
        } catch (e: Throwable) {
            // Fail closed: if anything breaks, show nothing from the camera.
            Log.e("Pipeline", "frame failed", e)
            Bitmap.createBitmap(src.width, src.height, Bitmap.Config.ARGB_8888).also {
                Canvas(it).apply {
                    drawColor(Color.BLACK)
                    drawText("error: " + (e.message ?: e.javaClass.simpleName), 20f, 40f, dbg)
                }
            }
        }
    }

    private fun run(src: Bitmap): Bitmap {
        val W = src.width
        val H = src.height
        engine.blurThreshold = Prefs.blurThreshold(ctx)
        val debug = Prefs.debug(ctx)

        val px = IntArray(W * H)
        src.getPixels(px, 0, W, 0, 0, W, H)

        val active = tracker.update(yolo.detect(src))

        // Classify at most one person per frame (CLIP is the slow step on a phone).
        // Everyone is blurred until their first classification, so this never leaks.
        val next = active
            .filter { engine.needsClassification(it.id, frame) }
            .minWithOrNull(compareBy<Track>({ engine.state(it.id).nObs > 0 }, { engine.state(it.id).lastCls }))
        if (next != null) {
            val crop = crop(src, next.det)
            if (crop != null) {
                engine.update(next.id, clip.predictFemale(crop), frame)
                crop.recycle()
            }
        }

        val activeIds = HashSet<Int>()
        for (t in active) {
            activeIds.add(t.id)
            if (engine.shouldBlur(t.id)) Blur.blurMasked(px, W, H, t.det.mask, strength)
        }
        // Keep blurring recently lost blurred tracks so detector dropouts never flash someone through.
        for (t in tracker.tracks) {
            if (t.lost in 1..holdFrames && engine.shouldBlur(t.id)) {
                Blur.blurMasked(px, W, H, t.det.mask, strength)
            }
            activeIds.add(t.id)
        }
        engine.prune(activeIds)

        val out = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_8888)
        out.setPixels(px, 0, W, 0, 0, W, H)

        frame++
        val now = System.nanoTime()
        val inst = 1e9f / (now - lastT).coerceAtLeast(1)
        fps = if (fps == 0f) inst else 0.9f * fps + 0.1f * inst
        lastT = now

        if (debug) {
            val c = Canvas(out)
            c.drawText("%.1f fps".format(fps), 16f, 34f, dbg)
            for (t in active) {
                val s = engine.state(t.id)
                val tag = if (engine.shouldBlur(t.id)) "blur" else "ok"
                c.drawText("#${t.id} $tag pF=%.2f".format(s.ema), t.det.x1, (t.det.y1 - 6f).coerceAtLeast(30f), dbg)
            }
        }
        return out
    }

    private fun crop(src: Bitmap, d: Det): Bitmap? {
        val pw = ((d.x2 - d.x1) * 0.05f).toInt()
        val ph = ((d.y2 - d.y1) * 0.05f).toInt()
        val x1 = (d.x1.toInt() - pw).coerceIn(0, src.width - 1)
        val y1 = (d.y1.toInt() - ph).coerceIn(0, src.height - 1)
        val x2 = (d.x2.toInt() + pw).coerceIn(x1 + 1, src.width)
        val y2 = (d.y2.toInt() + ph).coerceIn(y1 + 1, src.height)
        if (x2 - x1 < 8 || y2 - y1 < 8) return null
        return Bitmap.createBitmap(src, x1, y1, x2 - x1, y2 - y1)
    }
}
