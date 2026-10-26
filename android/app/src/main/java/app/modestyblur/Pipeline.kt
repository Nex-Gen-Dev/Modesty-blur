package app.modestyblur

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.SystemClock
import android.util.Log
import kotlin.math.max
import kotlin.math.min

/**
 * Per frame (fast, on the camera thread): segment -> track -> blur.
 * In the background (slow, never blocks the view): CLIP classification and family face matching.
 * Fail-closed: anyone not yet cleared by the background models is blurred.
 */
class Pipeline(private val ctx: Context) {
    private val clip = ClipClassifier(ctx)
    private val face: FaceEngine? = FaceEngine.create(ctx)
    private val worker = Worker(clip, face)
    private val tracker = Tracker()
    private val engine = DecisionEngine()
    private val inFlight = HashMap<Int, String>()

    private var yolo: YoloSeg? = null
    private var preset = -1
    private var frame = 0
    private var fps = 0f
    private var lastStart = System.nanoTime()
    private var segMs = 0f
    private var blurMs = 0f
    private var totalMs = 0f

    private val holdFrames = 10
    private val strength = 0.14f

    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE }
    private val shadow = Paint().apply { color = Color.argb(150, 0, 0, 0); style = Paint.Style.FILL }
    private val box = Paint().apply { style = Paint.Style.STROKE }

    init {
        FamilyStore.load(ctx)
    }

    fun process(src: Bitmap): Bitmap {
        return try {
            run(src)
        } catch (e: Throwable) {
            // Fail closed: if anything breaks, show nothing from the camera.
            Log.e("Pipeline", "frame failed", e)
            Bitmap.createBitmap(src.width, src.height, Bitmap.Config.ARGB_8888).also {
                if (Prefs.debug(ctx)) {
                    Canvas(it).apply {
                        drawColor(Color.BLACK)
                        text.textSize = src.width * 0.03f
                        drawText("error: " + (e.message ?: e.javaClass.simpleName), 20f, 60f, text)
                    }
                } else {
                    Canvas(it).drawColor(Color.BLACK)
                }
            }
        }
    }

    private fun run(src: Bitmap): Bitmap {
        val tStart = System.nanoTime()
        val now = SystemClock.elapsedRealtime()

        val p = Prefs.preset(ctx)
        if (p != preset || yolo == null) {
            yolo?.close()
            yolo = YoloSeg(ctx, Presets.yoloAsset(p))
            preset = p
        }
        val model = yolo!!
        engine.blurThreshold = Prefs.blurThreshold(ctx)
        val famThr = Prefs.familyThreshold(ctx)
        val debug = Prefs.debug(ctx)

        val bmp = if (src.isMutable) src else src.copy(Bitmap.Config.ARGB_8888, true)
        val W = bmp.width
        val H = bmp.height

        drainResults(now)

        // 1) segmentation + tracking (the only heavy step on the frame path)
        val t0 = System.nanoTime()
        val dilate = (min(W, H) * 0.012f).toInt().coerceAtLeast(6)
        val active = tracker.update(model.detect(bmp, 0.30f, dilate))
        val t1 = System.nanoTime()

        // 2) hand at most one job per frame to the background worker (crops come from the clean frame)
        schedule(bmp, active, now, famThr)

        // 3) blur everyone not cleared
        for (t in active) {
            if (engine.shouldBlur(t.id)) Blur.blurRegion(bmp, t.det.mask, strength)
        }
        // keep blurring recently lost blurred tracks so detector dropouts never flash someone through
        val alive = HashSet<Int>()
        for (t in tracker.tracks) {
            alive.add(t.id)
            if (t.lost in 1..holdFrames && engine.shouldBlur(t.id)) {
                Blur.blurRegion(bmp, t.det.mask, strength)
            }
        }
        engine.prune(alive)
        inFlight.keys.retainAll(alive)
        val t2 = System.nanoTime()

        frame++
        val total = (t2 - tStart) / 1e6f
        segMs = smooth(segMs, (t1 - t0) / 1e6f)
        blurMs = smooth(blurMs, (t2 - t1) / 1e6f)
        totalMs = smooth(totalMs, total)
        val inst = 1e9f / (tStart - lastStart).coerceAtLeast(1)
        fps = if (fps == 0f) inst else 0.9f * fps + 0.1f * inst
        lastStart = tStart

        if (debug) drawDebug(bmp, active, model)
        return bmp
    }

    private fun smooth(old: Float, v: Float) = if (old == 0f) v else 0.9f * old + 0.1f * v

    private fun known(id: Int) = tracker.tracks.any { it.id == id }

    private fun drainResults(now: Long) {
        while (true) {
            val r = worker.results.poll() ?: break
            when (r) {
                is Worker.Result.Cls -> {
                    inFlight.remove(r.id)
                    if (known(r.id)) engine.update(r.id, r.p, now)
                }
                is Worker.Result.FaceRes -> {
                    inFlight.remove(r.id)
                    if (known(r.id)) engine.updateFamily(r.id, r.status, r.name, r.sim, now)
                }
                is Worker.Result.Fail -> inFlight.remove(r.id)
            }
        }
    }

    private fun schedule(bmp: Bitmap, active: List<Track>, now: Long, famThr: Float) {
        if (worker.pending() >= 2) return
        val order = active.sortedWith(
            compareBy<Track>({ engine.state(it.id).nObs > 0 }, { engine.state(it.id).lastClsMs })
        )
        val useFace = worker.faceAvailable && !FamilyStore.isEmpty()
        for (t in order) {
            if (inFlight.containsKey(t.id)) continue
            if (engine.needsClassification(t.id, now)) {
                crop(bmp, t.det, 1f)?.let {
                    inFlight[t.id] = "classifying (CLIP)"
                    worker.classify(t.id, it)
                    return
                }
            } else if (useFace && engine.needsFamilyCheck(t.id, now)) {
                crop(bmp, t.det, 0.55f)?.let {          // faces live in the upper part of the person
                    inFlight[t.id] = "face check"
                    worker.faceCheck(t.id, it, famThr)
                    return
                }
            }
        }
    }

    private fun crop(src: Bitmap, d: Det, heightFraction: Float): Bitmap? {
        val pw = ((d.x2 - d.x1) * 0.05f).toInt()
        val ph = ((d.y2 - d.y1) * 0.05f).toInt()
        val x1 = (d.x1.toInt() - pw).coerceIn(0, src.width - 1)
        val y1 = (d.y1.toInt() - ph).coerceIn(0, src.height - 1)
        val x2 = (d.x2.toInt() + pw).coerceIn(x1 + 1, src.width)
        val yEnd = d.y1 + (d.y2 - d.y1) * heightFraction
        val y2 = (yEnd.toInt() + ph).coerceIn(y1 + 1, src.height)
        if (x2 - x1 < 12 || y2 - y1 < 12) return null
        return Bitmap.createBitmap(src, x1, y1, x2 - x1, y2 - y1)
    }

    // ---- debug overlay ----------------------------------------------------------------------------

    private fun drawDebug(bmp: Bitmap, active: List<Track>, model: YoloSeg) {
        val c = Canvas(bmp)
        val ts = max(bmp.width * 0.024f, 14f)
        text.textSize = ts
        box.strokeWidth = max(2f, ts / 8f)

        for (t in active) {
            val s = engine.state(t.id)
            val (label, color) = when {
                s.isFamily -> "FAMILY ${s.familyName}" to Color.CYAN
                s.cleared -> "CLEAR (not female)" to Color.GREEN
                s.nObs == 0 -> "BLUR (not classified yet)" to Color.RED
                s.ema >= engine.blurThreshold -> "BLUR (female)" to Color.RED
                else -> "BLUR (unsure)" to Color.rgb(255, 160, 0)
            }
            val action = inFlight[t.id] ?: "idle"
            val l1 = "#${t.id} $label"
            val l2 = "pF=%.2f  n=%d  %s".format(s.ema, s.nObs, action)
            val l3 = if (s.lastSim > 0f || s.familyHits > 0) "face sim=%.2f hits=%d".format(s.lastSim, s.familyHits) else ""

            box.color = color
            c.drawRect(t.det.x1, t.det.y1, t.det.x2, t.det.y2, box)

            val lines = if (l3.isEmpty()) listOf(l1, l2) else listOf(l1, l2, l3)
            val bw = lines.maxOf { text.measureText(it) } + ts * 0.6f
            val bh = ts * 1.25f * lines.size + ts * 0.3f
            val bx = t.det.x1.coerceIn(0f, max(0f, bmp.width - bw))
            val by = (t.det.y1 - bh).let { if (it < 0f) t.det.y1 else it }
            c.drawRect(bx, by, bx + bw, by + bh, shadow)
            text.color = color
            for ((i, line) in lines.withIndex()) {
                c.drawText(line, bx + ts * 0.3f, by + ts * (1.25f * (i + 1)), text)
            }
        }

        val famCount = FamilyStore.list().size
        val panel = listOf(
            "%.1f fps   frame %.0f ms  (segment %.0f, blur %.0f)".format(fps, totalMs, segMs, blurMs),
            "people %d   background queue %d   clip %d ms   face %d ms".format(
                active.size, worker.pending(), worker.lastClipMs, worker.lastFaceMs
            ),
            "model ${Presets.names[preset]} ${model.inW}x${model.inH}   frame ${bmpSize(active, bmp)}",
            "blur if pF>=%.2f   family: %s".format(
                engine.blurThreshold,
                if (!worker.faceAvailable) "face models missing" else if (famCount == 0) "nobody enrolled" else "$famCount enrolled"
            ),
        )
        val pw = panel.maxOf { text.measureText(it) } + ts * 0.8f
        val ph = ts * 1.25f * panel.size + ts * 0.4f
        c.drawRect(0f, 0f, pw, ph, shadow)
        text.color = Color.WHITE
        for ((i, line) in panel.withIndex()) c.drawText(line, ts * 0.4f, ts * 1.25f * (i + 1), text)
    }

    private fun bmpSize(active: List<Track>, bmp: Bitmap) = "${bmp.width}x${bmp.height}"
}
