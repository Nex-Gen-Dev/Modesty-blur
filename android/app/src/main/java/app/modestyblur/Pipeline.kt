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

/** What the app decided about one person (used by the Test lab). */
class PersonReport(
    val id: Int,
    val blurred: Boolean,
    val decision: String,
    val pFemale: Float,
    val facePFemale: Float,
    val family: String,
)

/** One shared pipeline so the camera screen and the Test lab never load the models twice. */
object PipelineHolder {
    @Volatile private var p: Pipeline? = null

    @Synchronized
    fun get(ctx: Context): Pipeline = p ?: Pipeline(ctx.applicationContext).also { p = it }
}

/**
 * Per frame (fast, on the camera thread): segment -> track -> blur.
 * In the background (never blocks the view): the person classifier and face matching.
 * Fail-closed: anyone not yet cleared by the background models is blurred.
 */
class Pipeline(private val ctx: Context) {
    private val personCls: PersonClassifier? = PersonClassifier.create(ctx)
    private val clip: ClipClassifier? = if (personCls == null) ClipClassifier(ctx) else null
    private val face: FaceEngine? = FaceEngine.create(ctx)
    private val worker = Worker({ b -> personCls?.predictFemale(b) ?: clip!!.predictFemale(b) }, face)
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
    private var lastActiveIds: List<Int> = emptyList()

    /** The Test lab turns this on so the labels are always drawn. */
    @Volatile var forceDebug = false
    @Volatile var lastReport: List<PersonReport> = emptyList()
        private set

    private val motion = MotionEstimator()
    private var appliedX = 0
    private var appliedY = 0
    /** How long a person who just vanished from the detector keeps being blurred (following the scene). */
    private val holdMs = 350L
    private val maxJobsPerFrame = 3
    private val strength = 0.14f
    val classifierName get() = if (personCls != null) "trained model" else "CLIP zero-shot"

    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE }
    private val shadow = Paint().apply { color = Color.argb(150, 0, 0, 0); style = Paint.Style.FILL }
    private val box = Paint().apply { style = Paint.Style.STROKE }

    init {
        FamilyStore.load(ctx)
        engine.steadyClassifyMs = if (personCls != null) 500L else 3000L
    }

    @Synchronized
    fun reset() {
        tracker.clear()
        motion.reset()
        appliedX = 0
        appliedY = 0
        engine.reset()
        inFlight.clear()
        lastActiveIds = emptyList()
        lastReport = emptyList()
    }

    @Synchronized
    fun process(src: Bitmap): Bitmap {
        return try {
            run(src)
        } catch (e: Throwable) {
            // Fail closed: if anything breaks, show nothing from the camera.
            Log.e("Pipeline", "frame failed", e)
            Bitmap.createBitmap(src.width, src.height, Bitmap.Config.ARGB_8888).also {
                val c = Canvas(it)
                c.drawColor(Color.BLACK)
                if (Prefs.debug(ctx) || forceDebug) {
                    text.textSize = src.width * 0.03f
                    c.drawText("error: " + (e.message ?: e.javaClass.simpleName), 20f, 60f, text)
                }
            }
        }
    }

    /**
     * Test lab: run one still photo until the background models have settled, then return the final
     * blurred image with the debug labels drawn. [lastReport] holds the per-person decisions.
     */
    fun runStill(photo: Bitmap, maxMs: Long = 8000): Bitmap {
        reset()
        forceDebug = true
        try {
            val start = SystemClock.elapsedRealtime()
            while (true) {
                process(photo.copy(Bitmap.Config.ARGB_8888, true))
                if (settled() || SystemClock.elapsedRealtime() - start > maxMs) break
                Thread.sleep(25)
            }
            return process(photo.copy(Bitmap.Config.ARGB_8888, true))
        } finally {
            forceDebug = false
        }
    }

    @Synchronized
    private fun settled(): Boolean {
        if (frame < 3 || inFlight.isNotEmpty() || worker.pending() > 0) return false
        return lastActiveIds.all {
            val s = engine.state(it)
            s.nObs >= 2 && (!worker.faceAvailable || s.faceChecks >= 1)
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
        engine.familyThreshold = Prefs.familyThreshold(ctx)
        val debug = Prefs.debug(ctx) || forceDebug

        val bmp = if (src.isMutable) src else src.copy(Bitmap.Config.ARGB_8888, true)
        val W = bmp.width
        val H = bmp.height

        drainResults(now)

        // 0) how far did the scene move since the last frame? (a few ms; lets blur follow a panning camera)
        motion.update(bmp)
        val tx = Math.round(motion.gx)
        val ty = Math.round(motion.gy)
        val mdx = tx - appliedX
        val mdy = ty - appliedY
        appliedX = tx
        appliedY = ty

        // 1) segmentation + tracking (the only heavy step on the frame path)
        val t0 = System.nanoTime()
        val dilate = (min(W, H) * 0.012f).toInt().coerceAtLeast(6)
        val active = tracker.update(model.detect(bmp, Prefs.detectConf(ctx), dilate), mdx, mdy, now)
        lastActiveIds = active.map { it.id }
        val t1 = System.nanoTime()

        // 2) hand at most one job per frame to the background worker (crops come from the clean frame)
        schedule(bmp, active, now)

        // 3) blur everyone not cleared
        for (t in active) {
            if (engine.shouldBlur(t.id)) Blur.blurRegion(bmp, t.det.mask, strength)
        }
        // keep blurring recently lost blurred tracks so detector dropouts never flash someone through
        val alive = HashSet<Int>()
        for (t in tracker.tracks) {
            alive.add(t.id)
            if (t.lost < 1 || now - t.lastSeenMs > holdMs || !engine.shouldBlur(t.id)) continue
            // the ghost has been shifted with the scene; skip it if a live detection already covers it
            if (active.any { detIou(it.det, t.det) > 0.3f }) continue
            val m = t.det.mask.clippedTo(W, H) ?: continue
            Blur.blurRegion(bmp, m, strength)
        }
        engine.prune(alive)
        inFlight.keys.retainAll(alive)
        val t2 = System.nanoTime()

        lastReport = active.map { t ->
            val s = engine.state(t.id)
            PersonReport(t.id, engine.shouldBlur(t.id), statusLabel(s), s.ema, s.facePFemale, s.familyName)
        }

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
                    if (known(r.id)) engine.updateFace(r.id, r.status, r.name, r.sim, r.pFemale, now)
                }
                is Worker.Result.Fail -> inFlight.remove(r.id)
            }
        }
    }

    /** Hand the most useful background jobs (up to [maxJobsPerFrame]) to the worker for this frame. */
    private fun schedule(bmp: Bitmap, active: List<Track>, now: Long) {
        val useFace = worker.faceAvailable
        var started = 0
        // unresolved people first, newest evidence-poor first
        val order = active.sortedWith(
            compareBy<Track>({ !engine.shouldBlur(it.id) }, { engine.state(it.id).nObs > 0 }, { engine.state(it.id).lastClsMs })
        )
        for (t in order) {
            if (started >= maxJobsPerFrame || worker.pending() >= 4) return
            if (inFlight.containsKey(t.id)) continue
            val s = engine.state(t.id)
            val wantFace = useFace && engine.needsFaceCheck(t.id, now)
            val wantCls = engine.needsClassification(t.id, now)
            // the very first look at a new person is the quick face pass (identity + face reading)
            val faceFirst = wantFace && s.faceChecks == 0
            if (faceFirst || (wantFace && !wantCls)) {
                val c = crop(bmp, t.det, 0.55f)          // faces live in the upper part of the person
                if (c != null) {
                    inFlight[t.id] = "face check"
                    worker.faceCheck(t.id, c, engine.familyThreshold)
                    started++
                    continue
                }
            }
            if (wantCls) {
                val c = crop(bmp, t.det, 1f)
                if (c != null) {
                    inFlight[t.id] = "classifying ($classifierName)"
                    worker.classify(t.id, c)
                    started++
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

    private fun statusLabel(s: DecisionEngine.State): String = when {
        s.isFamily -> "FAMILY ${s.familyName}"
        s.cleared -> "CLEAR (not female)"
        s.nObs == 0 -> "BLUR (not classified yet)"
        s.ema >= engine.blurThreshold -> "BLUR (female)"
        else -> "BLUR (unsure)"
    }

    private fun drawDebug(bmp: Bitmap, active: List<Track>, model: YoloSeg) {
        val c = Canvas(bmp)
        val ts = max(bmp.width * 0.024f, 14f)
        text.textSize = ts
        box.strokeWidth = max(2f, ts / 8f)

        for (t in active) {
            val s = engine.state(t.id)
            val color = when {
                s.isFamily -> Color.CYAN
                s.cleared -> Color.GREEN
                s.nObs > 0 && s.ema < engine.blurThreshold -> Color.rgb(255, 160, 0)
                else -> Color.RED
            }
            val action = inFlight[t.id] ?: "idle"
            val l1 = "#${t.id} ${statusLabel(s)}"
            val l2 = "pF=%.2f  n=%d  %s".format(s.ema, s.nObs, action)
            val l3 = if (s.faceChecks > 0) {
                val g = if (s.facePFemale >= 0f) "face F=%.2f".format(s.facePFemale) else "no face"
                "$g  sim=%.2f  hits=%d".format(s.lastSim, s.familyHits)
            } else ""

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
            "people %d   queue %d   classify %d ms   face %d ms   motion %+d,%+d px%s".format(
                active.size, worker.pending(), worker.lastClsMs, worker.lastFaceMs,
                appliedX, appliedY, if (motion.tooFast) " FAST" else ""
            ),
            "${Presets.names[preset]}  model ${model.inW}x${model.inH}  frame ${bmp.width}x${bmp.height}  classifier: $classifierName",
            "blur if pF>=%.2f  detect>=%.2f  family: %s".format(
                engine.blurThreshold, Prefs.detectConf(ctx),
                if (!worker.faceAvailable) "face models missing" else if (famCount == 0) "nobody enrolled" else "$famCount enrolled"
            ),
        )
        val pw = panel.maxOf { text.measureText(it) } + ts * 0.8f
        val ph = ts * 1.25f * panel.size + ts * 0.4f
        c.drawRect(0f, 0f, pw, ph, shadow)
        text.color = Color.WHITE
        for ((i, line) in panel.withIndex()) c.drawText(line, ts * 0.4f, ts * 1.25f * (i + 1), text)
    }
}
