package app.modestyblur

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.SystemClock
import android.util.Log
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

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
 * Two ways to run, sharing the same tracker and decision engine:
 *
 *  - [live] (camera screen): the screen is drawn at camera speed. Detection runs on its own thread every
 *    ~100 ms; between detections each blur follows the camera's motion and the person's walking speed.
 *    The blur itself is drawn by the GPU (see [BlurView]).
 *  - [process] / [runStill] (Test lab): detect, then blur on the CPU, one photo at a time.
 *
 * Classification and face matching always run on a background worker. Fail-closed: anyone not yet cleared by
 * the background models is blurred, and any error shows a black screen instead of the camera.
 */
class Pipeline(private val ctx: Context) {
    private val personCls: PersonClassifier? = PersonClassifier.create(ctx)
    private val clip: ClipClassifier? = if (personCls == null) ClipClassifier(ctx) else null
    private val face: FaceEngine? = FaceEngine.create(ctx)
    private val worker = Worker({ b -> personCls?.predictFemale(b) ?: clip!!.predictFemale(b) }, face)
    private val tracker = Tracker()
    private val engine = DecisionEngine()
    private val inFlight = HashMap<Int, String>()

    private val modelLock = Any()
    @Volatile private var yolo: YoloSeg? = null
    private var preset = -1
    private var frame = 0
    private var fps = 0f
    private var lastStart = System.nanoTime()
    private var segMs = 0f
    private var blurMs = 0f
    private var totalMs = 0f
    private var lastActiveIds: List<Int> = emptyList()

    // live-mode state
    private val detExec = Executors.newSingleThreadExecutor()
    @Volatile private var detBusy = false
    private var detMs = 0f
    private var detGapMs = 0f
    private var lastDetEnd = 0L

    /** The Test lab turns this on so the labels are always drawn. */
    @Volatile var forceDebug = false
    @Volatile var lastReport: List<PersonReport> = emptyList()
        private set

    private val motion = MotionEstimator()
    private var appliedX = 0
    private var appliedY = 0
    /** How long a person who just vanished from the detector keeps being blurred (following the scene). */
    private val holdMs = 350L
    private val liveHoldMs = 500L
    private val maxJobsPerFrame = 3
    private val strength = 0.14f
    val classifierName get() = if (personCls != null) "trained model" else "CLIP zero-shot"

    init {
        FamilyStore.load(ctx)
        engine.steadyClassifyMs = if (personCls != null) 500L else 3000L
    }

    private fun ensureModel(): YoloSeg = synchronized(modelLock) {
        val p = Prefs.preset(ctx)
        if (p != preset || yolo == null) {
            yolo?.close()
            yolo = YoloSeg(ctx, Presets.yoloAsset(p))
            preset = p
        }
        yolo!!
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

    // =============================================================================================
    //  LIVE PATH (camera screen)
    // =============================================================================================

    private class Pending(val bmp: Bitmap, val capX: Int, val capY: Int, val capMs: Long)

    /** Returns immediately with what to draw for this camera frame. Never waits for detection. */
    fun live(src: Bitmap): LiveFrame {
        return try {
            liveInner(src)
        } catch (e: Throwable) {
            Log.e("Pipeline", "live frame failed", e)
            LiveFrame(null, emptyList(), null)          // fail closed: black screen
        }
    }

    private fun liveInner(src: Bitmap): LiveFrame {
        val tStart = System.nanoTime()
        val now = SystemClock.elapsedRealtime()
        val W = src.width
        val H = src.height

        // frame shrunk to 1/6: the colours for the blur come from here
        val tw = max(8, W / LiveRender.CELL)
        val th = max(8, H / LiveRender.CELL)
        val small = Bitmap.createScaledBitmap(src, tw, th, true)
        val thumb = IntArray(tw * th)
        small.getPixels(thumb, 0, tw, 0, 0, tw, th)
        if (small !== src) small.recycle()
        val cw = W.toFloat() / tw
        val ch = H.toFloat() / th

        motion.update(src)
        val debug = Prefs.debug(ctx)
        val sprites = ArrayList<Sprite>()
        var info: DebugInfo? = null

        synchronized(this) {
            engine.blurThreshold = Prefs.blurThreshold(ctx)
            engine.familyThreshold = Prefs.familyThreshold(ctx)

            val tx = Math.round(motion.gx)
            val ty = Math.round(motion.gy)
            val mdx = tx - appliedX
            val mdy = ty - appliedY
            appliedX = tx
            appliedY = ty
            tracker.shift(mdx, mdy)                 // everyone moves with the scene

            drainResults(now)

            if (!detBusy) {
                detBusy = true
                val job = Pending(src, appliedX, appliedY, now)
                detExec.execute {
                    try {
                        detect(job)
                    } catch (e: Throwable) {
                        Log.e("Pipeline", "detection failed", e)
                    } finally {
                        detBusy = false
                    }
                }
            }

            val active = tracker.tracks.filter { it.lost == 0 }
            for (t in tracker.tracks) {
                if (!engine.shouldBlur(t.id)) continue
                if (t.lost >= 1) {
                    if (now - t.lastSeenMs > liveHoldMs) continue
                    if (active.any { detIou(it.det, t.det) > 0.3f }) continue
                }
                val d = predicted(t, now)
                LiveRender.sprite(d.mask, thumb, tw, th, cw, ch, strength)?.let { sprites.add(it) }
            }

            frame++
            val total = (System.nanoTime() - tStart) / 1e6f
            totalMs = smooth(totalMs, total)
            val inst = 1e9f / (tStart - lastStart).coerceAtLeast(1)
            fps = if (fps == 0f) inst else 0.9f * fps + 0.1f * inst
            lastStart = tStart

            if (debug) info = buildDebug(active, yolo, true)
        }
        return LiveFrame(src, sprites, info)
    }

    /** Where the person probably is right now: last detection plus walking speed times the time since. */
    private fun predicted(t: Track, now: Long): Det {
        val dt = (now - t.lastSeenMs).coerceIn(0L, 300L)
        if (dt == 0L || (t.vx == 0f && t.vy == 0f)) return t.det
        val bw = t.det.x2 - t.det.x1
        val bh = t.det.y2 - t.det.y1
        val ox = (t.vx * dt).coerceIn(-0.2f * bw, 0.2f * bw).roundToInt()
        val oy = (t.vy * dt).coerceIn(-0.2f * bh, 0.2f * bh).roundToInt()
        return t.det.shifted(ox, oy)
    }

    /** Runs on the detection thread. */
    private fun detect(job: Pending) {
        val model = ensureModel()
        val W = job.bmp.width
        val H = job.bmp.height
        val dilate = (min(W, H) * 0.012f).toInt().coerceAtLeast(6)
        val t0 = System.nanoTime()
        val dets = model.detect(job.bmp, Prefs.detectConf(ctx), dilate)
        val ms = (System.nanoTime() - t0) / 1e6f
        val end = SystemClock.elapsedRealtime()

        synchronized(this) {
            // the camera may have moved since this frame was captured: bring the result up to date
            val dx = appliedX - job.capX
            val dy = appliedY - job.capY
            val moved = if (dx != 0 || dy != 0) dets.map { it.shifted(dx, dy) } else dets
            val active = tracker.update(moved, 0, 0, job.capMs)
            lastActiveIds = active.map { it.id }
            schedule(job.bmp, active, end, dx, dy)

            val alive = HashSet<Int>()
            for (t in tracker.tracks) alive.add(t.id)
            engine.prune(alive)
            inFlight.keys.retainAll(alive)

            detMs = smooth(detMs, ms)
            if (lastDetEnd != 0L) detGapMs = smooth(detGapMs, (end - lastDetEnd).toFloat())
            lastDetEnd = end
        }
    }

    // =============================================================================================
    //  BLOCKING PATH (Test lab)
    // =============================================================================================

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
                    val p = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE; textSize = src.width * 0.03f }
                    c.drawText("error: " + (e.message ?: e.javaClass.simpleName), 20f, 60f, p)
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

        val model = ensureModel()
        engine.blurThreshold = Prefs.blurThreshold(ctx)
        engine.familyThreshold = Prefs.familyThreshold(ctx)
        val debug = Prefs.debug(ctx) || forceDebug

        val bmp = if (src.isMutable) src else src.copy(Bitmap.Config.ARGB_8888, true)
        val W = bmp.width
        val H = bmp.height

        drainResults(now)

        motion.update(bmp)
        val tx = Math.round(motion.gx)
        val ty = Math.round(motion.gy)
        val mdx = tx - appliedX
        val mdy = ty - appliedY
        appliedX = tx
        appliedY = ty

        val t0 = System.nanoTime()
        val dilate = (min(W, H) * 0.012f).toInt().coerceAtLeast(6)
        val active = tracker.update(model.detect(bmp, Prefs.detectConf(ctx), dilate), mdx, mdy, now)
        lastActiveIds = active.map { it.id }
        val t1 = System.nanoTime()

        schedule(bmp, active, now, 0, 0)

        for (t in active) {
            if (engine.shouldBlur(t.id)) Blur.blurRegion(bmp, t.det.mask, strength)
        }
        val alive = HashSet<Int>()
        for (t in tracker.tracks) {
            alive.add(t.id)
            if (t.lost < 1 || now - t.lastSeenMs > holdMs || !engine.shouldBlur(t.id)) continue
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

        if (debug) buildDebug(active, model, false).draw(Canvas(bmp), W)
        return bmp
    }

    // =============================================================================================
    //  shared
    // =============================================================================================

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

    /**
     * Hand the most useful background jobs (up to [maxJobsPerFrame]) to the worker. [ox],[oy]: how far the
     * tracks have moved since [bmp] was captured, so crops are cut from the right place in that image.
     */
    private fun schedule(bmp: Bitmap, active: List<Track>, now: Long, ox: Int, oy: Int) {
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
                val c = crop(bmp, t.det, 0.55f, ox, oy)          // faces live in the upper part of the person
                if (c != null) {
                    inFlight[t.id] = "face check"
                    worker.faceCheck(t.id, c, engine.familyThreshold)
                    started++
                    continue
                }
            }
            if (wantCls) {
                val c = crop(bmp, t.det, 1f, ox, oy)
                if (c != null) {
                    inFlight[t.id] = "classifying ($classifierName)"
                    worker.classify(t.id, c)
                    started++
                }
            }
        }
    }

    private fun crop(src: Bitmap, d: Det, heightFraction: Float, ox: Int, oy: Int): Bitmap? {
        val pw = ((d.x2 - d.x1) * 0.05f).toInt()
        val ph = ((d.y2 - d.y1) * 0.05f).toInt()
        val bx1 = d.x1.toInt() - ox
        val by1 = d.y1.toInt() - oy
        val bx2 = d.x2.toInt() - ox
        val x1 = (bx1 - pw).coerceIn(0, src.width - 1)
        val y1 = (by1 - ph).coerceIn(0, src.height - 1)
        val x2 = (bx2 + pw).coerceIn(x1 + 1, src.width)
        val yEnd = (d.y1 - oy) + (d.y2 - d.y1) * heightFraction
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

    private fun buildDebug(active: List<Track>, model: YoloSeg?, live: Boolean): DebugInfo {
        val items = active.map { t ->
            val s = engine.state(t.id)
            val color = when {
                s.isFamily -> Color.CYAN
                s.cleared -> Color.GREEN
                s.nObs > 0 && s.ema < engine.blurThreshold -> Color.rgb(255, 160, 0)
                else -> Color.RED
            }
            val action = inFlight[t.id] ?: "idle"
            val lines = ArrayList<String>()
            lines.add("#${t.id} ${statusLabel(s)}")
            lines.add("pF=%.2f  n=%d  %s".format(s.ema, s.nObs, action))
            if (s.faceChecks > 0) {
                val g = if (s.facePFemale >= 0f) "face F=%.2f".format(s.facePFemale) else "no face"
                lines.add("$g  sim=%.2f  hits=%d".format(s.lastSim, s.familyHits))
            }
            DebugItem(t.det.x1, t.det.y1, t.det.x2, t.det.y2, color, lines)
        }

        val famCount = FamilyStore.list().size
        val panel = ArrayList<String>()
        if (live) {
            panel.add("%.0f fps screen   detect %.0f ms every %.0f ms   frame %.1f ms".format(fps, detMs, detGapMs, totalMs))
        } else {
            panel.add("%.1f fps   frame %.0f ms  (segment %.0f, blur %.0f)".format(fps, totalMs, segMs, blurMs))
        }
        panel.add(
            "people %d   queue %d   classify %d ms   face %d ms   motion %+d,%+d px%s".format(
                active.size, worker.pending(), worker.lastClsMs, worker.lastFaceMs,
                appliedX, appliedY, if (motion.tooFast) " FAST" else ""
            )
        )
        val presetName = if (preset in Presets.names.indices) Presets.names[preset] else "?"
        panel.add(
            "$presetName  model ${model?.inW}x${model?.inH}  runs on ${model?.accel} (${"%.0f".format(model?.accelMs ?: 0f)} ms test)  classifier: $classifierName"
        )
        panel.add(
            "blur if pF>=%.2f  detect>=%.2f  family: %s".format(
                engine.blurThreshold, Prefs.detectConf(ctx),
                if (!worker.faceAvailable) "face models missing" else if (famCount == 0) "nobody enrolled" else "$famCount enrolled"
            )
        )
        return DebugInfo(items, panel)
    }
}
