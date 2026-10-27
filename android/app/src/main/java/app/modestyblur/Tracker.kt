package app.modestyblur

/**
 * Minimal IoU tracker with camera-motion compensation. A new ID is always blurred until classified, so an
 * ID switch can never leak anyone through. Lost tracks stay in the list briefly so the pipeline can keep
 * blurring their (motion-compensated) last silhouette ("ghosts").
 *
 * Before matching, every existing track is shifted by the camera motion since the previous frame, so a
 * person keeps the same ID while you pan the phone.
 */
class Tracker(
    private val maxLost: Int = 30,
    private val maxLostMs: Long = 1200L,
    private val minIou: Float = 0.15f,
) {
    val tracks = ArrayList<Track>()
    private var nextId = 1

    /** Forget everyone (IDs keep counting up so stale background results can never collide). */
    fun clear() {
        tracks.clear()
    }

    /** Move every track with the scene (camera panned by dx, dy pixels). */
    fun shift(dx: Int, dy: Int) {
        if (dx == 0 && dy == 0) return
        for (t in tracks) t.det = t.det.shifted(dx, dy)
    }

    /** Returns the tracks matched to a detection in this frame. [dx],[dy]: scene motion since last frame. */
    fun update(dets: List<Det>, dx: Int, dy: Int, now: Long): List<Track> {
        class Match(val score: Float, val t: Int, val d: Int)

        if (dx != 0 || dy != 0) for (t in tracks) t.det = t.det.shifted(dx, dy)

        val pairs = ArrayList<Match>()
        for (ti in tracks.indices) for (di in dets.indices) {
            val v = detIou(tracks[ti].det, dets[di])
            if (v >= minIou) pairs.add(Match(v, ti, di))
        }
        pairs.sortByDescending { it.score }

        val usedT = BooleanArray(tracks.size)
        val usedD = BooleanArray(dets.size)
        for (p in pairs) {
            if (usedT[p.t] || usedD[p.d]) continue
            usedT[p.t] = true
            usedD[p.d] = true
            val tr = tracks[p.t]
            val nd = dets[p.d]
            val dtMs = now - tr.lastSeenMs
            if (dtMs in 1..600) {
                val vx = ((nd.x1 + nd.x2) - (tr.det.x1 + tr.det.x2)) / 2f / dtMs
                val vy = ((nd.y1 + nd.y2) - (tr.det.y1 + tr.det.y2)) / 2f / dtMs
                tr.vx = (0.5f * tr.vx + 0.5f * vx).coerceIn(-1.5f, 1.5f)
                tr.vy = (0.5f * tr.vy + 0.5f * vy).coerceIn(-1.5f, 1.5f)
            }
            tr.det = nd
            tr.lost = 0
            tr.lastSeenMs = now
        }
        for (ti in tracks.indices) if (!usedT[ti]) tracks[ti].lost++
        tracks.removeAll { it.lost > maxLost || now - it.lastSeenMs > maxLostMs }

        for (di in dets.indices) if (!usedD[di]) tracks.add(Track(nextId++, dets[di]).also { it.lastSeenMs = now })
        return tracks.filter { it.lost == 0 }
    }
}
