package app.modestyblur

import kotlin.math.max
import kotlin.math.min

/**
 * Minimal IoU tracker. A new ID is always blurred until classified, so an ID switch
 * can never leak anyone through; lost tracks stay in the list for a while so the
 * pipeline can keep blurring their last silhouette ("ghosts").
 */
class Tracker(private val maxLost: Int = 15, private val minIou: Float = 0.2f) {
    val tracks = ArrayList<Track>()
    private var nextId = 1

    /** Forget everyone (IDs keep counting up so stale background results can never collide). */
    fun clear() {
        tracks.clear()
    }

    /** Returns the tracks matched to a detection in this frame. */
    fun update(dets: List<Det>): List<Track> {
        class Match(val iou: Float, val t: Int, val d: Int)

        val pairs = ArrayList<Match>()
        for (ti in tracks.indices) for (di in dets.indices) {
            val v = iou(tracks[ti].det, dets[di])
            if (v >= minIou) pairs.add(Match(v, ti, di))
        }
        pairs.sortByDescending { it.iou }

        val usedT = BooleanArray(tracks.size)
        val usedD = BooleanArray(dets.size)
        for (p in pairs) {
            if (usedT[p.t] || usedD[p.d]) continue
            usedT[p.t] = true
            usedD[p.d] = true
            tracks[p.t].det = dets[p.d]
            tracks[p.t].lost = 0
        }
        for (ti in tracks.indices) if (!usedT[ti]) tracks[ti].lost++
        tracks.removeAll { it.lost > maxLost }

        for (di in dets.indices) if (!usedD[di]) tracks.add(Track(nextId++, dets[di]))
        return tracks.filter { it.lost == 0 }
    }

    private fun iou(a: Det, b: Det): Float {
        val ix = max(0f, min(a.x2, b.x2) - max(a.x1, b.x1))
        val iy = max(0f, min(a.y2, b.y2) - max(a.y1, b.y1))
        val inter = ix * iy
        val ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
        return if (ua <= 0f) 0f else inter / ua
    }
}
