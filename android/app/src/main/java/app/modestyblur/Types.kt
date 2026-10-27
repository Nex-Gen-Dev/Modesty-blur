package app.modestyblur

import android.graphics.Bitmap
import kotlin.math.max
import kotlin.math.min

/** GPU-ready copy of a mask; shared by every shifted copy of the same silhouette. */
class MaskCache {
    @Volatile var alpha: Bitmap? = null
}

/** Binary silhouette (0/1 bytes) covering the rectangle [x0, x0+w) x [y0, y0+h) of the frame. */
class Mask(
    val x0: Int, val y0: Int, val w: Int, val h: Int, val data: ByteArray,
    val cache: MaskCache = MaskCache(),
)

class Det(
    val x1: Float, val y1: Float, val x2: Float, val y2: Float,
    val score: Float, val mask: Mask,
)

class Track(val id: Int, var det: Det) {
    var lost: Int = 0
    /** Wall-clock time (ms) of the last frame this person was actually detected. */
    var lastSeenMs: Long = 0L
    /** Walking speed of the person in the scene, pixels per millisecond (camera motion removed). */
    var vx: Float = 0f
    var vy: Float = 0f
}

/** The same detection moved by (dx, dy) pixels: used to follow the scene when the camera pans. */
fun Det.shifted(dx: Int, dy: Int): Det =
    if (dx == 0 && dy == 0) this
    else Det(x1 + dx, y1 + dy, x2 + dx, y2 + dy, score, Mask(mask.x0 + dx, mask.y0 + dy, mask.w, mask.h, mask.data, mask.cache))

fun detIou(a: Det, b: Det): Float {
    val ix = max(0f, min(a.x2, b.x2) - max(a.x1, b.x1))
    val iy = max(0f, min(a.y2, b.y2) - max(a.y1, b.y1))
    val inter = ix * iy
    val ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return if (ua <= 0f) 0f else inter / ua
}

/** Crop a (possibly partly off-screen) mask to the frame. Returns null when nothing is left on screen. */
fun Mask.clippedTo(W: Int, H: Int): Mask? {
    if (w <= 0 || h <= 0) return null
    val cx0 = max(x0, 0)
    val cy0 = max(y0, 0)
    val cx1 = min(x0 + w, W)
    val cy1 = min(y0 + h, H)
    if (cx1 <= cx0 || cy1 <= cy0) return null
    if (cx0 == x0 && cy0 == y0 && cx1 == x0 + w && cy1 == y0 + h) return this
    val nw = cx1 - cx0
    val nh = cy1 - cy0
    val out = ByteArray(nw * nh)
    for (y in 0 until nh) {
        System.arraycopy(data, (cy0 - y0 + y) * w + (cx0 - x0), out, y * nw, nw)
    }
    return Mask(cx0, cy0, nw, nh, out)
}
