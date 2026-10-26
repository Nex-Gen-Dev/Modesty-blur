package app.modestyblur

import android.graphics.Bitmap
import kotlin.math.abs

/**
 * Cheap global motion estimate between consecutive camera frames (a few milliseconds).
 *
 * Compares small grayscale thumbnails over a +-[r] pixel search window and keeps a running total
 * ([gx], [gy]) of how far the scene content has moved in full-frame pixels. Blur sprites are attached to
 * the content, so when you pan the camera they move with it instead of staying where they were.
 */
class MotionEstimator {
    private val tw = 96
    private val th = 170
    private val r = 5
    private var prev: IntArray? = null

    /** Accumulated content motion in full-frame pixels (content moving right/down = positive). */
    var gx = 0f
        private set
    var gy = 0f
        private set

    /** True when the best match sat on the edge of the search window (the camera moved very fast). */
    var tooFast = false
        private set

    fun reset() {
        prev = null
        gx = 0f
        gy = 0f
        tooFast = false
    }

    fun update(bmp: Bitmap) {
        val small = Bitmap.createScaledBitmap(bmp, tw, th, true)
        val px = IntArray(tw * th)
        small.getPixels(px, 0, tw, 0, 0, tw, th)
        if (small !== bmp) small.recycle()
        val cur = IntArray(tw * th) { i ->
            val p = px[i]
            (((p shr 16) and 0xFF) * 3 + ((p shr 8) and 0xFF) * 6 + (p and 0xFF)) / 10
        }
        val old = prev
        prev = cur
        if (old == null) return

        val n = 2 * r + 1
        val sad = FloatArray(n * n)
        var best = Float.MAX_VALUE
        var bdx = 0
        var bdy = 0
        for (dy in -r..r) {
            for (dx in -r..r) {
                var s = 0
                var y = r
                while (y < th - r) {
                    val rowC = y * tw
                    val rowP = (y - dy) * tw - dx
                    var x = r
                    while (x < tw - r) {
                        s += abs(cur[rowC + x] - old[rowP + x])
                        x += 2
                    }
                    y += 2
                }
                val v = s.toFloat()
                sad[(dy + r) * n + (dx + r)] = v
                if (v < best) { best = v; bdx = dx; bdy = dy }
            }
        }

        val sad0 = sad[r * n + r]
        if (sad0 <= best * 1.02f) {          // prefer "no motion" when the evidence is ambiguous
            tooFast = false
            return
        }
        tooFast = abs(bdx) == r || abs(bdy) == r

        var fdx = bdx.toFloat()
        var fdy = bdy.toFloat()
        if (abs(bdx) < r) {
            val a = sad[(bdy + r) * n + (bdx - 1 + r)]
            val c = sad[(bdy + r) * n + (bdx + 1 + r)]
            fdx += subpixel(a, best, c)
        }
        if (abs(bdy) < r) {
            val a = sad[(bdy - 1 + r) * n + (bdx + r)]
            val c = sad[(bdy + 1 + r) * n + (bdx + r)]
            fdy += subpixel(a, best, c)
        }
        gx += fdx * bmp.width / tw
        gy += fdy * bmp.height / th
    }

    /** Parabola through three SAD samples; returns the offset of the minimum in [-0.5, 0.5]. */
    private fun subpixel(a: Float, b: Float, c: Float): Float {
        val d = a - 2f * b + c
        if (d <= 1e-6f) return 0f
        return (0.5f * (a - c) / d).coerceIn(-0.5f, 0.5f)
    }
}
