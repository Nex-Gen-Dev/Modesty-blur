package app.modestyblur

import android.graphics.Bitmap
import kotlin.math.max
import kotlin.math.min

/**
 * Full-silhouette blur using normalized convolution: colors are averaged only from pixels
 * inside the mask, so nothing bleeds in from the background and the result is a smooth,
 * unrecognizable blob in the exact shape of the body.
 *
 * Fast path: the work happens on a coarse grid (about 40 cells across the short side of the
 * person), then the grid is interpolated back up. Cost is roughly one pass over the person's pixels.
 */
object Blur {

    /** Grow a 0/1 mask by r pixels (square structuring element), in place. */
    fun dilate(m: ByteArray, w: Int, h: Int, r: Int) {
        if (r <= 0 || w <= 0 || h <= 0) return
        val tmp = ByteArray(w * h)
        for (y in 0 until h) {
            var last = -1_000_000
            for (x in 0 until w) {
                if (m[y * w + x].toInt() != 0) last = x
                if (x - last <= r) tmp[y * w + x] = 1
            }
            var next = 1_000_000
            for (x in w - 1 downTo 0) {
                if (m[y * w + x].toInt() != 0) next = x
                if (next - x <= r) tmp[y * w + x] = 1
            }
        }
        java.util.Arrays.fill(m, 0)
        for (x in 0 until w) {
            var last = -1_000_000
            for (y in 0 until h) {
                if (tmp[y * w + x].toInt() != 0) last = y
                if (y - last <= r) m[y * w + x] = 1
            }
            var next = 1_000_000
            for (y in h - 1 downTo 0) {
                if (tmp[y * w + x].toInt() != 0) next = y
                if (next - y <= r) m[y * w + x] = 1
            }
        }
    }

    /** Blur the silhouette described by [m] directly on [bmp] (must be mutable). */
    fun blurRegion(bmp: Bitmap, m: Mask, strength: Float) {
        val w = m.w
        val h = m.h
        if (w <= 0 || h <= 0) return
        if (m.x0 < 0 || m.y0 < 0 || m.x0 + w > bmp.width || m.y0 + h > bmp.height) return
        val roi = IntArray(w * h)
        bmp.getPixels(roi, 0, w, m.x0, m.y0, w, h)
        blurRoi(roi, w, h, m.data, strength)
        bmp.setPixels(roi, 0, w, m.x0, m.y0, w, h)
    }

    /** Blur the masked pixels of [roi] (ARGB ints, w x h) in place. */
    fun blurRoi(roi: IntArray, w: Int, h: Int, mask: ByteArray, strength: Float) {
        val d = max(2, min(w, h) / 40)                 // cell size in pixels
        val sw = (w + d - 1) / d
        val sh = (h + d - 1) / d
        val step = max(1, d / 4)                       // sample a subset of pixels per cell
        val sr = FloatArray(sw * sh)
        val sg = FloatArray(sw * sh)
        val sb = FloatArray(sw * sh)
        val sm = FloatArray(sw * sh)

        var y = 0
        while (y < h) {
            val row = y * w
            val cellRow = (y / d) * sw
            var x = 0
            while (x < w) {
                if (mask[row + x].toInt() != 0) {
                    val p = roi[row + x]
                    val i = cellRow + x / d
                    sr[i] += ((p shr 16) and 0xFF).toFloat()
                    sg[i] += ((p shr 8) and 0xFF).toFloat()
                    sb[i] += (p and 0xFF).toFloat()
                    sm[i] += 1f
                }
                x += step
            }
            y += step
        }

        val rad = max(1, (strength * min(sw, sh)).toInt())
        for (a in arrayOf(sr, sg, sb, sm)) boxBlur3(a, sw, sh, rad)

        // per-cell mean color of the masked pixels only
        for (i in sr.indices) {
            val den = max(sm[i], 1e-3f)
            sr[i] /= den
            sg[i] /= den
            sb[i] /= den
        }

        val ix0 = IntArray(w)
        val ix1 = IntArray(w)
        val fxs = FloatArray(w)
        for (x in 0 until w) {
            val g = ((x + 0.5f) / d - 0.5f).coerceIn(0f, (sw - 1).toFloat())
            val i0 = g.toInt()
            ix0[x] = i0
            ix1[x] = min(i0 + 1, sw - 1)
            fxs[x] = g - i0
        }
        for (yy in 0 until h) {
            val g = ((yy + 0.5f) / d - 0.5f).coerceIn(0f, (sh - 1).toFloat())
            val j0 = g.toInt()
            val j1 = min(j0 + 1, sh - 1)
            val fy = g - j0
            val r0 = j0 * sw
            val r1 = j1 * sw
            val row = yy * w
            for (x in 0 until w) {
                if (mask[row + x].toInt() == 0) continue
                val a = ix0[x]
                val b = ix1[x]
                val tx = fxs[x]
                val w00 = (1 - tx) * (1 - fy)
                val w10 = tx * (1 - fy)
                val w01 = (1 - tx) * fy
                val w11 = tx * fy
                val r = (sr[r0 + a] * w00 + sr[r0 + b] * w10 + sr[r1 + a] * w01 + sr[r1 + b] * w11).toInt().coerceIn(0, 255)
                val g2 = (sg[r0 + a] * w00 + sg[r0 + b] * w10 + sg[r1 + a] * w01 + sg[r1 + b] * w11).toInt().coerceIn(0, 255)
                val b2 = (sb[r0 + a] * w00 + sb[r0 + b] * w10 + sb[r1 + a] * w01 + sb[r1 + b] * w11).toInt().coerceIn(0, 255)
                roi[row + x] = (0xFF shl 24) or (r shl 16) or (g2 shl 8) or b2
            }
        }
    }

    /** Three passes of zero-padded box blur (approximates a Gaussian). */
    private fun boxBlur3(a: FloatArray, w: Int, h: Int, r: Int) {
        val tmp = FloatArray(a.size)
        repeat(3) {
            boxPassH(a, tmp, w, h, r)
            boxPassV(tmp, a, w, h, r)
        }
    }

    private fun boxPassH(src: FloatArray, dst: FloatArray, w: Int, h: Int, r: Int) {
        val pre = FloatArray(w + 1)
        for (y in 0 until h) {
            val o = y * w
            for (x in 0 until w) pre[x + 1] = pre[x] + src[o + x]
            for (x in 0 until w) {
                val lo = max(x - r, 0)
                val hi = min(x + r, w - 1)
                dst[o + x] = pre[hi + 1] - pre[lo]
            }
        }
    }

    private fun boxPassV(src: FloatArray, dst: FloatArray, w: Int, h: Int, r: Int) {
        val pre = FloatArray(h + 1)
        for (x in 0 until w) {
            for (y in 0 until h) pre[y + 1] = pre[y] + src[y * w + x]
            for (y in 0 until h) {
                val lo = max(y - r, 0)
                val hi = min(y + r, h - 1)
                dst[y * w + x] = pre[hi + 1] - pre[lo]
            }
        }
    }

    /** Bilinear sample of a float grid (used for mask logits). */
    fun sample(a: FloatArray, w: Int, h: Int, fx: Float, fy: Float): Float {
        val x = fx.coerceIn(0f, (w - 1).toFloat())
        val y = fy.coerceIn(0f, (h - 1).toFloat())
        val x0 = x.toInt()
        val y0 = y.toInt()
        val x1 = min(x0 + 1, w - 1)
        val y1 = min(y0 + 1, h - 1)
        val tx = x - x0
        val ty = y - y0
        val top = a[y0 * w + x0] * (1 - tx) + a[y0 * w + x1] * tx
        val bot = a[y1 * w + x0] * (1 - tx) + a[y1 * w + x1] * tx
        return top * (1 - ty) + bot * ty
    }
}
