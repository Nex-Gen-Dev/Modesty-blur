package app.modestyblur

import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Full-silhouette blur using normalized convolution: colors are averaged only from pixels
 * inside the mask, so nothing bleeds in from the background and the result is a smooth,
 * unrecognizable blob in the exact shape of the body.
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

    /** Blur the masked area of [px] (ARGB ints, frame size W x H) in place. */
    fun blurMasked(px: IntArray, frameW: Int, frameH: Int, m: Mask, strength: Float) {
        val w = m.w
        val h = m.h
        if (w <= 0 || h <= 0) return
        val d = if (min(w, h) > 80) 4 else 1
        val sw = (w + d - 1) / d
        val sh = (h + d - 1) / d
        val sr = FloatArray(sw * sh)
        val sg = FloatArray(sw * sh)
        val sb = FloatArray(sw * sh)
        val sm = FloatArray(sw * sh)

        for (y in 0 until h) {
            val fy = m.y0 + y
            if (fy < 0 || fy >= frameH) continue
            for (x in 0 until w) {
                if (m.data[y * w + x].toInt() == 0) continue
                val fx = m.x0 + x
                if (fx < 0 || fx >= frameW) continue
                val p = px[fy * frameW + fx]
                val i = (y / d) * sw + (x / d)
                sr[i] += ((p shr 16) and 0xFF).toFloat()
                sg[i] += ((p shr 8) and 0xFF).toFloat()
                sb[i] += (p and 0xFF).toFloat()
                sm[i] += 1f
            }
        }

        val rad = max(2, (strength * min(sw, sh)).toInt())
        for (a in arrayOf(sr, sg, sb, sm)) boxBlur3(a, sw, sh, rad)

        for (y in 0 until h) {
            val fy = m.y0 + y
            if (fy < 0 || fy >= frameH) continue
            val gy = (y + 0.5f) / d - 0.5f
            for (x in 0 until w) {
                if (m.data[y * w + x].toInt() == 0) continue
                val fx = m.x0 + x
                if (fx < 0 || fx >= frameW) continue
                val gx = (x + 0.5f) / d - 0.5f
                val den = max(sample(sm, sw, sh, gx, gy), 1e-3f)
                val r = (sample(sr, sw, sh, gx, gy) / den).toInt().coerceIn(0, 255)
                val g = (sample(sg, sw, sh, gx, gy) / den).toInt().coerceIn(0, 255)
                val b = (sample(sb, sw, sh, gx, gy) / den).toInt().coerceIn(0, 255)
                px[fy * frameW + fx] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
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

    fun sample(a: FloatArray, w: Int, h: Int, fx: Float, fy: Float): Float {
        val x = fx.coerceIn(0f, (w - 1).toFloat())
        val y = fy.coerceIn(0f, (h - 1).toFloat())
        val x0 = floor(x).toInt()
        val y0 = floor(y).toInt()
        val x1 = min(x0 + 1, w - 1)
        val y1 = min(y0 + 1, h - 1)
        val tx = x - x0
        val ty = y - y0
        val top = a[y0 * w + x0] * (1 - tx) + a[y0 * w + x1] * tx
        val bot = a[y1 * w + x0] * (1 - tx) + a[y1 * w + x1] * tx
        return top * (1 - ty) + bot * ty
    }
}
