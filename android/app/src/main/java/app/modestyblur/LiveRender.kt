package app.modestyblur

import android.graphics.Bitmap
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * One blurred person, ready for the GPU: [grid] is a tiny image of the blurred colours (from the 1/6-size
 * frame), [mask] is the silhouette. The view stretches the grid over the silhouette with bilinear
 * filtering and cuts it out with the mask, which costs the CPU almost nothing.
 */
class Sprite(
    val mask: Bitmap, val grid: Bitmap,
    val mx: Float, val my: Float, val mw: Float, val mh: Float,
    val gx: Float, val gy: Float, val gw: Float, val gh: Float,
)

object LiveRender {
    /** The frame is shrunk by this factor to get blur colours. */
    const val CELL = 6

    /** Silhouette as a small alpha image (cached per detection). */
    private fun maskBitmap(m: Mask): Bitmap? {
        m.cache.alpha?.let { return it }
        if (m.w <= 0 || m.h <= 0) return null
        val s = if (max(m.w, m.h) > 240) 3 else 2
        val bw = (m.w + s - 1) / s
        val bh = (m.h + s - 1) / s
        val px = IntArray(bw * bh)
        for (j in 0 until bh) {
            val ys = j * s
            val ye = min(ys + s, m.h)
            for (i in 0 until bw) {
                val xs = i * s
                val xe = min(xs + s, m.w)
                var c = 0
                var n = 0
                for (yy in ys until ye) {
                    val row = yy * m.w
                    for (xx in xs until xe) {
                        n++
                        if (m.data[row + xx].toInt() != 0) c++
                    }
                }
                px[j * bw + i] = ((c * 255 / max(n, 1)) shl 24)
            }
        }
        val bmp = Bitmap.createBitmap(px, bw, bh, Bitmap.Config.ARGB_8888)
        m.cache.alpha = bmp
        return bmp
    }

    /**
     * Build the sprite for one silhouette. [thumb] is the frame shrunk to tw x th; (cw, ch) is the size of one
     * thumb pixel in full-frame pixels. The blur is a normalized convolution: colours are averaged only over
     * the body, so nothing leaks in from the background.
     */
    fun sprite(m: Mask, thumb: IntArray, tw: Int, th: Int, cw: Float, ch: Float, strength: Float): Sprite? {
        val maskBmp = maskBitmap(m) ?: return null
        val gx0 = floor(m.x0 / cw).toInt().coerceIn(0, tw)
        val gy0 = floor(m.y0 / ch).toInt().coerceIn(0, th)
        val gx1 = ceil((m.x0 + m.w) / cw).toInt().coerceIn(0, tw)
        val gy1 = ceil((m.y0 + m.h) / ch).toInt().coerceIn(0, th)
        val gw = gx1 - gx0
        val gh = gy1 - gy0
        if (gw <= 0 || gh <= 0) return null

        val sr = FloatArray(gw * gh)
        val sg = FloatArray(gw * gh)
        val sb = FloatArray(gw * gh)
        val sm = FloatArray(gw * gh)
        for (gy in 0 until gh) {
            for (gx in 0 until gw) {
                var cov = 0
                for (sy in 0..1) {
                    val my = ((gy0 + gy + 0.25f + 0.5f * sy) * ch).toInt() - m.y0
                    if (my < 0 || my >= m.h) continue
                    for (sx in 0..1) {
                        val mx = ((gx0 + gx + 0.25f + 0.5f * sx) * cw).toInt() - m.x0
                        if (mx >= 0 && mx < m.w && m.data[my * m.w + mx].toInt() != 0) cov++
                    }
                }
                if (cov == 0) continue
                val wgt = cov / 4f
                val p = thumb[(gy0 + gy) * tw + gx0 + gx]
                val i = gy * gw + gx
                sr[i] = ((p shr 16) and 0xFF) * wgt
                sg[i] = ((p shr 8) and 0xFF) * wgt
                sb[i] = (p and 0xFF) * wgt
                sm[i] = wgt
            }
        }
        val rad = max(1, (strength * min(gw, gh)).toInt())
        for (a in arrayOf(sr, sg, sb, sm)) Blur.boxBlur3(a, gw, gh, rad)

        val px = IntArray(gw * gh)
        for (i in px.indices) {
            val den = max(sm[i], 1e-3f)
            val r = (sr[i] / den).toInt().coerceIn(0, 255)
            val g = (sg[i] / den).toInt().coerceIn(0, 255)
            val b = (sb[i] / den).toInt().coerceIn(0, 255)
            px[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        val grid = Bitmap.createBitmap(px, gw, gh, Bitmap.Config.ARGB_8888)
        return Sprite(
            maskBmp, grid,
            m.x0.toFloat(), m.y0.toFloat(), m.w.toFloat(), m.h.toFloat(),
            gx0 * cw, gy0 * ch, gw * cw, gh * ch,
        )
    }
}
