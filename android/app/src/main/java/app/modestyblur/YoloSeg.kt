package app.modestyblur

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import java.nio.FloatBuffer
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

private class Cand(val x1: Float, val y1: Float, val x2: Float, val y2: Float, val score: Float, val idx: Int)

private fun iou(a: Cand, b: Cand): Float {
    val ix = max(0f, min(a.x2, b.x2) - max(a.x1, b.x1))
    val iy = max(0f, min(a.y2, b.y2) - max(a.y1, b.y1))
    val inter = ix * iy
    val ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return if (ua <= 0f) 0f else inter / ua
}

/**
 * YOLO11-seg (ONNX) person detector + instance masks. The model's input size is read from the file,
 * so the Fast / Balanced / Quality presets are just different exported models.
 * Output layout: [1, 116, N] (4 box + 80 classes + 32 mask coefficients) and [1, 32, H/4, W/4].
 */
class YoloSeg(ctx: Context, asset: String) {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    val inH: Int
    val inW: Int

    init {
        val bytes = ctx.assets.open(asset).use { it.readBytes() }
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(4)
        session = env.createSession(bytes, opts)
        inputName = session.inputNames.first()
        val shape = (session.inputInfo[inputName]!!.info as TensorInfo).shape   // [1, 3, H, W]
        inH = shape[2].toInt()
        inW = shape[3].toInt()
    }

    fun close() {
        try { session.close() } catch (_: Exception) {}
    }

    fun detect(src: Bitmap, conf: Float, dilate: Int): List<Det> {
        val W = src.width
        val H = src.height
        val scale = min(inW.toFloat() / W, inH.toFloat() / H)
        val nw = (W * scale).roundToInt().coerceIn(1, inW)
        val nh = (H * scale).roundToInt().coerceIn(1, inH)
        val padX = (inW - nw) / 2
        val padY = (inH - nh) / 2

        val resized = Bitmap.createScaledBitmap(src, nw, nh, true)
        val rp = IntArray(nw * nh)
        resized.getPixels(rp, 0, nw, 0, 0, nw, nh)
        if (resized !== src) resized.recycle()

        val plane = inW * inH
        val input = FloatArray(3 * plane) { 114f / 255f }
        for (y in 0 until nh) {
            val row = (y + padY) * inW + padX
            for (x in 0 until nw) {
                val p = rp[y * nw + x]
                val i = row + x
                input[i] = ((p shr 16) and 0xFF) / 255f
                input[plane + i] = ((p shr 8) and 0xFF) / 255f
                input[2 * plane + i] = (p and 0xFF) / 255f
            }
        }

        val tensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(input), longArrayOf(1, 3, inH.toLong(), inW.toLong())
        )
        return tensor.use { t ->
            session.run(mapOf(inputName to t)).use { res ->
                val o0 = res.get(0) as OnnxTensor
                val o1 = res.get(1) as OnnxTensor
                parse(o0, o1, W, H, scale, padX, padY, conf, dilate)
            }
        }
    }

    private fun parse(
        o0: OnnxTensor, o1: OnnxTensor, W: Int, H: Int,
        scale: Float, padX: Int, padY: Int, conf: Float, dilate: Int,
    ): List<Det> {
        val s0 = o0.info.shape            // [1, 4 + 80 + nm, N]
        val c = s0[1].toInt()
        val n = s0[2].toInt()
        val nm = c - 84
        val b0 = o0.floatBuffer

        val cands = ArrayList<Cand>()
        for (i in 0 until n) {
            val sc = b0.get(4 * n + i)         // class 0 = person
            if (sc < conf) continue
            val cx = b0.get(i)
            val cy = b0.get(n + i)
            val bw = b0.get(2 * n + i)
            val bh = b0.get(3 * n + i)
            val x1 = ((cx - bw / 2 - padX) / scale).coerceIn(0f, W.toFloat())
            val y1 = ((cy - bh / 2 - padY) / scale).coerceIn(0f, H.toFloat())
            val x2 = ((cx + bw / 2 - padX) / scale).coerceIn(0f, W.toFloat())
            val y2 = ((cy + bh / 2 - padY) / scale).coerceIn(0f, H.toFloat())
            if (x2 - x1 < 4f || y2 - y1 < 4f) continue
            cands.add(Cand(x1, y1, x2, y2, sc, i))
        }
        if (cands.isEmpty()) return emptyList()
        cands.sortByDescending { it.score }
        val keep = ArrayList<Cand>()
        for (cd in cands) {
            if (keep.size >= 40) break          // crowds: never silently leave anyone unblurred
            if (keep.none { iou(it, cd) > 0.5f }) keep.add(cd)
        }

        val s1 = o1.info.shape             // [1, nm, ph, pw]
        val ph = s1[2].toInt()
        val pw = s1[3].toInt()
        val b1 = o1.floatBuffer
        val strideX = inW.toFloat() / pw
        val strideY = inH.toFloat() / ph

        return keep.map { cd ->
            Det(
                cd.x1, cd.y1, cd.x2, cd.y2, cd.score,
                buildMask(cd, b0, b1, n, nm, ph, pw, strideX, strideY, W, H, scale, padX, padY, dilate)
            )
        }
    }

    private fun buildMask(
        cd: Cand, b0: FloatBuffer, b1: FloatBuffer, n: Int, nm: Int, ph: Int, pw: Int,
        strideX: Float, strideY: Float, W: Int, H: Int, scale: Float, padX: Int, padY: Int, dilate: Int,
    ): Mask {
        val x0 = max(floor(cd.x1).toInt() - dilate, 0)
        val y0 = max(floor(cd.y1).toInt() - dilate, 0)
        val x1 = min(ceil(cd.x2).toInt() + dilate, W)
        val y1 = min(ceil(cd.y2).toInt() + dilate, H)
        val w = x1 - x0
        val h = y1 - y0
        if (w <= 0 || h <= 0) return Mask(0, 0, 0, 0, ByteArray(0))

        val coef = FloatArray(nm) { b0.get((84 + it) * n + cd.idx) }

        // logits over the part of the prototype grid that covers the box
        val px0 = (floor((cd.x1 * scale + padX) / strideX).toInt() - 1).coerceIn(0, pw - 1)
        val px1 = (ceil((cd.x2 * scale + padX) / strideX).toInt() + 1).coerceIn(0, pw - 1)
        val py0 = (floor((cd.y1 * scale + padY) / strideY).toInt() - 1).coerceIn(0, ph - 1)
        val py1 = (ceil((cd.y2 * scale + padY) / strideY).toInt() + 1).coerceIn(0, ph - 1)
        val lw = px1 - px0 + 1
        val lh = py1 - py0 + 1
        val logit = FloatArray(lw * lh)
        val plane = ph * pw
        for (yy in 0 until lh) {
            for (xx in 0 until lw) {
                val off = (py0 + yy) * pw + (px0 + xx)
                var sum = 0f
                for (k in 0 until nm) sum += coef[k] * b1.get(k * plane + off)
                logit[yy * lw + xx] = sum
            }
        }

        val data = ByteArray(w * h)
        for (y in 0 until h) {
            val oy = y0 + y
            if (oy < cd.y1 || oy > cd.y2) continue
            val fy = ((oy + 0.5f) * scale + padY) / strideY - 0.5f - py0
            for (x in 0 until w) {
                val ox = x0 + x
                if (ox < cd.x1 || ox > cd.x2) continue
                val fx = ((ox + 0.5f) * scale + padX) / strideX - 0.5f - px0
                // slightly generous threshold (sigmoid ~0.31) so the silhouette errs on the big side
                if (Blur.sample(logit, lw, lh, fx, fy) > -0.8f) data[y * w + x] = 1
            }
        }
        Blur.dilate(data, w, h, dilate)
        return Mask(x0, y0, w, h, data)
    }
}
