package app.modestyblur

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

class Face(
    val x1: Float, val y1: Float, val x2: Float, val y2: Float,
    val score: Float, val kps: FloatArray,
) {
    val area get() = (x2 - x1) * (y2 - y1)
}

/** Everything one face pass learns: identity embedding, face-gender reading (-1 if unavailable), face width. */
class FaceInfo(val emb: FloatArray, val pFemale: Float, val faceW: Float)

/**
 * Face detection (SCRFD det_500m), recognition (MobileFaceNet w600k_mbf) and an optional face-gender
 * reading (genderage), all from the InsightFace "buffalo_s" pack.
 *
 * Embeddings are 512 numbers; two photos of the same person score a cosine similarity of roughly
 * 0.5-0.9, different people stay below about 0.3. The gender reading is NOT trusted to clear anyone:
 * it was seen to call a woman in a uniform and cap "male" with 100% confidence.
 */
class FaceEngine private constructor(
    private val env: OrtEnvironment,
    private val det: OrtSession,
    private val rec: OrtSession,
    private val gen: OrtSession?,
) {
    private val detInput = det.inputNames.first()
    private val recInput = rec.inputNames.first()
    private val genInput = gen?.inputNames?.first()

    companion object {
        /** Returns null if the face models are not bundled in this build. */
        fun create(ctx: Context): FaceEngine? = try {
            val env = OrtEnvironment.getEnvironment()
            fun opts() = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) }
            val det = env.createSession(ctx.assets.open("det_500m.onnx").use { it.readBytes() }, opts())
            val rec = env.createSession(ctx.assets.open("w600k_mbf.onnx").use { it.readBytes() }, opts())
            val gen = try {
                env.createSession(ctx.assets.open("genderage.onnx").use { it.readBytes() }, opts())
            } catch (e: Exception) {
                null
            }
            FaceEngine(env, det, rec, gen)
        } catch (e: Exception) {
            null
        }

        private val TEMPLATE = floatArrayOf(
            38.2946f, 51.6963f, 73.5318f, 51.5014f, 56.0252f,
            71.7366f, 41.5493f, 92.3655f, 70.7299f, 92.2041f,
        )
    }

    /** Detect faces; the image is letterboxed (top-left) into a size x size square. */
    fun detect(bmp: Bitmap, size: Int, thr: Float = 0.5f): List<Face> {
        val scale = min(size.toFloat() / bmp.width, size.toFloat() / bmp.height)
        val nw = max(1, (bmp.width * scale).roundToInt())
        val nh = max(1, (bmp.height * scale).roundToInt())
        val resized = Bitmap.createScaledBitmap(bmp, nw, nh, true)
        val rp = IntArray(nw * nh)
        resized.getPixels(rp, 0, nw, 0, 0, nw, nh)
        if (resized !== bmp) resized.recycle()

        val plane = size * size
        val input = FloatArray(3 * plane) { (0f - 127.5f) / 128f }
        for (y in 0 until nh) {
            for (x in 0 until nw) {
                val p = rp[y * nw + x]
                val i = y * size + x
                input[i] = (((p shr 16) and 0xFF) - 127.5f) / 128f
                input[plane + i] = (((p shr 8) and 0xFF) - 127.5f) / 128f
                input[2 * plane + i] = ((p and 0xFF) - 127.5f) / 128f
            }
        }

        val faces = ArrayList<Face>()
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, 3, size.toLong(), size.toLong()))
        tensor.use { t ->
            det.run(mapOf(detInput to t)).use { res ->
                // outputs: scores x3, boxes x3, keypoints x3 for strides 8, 16, 32 (2 anchors per cell)
                for (idx in 0 until 3) {
                    val stride = 8 shl idx
                    val sc = (res.get(idx) as OnnxTensor).floatBuffer
                    val bb = (res.get(idx + 3) as OnnxTensor).floatBuffer
                    val kp = (res.get(idx + 6) as OnnxTensor).floatBuffer
                    val gw = size / stride
                    val count = gw * gw * 2
                    for (i in 0 until count) {
                        val s = sc.get(i)
                        if (s < thr) continue
                        val pos = i / 2
                        val cx = ((pos % gw) * stride).toFloat()
                        val cy = ((pos / gw) * stride).toFloat()
                        val x1 = (cx - bb.get(i * 4) * stride) / scale
                        val y1 = (cy - bb.get(i * 4 + 1) * stride) / scale
                        val x2 = (cx + bb.get(i * 4 + 2) * stride) / scale
                        val y2 = (cy + bb.get(i * 4 + 3) * stride) / scale
                        val k = FloatArray(10) { j ->
                            val v = kp.get(i * 10 + j) * stride
                            if (j % 2 == 0) (cx + v) / scale else (cy + v) / scale
                        }
                        faces.add(Face(x1, y1, x2, y2, s, k))
                    }
                }
            }
        }
        return nms(faces, 0.4f)
    }

    private fun nms(faces: List<Face>, thr: Float): List<Face> {
        val sorted = faces.sortedByDescending { it.score }
        val keep = ArrayList<Face>()
        for (f in sorted) {
            var ok = true
            for (k in keep) {
                val ix = max(0f, min(f.x2, k.x2) - max(f.x1, k.x1))
                val iy = max(0f, min(f.y2, k.y2) - max(f.y1, k.y1))
                val inter = ix * iy
                val u = f.area + k.area - inter
                if (u > 0f && inter / u > thr) { ok = false; break }
            }
            if (ok) keep.add(f)
        }
        return keep
    }

    /** Bilinear RGB sample (0..255 floats) with a black border, written into [out]. */
    private fun sampleRgb(px: IntArray, w: Int, h: Int, sx: Float, sy: Float, out: FloatArray) {
        val x0 = floor(sx.toDouble()).toInt()
        val y0 = floor(sy.toDouble()).toInt()
        val fx = sx - x0
        val fy = sy - y0
        var r = 0f; var g = 0f; var b = 0f
        for (dy in 0..1) {
            for (dx in 0..1) {
                val xx = x0 + dx
                val yy = y0 + dy
                if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue
                val wgt = (if (dx == 0) 1 - fx else fx) * (if (dy == 0) 1 - fy else fy)
                val p = px[yy * w + xx]
                r += ((p shr 16) and 0xFF) * wgt
                g += ((p shr 8) and 0xFF) * wgt
                b += (p and 0xFF) * wgt
            }
        }
        out[0] = r; out[1] = g; out[2] = b
    }

    /** Align the face to the 112x112 ArcFace template and return its normalized 512-d embedding. */
    private fun embedPx(px: IntArray, w: Int, h: Int, face: Face): FloatArray {
        // least-squares similarity transform (rotation + uniform scale + shift) from landmarks to template
        var msx = 0f; var msy = 0f; var mdx = 0f; var mdy = 0f
        for (i in 0 until 5) {
            msx += face.kps[2 * i]; msy += face.kps[2 * i + 1]
            mdx += TEMPLATE[2 * i]; mdy += TEMPLATE[2 * i + 1]
        }
        msx /= 5; msy /= 5; mdx /= 5; mdy /= 5
        var num1 = 0f; var num2 = 0f; var den = 0f
        for (i in 0 until 5) {
            val sx = face.kps[2 * i] - msx
            val sy = face.kps[2 * i + 1] - msy
            val dx = TEMPLATE[2 * i] - mdx
            val dy = TEMPLATE[2 * i + 1] - mdy
            num1 += sx * dx + sy * dy
            num2 += sx * dy - sy * dx
            den += sx * sx + sy * sy
        }
        val a = num1 / max(den, 1e-6f)
        val b = num2 / max(den, 1e-6f)
        val tx = mdx - (a * msx - b * msy)
        val ty = mdy - (b * msx + a * msy)
        val s2 = max(a * a + b * b, 1e-9f)

        val plane = 112 * 112
        val input = FloatArray(3 * plane)
        val rgb = FloatArray(3)
        for (v in 0 until 112) {
            for (u in 0 until 112) {
                // inverse mapping: template pixel (u, v) -> source pixel
                val sx = (a * (u - tx) + b * (v - ty)) / s2
                val sy = (-b * (u - tx) + a * (v - ty)) / s2
                sampleRgb(px, w, h, sx, sy, rgb)
                val i = v * 112 + u
                input[i] = (rgb[0] - 127.5f) / 127.5f
                input[plane + i] = (rgb[1] - 127.5f) / 127.5f
                input[2 * plane + i] = (rgb[2] - 127.5f) / 127.5f
            }
        }

        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, 3, 112, 112))
        val emb = tensor.use { t ->
            rec.run(mapOf(recInput to t)).use { res ->
                val buf = (res.get(0) as OnnxTensor).floatBuffer
                val out = FloatArray(buf.remaining())
                buf.get(out)
                out
            }
        }
        var norm = 0f
        for (v in emb) norm += v * v
        norm = sqrt(norm).coerceAtLeast(1e-6f)
        for (i in emb.indices) emb[i] /= norm
        return emb
    }

    /** P(female) from the face, or -1 if the gender model is not bundled. Raw 0-255 RGB, 96x96 crop. */
    private fun genderPx(px: IntArray, w: Int, h: Int, f: Face): Float {
        val g = gen ?: return -1f
        val cx = (f.x1 + f.x2) / 2f
        val cy = (f.y1 + f.y2) / 2f
        val sc = 96f / (max(f.x2 - f.x1, f.y2 - f.y1) * 1.5f).coerceAtLeast(1f)
        val plane = 96 * 96
        val input = FloatArray(3 * plane)
        val rgb = FloatArray(3)
        for (v in 0 until 96) {
            for (u in 0 until 96) {
                sampleRgb(px, w, h, (u - 48f) / sc + cx, (v - 48f) / sc + cy, rgb)
                val i = v * 96 + u
                input[i] = rgb[0]
                input[plane + i] = rgb[1]
                input[2 * plane + i] = rgb[2]
            }
        }
        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, 3, 96, 96))
        val logits = tensor.use { t ->
            g.run(mapOf(genInput!! to t)).use { res ->
                val buf = (res.get(0) as OnnxTensor).floatBuffer
                floatArrayOf(buf.get(0), buf.get(1))      // [female, male]
            }
        }
        val m = max(logits[0], logits[1])
        val e0 = exp(logits[0] - m)
        val e1 = exp(logits[1] - m)
        return e0 / (e0 + e1)
    }

    /** One pass over the biggest face in [bmp]: embedding + face-gender. Null if no face >= [minFace] px wide. */
    fun analyzeLargest(bmp: Bitmap, detSize: Int, minFace: Float): FaceInfo? {
        val face = detect(bmp, detSize).maxByOrNull { it.area } ?: return null
        val fw = face.x2 - face.x1
        if (fw < minFace) return null
        val w = bmp.width
        val h = bmp.height
        val px = IntArray(w * h)
        bmp.getPixels(px, 0, w, 0, 0, w, h)
        return FaceInfo(embedPx(px, w, h, face), genderPx(px, w, h, face), fw)
    }

    /** Embedding only (used when enrolling). */
    fun embedLargest(bmp: Bitmap, detSize: Int, minFace: Float): FloatArray? =
        analyzeLargest(bmp, detSize, minFace)?.emb
}
