package app.modestyblur

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import java.nio.FloatBuffer
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

/**
 * Face detection (SCRFD det_500m) and recognition (MobileFaceNet w600k_mbf), both from the
 * InsightFace "buffalo_s" pack. Embeddings are 512 numbers; two photos of the same person
 * have a cosine similarity of roughly 0.5-0.8, different people stay below about 0.3.
 */
class FaceEngine private constructor(
    private val env: OrtEnvironment,
    private val det: OrtSession,
    private val rec: OrtSession,
) {
    private val detInput = det.inputNames.first()
    private val recInput = rec.inputNames.first()

    companion object {
        /** Returns null if the face models are not bundled in this build. */
        fun create(ctx: Context): FaceEngine? = try {
            val env = OrtEnvironment.getEnvironment()
            val o1 = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) }
            val o2 = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) }
            val det = env.createSession(ctx.assets.open("det_500m.onnx").use { it.readBytes() }, o1)
            val rec = env.createSession(ctx.assets.open("w600k_mbf.onnx").use { it.readBytes() }, o2)
            FaceEngine(env, det, rec)
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

    /** Align the face to the 112x112 ArcFace template and return its normalized 512-d embedding. */
    fun embed(bmp: Bitmap, face: Face): FloatArray {
        val w = bmp.width
        val h = bmp.height
        val px = IntArray(w * h)
        bmp.getPixels(px, 0, w, 0, 0, w, h)

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
        for (v in 0 until 112) {
            for (u in 0 until 112) {
                // inverse mapping: template pixel (u, v) -> source pixel
                val sx = (a * (u - tx) + b * (v - ty)) / s2
                val sy = (-b * (u - tx) + a * (v - ty)) / s2
                val x0 = Math.floor(sx.toDouble()).toInt()
                val y0 = Math.floor(sy.toDouble()).toInt()
                val fx = sx - x0
                val fy = sy - y0
                var r = 0f; var g = 0f; var bl = 0f
                for (dyi in 0..1) {
                    for (dxi in 0..1) {
                        val xx = x0 + dxi
                        val yy = y0 + dyi
                        if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue
                        val wgt = (if (dxi == 0) 1 - fx else fx) * (if (dyi == 0) 1 - fy else fy)
                        val p = px[yy * w + xx]
                        r += ((p shr 16) and 0xFF) * wgt
                        g += ((p shr 8) and 0xFF) * wgt
                        bl += (p and 0xFF) * wgt
                    }
                }
                val i = v * 112 + u
                input[i] = (r - 127.5f) / 127.5f
                input[plane + i] = (g - 127.5f) / 127.5f
                input[2 * plane + i] = (bl - 127.5f) / 127.5f
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

    /** Embedding of the biggest face found, or null if there is no face at least [minFace] px wide. */
    fun embedLargest(bmp: Bitmap, detSize: Int, minFace: Float): FloatArray? {
        val face = detect(bmp, detSize).maxByOrNull { it.area } ?: return null
        if (face.x2 - face.x1 < minFace) return null
        return embed(bmp, face)
    }
}
