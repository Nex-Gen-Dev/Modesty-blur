package app.modestyblur

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import org.json.JSONObject
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt

/** CLIP zero-shot P(female) from a person crop. Prompts: woman, girl, man, boy (first two = female). */
class ClipClassifier(ctx: Context) {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val text: Array<FloatArray>

    init {
        val bytes = ctx.assets.open("clip_visual.onnx").use { it.readBytes() }
        val opts = OrtSession.SessionOptions()
        opts.setIntraOpNumThreads(4)
        session = env.createSession(bytes, opts)
        inputName = session.inputNames.first()

        val json = JSONObject(ctx.assets.open("clip_text.json").bufferedReader().use { it.readText() })
        val arr = json.getJSONArray("text")
        text = Array(arr.length()) { i ->
            val row = arr.getJSONArray(i)
            FloatArray(row.length()) { j -> row.getDouble(j).toFloat() }
        }
    }

    fun predictFemale(crop: Bitmap): Float {
        val size = max(crop.width, crop.height)
        val sq = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(sq)
        canvas.drawColor(Color.rgb(124, 116, 104))
        canvas.drawBitmap(crop, (size - crop.width) / 2f, (size - crop.height) / 2f, null)
        val img = Bitmap.createScaledBitmap(sq, 224, 224, true)

        val px = IntArray(224 * 224)
        img.getPixels(px, 0, 224, 0, 0, 224, 224)
        val plane = 224 * 224
        val input = FloatArray(3 * plane)
        val mean = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
        val std = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)
        for (i in 0 until plane) {
            val p = px[i]
            input[i] = (((p shr 16) and 0xFF) / 255f - mean[0]) / std[0]
            input[plane + i] = (((p shr 8) and 0xFF) / 255f - mean[1]) / std[1]
            input[2 * plane + i] = ((p and 0xFF) / 255f - mean[2]) / std[2]
        }
        if (img !== sq) img.recycle()
        sq.recycle()

        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, 3, 224, 224))
        val emb: FloatArray = tensor.use { t ->
            session.run(mapOf(inputName to t)).use { res ->
                val buf = (res.get(0) as OnnxTensor).floatBuffer
                val out = FloatArray(buf.remaining())
                buf.get(out)
                out
            }
        }

        var norm = 0f
        for (v in emb) norm += v * v
        norm = sqrt(norm).coerceAtLeast(1e-6f)
        val logits = FloatArray(text.size) { k ->
            var dot = 0f
            for (j in emb.indices) dot += emb[j] / norm * text[k][j]
            100f * dot
        }
        val m = logits.max()
        val ex = FloatArray(logits.size) { exp(logits[it] - m) }
        val sum = ex.sum()
        return (ex[0] + ex[1]) / sum
    }
}
