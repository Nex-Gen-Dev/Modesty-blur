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
import kotlin.math.roundToInt

/**
 * Optional small, fast person classifier trained on your own labeled photos (see tools/train_person_cls.py).
 * If person_cls.onnx is bundled the app uses it instead of CLIP: it runs in a few milliseconds.
 * The model takes an RGB crop scaled to 0..1 and normalizes internally; output = class probabilities.
 */
class PersonClassifier private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val femaleIdx: IntArray,
    private val inW: Int,
    private val inH: Int,
) {
    private val inputName = session.inputNames.first()

    companion object {
        fun create(ctx: Context): PersonClassifier? = try {
            val meta = JSONObject(ctx.assets.open("person_cls.json").bufferedReader().use { it.readText() })
            val fa = meta.getJSONArray("female")
            val idx = IntArray(fa.length()) { fa.getInt(it) }
            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) }
            val s = env.createSession(ctx.assets.open("person_cls.onnx").use { it.readBytes() }, opts)
            PersonClassifier(env, s, idx, meta.getInt("width"), meta.getInt("height"))
        } catch (e: Exception) {
            null
        }
    }

    fun predictFemale(crop: Bitmap): Float {
        // pad (not stretch) the crop to the model's aspect ratio with the mean color, like in training
        val aspect = inW.toFloat() / inH
        val cw = crop.width
        val ch = crop.height
        val canvasW: Int
        val canvasH: Int
        if (cw.toFloat() / ch > aspect) {
            canvasW = cw
            canvasH = (cw / aspect).roundToInt()
        } else {
            canvasH = ch
            canvasW = (ch * aspect).roundToInt()
        }
        val padded = Bitmap.createBitmap(canvasW, canvasH, Bitmap.Config.ARGB_8888)
        val c = Canvas(padded)
        c.drawColor(Color.rgb(124, 116, 104))
        c.drawBitmap(crop, (canvasW - cw) / 2f, (canvasH - ch) / 2f, null)
        val img = Bitmap.createScaledBitmap(padded, inW, inH, true)

        val px = IntArray(inW * inH)
        img.getPixels(px, 0, inW, 0, 0, inW, inH)
        val plane = inW * inH
        val input = FloatArray(3 * plane)
        for (i in 0 until plane) {
            val p = px[i]
            input[i] = ((p shr 16) and 0xFF) / 255f
            input[plane + i] = ((p shr 8) and 0xFF) / 255f
            input[2 * plane + i] = (p and 0xFF) / 255f
        }
        if (img !== padded) img.recycle()
        padded.recycle()

        val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, 3, inH.toLong(), inW.toLong()))
        return tensor.use { t ->
            session.run(mapOf(inputName to t)).use { res ->
                val buf = (res.get(0) as OnnxTensor).floatBuffer
                var sum = 0f
                for (i in femaleIdx) sum += buf.get(i)
                sum
            }
        }
    }
}
