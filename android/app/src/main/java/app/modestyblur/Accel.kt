package app.modestyblur

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import ai.onnxruntime.providers.NNAPIFlags
import android.content.Context
import android.os.Build
import android.os.SystemClock
import android.util.Log
import java.nio.FloatBuffer
import java.util.EnumSet
import kotlin.math.abs

/**
 * Picks the fastest way to run a model on THIS phone: plain CPU, XNNPACK (optimized CPU kernels) or NNAPI
 * (the phone's NPU / GPU / DSP). The first launch times each one, rejects any whose answers differ from the
 * CPU's, keeps the winner, and remembers the choice for next time.
 */
object Accel {
    class Result(val session: OrtSession, val name: String, val ms: Float)

    private const val TAG = "Accel"

    private fun create(env: OrtEnvironment, bytes: ByteArray, kind: String, threads: Int): OrtSession {
        val o = OrtSession.SessionOptions()
        when (kind) {
            "XNNPACK" -> {
                o.setIntraOpNumThreads(1)
                o.addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
            }
            "NNAPI" -> {
                o.setIntraOpNumThreads(threads)
                o.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
            }
            else -> o.setIntraOpNumThreads(threads)
        }
        return env.createSession(bytes, o)
    }

    private class Bench(val ms: Float, val out: FloatArray)

    private fun bench(env: OrtEnvironment, s: OrtSession, shape: LongArray, input: FloatArray): Bench {
        val name = s.inputNames.first()
        OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape).use { t ->
            var first = FloatArray(0)
            val times = ArrayList<Float>()
            for (i in 0 until 5) {
                val t0 = SystemClock.elapsedRealtimeNanos()
                s.run(mapOf(name to t)).use { res ->
                    val o = (res.get(0) as OnnxTensor).floatBuffer
                    if (i == 1) {                       // keep one output for the correctness check
                        first = FloatArray(o.remaining())
                        o.duplicate().get(first)
                    }
                }
                if (i >= 1) times.add((SystemClock.elapsedRealtimeNanos() - t0) / 1e6f)
            }
            times.sort()
            return Bench(times[times.size / 2], first)
        }
    }

    private fun agrees(ref: FloatArray, got: FloatArray): Boolean {
        if (ref.isEmpty() || ref.size != got.size) return false
        var d = 0.0
        var m = 0.0
        for (i in ref.indices) {
            val g = got[i]
            if (g.isNaN() || g.isInfinite()) return false
            d += abs(ref[i] - g)
            m += abs(ref[i])
        }
        return d / ref.size < 0.15 * (m / ref.size) + 1e-4
    }

    fun open(ctx: Context, env: OrtEnvironment, key: String, bytes: ByteArray, threads: Int = 4): Result {
        val sp = ctx.getSharedPreferences("accel", Context.MODE_PRIVATE)
        val cacheKey = "$key:${bytes.size}:${Build.MODEL}:${Build.VERSION.SDK_INT}"
        sp.getString(cacheKey, null)?.let { saved ->
            val kind = saved.substringBefore('|')
            val ms = saved.substringAfter('|', "0").toFloatOrNull() ?: 0f
            try {
                return Result(create(env, bytes, kind, threads), kind, ms)
            } catch (e: Throwable) {
                Log.w(TAG, "saved accelerator $kind failed, re-testing", e)
            }
        }

        val cpu = create(env, bytes, "CPU", threads)
        val shape = (cpu.inputInfo[cpu.inputNames.first()]!!.info as TensorInfo).shape
        val rnd = java.util.Random(1234)
        val input = FloatArray(shape.fold(1L) { a, b -> a * b }.toInt()) { rnd.nextFloat() }
        val ref = bench(env, cpu, shape, input)

        var best = cpu
        var bestKind = "CPU"
        var bestMs = ref.ms
        for (kind in listOf("XNNPACK", "NNAPI")) {
            var s: OrtSession? = null
            try {
                s = create(env, bytes, kind, threads)
                val b = bench(env, s, shape, input)
                Log.i(TAG, "$key $kind %.1f ms (cpu %.1f)".format(b.ms, ref.ms))
                if (agrees(ref.out, b.out) && b.ms < bestMs * 0.9f) {
                    if (best !== cpu) best.close()
                    best = s
                    bestKind = kind
                    bestMs = b.ms
                    s = null
                }
            } catch (e: Throwable) {
                Log.w(TAG, "$kind unavailable", e)
            } finally {
                try { s?.close() } catch (_: Throwable) {}
            }
        }
        if (best !== cpu) cpu.close()
        sp.edit().putString(cacheKey, "$bestKind|$bestMs").apply()
        return Result(best, bestKind, bestMs)
    }
}
