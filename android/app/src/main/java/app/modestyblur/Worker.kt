package app.modestyblur

import android.graphics.Bitmap
import android.util.Log
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

/**
 * Runs the slow models (CLIP, face recognition) on a background thread so the live view never waits
 * for them. Until a result arrives the person simply stays blurred.
 */
class Worker(private val clip: ClipClassifier, private val face: FaceEngine?) {
    sealed class Result {
        class Cls(val id: Int, val p: Float) : Result()
        class FaceRes(val id: Int, val status: Int, val name: String, val sim: Float) : Result()
        class Fail(val id: Int) : Result()
    }

    val results = ConcurrentLinkedQueue<Result>()
    private val pool = Executors.newSingleThreadExecutor()
    private val pendingCount = AtomicInteger(0)

    @Volatile var lastClipMs = 0L
    @Volatile var lastFaceMs = 0L

    val faceAvailable get() = face != null
    fun pending() = pendingCount.get()

    fun classify(id: Int, crop: Bitmap) {
        pendingCount.incrementAndGet()
        pool.execute {
            try {
                val t0 = System.nanoTime()
                val p = clip.predictFemale(crop)
                lastClipMs = (System.nanoTime() - t0) / 1_000_000
                results.add(Result.Cls(id, p))
            } catch (e: Throwable) {
                Log.e("Worker", "classify failed", e)
                results.add(Result.Fail(id))
            } finally {
                crop.recycle()
                pendingCount.decrementAndGet()
            }
        }
    }

    /** Looks for a face in [crop] and compares it with the family gallery. */
    fun faceCheck(id: Int, crop: Bitmap, threshold: Float) {
        val fe = face
        if (fe == null) {
            crop.recycle()
            return
        }
        pendingCount.incrementAndGet()
        pool.execute {
            try {
                val t0 = System.nanoTime()
                val emb = fe.embedLargest(crop, 320, 22f)
                lastFaceMs = (System.nanoTime() - t0) / 1_000_000
                if (emb == null) {
                    results.add(Result.FaceRes(id, 0, "", 0f))
                } else {
                    val best = FamilyStore.best(emb)
                    if (best != null && best.second >= threshold) {
                        results.add(Result.FaceRes(id, 2, best.first, best.second))
                    } else {
                        results.add(Result.FaceRes(id, 1, "", best?.second ?: 0f))
                    }
                }
            } catch (e: Throwable) {
                Log.e("Worker", "face check failed", e)
                results.add(Result.Fail(id))
            } finally {
                crop.recycle()
                pendingCount.decrementAndGet()
            }
        }
    }
}
