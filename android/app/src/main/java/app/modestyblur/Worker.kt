package app.modestyblur

import android.graphics.Bitmap
import android.util.Log
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

/**
 * Runs the slow models (person classifier, face recognition) on a background thread so the live view
 * never waits for them. Until a result arrives the person simply stays blurred.
 */
class Worker(private val scorer: (Bitmap) -> Float, private val face: FaceEngine?) {
    sealed class Result {
        class Cls(val id: Int, val p: Float) : Result()
        class FaceRes(val id: Int, val status: Int, val name: String, val sim: Float, val pFemale: Float) : Result()
        class Fail(val id: Int) : Result()
    }

    val results = ConcurrentLinkedQueue<Result>()
    private val pool = Executors.newSingleThreadExecutor()
    private val pendingCount = AtomicInteger(0)

    @Volatile var lastClsMs = 0L
    @Volatile var lastFaceMs = 0L

    val faceAvailable get() = face != null
    fun pending() = pendingCount.get()

    fun classify(id: Int, crop: Bitmap) {
        pendingCount.incrementAndGet()
        pool.execute {
            try {
                val t0 = System.nanoTime()
                val p = scorer(crop)
                lastClsMs = (System.nanoTime() - t0) / 1_000_000
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

    /** Looks for a face in [crop]: identity vs the family gallery, plus the face-gender reading. */
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
                val info = fe.analyzeLargest(crop, 320, 22f)
                lastFaceMs = (System.nanoTime() - t0) / 1_000_000
                if (info == null) {
                    results.add(Result.FaceRes(id, 0, "", 0f, -1f))
                } else {
                    val best = FamilyStore.best(info.emb)
                    if (best != null && best.second >= threshold) {
                        results.add(Result.FaceRes(id, 2, best.first, best.second, info.pFemale))
                    } else {
                        results.add(Result.FaceRes(id, 1, "", best?.second ?: 0f, info.pFemale))
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
