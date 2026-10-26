package app.modestyblur

import kotlin.math.max

/**
 * Same fail-closed policy as the Python version, time-based because the slow models run asynchronously.
 *
 * Rules of thumb:
 *  - Everyone is blurred until cleared. Clearing needs strong evidence (family face match, or the
 *    person classifier repeatedly saying "not female").
 *  - A face-gender reading can only ADD blur (a face that looks female blocks clearing); it can never
 *    remove it, because face-gender models are known to be overconfident on hats, uniforms and angles.
 */
class DecisionEngine {
    var blurThreshold = 0.30f
    var familyThreshold = 0.45f
    var steadyClassifyMs = 3000L            // how often a settled person is re-classified

    private val clearThreshold = 0.12f
    private val minObsToClear = 2
    private val alpha = 0.5f
    private val familyHitsRequired = 2
    private val strongMarginForInstant = 0.12f   // a face match this far above the threshold is accepted at once

    class State {
        var ema = 1f
        var nObs = 0
        var cleared = false
        var lastClsMs = 0L
        var lastFamMs = 0L
        var familyHits = 0
        var isFamily = false
        var familyName = ""
        var lastSim = 0f
        var faceChecks = 0
        var lastFaceStatus = -1      // -1 never, 0 no face, 1 face but not family, 2 family match
        var facePFemale = -1f        // -1 = no face reading
    }

    private val states = HashMap<Int, State>()

    fun state(id: Int): State = states.getOrPut(id) { State() }

    fun reset() = states.clear()

    fun needsClassification(id: Int, nowMs: Long): Boolean {
        val s = state(id)
        return when {
            s.nObs == 0 -> true
            s.nObs < minObsToClear -> true                      // warm-up: classify back to back
            else -> nowMs - s.lastClsMs >= steadyClassifyMs
        }
    }

    fun needsFaceCheck(id: Int, nowMs: Long): Boolean {
        val s = state(id)
        val every = when {
            s.isFamily -> 4000L                                  // keep re-verifying accepted family
            s.cleared -> 3000L                                   // second opinion on people cleared as male
            s.familyHits in 1 until familyHitsRequired -> 0L     // confirm a promising match immediately
            s.lastFaceStatus == 0 -> 300L                        // no face yet: look again very soon
            s.faceChecks == 0 -> 0L
            else -> 1500L
        }
        return nowMs - s.lastFamMs >= every
    }

    fun update(id: Int, pFemale: Float, nowMs: Long) {
        val s = state(id)
        s.lastClsMs = nowMs
        s.ema = if (s.nObs == 0) pFemale else alpha * pFemale + (1 - alpha) * s.ema
        s.nObs++
        if (s.ema >= blurThreshold) {
            s.cleared = false
        } else if (s.nObs >= minObsToClear && s.ema <= clearThreshold && s.facePFemale < 0.5f) {
            s.cleared = true
        }
    }

    /**
     * Result of a face check. [status]: 0 no face found, 1 face but not family, 2 family match.
     * [pFemale] is the face-gender reading, or -1 if none.
     */
    fun updateFace(id: Int, status: Int, name: String, sim: Float, pFemale: Float, nowMs: Long) {
        val s = state(id)
        s.lastFamMs = nowMs
        s.faceChecks++
        s.lastFaceStatus = status
        s.lastSim = sim
        s.facePFemale = pFemale

        // blur-only signal
        if (pFemale >= 0.5f) {
            s.cleared = false
            s.ema = max(s.ema, pFemale)
        }

        when (status) {
            2 -> {
                s.familyHits++
                if (sim >= familyThreshold + strongMarginForInstant || s.familyHits >= familyHitsRequired) {
                    s.isFamily = true
                    s.familyName = name
                }
            }
            1 -> {                       // a visible face that does not match: reset and revoke
                s.familyHits = 0
                s.isFamily = false
                s.familyName = ""
            }
            else -> {}                   // no face visible: neither confirms nor revokes
        }
    }

    /** Pending, uncertain and female all return true. */
    fun shouldBlur(id: Int): Boolean {
        val s = state(id)
        return !(s.isFamily || s.cleared)
    }

    fun prune(keep: Set<Int>) {
        states.keys.retainAll(keep)
    }
}
