package app.modestyblur

/**
 * Same fail-closed policy as the Python version, now time-based because the slow models run
 * asynchronously: blur by default, clear only with strong evidence, re-arm on any doubt.
 */
class DecisionEngine {
    var blurThreshold = 0.30f
    private val clearThreshold = 0.12f
    private val minObsToClear = 3
    private val alpha = 0.5f
    private val steadyClassifyMs = 3000L       // re-check each person this often once warmed up
    private val familyCheckMs = 1500L          // face check cadence for blurred people
    private val familyRecheckMs = 4000L        // re-verify people already accepted as family
    private val familyHitsRequired = 3

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
    }

    private val states = HashMap<Int, State>()

    fun state(id: Int): State = states.getOrPut(id) { State() }

    fun needsClassification(id: Int, nowMs: Long): Boolean {
        val s = state(id)
        return when {
            s.nObs == 0 -> true
            s.nObs < minObsToClear -> true                      // warm-up: classify back to back
            else -> nowMs - s.lastClsMs >= steadyClassifyMs
        }
    }

    fun needsFamilyCheck(id: Int, nowMs: Long): Boolean {
        val s = state(id)
        if (s.nObs == 0 || s.cleared) return false
        val every = if (s.isFamily) familyRecheckMs else familyCheckMs
        return nowMs - s.lastFamMs >= every
    }

    fun update(id: Int, pFemale: Float, nowMs: Long) {
        val s = state(id)
        s.lastClsMs = nowMs
        s.ema = if (s.nObs == 0) pFemale else alpha * pFemale + (1 - alpha) * s.ema
        s.nObs++
        if (s.ema >= blurThreshold) {
            s.cleared = false
        } else if (s.nObs >= minObsToClear && s.ema <= clearThreshold) {
            s.cleared = true
        }
    }

    /** status: 0 no face found, 1 face found but not a family member, 2 family match. */
    fun updateFamily(id: Int, status: Int, name: String, sim: Float, nowMs: Long) {
        val s = state(id)
        s.lastFamMs = nowMs
        s.lastSim = sim
        when (status) {
            2 -> {
                s.familyHits++
                if (s.familyHits >= familyHitsRequired) {
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
