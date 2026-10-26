package app.modestyblur

/** Same fail-closed policy as the Python version: blur by default, clear only with strong evidence. */
class DecisionEngine(
    var blurThreshold: Float = 0.30f,
    private val clearThreshold: Float = 0.12f,
    private val minObsToClear: Int = 3,
    private val alpha: Float = 0.5f,
    private val classifyEvery: Int = 12,
) {
    class State {
        var ema = 1f
        var nObs = 0
        var cleared = false
        var lastCls = -1_000_000
    }

    private val states = HashMap<Int, State>()

    fun state(id: Int): State = states.getOrPut(id) { State() }

    fun needsClassification(id: Int, frame: Int): Boolean {
        val s = state(id)
        return s.nObs == 0 || frame - s.lastCls >= classifyEvery
    }

    fun update(id: Int, pFemale: Float, frame: Int) {
        val s = state(id)
        s.lastCls = frame
        s.ema = if (s.nObs == 0) pFemale else alpha * pFemale + (1 - alpha) * s.ema
        s.nObs++
        if (s.ema >= blurThreshold) {
            s.cleared = false
        } else if (s.nObs >= minObsToClear && s.ema <= clearThreshold) {
            s.cleared = true
        }
    }

    /** Pending, uncertain and female all return true. */
    fun shouldBlur(id: Int): Boolean = !state(id).cleared

    fun prune(keep: Set<Int>) {
        states.keys.retainAll(keep)
    }
}
