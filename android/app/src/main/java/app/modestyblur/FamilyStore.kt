package app.modestyblur

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * The family whitelist: a name plus one or more 512-number face embeddings per person.
 * Stored only in this app's private storage. Photos are never kept, only the numbers.
 */
object FamilyStore {
    class Person(val name: String, val embs: MutableList<FloatArray>)

    private val people = ArrayList<Person>()
    private var loaded = false

    private fun file(ctx: Context) = File(ctx.filesDir, "family.json")

    @Synchronized
    fun load(ctx: Context) {
        if (loaded) return
        loaded = true
        people.clear()
        try {
            val f = file(ctx)
            if (!f.exists()) return
            val arr = JSONArray(f.readText())
            for (i in 0 until arr.length()) {
                val o = arr.getJSONObject(i)
                val embs = ArrayList<FloatArray>()
                val ea = o.getJSONArray("embs")
                for (j in 0 until ea.length()) {
                    val row = ea.getJSONArray(j)
                    embs.add(FloatArray(row.length()) { k -> row.getDouble(k).toFloat() })
                }
                people.add(Person(o.getString("name"), embs))
            }
        } catch (e: Exception) {
            people.clear()
        }
    }

    @Synchronized
    private fun save(ctx: Context) {
        val arr = JSONArray()
        for (p in people) {
            val ea = JSONArray()
            for (e in p.embs) ea.put(JSONArray(e.map { it.toDouble() }))
            arr.put(JSONObject().put("name", p.name).put("embs", ea))
        }
        file(ctx).writeText(arr.toString())
    }

    @Synchronized
    fun isEmpty(): Boolean = people.isEmpty()

    @Synchronized
    fun list(): List<Pair<String, Int>> = people.map { it.name to it.embs.size }

    @Synchronized
    fun add(ctx: Context, name: String, embs: List<FloatArray>) {
        val existing = people.firstOrNull { it.name.equals(name, ignoreCase = true) }
        if (existing != null) existing.embs.addAll(embs) else people.add(Person(name, embs.toMutableList()))
        save(ctx)
    }

    @Synchronized
    fun remove(ctx: Context, name: String) {
        people.removeAll { it.name == name }
        save(ctx)
    }

    /** Best (name, cosine similarity) over everyone enrolled; embeddings are already normalized. */
    @Synchronized
    fun best(emb: FloatArray): Pair<String, Float>? {
        var bestName: String? = null
        var bestSim = -1f
        for (p in people) {
            for (e in p.embs) {
                if (e.size != emb.size) continue
                var dot = 0f
                for (i in e.indices) dot += e[i] * emb[i]
                if (dot > bestSim) { bestSim = dot; bestName = p.name }
            }
        }
        return bestName?.let { it to bestSim }
    }
}
