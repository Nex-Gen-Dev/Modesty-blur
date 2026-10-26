package app.modestyblur

import android.content.Context
import android.content.SharedPreferences

object Prefs {
    private fun sp(c: Context): SharedPreferences = c.getSharedPreferences("mb", Context.MODE_PRIVATE)

    fun autoUpdate(c: Context) = sp(c).getBoolean("auto_update", true)
    fun setAutoUpdate(c: Context, v: Boolean) = sp(c).edit().putBoolean("auto_update", v).apply()

    fun debug(c: Context) = sp(c).getBoolean("debug", false)
    fun setDebug(c: Context, v: Boolean) = sp(c).edit().putBoolean("debug", v).apply()

    /** Lower = safer (more people blurred). */
    fun blurThreshold(c: Context) = sp(c).getFloat("blur_threshold", 0.30f)
    fun setBlurThreshold(c: Context, v: Float) = sp(c).edit().putFloat("blur_threshold", v).apply()

    // An update the user postponed ("Later") stays here so Settings can always offer it.
    fun pendingCode(c: Context) = sp(c).getInt("pending_code", 0)
    fun pendingName(c: Context) = sp(c).getString("pending_name", "") ?: ""
    fun pendingUrl(c: Context) = sp(c).getString("pending_url", "") ?: ""
    fun setPending(c: Context, code: Int, name: String, url: String) =
        sp(c).edit().putInt("pending_code", code).putString("pending_name", name)
            .putString("pending_url", url).apply()

    fun lastCheck(c: Context) = sp(c).getLong("last_check", 0L)
    fun setLastCheck(c: Context, t: Long) = sp(c).edit().putLong("last_check", t).apply()
}
