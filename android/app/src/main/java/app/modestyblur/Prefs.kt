package app.modestyblur

import android.content.Context
import android.content.SharedPreferences
import android.util.Size

/** Speed/quality presets: camera resolution + segmentation model size. */
object Presets {
    val names = listOf("Fast", "Balanced", "Quality")
    val details = listOf(
        "Smoothest. 640x480 camera, small model.",
        "Recommended. 720p camera, medium model.",
        "Sharpest and most accurate edges. Slower.",
    )

    fun yoloAsset(p: Int) = listOf("yolo_fast.onnx", "yolo_balanced.onnx", "yolo_quality.onnx")[p.coerceIn(0, 2)]
    fun camSize(p: Int): Size = if (p == 0) Size(640, 480) else Size(1280, 720)
}

object Prefs {
    private fun sp(c: Context): SharedPreferences = c.getSharedPreferences("mb", Context.MODE_PRIVATE)

    fun autoUpdate(c: Context) = sp(c).getBoolean("auto_update", true)
    fun setAutoUpdate(c: Context, v: Boolean) = sp(c).edit().putBoolean("auto_update", v).apply()

    fun debug(c: Context) = sp(c).getBoolean("debug", false)
    fun setDebug(c: Context, v: Boolean) = sp(c).edit().putBoolean("debug", v).apply()

    /** 0 = Fast, 1 = Balanced, 2 = Quality. */
    fun preset(c: Context) = sp(c).getInt("preset", 1)
    fun setPreset(c: Context, v: Int) = sp(c).edit().putInt("preset", v).apply()

    /** Lower = safer (more people blurred). */
    fun blurThreshold(c: Context) = sp(c).getFloat("blur_threshold", 0.30f)
    fun setBlurThreshold(c: Context, v: Float) = sp(c).edit().putFloat("blur_threshold", v).apply()

    /** Minimum face similarity (0..1) to count as a family member. Higher = stricter. */
    fun familyThreshold(c: Context) = sp(c).getFloat("family_threshold", 0.45f)
    fun setFamilyThreshold(c: Context, v: Float) = sp(c).edit().putFloat("family_threshold", v).apply()

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
