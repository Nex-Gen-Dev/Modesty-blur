package app.modestyblur

import android.content.Context
import android.content.SharedPreferences
import android.util.Size

/** Speed/quality presets: camera resolution (always 16:9, shown in portrait) + segmentation model size. */
object Presets {
    val names = listOf("Fast", "Balanced", "Quality", "1080p")
    val details = listOf(
        "Smoothest. 360p camera, small model.",
        "Recommended. 720p camera, medium model.",
        "720p camera, larger model for cleaner edges.",
        "1080p camera and the largest model. Sharpest, but needs a fast phone.",
    )

    fun yoloAsset(p: Int) =
        listOf("yolo_fast.onnx", "yolo_balanced.onnx", "yolo_quality.onnx", "yolo_ultra.onnx")[p.coerceIn(0, 3)]

    /** Landscape sensor size; the app rotates frames to portrait. */
    fun camSize(p: Int): Size = when (p) {
        0 -> Size(640, 360)
        3 -> Size(1920, 1080)
        else -> Size(1280, 720)
    }
}

object Prefs {
    private fun sp(c: Context): SharedPreferences = c.getSharedPreferences("mb", Context.MODE_PRIVATE)

    fun autoUpdate(c: Context) = sp(c).getBoolean("auto_update", true)
    fun setAutoUpdate(c: Context, v: Boolean) = sp(c).edit().putBoolean("auto_update", v).apply()

    fun debug(c: Context) = sp(c).getBoolean("debug", false)
    fun setDebug(c: Context, v: Boolean) = sp(c).edit().putBoolean("debug", v).apply()

    /** 0 = Fast, 1 = Balanced, 2 = Quality, 3 = 1080p. */
    fun preset(c: Context) = sp(c).getInt("preset", 1).coerceIn(0, 3)
    fun setPreset(c: Context, v: Int) = sp(c).edit().putInt("preset", v).apply()

    /** Lower = safer (more people blurred). */
    fun blurThreshold(c: Context) = sp(c).getFloat("blur_threshold", 0.30f)
    fun setBlurThreshold(c: Context, v: Float) = sp(c).edit().putFloat("blur_threshold", v).apply()

    /** Minimum face similarity (0..1) to count as a family member. Higher = stricter. */
    fun familyThreshold(c: Context) = sp(c).getFloat("family_threshold", 0.45f)
    fun setFamilyThreshold(c: Context, v: Float) = sp(c).edit().putFloat("family_threshold", v).apply()

    /** Minimum detector confidence to treat something as a person. Higher = fewer blurred feet/floor patches. */
    fun detectConf(c: Context) = sp(c).getFloat("detect_conf", 0.35f)
    fun setDetectConf(c: Context, v: Float) = sp(c).edit().putFloat("detect_conf", v).apply()

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
