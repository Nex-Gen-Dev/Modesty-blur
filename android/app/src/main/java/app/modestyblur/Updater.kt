package app.modestyblur

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.Settings
import android.widget.Toast
import androidx.core.content.FileProvider
import org.json.JSONObject
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/** Checks GitHub Releases for a newer APK, downloads it and hands it to the system installer. */
object Updater {
    class Release(val code: Int, val name: String, val apkUrl: String)

    /** Latest release, or null if none / unreachable (a private repo answers 404). */
    fun fetchLatest(): Release? = try {
        val url = URL("https://api.github.com/repos/${BuildConfig.GITHUB_REPO}/releases/latest")
        val c = url.openConnection() as HttpURLConnection
        c.setRequestProperty("Accept", "application/vnd.github+json")
        c.setRequestProperty("User-Agent", "ModestyBlur")
        c.connectTimeout = 10_000
        c.readTimeout = 15_000
        if (c.responseCode != 200) {
            null
        } else {
            val j = JSONObject(c.inputStream.bufferedReader().use { it.readText() })
            val tag = j.getString("tag_name")                    // e.g. v0.1.17
            val code = tag.substringAfterLast('.').toIntOrNull()
            val assets = j.getJSONArray("assets")
            var apk: String? = null
            for (i in 0 until assets.length()) {
                val a = assets.getJSONObject(i)
                if (a.getString("name").endsWith(".apk")) apk = a.getString("browser_download_url")
            }
            if (code == null || apk == null) null else Release(code, tag.removePrefix("v"), apk)
        }
    } catch (e: Exception) {
        null
    }

    fun isNewer(r: Release) = r.code > BuildConfig.BUILD_NUMBER

    /** Check now (blocking, call off the main thread). Remembers a newer release in Prefs. */
    fun check(ctx: Context): Release? {
        val r = fetchLatest()
        Prefs.setLastCheck(ctx, System.currentTimeMillis())
        if (r != null && isNewer(r)) {
            Prefs.setPending(ctx, r.code, r.name, r.apkUrl)
            return r
        }
        if (r != null) Prefs.setPending(ctx, 0, "", "")
        return null
    }

    fun pending(ctx: Context): Release? {
        val code = Prefs.pendingCode(ctx)
        return if (code > BuildConfig.BUILD_NUMBER)
            Release(code, Prefs.pendingName(ctx), Prefs.pendingUrl(ctx)) else null
    }

    /** Download in the background, then open the system installer. */
    fun downloadAndInstall(act: Activity, r: Release, onStatus: (String) -> Unit) {
        if (!act.packageManager.canRequestPackageInstalls()) {
            Toast.makeText(act, "Allow installs from this app, then tap Install again", Toast.LENGTH_LONG).show()
            act.startActivity(
                Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES, Uri.parse("package:${act.packageName}"))
            )
            return
        }
        Thread {
            try {
                val dir = File(act.cacheDir, "updates").apply { mkdirs() }
                val f = File(dir, "update.apk")
                val c = URL(r.apkUrl).openConnection() as HttpURLConnection
                c.connectTimeout = 15_000
                c.readTimeout = 30_000
                val total = c.contentLengthLong
                c.inputStream.use { ins ->
                    f.outputStream().use { out ->
                        val buf = ByteArray(64 * 1024)
                        var read = 0L
                        var lastPct = -1
                        while (true) {
                            val n = ins.read(buf)
                            if (n < 0) break
                            out.write(buf, 0, n)
                            read += n
                            if (total > 0) {
                                val pct = (read * 100 / total).toInt()
                                if (pct != lastPct) {
                                    lastPct = pct
                                    act.runOnUiThread { onStatus("Downloading $pct%") }
                                }
                            }
                        }
                    }
                }
                act.runOnUiThread {
                    onStatus("Opening installer...")
                    install(act, f)
                }
            } catch (e: Exception) {
                act.runOnUiThread { onStatus("Download failed: ${e.message}") }
            }
        }.start()
    }

    private fun install(ctx: Context, f: File) {
        val uri = FileProvider.getUriForFile(ctx, "${ctx.packageName}.fileprovider", f)
        val i = Intent(Intent.ACTION_VIEW)
            .setDataAndType(uri, "application/vnd.android.package-archive")
            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_ACTIVITY_NEW_TASK)
        ctx.startActivity(i)
    }
}
