package app.modestyblur

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import kotlin.math.max

class DebugItem(
    val x1: Float, val y1: Float, val x2: Float, val y2: Float,
    val color: Int, val lines: List<String>,
)

/** A snapshot of what the debug overlay should show; drawn on any canvas in frame coordinates. */
class DebugInfo(val items: List<DebugItem>, val panel: List<String>) {
    private val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.WHITE }
    private val shadow = Paint().apply { color = Color.argb(150, 0, 0, 0); style = Paint.Style.FILL }
    private val box = Paint().apply { style = Paint.Style.STROKE }

    fun draw(c: Canvas, frameWidth: Int) {
        val ts = max(frameWidth * 0.024f, 14f)
        text.textSize = ts
        box.strokeWidth = max(2f, ts / 8f)

        for (it in items) {
            box.color = it.color
            c.drawRect(it.x1, it.y1, it.x2, it.y2, box)
            val bw = it.lines.maxOf { l -> text.measureText(l) } + ts * 0.6f
            val bh = ts * 1.25f * it.lines.size + ts * 0.3f
            val bx = it.x1.coerceIn(0f, max(0f, frameWidth - bw))
            val by = (it.y1 - bh).let { v -> if (v < 0f) it.y1 else v }
            c.drawRect(bx, by, bx + bw, by + bh, shadow)
            text.color = it.color
            for ((i, line) in it.lines.withIndex()) {
                c.drawText(line, bx + ts * 0.3f, by + ts * (1.25f * (i + 1)), text)
            }
        }

        val pw = panel.maxOf { text.measureText(it) } + ts * 0.8f
        val ph = ts * 1.25f * panel.size + ts * 0.4f
        c.drawRect(0f, 0f, pw, ph, shadow)
        text.color = Color.WHITE
        for ((i, line) in panel.withIndex()) c.drawText(line, ts * 0.4f, ts * 1.25f * (i + 1), text)
    }
}
