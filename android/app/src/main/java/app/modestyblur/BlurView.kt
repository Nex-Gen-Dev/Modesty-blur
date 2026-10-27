package app.modestyblur

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

/** What to show for one camera frame. A null [bmp] means "something went wrong": the screen goes black. */
class LiveFrame(val bmp: Bitmap?, val sprites: List<Sprite>, val debug: DebugInfo?)

/**
 * Draws the camera frame and the blurred silhouettes on the GPU. The camera image is never touched on the
 * CPU: each person is a small colour grid stretched over their silhouette and cut out with the mask.
 */
class BlurView(ctx: Context, attrs: AttributeSet? = null) : View(ctx, attrs) {
    @Volatile private var frame: LiveFrame? = null

    private val base = Paint(Paint.FILTER_BITMAP_FLAG)
    private val smooth = Paint(Paint.FILTER_BITMAP_FLAG)
    private val cut = Paint(Paint.FILTER_BITMAP_FLAG).apply {
        xfermode = PorterDuffXfermode(PorterDuff.Mode.DST_IN)
    }
    private val layerRect = RectF()
    private val dstA = RectF()
    private val dstB = RectF()

    /** Safe to call from any thread. */
    fun show(f: LiveFrame) {
        frame = f
        postInvalidateOnAnimation()
    }

    @Suppress("DEPRECATION")
    override fun onDraw(canvas: Canvas) {
        canvas.drawColor(Color.BLACK)
        val f = frame ?: return
        val b = f.bmp ?: return                       // fail closed: black
        val s = min(width.toFloat() / b.width, height.toFloat() / b.height)
        val ox = (width - b.width * s) / 2f
        val oy = (height - b.height * s) / 2f
        canvas.save()
        canvas.translate(ox, oy)
        canvas.scale(s, s)
        canvas.drawBitmap(b, 0f, 0f, base)
        for (sp in f.sprites) {
            layerRect.set(sp.mx, sp.my, sp.mx + sp.mw, sp.my + sp.mh)
            val layer = canvas.saveLayer(layerRect, null)
            dstA.set(sp.gx, sp.gy, sp.gx + sp.gw, sp.gy + sp.gh)
            canvas.drawBitmap(sp.grid, null, dstA, smooth)
            dstB.set(sp.mx, sp.my, sp.mx + sp.mw, sp.my + sp.mh)
            canvas.drawBitmap(sp.mask, null, dstB, cut)
            canvas.restoreToCount(layer)
        }
        f.debug?.draw(canvas, b.width)
        canvas.restore()
    }
}
