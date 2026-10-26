package app.modestyblur

/** Binary silhouette (0/1 bytes) covering the rectangle [x0, x0+w) x [y0, y0+h) of the frame. */
class Mask(val x0: Int, val y0: Int, val w: Int, val h: Int, val data: ByteArray)

class Det(
    val x1: Float, val y1: Float, val x2: Float, val y2: Float,
    val score: Float, val mask: Mask,
)

class Track(val id: Int, var det: Det) {
    var lost: Int = 0
}
