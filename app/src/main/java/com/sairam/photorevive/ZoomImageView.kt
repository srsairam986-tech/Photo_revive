package com.sairam.photorevive

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.util.AttributeSet
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import androidx.appcompat.widget.AppCompatImageView
import kotlin.math.max

/** Full-screen 922x2048 photo viewer: fills the frame, pinch / double-tap to zoom, drag to pan, tap = click. */
class ZoomImageView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : AppCompatImageView(context, attrs) {

    private val m = Matrix()
    private var zoom = 1f

    private val scaler = ScaleGestureDetector(context, object : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(d: ScaleGestureDetector): Boolean {
            val newZoom = (zoom * d.scaleFactor).coerceIn(1f, 10f)
            val f = newZoom / zoom
            zoom = newZoom
            m.postScale(f, f, d.focusX, d.focusY)
            keepInside()
            return true
        }
    })

    private val gestures = GestureDetector(context, object : GestureDetector.SimpleOnGestureListener() {
        override fun onScroll(e1: MotionEvent?, e2: MotionEvent, dx: Float, dy: Float): Boolean {
            if (zoom > 1f) { m.postTranslate(-dx, -dy); keepInside() }
            return true
        }

        override fun onDoubleTap(e: MotionEvent): Boolean {
            if (zoom > 1f) fitScreen() else {
                zoom = 3f
                m.postScale(3f, 3f, e.x, e.y)
                keepInside()
            }
            return true
        }

        override fun onSingleTapConfirmed(e: MotionEvent): Boolean {
            performClick()
            return true
        }
    })

    init {
        scaleType = ScaleType.MATRIX
    }

    override fun setImageBitmap(bm: Bitmap?) {
        super.setImageBitmap(bm)
        fitScreen()
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        fitScreen()
    }

    fun fitScreen() {
        val d = drawable ?: return
        val vw = width.toFloat()
        val vh = height.toFloat()
        if (vw <= 0f || vh <= 0f) return
        val dw = d.intrinsicWidth.toFloat()
        val dh = d.intrinsicHeight.toFloat()
        val s = max(vw / dw, vh / dh)          // fill the whole 922x2048 frame
        m.reset()
        m.postScale(s, s)
        m.postTranslate((vw - dw * s) / 2f, (vh - dh * s) / 2f)
        zoom = 1f
        imageMatrix = m
    }

    private fun keepInside() {
        val d = drawable ?: return
        val r = RectF(0f, 0f, d.intrinsicWidth.toFloat(), d.intrinsicHeight.toFloat())
        m.mapRect(r)
        var dx = 0f
        var dy = 0f
        if (r.width() <= width) dx = (width - r.width()) / 2f - r.left
        else if (r.left > 0) dx = -r.left
        else if (r.right < width) dx = width - r.right
        if (r.height() <= height) dy = (height - r.height()) / 2f - r.top
        else if (r.top > 0) dy = -r.top
        else if (r.bottom < height) dy = height - r.bottom
        m.postTranslate(dx, dy)
        imageMatrix = m
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(event: MotionEvent): Boolean {
        scaler.onTouchEvent(event)
        gestures.onTouchEvent(event)
        return true
    }

    override fun performClick(): Boolean = super.performClick()
}
