package com.exposurewatch.app.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

/** A 240-degree arc gauge showing a 0-100 score with a graded colour. */
class GaugeView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyle: Int = 0
) : View(context, attrs, defStyle) {

    private var value = 0
    private val startAngle = 150f
    private val sweepMax = 240f

    private val track = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND
        color = Color.parseColor("#1E2A38")
    }
    private val arc = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE; strokeCap = Paint.Cap.ROUND
        color = Color.parseColor("#22D3A6")
    }
    private val big = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE; textAlign = Paint.Align.CENTER; isFakeBoldText = true
    }
    private val small = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#7C8CA0"); textAlign = Paint.Align.CENTER
    }
    private val rect = RectF()

    fun setValue(v: Int) {
        value = v.coerceIn(0, 100)
        arc.color = colorFor(value)
        invalidate()
    }

    private fun colorFor(v: Int): Int = when {
        v >= 70 -> Color.parseColor("#FF5252")
        v >= 40 -> Color.parseColor("#FFB300")
        else -> Color.parseColor("#22D3A6")
    }

    override fun onDraw(canvas: Canvas) {
        val size = min(width, height).toFloat()
        val pad = size * 0.14f
        val stroke = size * 0.09f
        track.strokeWidth = stroke
        arc.strokeWidth = stroke
        rect.set(pad, pad, size - pad, size - pad)

        canvas.drawArc(rect, startAngle, sweepMax, false, track)
        val sweep = sweepMax * (value / 100f)
        canvas.drawArc(rect, startAngle, sweep, false, arc)

        big.textSize = size * 0.26f
        small.textSize = size * 0.075f
        val cx = size / 2f
        canvas.drawText(value.toString(), cx, size / 2f + big.textSize * 0.34f, big)
        canvas.drawText("EXPOSURE", cx, size * 0.74f, small)
    }
}
