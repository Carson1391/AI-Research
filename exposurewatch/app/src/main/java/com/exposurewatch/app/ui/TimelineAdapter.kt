package com.exposurewatch.app.ui

import android.graphics.Color
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.exposurewatch.app.databinding.ItemTimelineBinding
import com.exposurewatch.app.model.EventRecord
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class TimelineAdapter : RecyclerView.Adapter<TimelineAdapter.VH>() {

    private val items = ArrayList<EventRecord>()
    private val fmt = SimpleDateFormat("MMM d - HH:mm:ss", Locale.getDefault())

    fun submit(list: List<EventRecord>) {
        items.clear(); items.addAll(list); notifyDataSetChanged()
    }

    class VH(val b: ItemTimelineBinding) : RecyclerView.ViewHolder(b.root)

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val b = ItemTimelineBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return VH(b)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        val e = items[position]
        val b = holder.b
        b.txtTitle.text = e.title
        b.txtTime.text = fmt.format(Date(e.timestamp))
        b.txtScore.text = e.score.toString()
        b.chipScore.setCardBackgroundColor(colorFor(e.score))
        b.txtEvidence.text = e.evidence.joinToString("\n") { "\u2022 $it" }
    }

    override fun getItemCount(): Int = items.size

    private fun colorFor(v: Int): Int = when {
        v >= 70 -> Color.parseColor("#FF5252")
        v >= 40 -> Color.parseColor("#FFB300")
        else -> Color.parseColor("#22D3A6")
    }
}
