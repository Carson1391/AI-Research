package com.exposurewatch.app.ui

import android.graphics.Color
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.exposurewatch.app.databinding.ItemSignalBinding
import com.exposurewatch.app.engine.Vendors
import com.exposurewatch.app.model.RiskTier
import com.exposurewatch.app.model.Signal
import com.exposurewatch.app.model.SignalKind

class SignalsAdapter(
    private val onClick: (String) -> Unit = {}
) : RecyclerView.Adapter<SignalsAdapter.VH>() {

    private val items = ArrayList<Signal>()

    fun submit(list: List<Signal>) {
        items.clear()
        items.addAll(list.sortedWith(compareByDescending<Signal> { it.score }.thenByDescending { it.rssi }))
        notifyDataSetChanged()
    }

    class VH(val b: ItemSignalBinding) : RecyclerView.ViewHolder(b.root)

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val b = ItemSignalBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return VH(b)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        val s = items[position]
        val b = holder.b
        val kind = if (s.kind == SignalKind.WIFI) "Wi-Fi" else "BLE"
        val ref = if (s.kind == SignalKind.WIFI) -45 else -59
        val dist = Vendors.distanceBucket(s.rssi, ref)

        val title = s.label.ifBlank { s.id }
        b.txtName.text = if (s.trackerName.isNotBlank()) "\u26A0 $title" else title

        val vendor = when {
            s.trackerName.isNotBlank() -> s.trackerName
            s.vendor.isNotBlank() -> s.vendor
            s.randomized -> "randomized MAC"
            else -> "unknown vendor"
        }
        b.txtMeta.text = "$kind \u00B7 $vendor \u00B7 ${s.rssi} dBm \u00B7 $dist"
        b.txtNote.text = buildString {
            if (s.randomized) append("Hidden vendor (randomized). ")
            append(s.riskNote)
        }.trim()

        b.bar.setBackgroundColor(tierColor(s.riskTier))
        b.txtTier.text = when (s.riskTier) {
            RiskTier.SUSPECT -> "SUSPECT"
            RiskTier.WATCH -> "WATCH"
            RiskTier.NORMAL -> "OK"
        }
        b.txtTier.setTextColor(tierColor(s.riskTier))
        b.root.setOnClickListener { onClick(s.key) }
    }

    override fun getItemCount(): Int = items.size

    private fun tierColor(t: RiskTier): Int = when (t) {
        RiskTier.SUSPECT -> Color.parseColor("#FF5252")
        RiskTier.WATCH -> Color.parseColor("#FFB300")
        RiskTier.NORMAL -> Color.parseColor("#22D3A6")
    }
}
