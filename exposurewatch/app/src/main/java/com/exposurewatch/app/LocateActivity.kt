package com.exposurewatch.app

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.exposurewatch.app.databinding.ActivityLocateBinding
import com.exposurewatch.app.engine.Repository
import com.exposurewatch.app.engine.Vendors
import com.exposurewatch.app.model.SignalKind

/**
 * Source detail: separates the CLAIMED identity (MAC/vendor/name - all spoofable)
 * from the BEHAVIOUR (where/how often/RSSI shape/follows-you) that actually earns
 * trust, plus a live strength meter to walk the source down.
 */
class LocateActivity : AppCompatActivity() {

    private lateinit var b: ActivityLocateBinding
    private var key: String = ""
    private var peak = -200
    private val refresh = { runOnUiThread { render() } }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityLocateBinding.inflate(layoutInflater)
        setContentView(b.root)
        key = intent.getStringExtra(EXTRA_KEY) ?: ""
        b.btnBack.setOnClickListener { finish() }
        render()
    }

    override fun onStart() {
        super.onStart()
        Repository.addListener(refresh)
        render()
    }

    override fun onStop() {
        Repository.removeListener(refresh)
        super.onStop()
    }

    private fun render() {
        val live = Repository.live[key]
        val known = Repository.knownFor(key)
        val s = live ?: known ?: return
        val ref = if (s.kind == SignalKind.WIFI) -45 else -59

        val title = s.label.ifBlank { s.id }
        b.txtName.text = if (s.trackerName.isNotBlank()) "\u26A0 $title" else title

        b.txtClaimMac.text = "MAC/BSSID: ${s.id}"
        b.txtClaimVendor.text = "OUI vendor: " + when {
            s.randomized -> "randomized - vendor hidden"
            s.vendor.isNotBlank() -> s.vendor
            else -> "unknown"
        }
        b.txtClaimName.text = if (s.kind == SignalKind.WIFI)
            "SSID: ${s.label.ifBlank { "(hidden)" }}"
        else
            "Name: ${s.label.ifBlank { "(none)" }}"
        b.txtClaimType.text = if (s.kind == SignalKind.WIFI)
            "Type: Wi-Fi AP \u00B7 ${Vendors.securityOf(s.capabilities)} \u00B7 ${s.frequency} MHz"
        else
            "Type: BLE \u00B7 ${if (s.vendor.isNotBlank()) s.vendor else "unknown company"}" +
                if (s.trackerName.isNotBlank()) " \u00B7 ${s.trackerName}" else ""

        val beh = known ?: s
        val nonHome = beh.cells.count { !Repository.isHomeCell(it) }
        val homeOnly = beh.cells.isNotEmpty() && beh.cells.all { Repository.isHomeCell(it) }
        b.txtBehSeen.text = "Seen: ${beh.seenCount} times across ${beh.cells.size} location(s)"
        b.txtBehLocations.text = "Distinct non-home locations: $nonHome"
        b.txtBehRssi.text = "RSSI shape: ${beh.rssiMin}..${beh.rssiMax} dBm (now ${s.rssi})"
        b.txtBehFollows.text = "Follows you: " +
            if (nonHome >= 3) "YES - across $nonHome places" else "no evidence yet"
        b.txtBehHome.text = "Only near home: " + if (homeOnly) "yes" else "no"
        b.txtBehUnlock.text = "Appears with unlock/motion: " +
            if (beh.afterUnlockCount > 0 || beh.movingCount > beh.stillCount) "yes" else "no"
        b.txtBehContexts.text = "Baselines seen in: " +
            if (beh.contexts.isEmpty()) "-" else beh.contexts.joinToString(", ")

        if (live == null) {
            b.txtRssi.text = "out of range"
            b.txtDistance.text = "Not in range right now"
            b.barStrength.progress = 0
        } else {
            if (live.rssi > peak) peak = live.rssi
            b.txtRssi.text = "${live.rssi} dBm"
            b.barStrength.progress = ((live.rssi + 100) * 100 / 70).coerceIn(0, 100)
            b.txtDistance.text = Vendors.distanceBucket(live.rssi, ref)
            b.txtPeak.text = "peak: $peak dBm (closest you've been)"
        }
    }

    companion object {
        const val EXTRA_KEY = "key"
    }
}
