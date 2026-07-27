package com.exposurewatch.app.wifi

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.net.wifi.ScanResult
import android.net.wifi.WifiManager
import android.os.Build
import com.exposurewatch.app.engine.Fingerprint
import com.exposurewatch.app.engine.Vendors
import com.exposurewatch.app.model.Signal
import com.exposurewatch.app.model.SignalKind

/**
 * Wraps WifiManager scanning. Reads the latest available scan results and asks
 * the OS to refresh them (subject to Android's scan throttling).
 */
class WifiScanner(
    private val context: Context,
    private val onResults: (List<Signal>) -> Unit
) {
    private val wifi = context.applicationContext
        .getSystemService(Context.WIFI_SERVICE) as WifiManager
    private var registered = false

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(c: Context?, intent: Intent?) { emitLatest() }
    }

    fun start() {
        if (registered) return
        val filter = IntentFilter(WifiManager.SCAN_RESULTS_AVAILABLE_ACTION)
        if (Build.VERSION.SDK_INT >= 33) {
            context.registerReceiver(receiver, filter, Context.RECEIVER_EXPORTED)
        } else {
            @Suppress("UnspecifiedRegisterReceiverFlag")
            context.registerReceiver(receiver, filter)
        }
        registered = true
        emitLatest()
    }

    fun stop() {
        if (!registered) return
        runCatching { context.unregisterReceiver(receiver) }
        registered = false
    }

    /** Force-read whatever scan results are currently cached by the OS. */
    fun pollNow() = emitLatest()

    /** Kick a fresh scan; harmless if throttled - cached results are still read. */
    fun requestScan() {
        runCatching {
            @Suppress("DEPRECATION")
            wifi.startScan()
        }
    }

    private fun emitLatest() {
        val now = System.currentTimeMillis()
        val results: List<ScanResult> = runCatching { wifi.scanResults }.getOrDefault(emptyList())
        if (results.isEmpty()) return
        val signals = results.mapNotNull { r -> toSignal(r, now) }
        if (signals.isNotEmpty()) onResults(signals)
    }

    private fun toSignal(r: ScanResult, now: Long): Signal? {
        val bssid = r.BSSID ?: return null
        val ssid = ssidOf(r)
        val width = if (Build.VERSION.SDK_INT >= 23) r.channelWidth else -1
        val mc = Build.VERSION.SDK_INT >= 28 && runCatching { r.is80211mcResponder }.getOrDefault(false)
        val caps = r.capabilities ?: ""

        val standard = if (Build.VERSION.SDK_INT >= 30)
            runCatching { r.wifiStandard }.getOrDefault(0) else 0
        val ieList: List<Pair<Int, ByteArray>> = if (Build.VERSION.SDK_INT >= 30) {
            runCatching {
                r.informationElements.map { ie ->
                    val buf = ie.bytes.duplicate()
                    val arr = ByteArray(buf.remaining())
                    buf.get(arr)
                    ie.id to arr
                }
            }.getOrDefault(emptyList())
        } else emptyList()
        val ieHash = Fingerprint.ieHash(ieList)

        val fp = Fingerprint.wifi(r.frequency, width, caps, mc, standard, ieHash)
        return Signal(
            kind = SignalKind.WIFI,
            id = bssid,
            label = ssid,
            rssi = r.level,
            frequency = r.frequency,
            capabilities = caps,
            extra = "cw=$width;mc=$mc;std=$standard;ie=$ieHash",
            fingerprint = fp,
            firstSeen = now,
            lastSeen = now,
            vendor = Vendors.wifiVendor(bssid),
            randomized = Vendors.isRandomizedMac(bssid)
        )
    }

    private fun ssidOf(r: ScanResult): String {
        return if (Build.VERSION.SDK_INT >= 33) {
            runCatching { r.wifiSsid?.toString()?.trim('"') ?: "" }.getOrDefault("")
        } else {
            @Suppress("DEPRECATION")
            (r.SSID ?: "")
        }
    }
}
