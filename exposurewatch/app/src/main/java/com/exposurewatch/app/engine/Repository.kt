package com.exposurewatch.app.engine

import android.content.Context
import android.os.Handler
import android.os.Looper
import com.exposurewatch.app.model.EventRecord
import com.exposurewatch.app.model.ScanContext
import com.exposurewatch.app.model.Signal
import org.json.JSONArray
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Single source of truth. The scan service writes here; the UI reads and
 * subscribes. Live signals are volatile; the baseline and alert timeline are
 * persisted to app-private encrypted storage.
 */
object Repository {

    val live = ConcurrentHashMap<String, Signal>()
    private val known = ConcurrentHashMap<String, Signal>()

    private val ssidToBssids = ConcurrentHashMap<String, MutableSet<String>>()
    private val fingerprintToIds = ConcurrentHashMap<String, MutableSet<String>>()

    /** Sources captured during the learning window = the "normal" environment. */
    private val baselineKeys = ConcurrentHashMap.newKeySet<String>()

    val events = CopyOnWriteArrayList<EventRecord>()

    @Volatile var monitoring = false
    @Volatile var passiveMode = false
    @Volatile var lastExposureScore = 0
    @Volatile var lastScanAt = 0L
    @Volatile var baselineLearning = true

    @Volatile var homeCell: String = ""
    @Volatile var currentContexts: Set<String> = emptySet()
    @Volatile var currentCell: String = ""

    private val listeners = CopyOnWriteArrayList<() -> Unit>()
    private val main = Handler(Looper.getMainLooper())
    private var appCtx: Context? = null
    private var eventSeq = System.currentTimeMillis()

    fun init(context: Context) {
        if (appCtx != null) return
        appCtx = context.applicationContext
        load()
    }

    fun addListener(l: () -> Unit) { listeners.add(l) }
    fun removeListener(l: () -> Unit) { listeners.remove(l) }
    fun notifyChanged() { main.post { listeners.forEach { it() } } }

    fun knownFor(key: String): Signal? = known[key]
    fun bssidsForSsid(ssid: String): Set<String> = ssidToBssids[ssid] ?: emptySet()
    fun idsForFingerprint(fp: String): Set<String> = fingerprintToIds[fp] ?: emptySet()
    fun knownCount(): Int = known.size

    fun isBaseline(key: String): Boolean = baselineKeys.contains(key)
    fun isHomeCell(cell: String): Boolean = homeCell.isNotBlank() && cell == homeCell

    fun setHome(cell: String) {
        if (cell.isBlank()) return
        homeCell = cell
        persistMeta()
        notifyChanged()
    }

    /** Freeze the current known set as the trusted environment, stop learning. */
    fun finalizeBaseline() {
        baselineKeys.addAll(known.keys)
        baselineLearning = false
        persistKnown()
        persistBaselineKeys()
    }

    /** Fold a freshly observed source into the baseline, updating its behaviour. */
    fun remember(s: Signal, ctx: ScanContext) {
        val prior = known[s.key]
        val target: Signal
        if (prior == null) {
            s.rssiMin = s.rssi
            s.rssiMax = s.rssi
            known[s.key] = s
            target = s
        } else {
            prior.lastSeen = s.lastSeen
            prior.rssi = s.rssi
            prior.seenCount += 1
            prior.fingerprint = s.fingerprint
            if (s.vendor.isNotBlank()) prior.vendor = s.vendor
            if (s.trackerName.isNotBlank()) prior.trackerName = s.trackerName
            prior.randomized = s.randomized
            if (s.rssi < prior.rssiMin || prior.rssiMin == 0) prior.rssiMin = s.rssi
            if (s.rssi > prior.rssiMax) prior.rssiMax = s.rssi
            target = prior
        }
        target.contexts.addAll(ctx.contexts)
        if (ctx.cell.isNotBlank()) target.cells.add(ctx.cell)
        if (ctx.moving) target.movingCount += 1 else target.stillCount += 1
        if (ctx.recentUnlock) target.afterUnlockCount += 1

        if (s.label.isNotBlank()) {
            ssidToBssids.getOrPut(s.label) { ConcurrentHashMap.newKeySet() }.add(s.id)
        }
        fingerprintToIds.getOrPut(s.fingerprint) { ConcurrentHashMap.newKeySet() }.add(s.id)
    }

    fun addEvent(e: EventRecord) {
        events.add(0, e)
        while (events.size > 500) events.removeAt(events.size - 1)
        persistEvents()
        notifyChanged()
    }

    fun nextEventId(): Long = ++eventSeq

    fun clearAll() {
        live.clear(); known.clear(); events.clear()
        ssidToBssids.clear(); fingerprintToIds.clear(); baselineKeys.clear()
        lastExposureScore = 0; baselineLearning = true
        persistKnown(); persistEvents(); persistBaselineKeys(); notifyChanged()
    }

    // ---- persistence ------------------------------------------------------

    fun persistKnown() {
        val ctx = appCtx ?: return
        val arr = JSONArray()
        known.values.forEach { arr.put(it.toJson()) }
        CryptoStore.writeEncrypted(File(ctx.filesDir, "known.enc"), arr.toString())
    }

    private fun persistEvents() {
        val ctx = appCtx ?: return
        val arr = JSONArray()
        events.forEach { arr.put(it.toJson()) }
        CryptoStore.writeEncrypted(File(ctx.filesDir, "events.enc"), arr.toString())
    }

    private fun persistBaselineKeys() {
        val ctx = appCtx ?: return
        CryptoStore.writeEncrypted(
            File(ctx.filesDir, "baseline.enc"),
            JSONArray(baselineKeys.toList()).toString()
        )
    }

    private fun persistMeta() {
        val ctx = appCtx ?: return
        val o = org.json.JSONObject().put("homeCell", homeCell)
        CryptoStore.writeEncrypted(File(ctx.filesDir, "meta.enc"), o.toString())
    }

    private fun load() {
        val ctx = appCtx ?: return
        runCatching {
            val txt = CryptoStore.readDecrypted(File(ctx.filesDir, "known.enc"))
            if (txt != null) {
                val arr = JSONArray(txt)
                for (i in 0 until arr.length()) {
                    val s = Signal.fromJson(arr.getJSONObject(i))
                    known[s.key] = s
                    if (s.label.isNotBlank())
                        ssidToBssids.getOrPut(s.label) { ConcurrentHashMap.newKeySet() }.add(s.id)
                    fingerprintToIds.getOrPut(s.fingerprint) { ConcurrentHashMap.newKeySet() }.add(s.id)
                }
            }
        }
        runCatching {
            val txt = CryptoStore.readDecrypted(File(ctx.filesDir, "events.enc"))
            if (txt != null) {
                val arr = JSONArray(txt)
                for (i in 0 until arr.length()) events.add(EventRecord.fromJson(arr.getJSONObject(i)))
            }
        }
        runCatching {
            val txt = CryptoStore.readDecrypted(File(ctx.filesDir, "baseline.enc"))
            if (txt != null) {
                val arr = JSONArray(txt)
                for (i in 0 until arr.length()) baselineKeys.add(arr.getString(i))
            }
        }
        runCatching {
            val txt = CryptoStore.readDecrypted(File(ctx.filesDir, "meta.enc"))
            if (txt != null) homeCell = org.json.JSONObject(txt).optString("homeCell")
        }
        baselineLearning = baselineKeys.isEmpty()
    }
}
