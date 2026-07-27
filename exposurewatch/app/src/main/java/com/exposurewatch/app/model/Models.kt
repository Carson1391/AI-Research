package com.exposurewatch.app.model

import org.json.JSONArray
import org.json.JSONObject

enum class SignalKind { WIFI, BLE }
enum class RiskTier { NORMAL, WATCH, SUSPECT }

/**
 * The situation at the moment of a scan: which named baselines are active, a
 * coarse location cell, and whether the device is moving / was just unlocked.
 */
data class ScanContext(
    val contexts: Set<String>,
    val cell: String,
    val moving: Boolean,
    val recentUnlock: Boolean
)

/**
 * A single observed RF source (Wi-Fi AP or BLE advertiser).
 * `id` is the claimed identity (BSSID / device address). `fingerprint` is the
 * behavioural/radio "shape" that survives identity rotation and clone attempts.
 */
data class Signal(
    val kind: SignalKind,
    val id: String,
    val label: String,
    var rssi: Int,
    val frequency: Int,
    val capabilities: String,
    val extra: String,
    var fingerprint: String,
    val firstSeen: Long,
    var lastSeen: Long,
    var seenCount: Int = 1,
    var riskTier: RiskTier = RiskTier.NORMAL,
    var riskNote: String = "",
    var score: Int = 0,
    var vendor: String = "",
    var randomized: Boolean = false,
    var trackerName: String = "",
    var rssiMin: Int = 0,
    var rssiMax: Int = 0,
    var contexts: MutableSet<String> = mutableSetOf(),
    var cells: MutableSet<String> = mutableSetOf(),
    var movingCount: Int = 0,
    var stillCount: Int = 0,
    var afterUnlockCount: Int = 0
) {
    val key: String get() = "${kind.name}:$id"

    fun toJson(): JSONObject = JSONObject().apply {
        put("kind", kind.name)
        put("id", id)
        put("label", label)
        put("rssi", rssi)
        put("frequency", frequency)
        put("capabilities", capabilities)
        put("extra", extra)
        put("fingerprint", fingerprint)
        put("firstSeen", firstSeen)
        put("lastSeen", lastSeen)
        put("seenCount", seenCount)
        put("vendor", vendor)
        put("randomized", randomized)
        put("trackerName", trackerName)
        put("rssiMin", rssiMin)
        put("rssiMax", rssiMax)
        put("contexts", JSONArray(contexts.toList()))
        put("cells", JSONArray(cells.toList()))
        put("movingCount", movingCount)
        put("stillCount", stillCount)
        put("afterUnlockCount", afterUnlockCount)
    }

    companion object {
        private fun strSet(o: JSONObject, key: String): MutableSet<String> {
            val s = LinkedHashSet<String>()
            val arr = o.optJSONArray(key) ?: return s
            for (i in 0 until arr.length()) s.add(arr.getString(i))
            return s
        }

        fun fromJson(o: JSONObject): Signal = Signal(
            kind = SignalKind.valueOf(o.getString("kind")),
            id = o.getString("id"),
            label = o.optString("label"),
            rssi = o.optInt("rssi"),
            frequency = o.optInt("frequency"),
            capabilities = o.optString("capabilities"),
            extra = o.optString("extra"),
            fingerprint = o.optString("fingerprint"),
            firstSeen = o.optLong("firstSeen"),
            lastSeen = o.optLong("lastSeen"),
            seenCount = o.optInt("seenCount", 1),
            vendor = o.optString("vendor"),
            randomized = o.optBoolean("randomized"),
            trackerName = o.optString("trackerName"),
            rssiMin = o.optInt("rssiMin"),
            rssiMax = o.optInt("rssiMax"),
            contexts = strSet(o, "contexts"),
            cells = strSet(o, "cells"),
            movingCount = o.optInt("movingCount"),
            stillCount = o.optInt("stillCount"),
            afterUnlockCount = o.optInt("afterUnlockCount")
        )
    }
}

/** A scored exposure event shown on the Alerts timeline. */
data class EventRecord(
    val id: Long,
    val timestamp: Long,
    val title: String,
    val category: String,
    val score: Int,
    val evidence: List<String>
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("id", id)
        put("timestamp", timestamp)
        put("title", title)
        put("category", category)
        put("score", score)
        put("evidence", JSONArray(evidence))
    }

    companion object {
        fun fromJson(o: JSONObject): EventRecord {
            val ev = ArrayList<String>()
            val arr = o.optJSONArray("evidence") ?: JSONArray()
            for (i in 0 until arr.length()) ev.add(arr.getString(i))
            return EventRecord(
                id = o.getLong("id"),
                timestamp = o.getLong("timestamp"),
                title = o.getString("title"),
                category = o.optString("category"),
                score = o.optInt("score"),
                evidence = ev
            )
        }
    }
}
