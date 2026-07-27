package com.exposurewatch.app.engine

import java.security.MessageDigest

/**
 * Identity-independent fingerprints.
 *   MAC/BSSID = claimed identity
 *   RF behaviour = physical-ish signature
 */
object Fingerprint {

    fun wifi(
        frequency: Int,
        channelWidth: Int,
        capabilities: String,
        is80211mc: Boolean,
        wifiStandard: Int = 0,
        ieHash: String = ""
    ): String {
        val band = when {
            frequency in 2400..2500 -> "2G"
            frequency in 4900..5900 -> "5G"
            frequency in 5925..7125 -> "6G"
            frequency > 0 -> "B${frequency / 100}"
            else -> "0"
        }
        val sec = securityClass(capabilities)
        val rtt = if (is80211mc) "RTT" else "-"
        return sha("W|$band|$frequency|w$channelWidth|$sec|$rtt|std$wifiStandard|ie$ieHash")
    }

    /** Hash of the beacon's information elements - hard-to-spoof radio identity. */
    fun ieHash(elements: List<Pair<Int, ByteArray>>): String {
        if (elements.isEmpty()) return ""
        val md = MessageDigest.getInstance("SHA-256")
        for ((id, bytes) in elements.sortedBy { it.first }) {
            md.update(id.toByte())
            md.update(bytes)
        }
        val d = md.digest()
        val sb = StringBuilder()
        for (i in 0 until 6) sb.append("%02x".format(d[i]))
        return sb.toString()
    }

    fun ble(manufacturerData: String, serviceUuids: String, connectable: Boolean, txPower: Int): String {
        val tx = when {
            txPower == Int.MIN_VALUE -> "-"
            txPower >= 0 -> "P+"
            txPower >= -20 -> "P0"
            else -> "P-"
        }
        val conn = if (connectable) "C" else "N"
        return sha("B|$manufacturerData|$serviceUuids|$conn|$tx")
    }

    fun securityClass(capabilities: String): String {
        val c = capabilities.uppercase()
        return when {
            c.contains("WPA3") || c.contains("SAE") -> "WPA3"
            c.contains("WPA2") || c.contains("RSN") -> "WPA2"
            c.contains("WPA") -> "WPA"
            c.contains("WEP") -> "WEP"
            c.contains("OWE") -> "OWE"
            else -> "OPEN"
        }
    }

    private fun sha(s: String): String {
        val d = MessageDigest.getInstance("SHA-256").digest(s.toByteArray())
        val sb = StringBuilder()
        for (i in 0 until 6) sb.append("%02x".format(d[i]))
        return sb.toString()
    }
}
