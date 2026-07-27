package com.exposurewatch.app.engine

import kotlin.math.pow

/**
 * Identity resolution: turn a claimed MAC/advertisement into "who made this and
 * what is it". Also detects identity-hiding (randomized MAC) and known trackers.
 * Tables are curated, not exhaustive.
 */
object Vendors {

    private val OUI = mapOf(
        "00000C" to "Cisco", "001A2F" to "Cisco", "00D0BC" to "Cisco",
        "E0553D" to "Cisco Meraki", "0018BB" to "Cisco Meraki",
        "0027E3" to "Aruba/HPE", "6CF37F" to "Aruba/HPE", "204C03" to "Aruba/HPE",
        "24A43C" to "Ubiquiti", "802AA8" to "Ubiquiti", "744401" to "Ubiquiti",
        "F09FC2" to "Ubiquiti", "687251" to "Ubiquiti", "B4FBE4" to "Ubiquiti",
        "0418D6" to "Ubiquiti", "788A20" to "Ubiquiti", "FCECDA" to "Ubiquiti",
        "001018" to "Broadcom", "D8B190" to "Ruckus", "C0C520" to "Ruckus",
        "4C5E0C" to "Mikrotik", "6C3B6B" to "Mikrotik", "E48D8C" to "Mikrotik",
        "C4AD34" to "Ruckus", "94103E" to "Belkin",
        "00146C" to "Netgear", "A040A0" to "Netgear", "9C3DCF" to "Netgear",
        "3894ED" to "Netgear", "204E7F" to "Netgear",
        "50C7BF" to "TP-Link", "EC086B" to "TP-Link", "AC84C6" to "TP-Link",
        "9C5322" to "TP-Link", "6466B3" to "TP-Link", "003192" to "TP-Link",
        "1C3BF3" to "D-Link", "28107B" to "D-Link", "B8A386" to "D-Link",
        "0022B0" to "D-Link", "C8D3A3" to "D-Link", "B0C554" to "D-Link",
        "00408C" to "Axis Camera", "ACCC8E" to "Axis Camera", "B8A44F" to "Axis Camera",
        "44A642" to "Axis Camera",
        "C0561D" to "Hikvision", "4C11BF" to "Hikvision", "BCAD28" to "Hikvision",
        "18680B" to "Hikvision", "A4145E" to "Hikvision", "C074AD" to "Hikvision",
        "3C1F09" to "Dahua", "9CB79C" to "Dahua", "E059B0" to "Dahua",
        "902B34" to "Dahua", "4C6D58" to "Dahua",
        "00E04C" to "Realtek",
        "EC71DB" to "Reolink", "88DA1A" to "Reolink",
        "18B430" to "Nest/Google", "641666" to "Nest/Google", "F4F5D8" to "Google",
        "3C5AB4" to "Google", "A4778A" to "Google", "D8EB46" to "Google",
        "0071CC" to "Amazon", "44650D" to "Amazon", "FCA667" to "Amazon",
        "68544C" to "Amazon", "34D270" to "Amazon", "F0272D" to "Amazon Ring",
        "B47C9C" to "Amazon", "50DCE7" to "Amazon",
        "24628D" to "Wyze", "2CAA8E" to "Wyze", "7C78B2" to "Wyze",
        "D03F27" to "Espressif (ESP32)", "3C6105" to "Espressif (ESP32)",
        "8CAAB5" to "Espressif (ESP32)", "A0764E" to "Espressif (ESP32)",
        "246F28" to "Espressif (ESP32)", "7CDFA1" to "Espressif (ESP32)",
        "C8C9A3" to "Espressif (ESP32)", "84F3EB" to "Espressif (ESP32)",
        "B827EB" to "Raspberry Pi", "DCA632" to "Raspberry Pi", "E45F01" to "Raspberry Pi",
        "2CCF67" to "Raspberry Pi",
        "F0038C" to "Apple", "3C0754" to "Apple", "A85C2C" to "Apple",
        "D023DB" to "Apple", "40331A" to "Apple",
        "F8E61A" to "Samsung", "8CF5A3" to "Samsung", "5CF6DC" to "Samsung",
        "00166C" to "Samsung", "E8508B" to "Samsung"
    )

    private val BLE_COMPANY = mapOf(
        0x004C to "Apple", 0x0075 to "Samsung", 0x00E0 to "Google",
        0x0006 to "Microsoft", 0x000F to "Broadcom", 0x0059 to "Nordic",
        0x0499 to "Ruuvi", 0x0157 to "Amazfit/Huami",
        0x0001 to "Ericsson", 0x000D to "Texas Instruments", 0x0087 to "Garmin",
        0x004F to "APT/Fitbit", 0x0171 to "Amazon", 0x038F to "Xiaomi",
        0x05A7 to "Sonos", 0x0110 to "Tile"
    )

    fun wifiVendor(bssid: String): String {
        if (isRandomizedMac(bssid)) return ""
        val oui = bssid.replace(":", "").replace("-", "").uppercase().take(6)
        return OUI[oui] ?: "OUI $oui"
    }

    /** Locally-administered bit (bit 1 of first octet) => randomized/spoofed MAC. */
    fun isRandomizedMac(mac: String): Boolean {
        val hex = mac.replace(":", "").replace("-", "")
        if (hex.length < 2) return false
        val first = hex.substring(0, 2).toIntOrNull(16) ?: return false
        return (first and 0x02) != 0
    }

    fun bleCompanyName(companyId: Int): String = BLE_COMPANY[companyId] ?: ""

    /** Human-readable Wi-Fi security bucket for the detail view. */
    fun securityOf(capabilities: String): String = Fingerprint.securityClass(capabilities)

    /**
     * Heuristic tracker detection over the big families.
     * @param companyFirstByte companyId -> first byte of its manufacturer data
     * @param serviceUuids8 leading 8 hex chars of advertised service UUIDs
     */
    fun bleTracker(companyFirstByte: Map<Int, Int>, serviceUuids8: List<String>): String {
        companyFirstByte[0x004C]?.let { t ->
            if (t == 0x12 || t == 0x19) return "Apple Find My / AirTag"
        }
        if (serviceUuids8.any { it == "0000feed" || it == "0000feec" }) return "Tile tracker"
        if (serviceUuids8.any { it == "0000fd5a" }) return "Samsung SmartTag"
        return ""
    }

    /** Rough log-distance estimate as a bucket, not false precision. */
    fun distanceBucket(rssi: Int, refAt1m: Int): String {
        val d = distanceMeters(rssi, refAt1m)
        return when {
            d < 1.5 -> "very close (<1.5 m)"
            d < 4 -> "close (~${"%.0f".format(d)} m)"
            d < 10 -> "nearby (~${"%.0f".format(d)} m)"
            else -> "in range (>10 m)"
        }
    }

    fun distanceMeters(rssi: Int, refAt1m: Int): Double {
        if (rssi == 0) return -1.0
        val n = 2.7 // indoor path-loss exponent
        return 10.0.pow((refAt1m - rssi) / (10.0 * n))
    }
}
