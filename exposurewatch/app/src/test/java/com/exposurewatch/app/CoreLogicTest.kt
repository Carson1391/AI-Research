package com.exposurewatch.app

import com.exposurewatch.app.engine.Fingerprint
import com.exposurewatch.app.engine.Vendors
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/** Pure-JVM tests for the identity/behaviour math. */
class CoreLogicTest {

    @Test fun randomizedMac_localBitDetected() {
        assertTrue(Vendors.isRandomizedMac("DA:A1:19:33:44:55"))
        assertFalse(Vendors.isRandomizedMac("24:A4:3C:11:22:33"))
    }

    @Test fun wifiVendor_resolvesOuiAndHidesRandomized() {
        assertEquals("Ubiquiti", Vendors.wifiVendor("24:A4:3C:11:22:33"))
        assertEquals("", Vendors.wifiVendor("DA:A1:19:33:44:55"))
        assertTrue(Vendors.wifiVendor("08:11:22:33:44:55").startsWith("OUI"))
    }

    @Test fun bleTracker_detectsKnownFamilies() {
        assertEquals("Apple Find My / AirTag",
            Vendors.bleTracker(mapOf(0x004C to 0x12), emptyList()))
        assertEquals("Tile tracker",
            Vendors.bleTracker(emptyMap(), listOf("0000feed")))
        assertEquals("Samsung SmartTag",
            Vendors.bleTracker(emptyMap(), listOf("0000fd5a")))
        assertEquals("", Vendors.bleTracker(mapOf(0x004C to 0x07), emptyList()))
    }

    @Test fun distance_isMonotonicInRssi() {
        val near = Vendors.distanceMeters(-40, -59)
        val far = Vendors.distanceMeters(-85, -59)
        assertTrue("stronger signal must be closer", near < far)
    }

    @Test fun ieHash_isDeterministicAndOrderIndependent() {
        val a = listOf(0 to byteArrayOf(1, 2, 3), 221 to byteArrayOf(9, 9))
        val b = listOf(221 to byteArrayOf(9, 9), 0 to byteArrayOf(1, 2, 3))
        assertEquals(Fingerprint.ieHash(a), Fingerprint.ieHash(b))
        assertNotEquals(Fingerprint.ieHash(a), Fingerprint.ieHash(listOf(0 to byteArrayOf(1, 2, 4))))
        assertEquals("", Fingerprint.ieHash(emptyList()))
    }

    @Test fun wifiFingerprint_changesWithInformationElements() {
        val base = Fingerprint.wifi(2412, 1, "[WPA2-PSK-CCMP]", false, 6, "aaaa")
        val same = Fingerprint.wifi(2412, 1, "[WPA2-PSK-CCMP]", false, 6, "aaaa")
        val diffIe = Fingerprint.wifi(2412, 1, "[WPA2-PSK-CCMP]", false, 6, "bbbb")
        assertEquals(base, same)
        assertNotEquals(base, diffIe)
    }

    @Test fun securityClass_bucketsCorrectly() {
        assertEquals("WPA3", Vendors.securityOf("[WPA3-SAE-CCMP]"))
        assertEquals("WPA2", Vendors.securityOf("[WPA2-PSK-CCMP][RSN]"))
        assertEquals("OPEN", Vendors.securityOf("[ESS]"))
    }
}
