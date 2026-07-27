package com.exposurewatch.app.engine

import com.exposurewatch.app.model.RiskTier
import com.exposurewatch.app.model.ScanContext
import com.exposurewatch.app.model.Signal
import com.exposurewatch.app.model.SignalKind

/**
 * Decision logic:
 *   same SSID / new BSSID .......... different AP, extender, spoof or clone
 *   same BSSID / changed caps ...... possible evil-twin / MAC clone
 *   same fingerprint / new MAC ..... same hardware class or rotation
 *   BLE same addr / changed payload  rotating identity or spoof
 *   BLE follows across scans ....... tracker-like behaviour
 *
 * Trust repeated shape, not claimed identity.
 */
object ScoringEngine {

    data class Verdict(
        val tier: RiskTier,
        val note: String,
        val score: Int,
        val category: String,
        val title: String,
        val evidence: List<String>
    )

    private const val TRACKER_PERSIST = 6

    fun evaluate(s: Signal, learning: Boolean, ctx: ScanContext): Verdict {
        val base = if (s.kind == SignalKind.WIFI) evaluateWifi(s, learning) else evaluateBle(s, learning)
        if (learning) return base
        return applyBehavior(s, base, ctx)
    }

    /** Sharpen a verdict using where/when the source has been seen before. */
    private fun applyBehavior(s: Signal, base: Verdict, ctx: ScanContext): Verdict {
        val known = Repository.knownFor(s.key) ?: return base
        val cells = known.cells
        val nonHomeCells = cells.filter { !Repository.isHomeCell(it) }

        if (nonHomeCells.size >= 3 && !Repository.isBaseline(s.key)) {
            val ev = ArrayList(base.evidence)
            ev.add(0, "Seen with you across ${nonHomeCells.size} different locations")
            ev.add(1, "Persisting across locations is the strongest tracking signal")
            if (known.trackerName.isNotBlank()) ev.add(2, "Identified as ${known.trackerName}")
            return base.copy(
                tier = RiskTier.SUSPECT,
                score = maxOf(base.score, 84),
                category = "FOLLOW_LOCATIONS",
                title = "Followed across ${nonHomeCells.size} locations",
                note = "Follows you across locations",
                evidence = ev
            )
        }

        val homeOnly = cells.isNotEmpty() && cells.all { Repository.isHomeCell(it) }
        if (homeOnly && ctx.cell.isNotBlank() && !Repository.isHomeCell(ctx.cell) &&
            !Repository.isBaseline(s.key)
        ) {
            val ev = ArrayList(base.evidence)
            ev.add(0, "Normally seen only near your home - but it's here with you now")
            return base.copy(
                tier = RiskTier.SUSPECT,
                score = maxOf(base.score, 70),
                category = "HOME_DEVICE_ELSEWHERE",
                title = "Home device appeared away from home",
                note = "Home-only source seen elsewhere",
                evidence = ev
            )
        }

        if (known.contexts.contains("Night") && known.contexts.size == 1 &&
            known.seenCount >= 3 && !Repository.isBaseline(s.key) && base.tier == RiskTier.NORMAL
        ) {
            val ev = ArrayList(base.evidence)
            ev.add(0, "This source has only ever appeared at night")
            return base.copy(
                tier = RiskTier.WATCH,
                score = maxOf(base.score, 40),
                category = "NIGHT_ONLY",
                title = "Source appears only at night",
                note = "Night-only source",
                evidence = ev
            )
        }

        return base
    }

    // ---- Wi-Fi ------------------------------------------------------------

    private fun evaluateWifi(s: Signal, learning: Boolean): Verdict {
        val sec = Fingerprint.securityClass(s.capabilities)
        val rttCapable = s.extra.contains("mc=true")
        val prior = Repository.knownFor(s.key)
        val ev = ArrayList<String>()

        if (prior != null) {
            val priorSec = Fingerprint.securityClass(prior.capabilities)
            if (priorSec != sec && !learning) {
                ev += "BSSID ${s.id} previously advertised $priorSec, now advertises $sec"
                ev += "SSID \"${s.label}\" - capability profile changed at same identity"
                ev += "Consistent with evil-twin / MAC-clone of a known access point"
                return Verdict(RiskTier.SUSPECT, "Capability change on known BSSID", 74,
                    "WIFI_MAC_CLONE", "Possible Wi-Fi identity spoof / clone", ev)
            }
        }

        val knownBssids = Repository.bssidsForSsid(s.label)
        if (s.label.isNotBlank() && knownBssids.isNotEmpty() && s.id !in knownBssids && !learning) {
            val secMismatch = knownBssids.any { bssid ->
                Repository.knownFor("WIFI:$bssid")?.let {
                    Fingerprint.securityClass(it.capabilities) != sec
                } ?: false
            }
            if (secMismatch) {
                ev += "SSID \"${s.label}\" now broadcast by unfamiliar BSSID ${s.id}"
                ev += "Security profile ($sec) differs from established APs on this network"
                ev += "RSSI ${s.rssi} dBm does not match the usual map for this SSID"
                return Verdict(RiskTier.SUSPECT, "New BSSID + security mismatch on known SSID", 72,
                    "WIFI_EVIL_TWIN", "Possible evil-twin access point", ev)
            }
            ev += "SSID \"${s.label}\" broadcast by new BSSID ${s.id}"
            ev += "Could be a mesh node / extender, or an added AP"
            return Verdict(RiskTier.WATCH, "New BSSID for known SSID", 38,
                "WIFI_NEW_BSSID", "New access point on a known network", ev)
        }

        val sameShapeIds = Repository.idsForFingerprint(s.fingerprint)
        if (sameShapeIds.size > 1 && !learning) {
            ev += "Radio fingerprint ${s.fingerprint} shared across ${sameShapeIds.size} identities"
            ev += "Same hardware class, or one device rotating its BSSID"
            val score = if (rttCapable) 55 else 44
            return Verdict(RiskTier.WATCH, "Shared fingerprint across identities", score,
                "WIFI_ROTATION", "Same hardware behind multiple identities", ev)
        }

        if (prior == null && !learning) {
            val dist = Vendors.distanceBucket(s.rssi, -45)
            if (s.randomized) {
                ev += "New AP \"${s.label.ifBlank { s.id }}\" broadcasts a randomized/locally-administered MAC"
                ev += "Vendor is hidden - consistent with a portable hotspot, spoof, or privacy MAC"
                ev += "Distance: $dist - $sec"
                return Verdict(RiskTier.WATCH, "New AP hiding its vendor (randomized MAC)", 42,
                    "WIFI_RANDOM_MAC", "Access point hiding its identity", ev)
            }
            val vendorTxt = if (s.vendor.isNotBlank()) s.vendor else "unknown vendor"
            if (rttCapable) {
                ev += "New 802.11mc ranging-capable AP ${s.label.ifBlank { s.id }} ($vendorTxt) appeared"
                ev += "RTT/FTM hardware can actively measure your distance to the AP"
                ev += "Distance: $dist"
                return Verdict(RiskTier.WATCH, "New RTT-capable AP nearby", 48,
                    "WIFI_RTT", "Wi-Fi ranging-capable infrastructure nearby", ev)
            }
            ev += "New access point ${s.label.ifBlank { s.id }} entered the environment"
            ev += "Vendor: $vendorTxt - $sec - $dist"
            return Verdict(RiskTier.WATCH, "New access point", 22,
                "WIFI_NEW", "New Wi-Fi source", ev)
        }

        val note = if (rttCapable) "Known AP (ranging-capable)" else "Known AP"
        return Verdict(RiskTier.NORMAL, note, 5, "WIFI_KNOWN", "", emptyList())
    }

    // ---- BLE --------------------------------------------------------------

    private fun evaluateBle(s: Signal, learning: Boolean): Verdict {
        val prior = Repository.knownFor(s.key)
        val ev = ArrayList<String>()
        val connectable = s.extra.contains("conn=true")
        val dist = Vendors.distanceBucket(s.rssi, -59)

        if (s.trackerName.isNotBlank() && !learning && !Repository.isBaseline(s.key)) {
            ev += "${s.trackerName} detected near you (${s.rssi} dBm, $dist)"
            ev += "Advertisement matches a known tracking-tag signature"
            ev += "If this tag is not yours, it may be traveling with you"
            ev += "Re-check by moving away: a tracker's signal follows; a fixture's fades"
            return Verdict(RiskTier.SUSPECT, "Known tracker: ${s.trackerName}", 76,
                "BLE_KNOWN_TRACKER", "Known tracker tag near you", ev)
        }

        if (prior != null && prior.fingerprint != s.fingerprint && !learning) {
            ev += "BLE address ${s.id} kept identity but its advertisement payload changed"
            ev += "Manufacturer / service data shifted: possible rotation or spoof"
            return Verdict(RiskTier.WATCH, "Payload changed on stable BLE address", 46,
                "BLE_PAYLOAD", "BLE advertiser changed its payload", ev)
        }

        val persist = prior?.seenCount ?: 0
        if (persist >= TRACKER_PERSIST && !learning && !Repository.isBaseline(s.key)) {
            ev += "BLE source \"${s.label.ifBlank { s.id }}\" has persisted across $persist scans"
            ev += "Reappears with you rather than fading - tracker-like behaviour"
            if (!connectable) ev += "Non-connectable beacon advertising: consistent with a tag/beacon"
            ev += "RSSI ${s.rssi} dBm remaining stable as you move"
            return Verdict(RiskTier.SUSPECT, "Following across scans", 68,
                "BLE_TRACKER", "Possible tracker following you", ev)
        }

        val sameShapeIds = Repository.idsForFingerprint(s.fingerprint)
        if (sameShapeIds.size > 1 && !learning) {
            ev += "BLE fingerprint ${s.fingerprint} seen under ${sameShapeIds.size} rotating addresses"
            ev += "Same advertiser using address rotation"
            return Verdict(RiskTier.WATCH, "Rotating BLE identity", 40,
                "BLE_ROTATION", "One BLE advertiser rotating identities", ev)
        }

        if (prior == null && !learning) {
            val v = if (s.vendor.isNotBlank()) s.vendor
                else if (s.randomized) "randomized address" else "unknown"
            ev += "New BLE advertiser ${s.label.ifBlank { s.id }} appeared ($v)"
            ev += "Signal: ${s.rssi} dBm - $dist"
            return Verdict(RiskTier.WATCH, "New BLE source", 18, "BLE_NEW", "New BLE source", ev)
        }

        return Verdict(RiskTier.NORMAL, "Known BLE source", 5, "BLE_KNOWN", "", emptyList())
    }
}
