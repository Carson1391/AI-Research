package com.exposurewatch.app.ble

import android.bluetooth.BluetoothAdapter
import android.bluetooth.BluetoothManager
import android.bluetooth.le.BluetoothLeScanner
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.bluetooth.le.ScanSettings
import android.content.Context
import android.os.Build
import com.exposurewatch.app.engine.Fingerprint
import com.exposurewatch.app.engine.Vendors
import com.exposurewatch.app.model.Signal
import com.exposurewatch.app.model.SignalKind

/**
 * Wraps BluetoothLeScanner. Runs scan windows and coalesces advertisers into
 * [Signal]s (address, payload shape, tx power, connectable, tracker signature).
 */
class BleScanner(
    private val context: Context,
    private val onResults: (List<Signal>) -> Unit
) {
    private val adapter: BluetoothAdapter? =
        (context.getSystemService(Context.BLUETOOTH_SERVICE) as? BluetoothManager)?.adapter
    private var scanner: BluetoothLeScanner? = null
    private var scanning = false
    private val batch = LinkedHashMap<String, Signal>()

    private val callback = object : ScanCallback() {
        override fun onScanResult(callbackType: Int, result: ScanResult?) {
            result?.let { record(it) }
        }
        override fun onBatchScanResults(results: MutableList<ScanResult>?) {
            results?.forEach { record(it) }
        }
        override fun onScanFailed(errorCode: Int) { scanning = false }
    }

    fun isAvailable(): Boolean = adapter?.isEnabled == true

    fun startWindow(opportunistic: Boolean = false) {
        if (scanning || !isAvailable()) return
        scanner = adapter?.bluetoothLeScanner ?: return
        batch.clear()
        val mode = if (opportunistic) ScanSettings.SCAN_MODE_OPPORTUNISTIC
            else ScanSettings.SCAN_MODE_LOW_LATENCY
        val settings = ScanSettings.Builder().setScanMode(mode).build()
        runCatching {
            scanner?.startScan(null, settings, callback)
            scanning = true
        }
    }

    fun stopWindow() {
        if (!scanning) return
        runCatching { scanner?.stopScan(callback) }
        scanning = false
        if (batch.isNotEmpty()) onResults(batch.values.toList())
    }

    private fun record(r: ScanResult) {
        val now = System.currentTimeMillis()
        val dev = r.device ?: return
        val addr = dev.address ?: return
        val rec = r.scanRecord
        val name = rec?.deviceName ?: ""

        val mfg = StringBuilder()
        val companyFirstByte = HashMap<Int, Int>()
        rec?.manufacturerSpecificData?.let { sparse ->
            for (i in 0 until sparse.size()) {
                val companyId = sparse.keyAt(i)
                mfg.append(companyId).append(',')
                val data = sparse.valueAt(i)
                if (data != null && data.isNotEmpty()) {
                    companyFirstByte[companyId] = data[0].toInt() and 0xFF
                }
            }
        }
        val uuids8 = rec?.serviceUuids?.map { it.uuid.toString().substring(0, 8).lowercase() } ?: emptyList()
        val uuids = uuids8.joinToString(",")
        val connectable = if (Build.VERSION.SDK_INT >= 26)
            runCatching { r.isConnectable }.getOrDefault(true) else true
        val tx = runCatching { r.txPower }.getOrDefault(Int.MIN_VALUE)

        val companyName = companyFirstByte.keys.firstNotNullOfOrNull {
            val n = Vendors.bleCompanyName(it); if (n.isNotEmpty()) n else null
        } ?: ""
        val tracker = Vendors.bleTracker(companyFirstByte, uuids8)

        val fp = Fingerprint.ble(mfg.toString(), uuids, connectable, tx)
        val s = Signal(
            kind = SignalKind.BLE,
            id = addr,
            label = name,
            rssi = r.rssi,
            frequency = 0,
            capabilities = uuids,
            extra = "mfg=$mfg;svc=$uuids;conn=$connectable;tx=$tx",
            fingerprint = fp,
            firstSeen = now,
            lastSeen = now,
            vendor = companyName,
            randomized = Vendors.isRandomizedMac(addr),
            trackerName = tracker
        )
        val existing = batch[addr]
        if (existing == null || s.rssi > existing.rssi) batch[addr] = s
    }
}
