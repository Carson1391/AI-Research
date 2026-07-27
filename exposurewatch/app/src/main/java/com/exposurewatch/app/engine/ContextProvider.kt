package com.exposurewatch.app.engine

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Looper
import com.exposurewatch.app.model.ScanContext
import java.util.Calendar
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Turns phone sensors into a [ScanContext]: which named baselines are active
 * (Home / Driving / Walking / Night / Passive), a coarse location cell, and
 * whether the device is moving or was just unlocked. All local; location never
 * leaves the device.
 */
class ContextProvider(private val context: Context) {

    private val lm = context.getSystemService(Context.LOCATION_SERVICE) as? LocationManager
    private val sm = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager

    @Volatile private var lastLocation: Location? = null
    @Volatile private var lastMotionAt = 0L
    @Volatile private var lastUnlockAt = 0L

    private val locListener = object : LocationListener {
        override fun onLocationChanged(location: Location) { lastLocation = location }
        override fun onStatusChanged(provider: String?, status: Int, extras: android.os.Bundle?) {}
        override fun onProviderEnabled(provider: String) {}
        override fun onProviderDisabled(provider: String) {}
    }

    private val motionListener = object : SensorEventListener {
        override fun onSensorChanged(e: SensorEvent) {
            val m = sqrt(e.values[0] * e.values[0] + e.values[1] * e.values[1] + e.values[2] * e.values[2])
            if (abs(m - SensorManager.GRAVITY_EARTH) > 1.6f) lastMotionAt = System.currentTimeMillis()
        }
        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
    }

    private val unlockReceiver = object : BroadcastReceiver() {
        override fun onReceive(c: Context?, i: Intent?) {
            if (i?.action == Intent.ACTION_USER_PRESENT) lastUnlockAt = System.currentTimeMillis()
        }
    }

    fun start() {
        runCatching {
            lm?.let {
                val looper = Looper.getMainLooper()
                if (it.isProviderEnabled(LocationManager.NETWORK_PROVIDER))
                    it.requestLocationUpdates(LocationManager.NETWORK_PROVIDER, 5000L, 5f, locListener, looper)
                if (it.isProviderEnabled(LocationManager.GPS_PROVIDER))
                    it.requestLocationUpdates(LocationManager.GPS_PROVIDER, 5000L, 5f, locListener, looper)
                lastLocation = it.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
                    ?: it.getLastKnownLocation(LocationManager.GPS_PROVIDER)
            }
        }
        runCatching {
            sm?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)?.let {
                sm.registerListener(motionListener, it, SensorManager.SENSOR_DELAY_NORMAL)
            }
        }
        runCatching {
            context.registerReceiver(unlockReceiver, IntentFilter(Intent.ACTION_USER_PRESENT))
        }
    }

    fun stop() {
        runCatching { lm?.removeUpdates(locListener) }
        runCatching { sm?.unregisterListener(motionListener) }
        runCatching { context.unregisterReceiver(unlockReceiver) }
    }

    fun current(passive: Boolean): ScanContext {
        val loc = lastLocation
        val cell = if (loc != null) "%.3f,%.3f".format(loc.latitude, loc.longitude) else ""
        val mph = if (loc != null && loc.hasSpeed()) loc.speed * 2.2369 else 0.0
        val now = System.currentTimeMillis()
        val accelMoving = now - lastMotionAt < 3000
        val moving = accelMoving || mph > 1.0

        val ctx = LinkedHashSet<String>()
        if (cell.isNotBlank() && Repository.isHomeCell(cell)) ctx.add("Home")
        when {
            mph >= 15 -> ctx.add("Driving")
            mph in 1.0..15.0 -> ctx.add("Walking")
        }
        val hour = Calendar.getInstance().get(Calendar.HOUR_OF_DAY)
        if (hour >= 22 || hour < 6) ctx.add("Night")
        if (passive) ctx.add("Passive")

        return ScanContext(
            contexts = ctx,
            cell = cell,
            moving = moving,
            recentUnlock = now - lastUnlockAt < 15000
        )
    }

    fun currentCell(): String {
        val loc = lastLocation ?: return ""
        return "%.3f,%.3f".format(loc.latitude, loc.longitude)
    }
}
