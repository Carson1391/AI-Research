package com.exposurewatch.app

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import com.exposurewatch.app.ble.BleScanner
import com.exposurewatch.app.engine.ContextProvider
import com.exposurewatch.app.engine.Repository
import com.exposurewatch.app.engine.ScoringEngine
import com.exposurewatch.app.model.EventRecord
import com.exposurewatch.app.model.RiskTier
import com.exposurewatch.app.model.ScanContext
import com.exposurewatch.app.model.Signal
import com.exposurewatch.app.wifi.WifiScanner
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

/**
 * The always-on watcher. Each cycle: refresh Wi-Fi, open a BLE window, score
 * everything against the learned baseline, raise alerts for suspects, and keep
 * the exposure score current in the notification.
 */
class ExposureWatchService : Service() {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var loop: Job? = null

    private lateinit var wifi: WifiScanner
    private lateinit var ble: BleScanner
    private lateinit var contextProvider: ContextProvider

    @Volatile private var scanCtx: ScanContext = ScanContext(emptySet(), "", false, false)

    private val lastAlerted = HashMap<String, Long>()
    private var cycles = 0

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        Repository.init(this)
        wifi = WifiScanner(this) { ingest(it) }
        ble = BleScanner(this) { ingest(it) }
        contextProvider = ContextProvider(this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> { stopSelf(); return START_NOT_STICKY }
            ACTION_RELEARN -> {
                Repository.clearAll(); cycles = 0; Repository.baselineLearning = true
            }
        }
        startForegroundSafe(buildNotification())
        Repository.monitoring = true
        runCatching { contextProvider.start() }
        if (loop == null || loop?.isActive != true) loop = scope.launch { runLoop() }
        return START_STICKY
    }

    private suspend fun runLoop() {
        wifi.start()
        while (scope.isActive) {
            scanCtx = contextProvider.current(Repository.passiveMode)
            Repository.currentContexts = scanCtx.contexts
            Repository.currentCell = scanCtx.cell
            if (!Repository.passiveMode) wifi.requestScan()
            wifi.pollNow()
            if (ble.isAvailable()) {
                ble.startWindow(Repository.passiveMode)
                delay(BLE_WINDOW_MS)
                ble.stopWindow()
            } else {
                delay(BLE_WINDOW_MS)
            }
            afterCycle()
            delay(CYCLE_GAP_MS)
        }
    }

    @Synchronized
    private fun ingest(batch: List<Signal>) {
        val learning = Repository.baselineLearning
        val ctx = scanCtx
        for (s in batch) {
            val prior = Repository.live[s.key]
            if (prior != null) s.seenCount = prior.seenCount
            val v = ScoringEngine.evaluate(s, learning, ctx)
            s.riskTier = v.tier
            s.riskNote = v.note
            s.score = v.score
            Repository.live[s.key] = s
            Repository.remember(s, ctx)
            if (!learning && v.tier != RiskTier.NORMAL &&
                v.score >= ALERT_THRESHOLD && v.evidence.isNotEmpty()
            ) {
                maybeAlert(s, v)
            }
        }
        recomputeExposure()
        Repository.notifyChanged()
    }

    private fun maybeAlert(s: Signal, v: ScoringEngine.Verdict) {
        val dedupe = "${s.key}|${v.category}"
        val now = System.currentTimeMillis()
        val last = lastAlerted[dedupe] ?: 0L
        if (now - last < ALERT_COOLDOWN_MS) return
        lastAlerted[dedupe] = now
        Repository.addEvent(
            EventRecord(
                id = Repository.nextEventId(),
                timestamp = now,
                title = v.title,
                category = v.category,
                score = v.score,
                evidence = v.evidence
            )
        )
        updateNotification()
    }

    private fun recomputeExposure() {
        val vals = Repository.live.values
        val top = vals.maxOfOrNull { it.score } ?: 0
        val suspects = vals.count { it.riskTier == RiskTier.SUSPECT }
        Repository.lastExposureScore = (top + minOf(suspects, 3) * 4).coerceIn(0, 100)
        Repository.lastScanAt = System.currentTimeMillis()
    }

    private fun afterCycle() {
        cycles++
        if (Repository.baselineLearning && cycles >= BASELINE_CYCLES && Repository.knownCount() > 0) {
            Repository.finalizeBaseline()
        }
        if (!Repository.baselineLearning && cycles % 5 == 0) Repository.persistKnown()
        updateNotification()
        Repository.notifyChanged()
    }

    private fun startForegroundSafe(n: Notification) {
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(NOTIF_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_CONNECTED_DEVICE)
        } else {
            startForeground(NOTIF_ID, n)
        }
    }

    private fun updateNotification() {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.notify(NOTIF_ID, buildNotification())
    }

    private fun buildNotification(): Notification {
        val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= 26 && nm.getNotificationChannel(CHANNEL) == null) {
            nm.createNotificationChannel(
                NotificationChannel(CHANNEL, "Exposure monitoring", NotificationManager.IMPORTANCE_LOW)
                    .apply { description = "Live Wi-Fi / BLE / camera exposure watch" }
            )
        }
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val score = Repository.lastExposureScore
        val state = when {
            Repository.baselineLearning -> "Learning your normal environment..."
            score >= 70 -> "High exposure signals present"
            score >= 40 -> "Some sources worth watching"
            else -> "Environment looks clear"
        }
        val builder = if (Build.VERSION.SDK_INT >= 26)
            Notification.Builder(this, CHANNEL)
        else
            @Suppress("DEPRECATION") Notification.Builder(this)
        return builder
            .setContentTitle("ExposureWatch - score $score/100")
            .setContentText(state)
            .setSmallIcon(android.R.drawable.ic_menu_view)
            .setOngoing(true)
            .setContentIntent(open)
            .build()
    }

    override fun onDestroy() {
        Repository.monitoring = false
        runCatching { Repository.persistKnown() }
        runCatching { wifi.stop() }
        runCatching { ble.stopWindow() }
        runCatching { contextProvider.stop() }
        loop?.cancel()
        scope.cancel()
        Repository.notifyChanged()
        super.onDestroy()
    }

    companion object {
        const val ACTION_START = "com.exposurewatch.app.START"
        const val ACTION_STOP = "com.exposurewatch.app.STOP"
        const val ACTION_RELEARN = "com.exposurewatch.app.RELEARN"
        private const val CHANNEL = "exposure_watch"
        private const val NOTIF_ID = 4137

        private const val BLE_WINDOW_MS = 6000L
        private const val CYCLE_GAP_MS = 9000L
        private const val BASELINE_CYCLES = 3
        private const val ALERT_THRESHOLD = 45
        private const val ALERT_COOLDOWN_MS = 120_000L

        fun start(ctx: Context) {
            val i = Intent(ctx, ExposureWatchService::class.java).setAction(ACTION_START)
            if (Build.VERSION.SDK_INT >= 26) ctx.startForegroundService(i) else ctx.startService(i)
        }
        fun stop(ctx: Context) {
            ctx.startService(Intent(ctx, ExposureWatchService::class.java).setAction(ACTION_STOP))
        }
        fun relearn(ctx: Context) {
            val i = Intent(ctx, ExposureWatchService::class.java).setAction(ACTION_RELEARN)
            if (Build.VERSION.SDK_INT >= 26) ctx.startForegroundService(i) else ctx.startService(i)
        }
    }
}
