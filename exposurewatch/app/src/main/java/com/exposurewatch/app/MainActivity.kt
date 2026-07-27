package com.exposurewatch.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.View
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import com.exposurewatch.app.databinding.ActivityMainBinding
import com.exposurewatch.app.engine.Repository
import com.exposurewatch.app.ir.IrDetectorActivity
import com.exposurewatch.app.model.RiskTier
import com.exposurewatch.app.ui.SignalsAdapter
import com.exposurewatch.app.ui.TimelineAdapter

class MainActivity : AppCompatActivity() {

    private lateinit var b: ActivityMainBinding
    private lateinit var signalsAdapter: SignalsAdapter
    private val timelineAdapter = TimelineAdapter()
    private val refresh = { runOnUiThread { render() } }

    private val permLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { startMonitoring() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityMainBinding.inflate(layoutInflater)
        setContentView(b.root)
        Repository.init(this)

        if (!OnboardingActivity.isSeen(this)) {
            startActivity(Intent(this, OnboardingActivity::class.java))
        }

        signalsAdapter = SignalsAdapter { key ->
            startActivity(
                Intent(this, LocateActivity::class.java)
                    .putExtra(LocateActivity.EXTRA_KEY, key)
            )
        }
        b.recyclerSignals.layoutManager = LinearLayoutManager(this)
        b.recyclerSignals.adapter = signalsAdapter
        b.recyclerAlerts.layoutManager = LinearLayoutManager(this)
        b.recyclerAlerts.adapter = timelineAdapter

        b.switchPassive.isChecked = Repository.passiveMode
        b.switchPassive.setOnCheckedChangeListener { _, checked -> Repository.passiveMode = checked }

        b.bottomNav.setOnItemSelectedListener {
            when (it.itemId) {
                R.id.nav_dashboard -> show(0)
                R.id.nav_signals -> show(1)
                R.id.nav_alerts -> show(2)
            }
            true
        }
        show(0)

        b.btnToggle.setOnClickListener {
            if (Repository.monitoring) ExposureWatchService.stop(this) else requestScanPermissions()
        }
        b.btnRelearn.setOnClickListener { ExposureWatchService.relearn(this) }
        b.btnIr.setOnClickListener { startActivity(Intent(this, IrDetectorActivity::class.java)) }
        b.btnClear.setOnClickListener { Repository.clearAll(); render() }
        b.btnOpenSettings.setOnClickListener {
            startActivity(
                Intent(
                    android.provider.Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                    android.net.Uri.fromParts("package", packageName, null)
                )
            )
        }
        b.btnSetHome.setOnClickListener {
            val cell = Repository.currentCell
            if (cell.isBlank()) {
                android.widget.Toast.makeText(
                    this, "No location fix yet - start monitoring and wait a few seconds",
                    android.widget.Toast.LENGTH_SHORT
                ).show()
            } else {
                Repository.setHome(cell)
                android.widget.Toast.makeText(
                    this, "Home set. Sources here are now your home baseline.",
                    android.widget.Toast.LENGTH_SHORT
                ).show()
                render()
            }
        }
        render()
    }

    override fun onStart() {
        super.onStart()
        Repository.addListener(refresh)
        render()
    }

    override fun onStop() {
        Repository.removeListener(refresh)
        super.onStop()
    }

    private fun show(index: Int) {
        b.viewDashboard.visibility = if (index == 0) View.VISIBLE else View.GONE
        b.viewSignals.visibility = if (index == 1) View.VISIBLE else View.GONE
        b.viewAlerts.visibility = if (index == 2) View.VISIBLE else View.GONE
    }

    private fun requestScanPermissions() {
        val perms = ArrayList<String>()
        perms += Manifest.permission.ACCESS_FINE_LOCATION
        if (Build.VERSION.SDK_INT >= 31) perms += Manifest.permission.BLUETOOTH_SCAN
        if (Build.VERSION.SDK_INT >= 33) {
            perms += Manifest.permission.NEARBY_WIFI_DEVICES
            perms += Manifest.permission.POST_NOTIFICATIONS
        }
        val missing = perms.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isEmpty()) startMonitoring() else permLauncher.launch(missing.toTypedArray())
    }

    private fun startMonitoring() {
        ExposureWatchService.start(this)
        render()
    }

    private fun render() {
        val score = Repository.lastExposureScore
        b.gauge.setValue(score)

        val live = Repository.live.values.toList()
        signalsAdapter.submit(live)
        timelineAdapter.submit(Repository.events.toList())

        b.emptySignals.visibility = if (live.isEmpty()) View.VISIBLE else View.GONE
        b.emptyAlerts.visibility = if (Repository.events.isEmpty()) View.VISIBLE else View.GONE

        val locGranted = ContextCompat.checkSelfPermission(
            this, Manifest.permission.ACCESS_FINE_LOCATION
        ) == PackageManager.PERMISSION_GRANTED
        b.permBanner.visibility = if (locGranted) View.GONE else View.VISIBLE

        val suspects = live.count { it.riskTier == RiskTier.SUSPECT }
        val watch = live.count { it.riskTier == RiskTier.WATCH }

        b.txtStatus.text = when {
            !Repository.monitoring -> "Idle - press Start to watch your environment"
            Repository.baselineLearning -> "Learning your normal environment..."
            score >= 70 -> "High exposure signals present"
            score >= 40 -> "Some sources worth watching"
            else -> "Environment looks clear"
        }
        b.txtBaseline.text = buildString {
            append(
                if (Repository.baselineLearning)
                    "Baseline: learning (${Repository.knownCount()} sources filed)"
                else
                    "Baseline locked \u00B7 ${Repository.knownCount()} known sources"
            )
            append(if (Repository.homeCell.isNotBlank()) " \u00B7 home set" else " \u00B7 home not set")
            if (Repository.monitoring && Repository.currentContexts.isNotEmpty())
                append("\nHere now: ${Repository.currentContexts.joinToString(", ")}")
        }

        b.txtCounts.text = "Live: ${live.size}   \u00B7   Suspect: $suspects   \u00B7   Watch: $watch"
        b.btnToggle.text = if (Repository.monitoring) "Stop monitoring" else "Start monitoring"

        val topAlert = Repository.events.firstOrNull()
        b.txtRecent.text = if (topAlert != null)
            "Latest: ${topAlert.title}  (${topAlert.score}/100)"
        else
            "No alerts yet"

        b.badgeAlerts.text = Repository.events.size.toString()
        b.badgeAlerts.visibility = if (Repository.events.isEmpty()) View.GONE else View.VISIBLE
    }
}
