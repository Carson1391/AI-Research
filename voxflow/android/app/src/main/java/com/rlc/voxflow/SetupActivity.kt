package com.rlc.voxflow

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.provider.Settings
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.TextView

class SetupActivity : Activity() {

    private lateinit var status: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_setup)
        status = findViewById(R.id.setup_status)

        findViewById<Button>(R.id.btn_permission).setOnClickListener {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)
        }
        findViewById<Button>(R.id.btn_enable).setOnClickListener {
            startActivity(Intent(Settings.ACTION_INPUT_METHOD_SETTINGS))
        }
        findViewById<Button>(R.id.btn_switch).setOnClickListener {
            (getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager)
                .showInputMethodPicker()
        }
    }

    override fun onResume() {
        super.onResume()
        refreshStatus()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        refreshStatus()
    }

    private fun refreshStatus() {
        val micOk = checkSelfPermission(Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
        val enabledImes = Settings.Secure.getString(
            contentResolver, Settings.Secure.ENABLED_INPUT_METHODS
        ) ?: ""
        val imeEnabled = enabledImes.contains(packageName)

        status.text = buildString {
            append(if (micOk) "\u2705 Microphone permission granted\n"
                   else "\u274C Microphone permission missing\n")
            append(if (imeEnabled) "\u2705 VoxFlow keyboard enabled\n"
                   else "\u274C VoxFlow keyboard not enabled yet\n")
            append("\nFirst dictation after opening the keyboard waits for the ")
            append("model to load. The status line on the keyboard shows progress.")
        }
    }
}
