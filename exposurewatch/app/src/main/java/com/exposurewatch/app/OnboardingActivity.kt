package com.exposurewatch.app

import android.content.Context
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.exposurewatch.app.databinding.ActivityOnboardingBinding

/** One-time first-run explainer: what the app does, the three steps, privacy. */
class OnboardingActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val b = ActivityOnboardingBinding.inflate(layoutInflater)
        setContentView(b.root)
        b.btnGetStarted.setOnClickListener {
            markSeen(this)
            finish()
        }
    }

    companion object {
        private const val PREFS = "ew_prefs"
        private const val KEY_ONBOARDED = "onboarded"

        fun isSeen(ctx: Context): Boolean =
            ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean(KEY_ONBOARDED, false)

        fun markSeen(ctx: Context) {
            ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .edit().putBoolean(KEY_ONBOARDED, true).apply()
        }
    }
}
