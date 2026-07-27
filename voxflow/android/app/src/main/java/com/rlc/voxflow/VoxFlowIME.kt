package com.rlc.voxflow

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.inputmethodservice.InputMethodService
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.view.KeyEvent
import android.view.MotionEvent
import android.view.View
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.TextView
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OnlineModelConfig
import com.k2fsa.sherpa.onnx.OnlineRecognizer
import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig

class VoxFlowIME : InputMethodService() {

    companion object {
        private const val SAMPLE_RATE = 16000
        private const val TAIL_PAD_SEC = 1.0f      // silence fed before finalizing
        private const val RELEASE_TAIL_MS = 350L   // keep capturing after finger up
        private const val LANGUAGE = "en"          // used by Nemotron models; "auto" also works
    }

    private var recognizer: OnlineRecognizer? = null
    @Volatile private var modelReady = false
    @Volatile private var recording = false
    @Volatile private var finalizing = false

    private var audioRecord: AudioRecord? = null
    private var modelKind = "nemotron"
    private lateinit var vocab: Vocab

    private val mainHandler = Handler(Looper.getMainLooper())
    private var statusView: TextView? = null

    // ------------------------------------------------------------------ init

    override fun onCreate() {
        super.onCreate()
        vocab = Vocab(assets)
        Thread {
            try {
                modelKind = assets.open("model/meta.txt")
                    .bufferedReader().readText().trim()
            } catch (_: Exception) {
                modelKind = "nemotron"
            }
            try {
                val config = OnlineRecognizerConfig(
                    featConfig = FeatureConfig(sampleRate = SAMPLE_RATE, featureDim = 80),
                    modelConfig = OnlineModelConfig(
                        transducer = OnlineTransducerModelConfig(
                            encoder = "model/encoder.onnx",
                            decoder = "model/decoder.onnx",
                            joiner = "model/joiner.onnx",
                        ),
                        tokens = "model/tokens.txt",
                        numThreads = 4,
                        provider = "cpu",
                    ),
                    enableEndpoint = false,
                    decodingMethod = "greedy_search",
                )
                recognizer = OnlineRecognizer(assetManager = assets, config = config)
                modelReady = true
                setStatus(getString(R.string.ready))
            } catch (e: Exception) {
                setStatus("Model failed to load: ${e.message}. Run setup_android.py, rebuild.")
            }
        }.start()
    }

    override fun onDestroy() {
        recording = false
        recognizer?.release()
        recognizer = null
        super.onDestroy()
    }

    // ------------------------------------------------------------- keyboard

    @SuppressLint("InflateParams", "ClickableViewAccessibility")
    override fun onCreateInputView(): View {
        val root = layoutInflater.inflate(R.layout.keyboard_view, null)
        statusView = root.findViewById(R.id.status)
        statusView?.text =
            if (modelReady) getString(R.string.ready) else getString(R.string.loading_model)

        root.findViewById<Button>(R.id.btn_mic).setOnTouchListener { v, ev ->
            when (ev.actionMasked) {
                MotionEvent.ACTION_DOWN -> { v.isPressed = true; startDictation() }
                MotionEvent.ACTION_UP,
                MotionEvent.ACTION_CANCEL -> { v.isPressed = false; stopDictation() }
            }
            true
        }

        root.findViewById<Button>(R.id.btn_space).setOnClickListener {
            currentInputConnection?.commitText(" ", 1)
        }
        root.findViewById<Button>(R.id.btn_comma).setOnClickListener {
            currentInputConnection?.commitText(", ", 1)
        }
        root.findViewById<Button>(R.id.btn_period).setOnClickListener {
            currentInputConnection?.commitText(". ", 1)
        }
        root.findViewById<Button>(R.id.btn_enter).setOnClickListener {
            sendDownUpKeyEvents(KeyEvent.KEYCODE_ENTER)
        }
        root.findViewById<Button>(R.id.btn_globe).setOnClickListener {
            switchKeyboard()
        }

        // backspace: tap deletes one, hold repeats
        val backspace = root.findViewById<Button>(R.id.btn_backspace)
        val repeater = object : Runnable {
            override fun run() {
                sendDownUpKeyEvents(KeyEvent.KEYCODE_DEL)
                mainHandler.postDelayed(this, 60L)
            }
        }
        backspace.setOnTouchListener { v, ev ->
            when (ev.actionMasked) {
                MotionEvent.ACTION_DOWN -> {
                    v.isPressed = true
                    sendDownUpKeyEvents(KeyEvent.KEYCODE_DEL)
                    mainHandler.postDelayed(repeater, 400L)
                }
                MotionEvent.ACTION_UP,
                MotionEvent.ACTION_CANCEL -> {
                    v.isPressed = false
                    mainHandler.removeCallbacks(repeater)
                }
            }
            true
        }
        return root
    }

    private fun switchKeyboard() {
        if (Build.VERSION.SDK_INT >= 28) {
            if (!switchToNextInputMethod(false)) {
                (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager)
                    .showInputMethodPicker()
            }
        } else {
            (getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager)
                .showInputMethodPicker()
        }
    }

    private fun setStatus(text: String) {
        mainHandler.post { statusView?.text = text }
    }

    // ------------------------------------------------------------ dictation

    private fun startDictation() {
        if (!modelReady) { setStatus(getString(R.string.loading_model)); return }
        if (recording || finalizing) return
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            setStatus("Microphone permission needed, opening setup")
            startActivity(Intent(this, SetupActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            return
        }

        val rec = recognizer ?: return
        recording = true
        setStatus("Listening \u2026")

        Thread {
            val stream = rec.createStream()
            if (modelKind.startsWith("nemotron")) {
                try { stream.setOption("language", LANGUAGE) } catch (_: Exception) {}
            }

            val minBuf = AudioRecord.getMinBufferSize(
                SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
            )
            val ar = AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                maxOf(minBuf, SAMPLE_RATE) * 2
            )
            audioRecord = ar
            ar.startRecording()

            val shorts = ShortArray(1600) // 100 ms
            var partial = ""
            while (recording) {
                val n = ar.read(shorts, 0, shorts.size)
                if (n <= 0) continue
                val samples = FloatArray(n) { i -> shorts[i] / 32768.0f }
                stream.acceptWaveform(samples, SAMPLE_RATE)
                while (rec.isReady(stream)) rec.decode(stream)
                val text = rec.getResult(stream).text
                if (text != partial) {
                    partial = text
                    setStatus(text.takeLast(120))
                }
            }

            // finalize: tail padding gives the model right-context
            finalizing = true
            try {
                ar.stop()
            } catch (_: Exception) {}
            ar.release()
            audioRecord = null

            stream.acceptWaveform(
                FloatArray((SAMPLE_RATE * TAIL_PAD_SEC).toInt()), SAMPLE_RATE
            )
            stream.inputFinished()
            while (rec.isReady(stream)) rec.decode(stream)
            var text = rec.getResult(stream).text.trim()
            stream.release()

            text = vocab.repair(text)
            mainHandler.post {
                if (text.isNotEmpty()) {
                    currentInputConnection?.commitText("$text ", 1)
                    statusView?.text = getString(R.string.ready)
                } else {
                    statusView?.text = "Heard nothing. " + getString(R.string.ready)
                }
                finalizing = false
            }
        }.start()
    }

    private fun stopDictation() {
        if (!recording) return
        // keep capturing briefly so the last word has trailing audio
        mainHandler.postDelayed({ recording = false }, RELEASE_TAIL_MS)
    }
}
