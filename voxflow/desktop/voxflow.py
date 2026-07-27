#!/usr/bin/env python3
"""
VoxFlow: local dictation for Windows.

Pipeline:
    mic -> sherpa-onnx streaming ASR (Nemotron 3.5) -> vocabulary repair
        -> optional local LLM cleanup (Ollama) -> paste into focused window

Hotkeys (configurable below):
    tap  F6   open mic on/off: just talk, pauses commit each sentence
    hold F8   dictate, insert raw transcript (vocab repair only)
    hold F9   dictate, insert LLM-cleaned text
    tap  F10  undo the last insertion (sends backspaces)
    tap  F7   toggle language between LANGUAGE and "auto"
    Ctrl+C in this console quits

Run setup_desktop.py once first.
"""

import queue
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import keyboard
import pyperclip
import sherpa_onnx

# ============================== CONFIG =====================================

HERE = Path(__file__).resolve().parent

# Folder name inside desktop/models/ (created by setup_desktop.py)
MODEL_DIR = "sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11"

OPEN_MIC_KEY = "f6"     # tap to start hands-free mode, tap again to stop
RAW_KEY = "f8"          # hold to talk, raw output
CLEAN_KEY = "f9"        # hold to talk, LLM-cleaned output
UNDO_KEY = "f10"        # undo last insertion
LANG_KEY = "f7"         # toggle LANGUAGE <-> auto

LANGUAGE = "en"         # "en", "es", "fr", "ja", ... or "auto"
NUM_THREADS = 8         # CPU threads for the ONNX session
PROVIDER = "cpu"        # "cpu" is the reliable choice on Windows.
                        # "cuda" works only with a CUDA build of sherpa-onnx.

SAMPLE_RATE = 16000
CHUNK_SEC = 0.1         # mic chunk size fed to the recognizer
RELEASE_TAIL_SEC = 0.4  # keep capturing this long after key release
TAIL_PAD_SEC = 1.0      # silence padding before finalizing (streaming models
                        # need right-context to emit the last word)

# Open mic endpointing: a segment is committed after this much silence
# once you have said something. Lower = snappier, higher = tolerates
# longer mid-sentence pauses.
ENDPOINT_SILENCE_SEC = 1.0
ENDPOINT_MAX_UTT_SEC = 25.0   # force-commit segments longer than this
OPEN_MIC_CLEAN = False        # True routes every open mic segment through
                              # the Ollama cleanup pass (adds latency)

VOCAB_FILE = HERE.parent / "hotwords.txt"
FUZZY_THRESHOLD = 88    # rapidfuzz score needed to snap a word to vocab

OLLAMA_URL = "http://localhost:11434"
CLEANUP_MODEL = "auto"  # "auto" picks an installed model, or name one,
                        # e.g. "qwen3:4b". "" disables cleanup entirely.

RESTORE_CLIPBOARD = True
PASTE_SETTLE_SEC = 0.35

# ===========================================================================


def log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------- vocabulary ---------------------------------

class Vocab:
    """Post-ASR term repair.

    The Nemotron transducer in sherpa-onnx only supports greedy search, so
    contextual biasing is unavailable at decode time. This class fixes the
    transcript afterwards:
      forced rules   "spoken form => written form"  (regex, whole phrase)
      fuzzy snapping  near-miss words pulled to canonical spellings
    """

    def __init__(self, path: Path):
        self.rules: list[tuple[re.Pattern, str]] = []
        self.terms: list[str] = []
        self._load(path)
        try:
            from rapidfuzz import fuzz
            self._fuzz = fuzz
        except ImportError:
            self._fuzz = None
            log("[vocab] rapidfuzz not installed, fuzzy snapping disabled")

    def _load(self, path: Path) -> None:
        if not path.exists():
            log(f"[vocab] no vocab file at {path}, skipping")
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=>" in line:
                spoken, written = (s.strip() for s in line.split("=>", 1))
                pat = re.compile(r"\b" + re.escape(spoken) + r"\b", re.IGNORECASE)
                self.rules.append((pat, written))
            else:
                self.terms.append(line)
        log(f"[vocab] {len(self.terms)} terms, {len(self.rules)} rules loaded")

    def repair(self, text: str) -> str:
        for pat, written in self.rules:
            text = pat.sub(written, text)

        if self._fuzz and self.terms:
            fixed_words = []
            for word in text.split(" "):
                core = word.strip(".,!?;:()[]\"'")
                if len(core) >= 4:
                    best_term, best_score = None, 0
                    for term in self.terms:
                        if " " in term:
                            continue
                        score = self._fuzz.ratio(core.lower(), term.lower())
                        if score > best_score:
                            best_term, best_score = term, score
                    if best_term and best_score >= FUZZY_THRESHOLD:
                        word = word.replace(core, best_term)
                fixed_words.append(word)
            text = " ".join(fixed_words)
        return text


# ----------------------------- LLM cleanup --------------------------------

class Cleaner:
    def __init__(self, vocab_terms: list[str]):
        self.model = None
        if not CLEANUP_MODEL:
            return
        try:
            tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3).json()
            names = [m["name"] for m in tags.get("models", [])]
        except Exception:
            log("[cleanup] Ollama not reachable, F9 will fall back to raw output")
            return
        if not names:
            log("[cleanup] Ollama has no models installed, cleanup disabled")
            return
        if CLEANUP_MODEL != "auto":
            self.model = CLEANUP_MODEL if CLEANUP_MODEL in names else None
            if self.model is None:
                log(f"[cleanup] '{CLEANUP_MODEL}' not found in Ollama, "
                    f"available: {names}")
                return
        else:
            for pref in ("qwen3", "qwen", "llama", "gemma", "mistral", "phi"):
                match = next((n for n in names if pref in n.lower()), None)
                if match:
                    self.model = match
                    break
            if self.model is None:
                self.model = names[0]
        self.terms = ", ".join(vocab_terms[:60])
        log(f"[cleanup] using Ollama model: {self.model}")

    def clean(self, text: str) -> str:
        if not self.model:
            return text
        system = (
            "You clean up dictated speech into polished written text. "
            "Fix punctuation and capitalization. Remove filler words "
            "(um, uh, like, you know), false starts, and repeated words. "
            "When the speaker corrects themselves, keep only the corrected "
            "version. Never change numbers, units, equations, or technical "
            "terms. Keep these vocabulary terms exactly as written when they "
            f"appear: {self.terms}. "
            "Output only the cleaned text, nothing else."
        )
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": text},
                    ],
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0.1},
                },
                timeout=60,
            )
            out = r.json()["message"]["content"]
            out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL)
            out = out.strip().strip('"')
            return out if out else text
        except Exception as e:
            log(f"[cleanup] failed ({e}), inserting raw text")
            return text


# ----------------------------- text insertion -----------------------------

class Inserter:
    def __init__(self):
        self.last_len = 0

    def insert(self, text: str) -> None:
        if not text:
            return
        old_clip = None
        if RESTORE_CLIPBOARD:
            try:
                old_clip = pyperclip.paste()
            except Exception:
                old_clip = None
        pyperclip.copy(text)
        time.sleep(0.05)
        keyboard.send("ctrl+v")
        time.sleep(PASTE_SETTLE_SEC)
        if RESTORE_CLIPBOARD and old_clip is not None:
            try:
                pyperclip.copy(old_clip)
            except Exception:
                pass
        self.last_len = len(text)

    def undo(self) -> None:
        if self.last_len <= 0:
            return
        log(f"[undo] deleting last {self.last_len} characters")
        for _ in range(self.last_len):
            keyboard.send("backspace")
            time.sleep(0.004)
        self.last_len = 0


# ----------------------------- the app ------------------------------------

class VoxFlow:
    def __init__(self):
        model_path = HERE / "models" / MODEL_DIR
        if not (model_path / "tokens.txt").exists():
            log(f"Model not found at {model_path}")
            log("Run: python setup_desktop.py")
            sys.exit(1)

        log(f"Loading {MODEL_DIR} ({PROVIDER}, {NUM_THREADS} threads) ...")
        t0 = time.time()
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(model_path / "tokens.txt"),
            encoder=str(model_path / "encoder.int8.onnx"),
            decoder=str(model_path / "decoder.int8.onnx"),
            joiner=str(model_path / "joiner.int8.onnx"),
            num_threads=NUM_THREADS,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            decoding_method="greedy_search",
            provider=PROVIDER,
            # endpointing is only consulted in open mic mode; hold-to-talk
            # finalizes on key release regardless
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=ENDPOINT_SILENCE_SEC,
            rule3_min_utterance_length=ENDPOINT_MAX_UTT_SEC,
        )
        log(f"Model loaded in {time.time() - t0:.1f} s")

        self.vocab = Vocab(VOCAB_FILE)
        self.cleaner = Cleaner(self.vocab.terms)
        self.inserter = Inserter()

        self.language = LANGUAGE
        self.audio_q: queue.Queue[np.ndarray] = queue.Queue()
        self.capturing = False
        self.open_mic = False
        self.busy = threading.Lock()

        self.stream_in = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(SAMPLE_RATE * CHUNK_SEC),
            callback=self._audio_cb,
        )
        self.stream_in.start()

    # mic callback: runs on the audio thread, keep it tiny
    def _audio_cb(self, indata, frames, time_info, status):
        if self.capturing:
            self.audio_q.put(indata[:, 0].copy())

    def _drain_queue_into(self, s) -> None:
        while True:
            try:
                chunk = self.audio_q.get_nowait()
            except queue.Empty:
                break
            s.accept_waveform(SAMPLE_RATE, chunk)

    def dictate(self, hold_key: str, clean_mode: bool) -> None:
        if not self.busy.acquire(blocking=False):
            return
        try:
            mode = "clean" if clean_mode else "raw"
            log(f"\n[listening:{mode}] hold {hold_key.upper()} and speak ...")

            s = self.recognizer.create_stream()
            if self.language and self.language != "auto":
                s.set_option("language", self.language)

            with self.audio_q.mutex:
                self.audio_q.queue.clear()
            self.capturing = True
            t_start = time.time()

            partial = ""
            while keyboard.is_pressed(hold_key):
                try:
                    chunk = self.audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                s.accept_waveform(SAMPLE_RATE, chunk)
                while self.recognizer.is_ready(s):
                    self.recognizer.decode_stream(s)
                new_partial = self.recognizer.get_result(s)
                if new_partial != partial:
                    partial = new_partial
                    sys.stdout.write("\r  " + partial[-100:] + " " * 8)
                    sys.stdout.flush()

            # capture a short natural tail after key release
            time.sleep(RELEASE_TAIL_SEC)
            self.capturing = False
            self._drain_queue_into(s)

            spoken_sec = time.time() - t_start
            s.accept_waveform(
                SAMPLE_RATE,
                np.zeros(int(SAMPLE_RATE * TAIL_PAD_SEC), dtype=np.float32),
            )
            s.input_finished()
            while self.recognizer.is_ready(s):
                self.recognizer.decode_stream(s)

            text = self.recognizer.get_result(s).strip()
            print()
            if not text:
                log("[done] heard nothing")
                return

            text = self.vocab.repair(text)
            if clean_mode:
                log("[cleanup] rewriting ...")
                text = self.cleaner.clean(text)

            log(f"[insert] {text}")
            self.inserter.insert(text + " ")
            log(f"[done] {spoken_sec:.1f} s of speech")
        finally:
            self.capturing = False
            self.busy.release()

    def toggle_language(self) -> None:
        self.language = "auto" if self.language != "auto" else LANGUAGE
        log(f"[language] now: {self.language}")

    # ------------------------- open mic mode ------------------------------

    def toggle_open_mic(self) -> None:
        if self.open_mic:
            self.open_mic = False   # the loop notices and shuts down
            return
        if not self.busy.acquire(blocking=False):
            log("[open mic] busy with a hold-to-talk dictation, try again")
            return
        self.open_mic = True
        threading.Thread(target=self._open_mic_loop, daemon=True).start()

    def _commit_segment(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        text = self.vocab.repair(text)
        if OPEN_MIC_CLEAN:
            text = self.cleaner.clean(text)
        print()
        log(f"[insert] {text}")
        self.inserter.insert(text + " ")

    def _open_mic_loop(self) -> None:
        try:
            log(f"\n[open mic] ON. Just talk. A {ENDPOINT_SILENCE_SEC:.1f} s "
                f"pause commits the sentence. Tap {OPEN_MIC_KEY.upper()} "
                "again to stop.")
            s = self.recognizer.create_stream()
            if self.language and self.language != "auto":
                s.set_option("language", self.language)

            with self.audio_q.mutex:
                self.audio_q.queue.clear()
            self.capturing = True

            partial = ""
            while self.open_mic:
                try:
                    chunk = self.audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                s.accept_waveform(SAMPLE_RATE, chunk)
                while self.recognizer.is_ready(s):
                    self.recognizer.decode_stream(s)

                new_partial = self.recognizer.get_result(s)
                if new_partial != partial:
                    partial = new_partial
                    sys.stdout.write("\r  " + partial[-100:] + " " * 8)
                    sys.stdout.flush()

                if self.recognizer.is_endpoint(s):
                    self._commit_segment(self.recognizer.get_result(s))
                    self.recognizer.reset(s)
                    partial = ""

            # mode switched off: flush whatever is still in flight
            self.capturing = False
            self._drain_queue_into(s)
            s.accept_waveform(
                SAMPLE_RATE,
                np.zeros(int(SAMPLE_RATE * TAIL_PAD_SEC), dtype=np.float32),
            )
            s.input_finished()
            while self.recognizer.is_ready(s):
                self.recognizer.decode_stream(s)
            self._commit_segment(self.recognizer.get_result(s))
            log("[open mic] OFF")
        finally:
            self.capturing = False
            self.open_mic = False
            self.busy.release()

    def run(self) -> None:
        keyboard.on_press_key(
            RAW_KEY, lambda e: threading.Thread(
                target=self.dictate, args=(RAW_KEY, False), daemon=True).start())
        keyboard.on_press_key(
            CLEAN_KEY, lambda e: threading.Thread(
                target=self.dictate, args=(CLEAN_KEY, True), daemon=True).start())
        keyboard.on_press_key(UNDO_KEY, lambda e: self.inserter.undo())
        keyboard.on_press_key(LANG_KEY, lambda e: self.toggle_language())
        keyboard.on_press_key(OPEN_MIC_KEY, lambda e: self.toggle_open_mic())

        log("")
        log("=" * 62)
        log("VoxFlow ready.")
        log(f"  tap  {OPEN_MIC_KEY.upper()}  : open mic on/off (hands free, "
            "pauses commit text)")
        log(f"  hold {RAW_KEY.upper()}  : dictate (raw + vocab repair)")
        log(f"  hold {CLEAN_KEY.upper()}  : dictate (LLM cleanup)")
        log(f"  tap  {UNDO_KEY.upper()} : undo last insertion")
        log(f"  tap  {LANG_KEY.upper()}  : toggle language ({self.language} <-> auto)")
        log("  Ctrl+C here to quit")
        log("=" * 62)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log("\nbye")


if __name__ == "__main__":
    VoxFlow().run()
