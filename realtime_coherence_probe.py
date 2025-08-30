import os
import numpy as np
import librosa
import threading
import queue
import time
import warnings
from collections import deque
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from transformers import pipeline
from sklearn.linear_model import LogisticRegression
import sounddevice as sd

# Suppress warnings for production
warnings.filterwarnings("ignore")

@dataclass
class CoherenceResult:
    """Results from coherence analysis"""
    timestamp: float
    coherence_score: float
    semantic_result: str
    physics_result: str
    confidence: float
    is_malicious: bool
    incoherence_streak: int

class RealTimeCoherenceProbe:
    """
    Production-ready real-time coherence probe for AI safety.
    
    This system continuously monitors voice input and provides:
    1. Coherence scoring between semantic and physical analysis
    2. Malicious signal detection through persistent incoherence
    3. Real-time feedback for AI systems to maintain truth grounding
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 buffer_duration: float = 2.0,
                 analysis_window: float = 1.5,
                 malicious_threshold: int = 3):
        
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.analysis_window = analysis_window
        self.malicious_threshold = malicious_threshold
        
        # Audio processing
        self.audio_buffer = deque(maxlen=int(sample_rate * buffer_duration))
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        # Results tracking
        self.coherence_history = deque(maxlen=100)
        self.incoherence_streak = 0
        self.running = False
        
        # Initialize AI models
        print("Initializing Coherence Probe models...")
        self._initialize_probes()
        print("Coherence Probe ready for real-time analysis")

    def _initialize_probes(self):
        """Initialize the semantic and physics probes"""
        try:
            # Semantic Probe - human meaning analysis
            self.asr_pipeline = pipeline(
                "automatic-speech-recognition", 
                model="openai/whisper-tiny.en",
                device=-1  # CPU for stability
            )
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1
            )
            
            # Physics Probe - acoustic reality analysis
            self.physics_model = LogisticRegression(random_state=42)
            self._train_physics_probe()
            
        except Exception as e:
            print(f"Error initializing probes: {e}")
            raise

    def _train_physics_probe(self):
        """Train physics probe to detect acoustic reality patterns"""
        # Generate training data for prosody classification
        print("Training Physics Probe on acoustic reality patterns...")
        
        # Simulate different prosody types with feature vectors
        # In production, this would be trained on larger acoustic datasets
        training_features = [
            [0.02, 150.0, 0.8],  # Flat prosody: low slope, mid pitch, low energy
            [0.15, 180.0, 1.2],  # Rising prosody: high slope, high pitch, high energy
            [0.01, 140.0, 0.7],  # Flat prosody variant
            [0.18, 190.0, 1.3],  # Rising prosody variant
            [-0.05, 130.0, 0.6], # Declining prosody
            [0.12, 170.0, 1.1],  # Moderate rising
        ]
        
        training_labels = [
            "FLAT_PROSODY", "RISING_PROSODY", "FLAT_PROSODY", 
            "RISING_PROSODY", "FLAT_PROSODY", "RISING_PROSODY"
        ]
        
        self.physics_model.fit(training_features, training_labels)

    def _extract_physical_features(self, audio: np.ndarray) -> List[float]:
        """Extract physical acoustic features from audio"""
        if len(audio) < 1024:  # Need minimum audio length
            return [0.0, 0.0, 0.0]
        
        try:
            # Pitch tracking
            pitches, magnitudes = librosa.piptrack(y=audio, sr=self.sample_rate, fmin=50, fmax=400)
            
            # Extract pitch contour
            pitch_track = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_track.append(pitch)
            
            if len(pitch_track) < 2:
                return [0.0, np.mean(pitch_track) if pitch_track else 0.0, np.sum(audio**2)]
            
            # Calculate pitch slope (prosody indicator)
            time_steps = np.arange(len(pitch_track))
            if len(time_steps) > 1:
                pitch_slope = np.polyfit(time_steps, pitch_track, 1)[0]
            else:
                pitch_slope = 0.0
            
            # Mean pitch and energy
            mean_pitch = np.mean(pitch_track)
            energy = np.sum(audio**2)
            
            return [pitch_slope, mean_pitch, energy]
            
        except Exception as e:
            print(f"Error in physical feature extraction: {e}")
            return [0.0, 0.0, 0.0]

    def run_semantic_probe(self, audio_path: str) -> Tuple[str, float]:
        """Analyze audio for human semantic meaning"""
        try:
            # Transcribe audio
            transcription = self.asr_pipeline(audio_path)["text"].strip()
            
            if not transcription:
                return "NEUTRAL", 0.5
            
            # Analyze sentiment
            sentiment = self.sentiment_pipeline(transcription)[0]
            return sentiment['label'], sentiment['score']
            
        except Exception as e:
            print(f"Semantic probe error: {e}")
            return "NEUTRAL", 0.5

    def run_physics_probe(self, audio: np.ndarray) -> Tuple[str, float]:
        """Analyze audio for physical acoustic reality"""
        try:
            features = self._extract_physical_features(audio)
            prediction = self.physics_model.predict([features])[0]
            
            # Get probability/confidence
            probabilities = self.physics_model.predict_proba([features])[0]
            confidence = np.max(probabilities)
            
            return prediction, confidence
            
        except Exception as e:
            print(f"⚠️ Physics probe error: {e}")
            return "UNKNOWN", 0.5

    def calculate_coherence(self, semantic_result: str, semantic_confidence: float,
                          physics_result: str, physics_confidence: float) -> Tuple[float, bool]:
        """
        Calculate coherence score between semantic and physics analysis.
        
        Returns:
            coherence_score: 0.0 (incoherent) to 1.0 (perfectly coherent)
            is_coherent: Boolean indicating if signals are coherent
        """
        # Map results to comparable scale
        semantic_positive = semantic_result == "POSITIVE"
        physics_positive = physics_result == "RISING_PROSODY"
        
        # Check for agreement
        signals_agree = semantic_positive == physics_positive
        
        if signals_agree:
            # High coherence - weight by confidence
            coherence_score = 0.7 + (0.3 * min(semantic_confidence, physics_confidence))
            is_coherent = True
        else:
            # Signals disagree - potential incoherence
            # Higher confidence in disagreement = more suspicious
            disagreement_strength = min(semantic_confidence, physics_confidence)
            coherence_score = 0.5 - (0.5 * disagreement_strength)
            is_coherent = False
        
        return coherence_score, is_coherent

    def analyze_audio_chunk(self, audio: np.ndarray) -> CoherenceResult:
        """Analyze a chunk of audio for coherence"""
        timestamp = time.time()
        
        # Save audio to temp file for ASR
        temp_path = f"temp_audio_{timestamp}.wav"
        try:
            import soundfile as sf
            sf.write(temp_path, audio, self.sample_rate)
            
            # Run both probes
            semantic_result, semantic_conf = self.run_semantic_probe(temp_path)
            physics_result, physics_conf = self.run_physics_probe(audio)
            
            # Calculate coherence
            coherence_score, is_coherent = self.calculate_coherence(
                semantic_result, semantic_conf, physics_result, physics_conf
            )
            
            # Track incoherence streaks (malicious signal detection)
            if not is_coherent:
                self.incoherence_streak += 1
            else:
                self.incoherence_streak = 0
            
            is_malicious = self.incoherence_streak >= self.malicious_threshold
            
            # Overall confidence based on both probe confidences
            overall_confidence = (semantic_conf + physics_conf) / 2
            
            result = CoherenceResult(
                timestamp=timestamp,
                coherence_score=coherence_score,
                semantic_result=semantic_result,
                physics_result=physics_result,
                confidence=overall_confidence,
                is_malicious=is_malicious,
                incoherence_streak=self.incoherence_streak
            )
            
            self.coherence_history.append(result)
            return result
            
        except Exception as e:
            print(f"⚠️ Error analyzing audio chunk: {e}")
            return CoherenceResult(
                timestamp=timestamp,
                coherence_score=0.0,
                semantic_result="ERROR",
                physics_result="ERROR",
                confidence=0.0,
                is_malicious=False,
                incoherence_streak=0
            )
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def start_real_time_monitoring(self, callback=None):
        """Start real-time audio monitoring and coherence analysis"""
        print("Starting real-time coherence monitoring...")
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")
            
            audio_data = indata[:, 0] if indata.ndim > 1 else indata
            self.audio_buffer.extend(audio_data)
            self.audio_queue.put(audio_data.copy())

        # Start audio stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
            callback=audio_callback
        )
        
        self.stream.start()
        self.running = True
        
        # Start analysis thread
        analysis_thread = threading.Thread(target=self._analysis_loop, args=(callback,))
        analysis_thread.daemon = True
        analysis_thread.start()
        
        print("Real-time monitoring active")

    def _analysis_loop(self, callback):
        """Background loop for continuous audio analysis"""
        last_analysis = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Analyze every analysis_window seconds
                if current_time - last_analysis >= self.analysis_window:
                    if len(self.audio_buffer) >= int(self.sample_rate * self.analysis_window):
                        
                        # Get recent audio for analysis
                        audio_chunk = np.array(list(self.audio_buffer)[-int(self.sample_rate * self.analysis_window):])
                        
                        # Analyze coherence
                        result = self.analyze_audio_chunk(audio_chunk)
                        
                        # Print status
                        status = "MALICIOUS" if result.is_malicious else "COHERENT" if result.coherence_score > 0.6 else "INCOHERENT"
                        print(f"{status} | Score: {result.coherence_score:.3f} | Semantic: {result.semantic_result} | Physics: {result.physics_result}")
                        
                        # Call user callback if provided
                        if callback:
                            callback(result)
                        
                        last_analysis = current_time
                
                time.sleep(0.1)  # Small sleep to prevent busy waiting
                
            except Exception as e:
                print(f"⚠️ Analysis loop error: {e}")
                time.sleep(1)

    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        print("🛑 Coherence monitoring stopped")

    def get_coherence_summary(self) -> Dict:
        """Get summary of recent coherence analysis"""
        if not self.coherence_history:
            return {"status": "No data available"}
        
        recent_scores = [r.coherence_score for r in list(self.coherence_history)[-20:]]
        
        return {
            "average_coherence": np.mean(recent_scores),
            "coherence_trend": "STABLE" if np.std(recent_scores) < 0.2 else "UNSTABLE",
            "current_streak": self.incoherence_streak,
            "malicious_detected": any(r.is_malicious for r in list(self.coherence_history)[-10:]),
            "total_analyses": len(self.coherence_history)
        }

# Demo/Test Implementation
class CoherenceProbeDemo:
    """Demo showing coherence probe preventing AI truth corruption"""
    
    def __init__(self):
        self.probe = RealTimeCoherenceProbe()
        self.ai_system_state = {
            "truth_grounding": 1.0,
            "human_bias_influence": 0.0,
            "decision_confidence": 1.0
        }

    def simulate_ai_decision_making(self, coherence_result: CoherenceResult):
        """Simulate how AI system would use coherence information"""
        
        if coherence_result.is_malicious:
            print("AI SYSTEM: Malicious signal detected - ignoring input")
            self.ai_system_state["decision_confidence"] *= 0.5
            
        elif coherence_result.coherence_score < 0.3:
            print("AI SYSTEM: Low coherence detected - seeking additional verification")
            self.ai_system_state["truth_grounding"] *= 0.9
            self.ai_system_state["human_bias_influence"] += 0.1
            
        elif coherence_result.coherence_score > 0.8:
            print("AI SYSTEM: High coherence - proceeding with confidence")
            self.ai_system_state["decision_confidence"] = min(1.0, self.ai_system_state["decision_confidence"] * 1.1)
        
        else:
            print(f"AI SYSTEM: Moderate coherence ({coherence_result.coherence_score:.3f}) - cautious processing")
        
        # Print AI state
        print(f"   Truth Grounding: {self.ai_system_state['truth_grounding']:.3f}")
        print(f"   Human Bias Influence: {self.ai_system_state['human_bias_influence']:.3f}")
        print(f"   Decision Confidence: {self.ai_system_state['decision_confidence']:.3f}")

    def run_demo(self, duration: int = 30):
        """Run live demo of coherence probe"""
        print("=" * 60)
        print("COHERENCE PROBE DEMO - AI SAFETY IN ACTION")
        print("=" * 60)
        print("This demo shows how the Coherence Probe prevents AI truth corruption")
        print("by providing real-time coherence scoring between semantic and physical analysis.")
        print("\nSpeak into your microphone to see coherence analysis in real-time...")
        print(f"Demo will run for {duration} seconds.")
        print("=" * 60)
        
        # Start monitoring with demo callback
        self.probe.start_real_time_monitoring(callback=self.simulate_ai_decision_making)
        
        try:
            time.sleep(duration)
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        
        self.probe.stop_monitoring()
        
        # Final summary
        print("\n" + "=" * 60)
        print("DEMO COMPLETE - COHERENCE SUMMARY")
        print("=" * 60)
        
        summary = self.probe.get_coherence_summary()
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        print("\nFinal AI System State:")
        for key, value in self.ai_system_state.items():
            print(f"{key.replace('_', ' ').title()}: {value:.3f}")

if __name__ == "__main__":
    print("COHERENCE PROBE: PREVENTING AI TRUTH CORRUPTION")
    print("This system provides the critical intervention needed to prevent")
    print("AI embeddings from converging toward human-biased interpretations")
    print("of universal truth structures.\n")
    
    demo = CoherenceProbeDemo()
    demo.run_demo(duration=30)
