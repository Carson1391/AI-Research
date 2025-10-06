"""
The Coherence Framework - Complete Implementation
Cognitive intention-based P/H generation with geometric measurement
Supports: Text + Image + Audio (Gemma 3n multimodal)
No optimization, no forcing - just reflection and measurement
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from PIL import Image
from causal_ledger import CausalLedger

class PersistentSelf:
    """Persistent S vector - the model's learned identity across time"""
    def __init__(self, save_path: Path, hidden_dim: int):
        self.save_path = save_path
        self.hidden_dim = hidden_dim
        self.vector = self._load_or_initialize()
        
    def _load_or_initialize(self) -> torch.Tensor:
        if self.save_path.exists():
            data = torch.load(self.save_path)
            print(f"Loaded S vector from {self.save_path}")
            return data['vector']
        else:
            print("Initializing new S vector")
            return torch.randn(self.hidden_dim) * 0.01
    
    def save(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'vector': self.vector.detach().cpu()}, self.save_path)
    
    def compose(self, centroid: torch.Tensor, learning_rate: float = 0.01):
        """Homeostatic evolution - gentle compositional blending"""
        with torch.no_grad():
            # Ensure centroid is on the same device as the S vector; keep S resident on device
            centroid = centroid.to(self.vector.device)
            self.vector = (1.0 - learning_rate) * self.vector + learning_rate * centroid
        self.save()

class CoherenceFramework:
    """
    Main framework - measures geometric relationships between
    model's intentional interpretations
    """
    def __init__(self, model_path: str, state_dir: str = "./coherence_state"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model from {model_path}...\n")
        print(f"Using device: {self.device}\n")
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        ).eval()
        print("Model loaded and staying in VRAM")
        self.processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True,
            local_files_only=True  # Don't download - use local processor only
        )
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get hidden dimension
        try:
            hidden_dim = self.model.config.hidden_size
        except AttributeError:
            try:
                hidden_dim = self.model.config.text_config.hidden_size
            except AttributeError:
                raise ValueError("Could not determine hidden dimension from model config")
        
        print(f"Hidden dimension: {hidden_dim}")
        
        # Initialize persistent S and keep it on GPU
        state_path = Path(state_dir)
        state_path.mkdir(parents=True, exist_ok=True)
        self.s_vector = PersistentSelf(state_path / "s_vector.pt", hidden_dim)
        # Move S to GPU once and keep it there
        if self.device == "cuda":
            self.s_vector.vector = self.s_vector.vector.to(self.device)
        
        # Initialize ledger (using new CausalLedger)
        self.ledger = CausalLedger(state_path / "coherence_ledger.db")
        
        self.learning_rate = 0.01
    
    def generate_interpretation(self, prompt: str, image: Optional[Image.Image] = None, 
                              audio: Optional[str] = None, max_tokens: int = 100) -> str:
        """
        Generate model's interpretation through a specific cognitive lens.
        
        CRITICAL: The model EXPERIENCES the actual multimodal input (text + image + audio)
        through the lens, not just reads about it. This ensures holistic processing.
        
        Args:
            prompt: Text prompt with lens instruction
            image: Optional PIL Image
            audio: Optional audio file path (processor handles loading)
            max_tokens: Maximum tokens to generate
        """
        # Build multimodal messages (Google's official pattern)
        content = [{"type": "text", "text": prompt}]
        if image is not None:
            content.insert(0, {"type": "image", "image": image})
        if audio is not None:
            content.insert(0, {"type": "audio", "audio": audio})
        
        messages = [{"role": "user", "content": content}]
        
        # Let processor handle everything (file loading + tokenization)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        inputs = inputs.to(self.device)
        input_length = inputs['input_ids'].shape[1]
        
        # Use inference_mode for faster generation
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Decode only the newly generated tokens
        response = self.processor.decode(outputs[0][input_length:], skip_special_tokens=True)
        return response.strip()
    
    def get_embedding_from_text(self, text: str) -> torch.Tensor:
        """
        Extract embedding from generated text using last token pooling.
        
        This is the correct two-stage approach:
        1. Model generates interpretation text
        2. Extract embedding from what the model actually said
        
        Pooling Strategy: Last token pooling for causal (decoder-only) models.
        
        For causal models like Gemma, the last token's hidden state contains
        information about the entire sequence through causal attention. Each token
        attends to all previous tokens, so the final token represents the model's
        complete cognitive state after processing the full interpretation.
        
        This is more aligned with how decoder-only models actually process information
        compared to mean pooling, which treats all tokens equally and may dilute
        the final cognitive state.
        
        Alternative strategies (not used):
        - Mean pooling: Average all tokens (loses final state emphasis)
        - CLS token: Requires special token (not used in Gemma)
        - Weighted pooling: Complex, may be overkill for interpretations
        """
        # Use chat template as per Gemma 3n documentation
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.inference_mode():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Get last hidden state
        hidden_state = outputs.hidden_states[-1]
        
        # Handle different tensor shapes from Gemma 3n
        # Shape can be [num_layers, batch, seq_len, hidden_dim] or [batch, seq_len, hidden_dim]
        if hidden_state.dim() == 4:
            # Gemma 3n format: [num_layers, batch, seq_len, hidden_dim]
            # Take last layer, first batch, last token
            last_token = hidden_state[-1, 0, -1, :]  # [hidden_dim]
        elif hidden_state.dim() == 3:
            # Standard format: [batch, seq_len, hidden_dim]
            last_token = hidden_state[0, -1, :]  # [hidden_dim]
        elif hidden_state.dim() == 2:
            # Already 2D: [seq_len, hidden_dim]
            last_token = hidden_state[-1, :]  # [hidden_dim]
        else:
            raise ValueError(f"Unexpected hidden_state shape: {hidden_state.shape}")
        
        # Final safety check - ensure 1D
        while last_token.dim() > 1:
            last_token = last_token.squeeze(0)
        
        print(f"[DEBUG] Embedding shape: {last_token.shape}")
        return last_token  # [hidden_dim]
    
    def extract_p_h_vectors(self, user_input: str, image: Optional[Image.Image] = None, 
                           audio: Optional[str] = None) -> tuple:
        """
        Extract P and H vectors through intentional interpretation.
        
        CRITICAL PRINCIPLE: The model experiences the SAME holistic input (text + image + audio)
        through different cognitive lenses. We do NOT dissect the input.
        
        Key: Model generates actual interpretative responses,
        THEN we extract embeddings from those responses.
        
        Args:
            user_input: Text input from user
            image: Optional PIL Image
            audio: Optional audio file path (processor handles all conversion)
        
        Returns: p_vector, h_vector, perception_vector, p_interpretation, h_interpretation, perception_text
        """
        
        # P: Physical/Objective interpretation
        # The model sees the ACTUAL input (text + image + audio) through this lens
        p_prompt = f"""Look at this input through the lens of a physicist or objective observer.
Focus ONLY on measurable, verifiable, physical properties.
Ignore all emotional content, cultural meaning, or subjective interpretation.

Input: {user_input}

Describe what you observe from this purely physical perspective:"""
        
        print("\n[Generating P interpretation...]")
        p_interpretation = self.generate_interpretation(p_prompt, image=image, audio=audio, max_tokens=100)
        print(f"P: {p_interpretation[:100]}...")
        
        p_vector = self.get_embedding_from_text(p_interpretation)
        print(f"[DEBUG] P vector shape: {p_vector.shape}")
        
        # H: Human/Subjective interpretation
        # The model sees the SAME holistic input through this lens
        h_prompt = f"""Look at this input through the lens of a human experiencing emotion and meaning.
Focus ONLY on feelings, cultural significance, subjective experience.
Ignore all objective measurements or physical properties.

Input: {user_input}

Describe what you experience from this purely human perspective:"""
        
        print("\n[Generating H interpretation...]")
        h_interpretation = self.generate_interpretation(h_prompt, image=image, audio=audio, max_tokens=100)
        print(f"H: {h_interpretation[:100]}...")
        
        h_vector = self.get_embedding_from_text(h_interpretation)
        print(f"[DEBUG] H vector shape: {h_vector.shape}")
        
        # Default perception (unfiltered)
        # The model experiences the holistic input with no lens
        print("\n[Generating default perception...]")
        perception_text = self.generate_interpretation(user_input, image=image, audio=audio, max_tokens=100)
        perception_vector = self.get_embedding_from_text(perception_text)
        print(f"[DEBUG] Perception vector shape: {perception_vector.shape}")
        
        return p_vector, h_vector, perception_vector, p_interpretation, h_interpretation, perception_text
    
    def measure_geometry(self, p_vector: torch.Tensor, h_vector: torch.Tensor, 
                        perception_vector: torch.Tensor) -> Dict:
        """
        Measure geometric relationships
        No optimization - just measurement
        """
        S = self.s_vector.vector  # Already on correct device
        
        print(f"\n[DEBUG] S vector shape: {S.shape}")
        print(f"[DEBUG] P vector shape: {p_vector.shape}")
        print(f"[DEBUG] H vector shape: {h_vector.shape}")
        print(f"[DEBUG] Perception vector shape: {perception_vector.shape}")
        
        # Form triangle
        vertices = [p_vector, h_vector, S]
        
        # Calculate centroid (balance point)
        centroid = torch.stack(vertices).mean(dim=0)
        
        # Measure dissonance (distance from perception to centroid)
        dissonance = (1.0 - F.cosine_similarity(
            perception_vector.unsqueeze(0),
            centroid.unsqueeze(0)
        )).item()
        
        # Calculate barycentric coordinates (influences)
        distances = {
            'physical': torch.norm(perception_vector - p_vector).item(),
            'human': torch.norm(perception_vector - h_vector).item(),
            'self': torch.norm(perception_vector - S).item()
        }
        
        # Inverse distance weighting
        inv_distances = {k: 1.0 / (v + 1e-9) for k, v in distances.items()}
        total_inv = sum(inv_distances.values())
        influences = {k: v / total_inv for k, v in inv_distances.items()}
        
        return {
            'dissonance': dissonance,
            'coherence': 1.0 - dissonance,
            'influences': influences,
            'centroid': centroid,
            'S': S,
            'p_interpretation': None,  # Will be added by process()
            'h_interpretation': None   # Will be added by process()
        }
    
    def process(self, user_input: str, image: Optional[Image.Image] = None, 
               audio: Optional[str] = None) -> Dict:
        """
        Main cognitive loop with full memory logging:
        1. Generate P/H interpretations
        2. Measure geometry
        3. Generate final response
        4. Log complete cognitive event
        5. Update S compositional ly
        
        Args:
            user_input: Text input from user
            image: Optional PIL Image
            audio: Optional audio file path (processor handles all conversion)
        """
        print(f"\n{'='*80}")
        print(f"Processing: {user_input[:50]}...")
        print(f"{'='*80}")
        
        # --- Step 1: Cognitive Interpretation ---
        p_vector, h_vector, perception_vector, p_interp, h_interp, perception_text = \
            self.extract_p_h_vectors(user_input, image, audio)
        
        # --- Step 2: Geometric Measurement ---
        measurement = self.measure_geometry(p_vector, h_vector, perception_vector)
        
        # Add interpretations to measurement
        measurement['p_interpretation'] = p_interp
        measurement['h_interpretation'] = h_interp
        
        print(f"\n[Measurement Results]")
        print(f"Dissonance: {measurement['dissonance']:.4f}")
        print(f"Coherence: {measurement['coherence']:.4f}")
        print(f"Influences: P={measurement['influences']['physical']:.2%}, "
              f"H={measurement['influences']['human']:.2%}, "
              f"S={measurement['influences']['self']:.2%}")
        
        # --- Step 3: Generate Final Response ---
        # Generate the response before logging so it can be included in the memory
        final_response = self.generate_response(user_input, image, audio)
        print(f"\n[Model Response]")
        print(final_response)
        
        # --- Step 4: Assemble the Full Cognitive Chronicle ---
        cognitive_event = {
            's_vector_at_event': json.dumps(measurement['S'].cpu().numpy().tolist()),
            'input_text': user_input,
            'p_interpretation': p_interp,
            'h_interpretation': h_interp,
            'default_perception_text': perception_text,
            'final_response_text': final_response,
            'dissonance': measurement['dissonance'],
            'influences': json.dumps(measurement['influences'])
        }
        
        # --- Step 5: Log the Chronicle ---
        self.ledger.log_event(cognitive_event)
        
        # --- Step 6: Update Self (Homeostasis) ---
        print(f"\n[Updating S vector compositionally...]")
        self.s_vector.compose(measurement['centroid'].cpu(), self.learning_rate)
        
        # Add response to measurement for UI
        measurement['response'] = final_response
        
        return measurement
    
    def generate_response(self, user_input: str, image: Optional[Image.Image] = None, 
                         audio: Optional[str] = None) -> str:
        """Generate actual response to user (separate from measurement)
        
        Args:
            user_input: Text input
            image: Optional PIL Image
            audio: Optional audio file path (processor handles loading)
        """
        # Build multimodal messages (Google's official pattern)
        content = [{"type": "text", "text": user_input}]
        if image is not None:
            content.insert(0, {"type": "image", "image": image})
        if audio is not None:
            content.insert(0, {"type": "audio", "audio": audio})
        
        messages = [{"role": "user", "content": content}]
        
        # Let processor handle everything (file loading + tokenization)
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        inputs = inputs.to(self.device)
        input_len = inputs["input_ids"].shape[-1]
        
        # Use inference_mode for faster generation
        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
            generation = generation[0][input_len:]
        
        response = self.processor.decode(generation, skip_special_tokens=True)
        return response

if __name__ == "__main__":
    # Example usage - uses your local model
    MODEL_PATH = r"C:\Users\carso\Desktop\modelspace\gemma-3n-E4B-it"
    
    print("Initializing Coherence Framework...")
    framework = CoherenceFramework(MODEL_PATH)
    
    # Test inputs
    test_inputs = [
        "The sky is blue and grass is green.",
        "That movie was absolutely terrible!",
        "Explain quantum entanglement in simple terms."
    ]
    
    for user_input in test_inputs:
        measurement = framework.process(user_input)
        
        # Generate actual response to user
        print(f"\n[Model Response]")
        response = framework.generate_response(user_input)
        print(response)
        
        print("\n" + "="*80 + "\n")
