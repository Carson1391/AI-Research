import torch
from PIL import Image
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

class QwenVLManager:
    def __init__(self, model_dir: str = None):
        """Initialize the Qwen model and core components"""
        self.model_path = str(Path(__file__).parent.parent)
        self.logger = logging.getLogger("QwenVLManager")
        
        # Initialize model components
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model using provided dir or default path
        self.load_model(model_dir or self.model_path)

    def load_model(self, model_dir: str):
        """Load Qwen2-VL model from directory"""
        self.logger.info(f"Loading Qwen VL from {model_dir}")
        
        # Initialize tokenizer and processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_dir)
        
        # Initialize model
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        ).eval()
        
        self.device = next(self.model.parameters()).device
        self.logger.info("Model loaded successfully")

    async def process_image(self, image_path: str, prompt: str) -> str:
        """Process image with text prompt"""
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Prepare for inference
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)

            # Generate response
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            response = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return f"Error: {str(e)}"