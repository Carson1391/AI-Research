import torch
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from pathlib import Path
import logging
from datetime import datetime
from enhanced_learning import EnhancedLearning
from memory_integration import IntegratedMemory
from continuous_learning import CryptoNet

@dataclass
class LearningPersonality:
    """Personality traits that influence learning behavior"""
    curiosity: float = 0.8
    caution: float = 0.6
    adaptability: float = 0.7
    memory_weight: float = 0.8

class IntegratedLearning:
   
    def __init__(self, name: str = "CryptoNet"):
        self.name = name
        self.logger = logging.getLogger(f"IntegratedLearning_{name}")
        handler = logging.FileHandler('logs/learning.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Core components
        self.crypto_net = None
        self.enhanced_learning = None
        self.integrated_memory = None
        
        # Tracking
        self.insights = []
        self.patterns = {}
        
    def connect_systems(self, enhanced_learning: EnhancedLearning, 
                       crypto_net: CryptoNet,
                       integrated_memory: IntegratedMemory):
        self.enhanced_learning = enhanced_learning
        self.crypto_net = crypto_net
        self.integrated_memory = integrated_memory
        self.logger.info("Connected to all learning systems")

    async def process_website_data(self, url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process new data through all learning systems"""
        try:
            timestamp = datetime.now().isoformat()
            processed_data = {
                'url': url,
                'timestamp': timestamp,
                'patterns': [],
                'insights': [],
                'confidence': 0.0
            }

            # Process through enhanced learning
            if self.enhanced_learning:
                enhanced_results = self.enhanced_learning.process_content(
                    data, category=data.get('type', 'general')
                )
                processed_data.update(enhanced_results)

            # Process through CryptoNet if appropriate
            if self.crypto_net and 'features' in data:
                features = torch.tensor(data['features'], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    predictions = self.crypto_net(features)
                    processed_data['neural_predictions'] = predictions.squeeze(0).tolist()

            # Store in memory if significant
            if processed_data['confidence'] > 0.6 and self.integrated_memory:
                await self.integrated_memory.process_new_data(
                    data_type='website_analysis',
                    content=processed_data,
                    metadata={
                        'importance': processed_data['confidence'],
                        'category': data.get('type', 'general')
                    }
                )

            return processed_data

        except Exception as e:
            self.logger.error(f"Error processing website data: {e}")
            return {}

    async def process_training_data(self, raw_data: Dict) -> Optional[Dict]:
        """Process data specifically for training purposes"""
        try:
            # First pass through enhanced learning
            if self.enhanced_learning:
                processed = self.enhanced_learning.process_content(
                    raw_data, 
                    category='training_data'
                )
                
                if processed and processed.get('confidence', 0) > 0.3:  # Lower threshold for training
                    # Structure as a training pair
                    training_pair = {
                        'X1': processed.get('features', []),
                        'X2': processed.get('next_features', []),
                        'Y': processed.get('labels', []),
                        'metadata': {
                            'timestamp': datetime.now().isoformat(),
                            'confidence': processed.get('confidence', 0),
                            'source': raw_data.get('source', 'unknown')
                        }
                    }
                    
                    # Store through memory system
                    if self.integrated_memory:
                        await self.integrated_memory.process_new_data(
                            data_type='training_pair',
                            content=training_pair,
                            metadata={'purpose': 'model_training'}
                        )
                    
                    return training_pair
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing training data: {e}")
            return None

    def get_learning_status(self) -> Dict:
        """Get current learning status"""
        return {
            'total_insights': len(self.insights),
            'total_patterns': len(self.patterns),
            'personality': {
                'curiosity': self.personality.curiosity,
                'caution': self.personality.caution,
                'adaptability': self.personality.adaptability,
                'memory_weight': self.personality.memory_weight
            }
        }