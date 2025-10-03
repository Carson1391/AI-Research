from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass, field

@dataclass
class LearningPattern:
    """Pattern detected during learning"""
    pattern_id: str
    pattern_type: str
    description: str
    first_seen: datetime
    last_seen: datetime
    confidence: float = 0.5
    verification_count: int = 0
    success_count: int = 0
    fail_count: int = 0
    metadata: Dict = field(default_factory=dict)
    related_patterns: List[str] = field(default_factory=list)

@dataclass
class MarketInsight:
    """Market insight gained from learning"""
    insight_id: str
    category: str
    content: str
    confidence: float
    timestamp: datetime
    source_patterns: List[str]
    verification_status: str = "unverified"
    impact_score: float = 0.0
    metadata: Dict = field(default_factory=dict)

class LearningMemory:
    def __init__(self, memory_system, persistent_memory, learning_dir: str = "data/learning"):
        self.memory_system = memory_system  # Reference to main memory system
        self.persistent = persistent_memory  # Reference to persistent memory
        self.learning_dir = Path(learning_dir)
        self.learning_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pattern storage
        self.pattern_dir = self.learning_dir / "patterns"
        self.pattern_dir.mkdir(exist_ok=True)
        
        # Initialize insight storage
        self.insight_dir = self.learning_dir / "insights"
        self.insight_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger('LearningMemory')
        handler = logging.FileHandler(self.learning_dir / 'learning.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Load existing patterns and insights
        self.patterns: Dict[str, LearningPattern] = self._load_patterns()
        self.insights: Dict[str, MarketInsight] = self._load_insights()
        
        # Initialize learning metrics
        self.metrics = {
            'total_patterns': len(self.patterns),
            'verified_patterns': len([p for p in self.patterns.values() if p.verification_count > 0]),
            'total_insights': len(self.insights),
            'avg_confidence': self._calculate_avg_confidence()
        }
        
        self.logger.info("Learning memory system initialized")

    def _load_patterns(self) -> Dict[str, LearningPattern]:
        """Load existing patterns from storage"""
        patterns = {}
        try:
            if (self.pattern_dir / 'patterns.json').exists():
                with open(self.pattern_dir / 'patterns.json', 'r') as f:
                    pattern_data = json.load(f)
                    for p_id, p_data in pattern_data.items():
                        patterns[p_id] = LearningPattern(
                            pattern_id=p_id,
                            pattern_type=p_data['type'],
                            description=p_data['description'],
                            first_seen=datetime.fromisoformat(p_data['first_seen']),
                            last_seen=datetime.fromisoformat(p_data['last_seen']),
                            confidence=p_data['confidence'],
                            verification_count=p_data['verification_count'],
                            success_count=p_data['success_count'],
                            fail_count=p_data['fail_count'],
                            metadata=p_data.get('metadata', {}),
                            related_patterns=p_data.get('related_patterns', [])
                        )
            return patterns
        except Exception as e:
            self.logger.error(f"Error loading patterns: {e}")
            return {}

    def _load_insights(self) -> Dict[str, MarketInsight]:
        """Load existing insights from storage"""
        insights = {}
        try:
            if (self.insight_dir / 'insights.json').exists():
                with open(self.insight_dir / 'insights.json', 'r') as f:
                    insight_data = json.load(f)
                    for i_id, i_data in insight_data.items():
                        insights[i_id] = MarketInsight(
                            insight_id=i_id,
                            category=i_data['category'],
                            content=i_data['content'],
                            confidence=i_data['confidence'],
                            timestamp=datetime.fromisoformat(i_data['timestamp']),
                            source_patterns=i_data['source_patterns'],
                            verification_status=i_data.get('verification_status', "unverified"),
                            impact_score=i_data.get('impact_score', 0.0),
                            metadata=i_data.get('metadata', {})
                        )
            return insights
        except Exception as e:
            self.logger.error(f"Error loading insights: {e}")
            return {}

    def store_pattern(self, pattern_type: str, description: str, 
                     metadata: Optional[Dict] = None) -> str:
        """Store a new pattern or update existing one"""
        try:
            now = datetime.now()
            pattern_id = f"{pattern_type}_{now.strftime('%Y%m%d_%H%M%S')}"
            
            pattern = LearningPattern(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                description=description,
                first_seen=now,
                last_seen=now,
                metadata=metadata or {},
                confidence=metadata.get('initial_confidence', 0.5) if metadata else 0.5
            )
            
            # Store in memory
            self.patterns[pattern_id] = pattern
            
            # Store in persistent memory
            self.persistent.store(
                'pattern',
                pattern_type,
                {
                    'description': description,
                    'metadata': metadata,
                    'timestamp': now.isoformat()
                }
            )
            
            # Save to file
            self._save_patterns()
            
            # Update metrics
            self.metrics['total_patterns'] = len(self.patterns)
            
            self.logger.info(f"Stored new pattern: {pattern_id}")
            return pattern_id
            
        except Exception as e:
            self.logger.error(f"Error storing pattern: {e}")
            return ""

    def verify_pattern(self, pattern_id: str, success: bool, 
                      confidence_change: float = 0.1):
        """Verify a pattern's accuracy"""
        try:
            if pattern_id not in self.patterns:
                self.logger.warning(f"Pattern not found: {pattern_id}")
                return
                
            pattern = self.patterns[pattern_id]
            pattern.verification_count += 1
            pattern.last_seen = datetime.now()
            
            if success:
                pattern.success_count += 1
                pattern.confidence = min(1.0, pattern.confidence + confidence_change)
            else:
                pattern.fail_count += 1
                pattern.confidence = max(0.0, pattern.confidence - confidence_change)
            
            # Update persistence
            self.persistent.store(
                'pattern_verification',
                pattern.pattern_type,
                {
                    'pattern_id': pattern_id,
                    'success': success,
                    'confidence': pattern.confidence,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # Save updated patterns
            self._save_patterns()
            
            # Update metrics
            self.metrics['verified_patterns'] = len(
                [p for p in self.patterns.values() if p.verification_count > 0]
            )
            self.metrics['avg_confidence'] = self._calculate_avg_confidence()
            
            self.logger.info(
                f"Verified pattern {pattern_id}: success={success}, "
                f"new confidence={pattern.confidence:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error verifying pattern: {e}")

    def _calculate_avg_confidence(self) -> float:
        """Calculate average confidence across all patterns"""
        try:
            if not self.patterns:
                return 0.0
            confidences = [p.confidence for p in self.patterns.values()]
            return float(np.mean(confidences))
        except Exception as e:
            self.logger.error(f"Error calculating average confidence: {e}")
            return 0.0

    def store_insight(self, category: str, content: str,
                     source_patterns: List[str], metadata: Optional[Dict] = None) -> str:
        """Store a new market insight"""
        try:
            now = datetime.now()
            insight_id = f"insight_{now.strftime('%Y%m%d_%H%M%S')}"
            
            # Calculate initial confidence based on source patterns
            source_confidences = [
                self.patterns[p_id].confidence 
                for p_id in source_patterns 
                if p_id in self.patterns
            ]
            initial_confidence = float(np.mean(source_confidences)) if source_confidences else 0.5
            
            insight = MarketInsight(
                insight_id=insight_id,
                category=category,
                content=content,
                confidence=initial_confidence,
                timestamp=now,
                source_patterns=source_patterns,
                metadata=metadata or {}
            )
            
            # Store in memory
            self.insights[insight_id] = insight
            
            # Store in persistent memory
            self.persistent.store(
                'insight',
                category,
                {
                    'content': content,
                    'confidence': initial_confidence,
                    'source_patterns': source_patterns,
                    'metadata': metadata,
                    'timestamp': now.isoformat()
                }
            )
            
            # Save to file
            self._save_insights()
            
            # Update metrics
            self.metrics['total_insights'] = len(self.insights)
            
            self.logger.info(f"Stored new insight: {insight_id}")
            return insight_id
            
        except Exception as e:
            self.logger.error(f"Error storing insight: {e}")
            return ""
            
    def _save_patterns(self):
        """Save patterns to file"""
        try:
            pattern_data = {}
            for p_id, pattern in self.patterns.items():
                pattern_data[p_id] = {
                    'type': pattern.pattern_type,
                    'description': pattern.description,
                    'first_seen': pattern.first_seen.isoformat(),
                    'last_seen': pattern.last_seen.isoformat(),
                    'confidence': pattern.confidence,
                    'verification_count': pattern.verification_count,
                    'success_count': pattern.success_count,
                    'fail_count': pattern.fail_count,
                    'metadata': pattern.metadata,
                    'related_patterns': pattern.related_patterns
                }
            
            with open(self.pattern_dir / 'patterns.json', 'w') as f:
                json.dump(pattern_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving patterns: {e}")

    def _save_insights(self):
        """Save insights to file"""
        try:
            insight_data = {}
            for i_id, insight in self.insights.items():
                insight_data[i_id] = {
                    'category': insight.category,
                    'content': insight.content,
                    'confidence': insight.confidence,
                    'timestamp': insight.timestamp.isoformat(),
                    'source_patterns': insight.source_patterns,
                    'verification_status': insight.verification_status,
                    'impact_score': insight.impact_score,
                    'metadata': insight.metadata
                }
            
            with open(self.insight_dir / 'insights.json', 'w') as f:
                json.dump(insight_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving insights: {e}")

    def get_metrics(self) -> Dict:
        """Get current learning metrics"""
        return {
            **self.metrics,
            'timestamp': datetime.now().isoformat()
        }

    def cleanup(self):
        """Clean up old or low value patterns and insights"""
        try:
            now = datetime.now()
            
            # Remove low confidence patterns that haven't been verified recently
            self.patterns = {
                p_id: p for p_id, p in self.patterns.items()
                if p.confidence >= 0.3 or 
                   (now - p.last_seen).days < 30 or
                   p.verification_count > 5
            }
            
            # Remove old insights with low impact
            self.insights = {
                i_id: i for i_id, i in self.insights.items()
                if i.impact_score >= 0.3 or
                   (now - i.timestamp).days < 30
            }
            
            # Save cleaned up data
            self._save_patterns()
            self._save_insights()
            
            # Update metrics
            self.metrics['total_patterns'] = len(self.patterns)
            self.metrics['total_insights'] = len(self.insights)
            self.metrics['avg_confidence'] = self._calculate_avg_confidence()
            
            self.logger.info("Completed memory cleanup")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")