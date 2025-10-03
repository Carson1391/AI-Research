from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import threading
import time

@dataclass
class MemoryConfig:
    """Configuration for memory system"""
    base_dir: str = "data/memory"
    memory_file: str = "data/memory/crypto_memory.json"
    max_short_term: int = 1000
    max_patterns: int = 5000
    min_confidence: float = 0.3

def setup_directory_structure():
    """Create necessary directory structure"""
    # Create base directories
    base = Path("data")
    dirs = [
        base,
        base / "memory",
        base / "storage",
        base / "models",
        base / "decision_data",
        base / "ai_dataset",
        base / "logs"
    ]
    
    # Create AI dataset subdirectories
    for layer in ['early', 'middle', 'later']:
        for dtype in ['abstract', 'concrete', 'graphs']:
            dirs.append(base / "ai_dataset" / layer / dtype)
    
    # Create all directories
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return {
        'base_dir': str(base),
        'memory_dir': str(base / "memory"),
        'storage_dir': str(base / "storage"),
        'models_dir': str(base / "models"),
        'decision_dir': str(base / "decision_data"),
        'dataset_dir': str(base / "ai_dataset"),
        'logs_dir': str(base / "logs")
    }

class IntegratedMemory:
    """Integration layer for all memory systems"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.base_dir = Path(config.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger('IntegratedMemory')
        self._setup_logging()
        
        # Initialize memory systems
        self.memory_system = None
        self.persistent_memory = None
        self.learning_memory = None
        
        # Initialize memory stats
        self.stats = {
            'patterns_learned': 0,
            'insights_generated': 0,
            'total_memories': 0,
            'avg_confidence': 0.0
        }
        
        # Memory synchronization
        self._sync_lock = threading.Lock()
        self._last_sync = time.time()
        
        # Start background tasks
        self._start_background_tasks()
        
        self.logger.info("Integrated memory system initialized")

    def _setup_logging(self):
        """Setup logging for integrated memory"""
        handler = logging.FileHandler(self.base_dir / 'integrated_memory.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def initialize_systems(self, memory_system, persistent_memory, learning_memory):
        """Initialize all memory subsystems"""
        self.memory_system = memory_system
        self.persistent_memory = persistent_memory
        self.learning_memory = learning_memory
        
        # Perform initial sync
        self._sync_memories()
        self.logger.info("All memory systems initialized and synced")

    def process_new_data(self, data_type: str, content: Any, 
                        metadata: Optional[Dict] = None) -> bool:
        """Process new data through all memory systems"""
        try:
            timestamp = datetime.now().isoformat()
            success = True
            
            # Store in base memory
            if self.memory_system:
                success &= bool(self.memory_system.remember(data_type, content, metadata))
            
            # Store in persistent memory if important
            if self.persistent_memory and self._should_persist(content, metadata):
                success &= self.persistent_memory.store(
                    data_type,
                    metadata.get('category', 'general'),
                    content,
                    metadata
                )
            
            # Process through learning system if applicable
            if self.learning_memory and self._should_learn(content, metadata):
                pattern_id = self.learning_memory.store_pattern(
                    data_type,
                    str(content),
                    metadata
                )
                if pattern_id:
                    self.stats['patterns_learned'] += 1
            
            # Update stats
            self.stats['total_memories'] += 1
            self._update_confidence_stats()
            
            # Trigger sync if needed
            if time.time() - self._last_sync > 300:  # 5 minutes
                self._sync_memories()
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error processing new data: {e}")
            return False

    def _should_persist(self, content: Any, metadata: Optional[Dict]) -> bool:
        """Determine if data should be persistently stored"""
        if not metadata:
            return False
            
        # Check importance threshold
        if metadata.get('importance', 0) > 0.7:
            return True
            
        # Check confidence threshold
        if metadata.get('confidence', 0) > 0.8:
            return True
            
        # Check verification count
        if metadata.get('verification_count', 0) > 5:
            return True
            
        return False

    def _should_learn(self, content: Any, metadata: Optional[Dict]) -> bool:
        """Determine if data should be processed for learning"""
        if not metadata:
            return False
            
        # Check if it's a learnable pattern
        if metadata.get('pattern_type') in ['market', 'technical', 'sentiment']:
            return True
            
        # Check if it has verification data
        if metadata.get('verification_data'):
            return True
            
        # Check if it's related to existing patterns
        if metadata.get('related_patterns'):
            return True
            
        return False

    def retrieve_memory(self, query: Dict) -> Dict[str, List]:
        """Retrieve memories from all systems"""
        try:
            results = {
                'base': [],
                'persistent': [],
                'learned': []
            }
            
            # Query base memory
            if self.memory_system:
                base_results = self.memory_system.recall(
                    query.get('type'),
                    query.get('query')
                )
                results['base'] = base_results.get('short_term', [])
            
            # Query persistent memory
            if self.persistent_memory:
                persistent_results = self.persistent_memory.retrieve(
                    query.get('type'),
                    query.get('category'),
                    query
                )
                results['persistent'] = persistent_results
            
            # Query learning memory
            if self.learning_memory:
                if query.get('type') == 'pattern':
                    results['learned'] = [
                        pattern for pattern in self.learning_memory.patterns.values()
                        if pattern.confidence >= query.get('min_confidence', 0.0)
                    ]
                elif query.get('type') == 'insight':
                    results['learned'] = [
                        insight for insight in self.learning_memory.insights.values()
                        if insight.confidence >= query.get('min_confidence', 0.0)
                    ]
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving memory: {e}")
            return {}

    def _sync_memories(self):
        """Synchronize all memory systems"""
        try:
            with self._sync_lock:
                if not all([self.memory_system, self.persistent_memory, self.learning_memory]):
                    return
                
                # Sync high-confidence patterns to persistent storage
                for pattern in self.learning_memory.patterns.values():
                    if pattern.confidence >= 0.8:
                        self.persistent_memory.store(
                            'pattern',
                            pattern.pattern_type,
                            {
                                'description': pattern.description,
                                'confidence': pattern.confidence,
                                'verification_count': pattern.verification_count,
                                'metadata': pattern.metadata
                            }
                        )
                
                # Sync verified insights
                for insight in self.learning_memory.insights.values():
                    if insight.verification_status == "verified":
                        self.persistent_memory.store(
                            'insight',
                            insight.category,
                            {
                                'content': insight.content,
                                'confidence': insight.confidence,
                                'impact_score': insight.impact_score,
                                'metadata': insight.metadata
                            }
                        )
                
                self._last_sync = time.time()
                self.logger.info("Memory systems synchronized")
                
        except Exception as e:
            self.logger.error(f"Error syncing memories: {e}")

    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        def cleanup_task():
            while True:
                try:
                    time.sleep(self.config.cleanup_interval)
                    self._cleanup_memories()
                except Exception as e:
                    self.logger.error(f"Error in cleanup task: {e}")
                    time.sleep(300)

        def backup_task():
            while True:
                try:
                    time.sleep(self.config.backup_interval)
                    self._backup_memories()
                except Exception as e:
                    self.logger.error(f"Error in backup task: {e}")
                    time.sleep(300)

        threading.Thread(target=cleanup_task, daemon=True).start()
        threading.Thread(target=backup_task, daemon=True).start()

    def _cleanup_memories(self):
        """Clean up all memory systems"""
        try:
            if self.memory_system:
                self.memory_system.cleanup()
            
            if self.persistent_memory:
                self.persistent_memory._cleanup_old_files()
            
            if self.learning_memory:
                self.learning_memory.cleanup()
            
            # Update stats after cleanup
            self._update_stats()
            self.logger.info("Completed memory cleanup")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

    def _backup_memories(self):
        """Backup important memories"""
        try:
            backup_dir = self.base_dir / 'backups' / datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup patterns
            if self.learning_memory:
                with open(backup_dir / 'patterns.json', 'w') as f:
                    json.dump(
                        {p_id: self._pattern_to_dict(p) 
                         for p_id, p in self.learning_memory.patterns.items()},
                        f, indent=2
                    )
            
            # Backup insights
            if self.learning_memory:
                with open(backup_dir / 'insights.json', 'w') as f:
                    json.dump(
                        {i_id: self._insight_to_dict(i)
                         for i_id, i in self.learning_memory.insights.items()},
                        f, indent=2
                    )
            
            # Backup stats
            with open(backup_dir / 'stats.json', 'w') as f:
                json.dump(self.stats, f, indent=2)
            
            self.logger.info(f"Created backup at {backup_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")

    def _update_stats(self):
        """Update memory statistics"""
        try:
            self.stats.update({
                'patterns_learned': len(self.learning_memory.patterns) if self.learning_memory else 0,
                'insights_generated': len(self.learning_memory.insights) if self.learning_memory else 0,
                'total_memories': sum([
                    len(self.learning_memory.patterns) if self.learning_memory else 0,
                    len(self.learning_memory.insights) if self.learning_memory else 0
                ]),
                'last_updated': datetime.now().isoformat()
            })
            
            self._update_confidence_stats()
            
        except Exception as e:
            self.logger.error(f"Error updating stats: {e}")

    def _update_confidence_stats(self):
        """Update confidence-related statistics"""
        try:
            if self.learning_memory:
                pattern_confidences = [p.confidence for p in self.learning_memory.patterns.values()]
                insight_confidences = [i.confidence for i in self.learning_memory.insights.values()]
                
                if pattern_confidences or insight_confidences:
                    all_confidences = pattern_confidences + insight_confidences
                    self.stats['avg_confidence'] = sum(all_confidences) / len(all_confidences)
                
        except Exception as e:
            self.logger.error(f"Error updating confidence stats: {e}")

    def _pattern_to_dict(self, pattern) -> Dict:
        """Convert pattern to dictionary for storage"""
        return {
            'pattern_id': pattern.pattern_id,
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

    def _insight_to_dict(self, insight) -> Dict:
        """Convert insight to dictionary for storage"""
        return {
            'insight_id': insight.insight_id,
            'category': insight.category,
            'content': insight.content,
            'confidence': insight.confidence,
            'timestamp': insight.timestamp.isoformat(),
            'source_patterns': insight.source_patterns,
            'verification_status': insight.verification_status,
            'impact_score': insight.impact_score,
            'metadata': insight.metadata
        }

    def get_stats(self) -> Dict:
        """Get current memory statistics"""
        self._update_stats()
        return self.stats