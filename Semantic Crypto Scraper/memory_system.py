from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from pathlib import Path
import logging
import sqlite3
import os
import aiofiles

@dataclass
class MemoryTypes:
    """Types of information to remember"""
    CONVERSATION = "conversation"
    DATASET = "dataset"
    COIN = "coin"
    GROWTH_PATTERN = "growth_pattern"
    SOURCE = "source"
    INSIGHT = "insight"

class MemorySystem:
    def __init__(self, memory_dir: str = "data/memory"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.memory_files = {
            'crypto': self.memory_dir / 'crypto_memory.json',
            'patterns': self.memory_dir / 'patterns.json',
            'insights': self.memory_dir / 'insights.json',
            'conversations': self.memory_dir / 'conversations.db'
        }

        self.logger = logging.getLogger('MemorySystem')
        handler = logging.FileHandler(self.memory_dir / 'memory.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.memory = self._initialize_memory()
        self._setup_database()

        self.logger.info("Memory system initialized")

    def _initialize_memory(self) -> Dict:
        try:
            memory = {
                'short_term': {
                    'recent_insights': [],
                    'active_patterns': {},
                    'current_focus': None
                },
                'long_term': {
                    'patterns': self._load_patterns(),
                    'insights': self._load_insights(),
                    'crypto_data': self._load_crypto_data()
                },
                'stats': {
                    'total_patterns': 0,
                    'verified_patterns': 0,
                    'total_insights': 0,
                    'learning_cycles': 0
                }
            }
            return memory
        except Exception as e:
            self.logger.error(f"Error initializing memory: {e}")
            return {
                'short_term': {'recent_insights': [], 'active_patterns': {}, 'current_focus': None},
                'long_term': {'patterns': {}, 'insights': {}, 'crypto_data': {}},
                'stats': {'total_patterns': 0, 'verified_patterns': 0, 'total_insights': 0}
            }

    def _setup_database(self):
        try:
            with sqlite3.connect(self.memory_files['conversations']) as conn:
                # Enable foreign keys
                conn.execute("PRAGMA foreign_keys = ON")

                # Create conversations table with indices
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        context TEXT,
                        processed BOOLEAN DEFAULT FALSE
                    )
                """)

                # Add indices for common queries
                conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_role ON conversations(role)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_processed ON conversations(processed)")

                # Create metrics table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        metric_type TEXT NOT NULL,
                        value REAL NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        context TEXT
                    )
                """)

                # Add index for metrics
                conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_type_timestamp ON metrics(metric_type, timestamp)")

                conn.commit()
                self.logger.info("Database setup completed successfully")
        except sqlite3.Error as e:
            self.logger.error(f"SQLite error during database setup: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during database setup: {e}")
            raise

    def remember(self, MemoryTypes: str, content: Any, importance: float = 0.5,
                context: Optional[Dict] = None):
        try:
            timestamp = datetime.now().isoformat()

            if MemoryTypes == MemoryTypes.GROWTH_PATTERN:
                self._store_pattern(content, importance, timestamp)
            elif MemoryTypes == MemoryTypes.INSIGHT:
                self._store_insight(content, importance, timestamp)
            elif MemoryTypes == MemoryTypes.COIN:
                self._store_crypto_data(content, timestamp)
            elif MemoryTypes == MemoryTypes.CONVERSATION:
                self._store_conversation(content, context, timestamp)

            # Only store in short-term memory if importance is high enough
            if importance is not None and importance > 0.7:
                self.memory['short_term']['recent_insights'].append({
                    'type': MemoryTypes,
                    'content': content,
                    'timestamp': timestamp,
                    'context': context
                })

                self.memory['short_term']['recent_insights'] = \
                    self.memory['short_term']['recent_insights'][-100:]

            self.logger.info(f"Stored {MemoryTypes} memory")

        except Exception as e:
            self.logger.error(f"Error storing memory: {e}")

    def _store_pattern(self, pattern: Dict, importance: float, timestamp: str):
        try:
            pattern_id = pattern.get('id') or f"pattern_{timestamp}"
            patterns = self._load_patterns()

            pattern_data = {
                **pattern,
                'first_seen': timestamp,
                'last_seen': timestamp,
                'verification_count': 1,
                'importance': importance,
                'confidence': pattern.get('confidence', 0.5),
                'success_count': 0,
                'fail_count': 0
            }

            patterns[pattern_id] = pattern_data

            with open(self.memory_files['patterns'], 'w') as f:
                json.dump(patterns, f, indent=2)

            self.memory['stats']['total_patterns'] += 1

        except Exception as e:
            self.logger.error(f"Error storing pattern: {e}")

    def recall(self, memory_type: Optional[str] = None, query: Optional[Dict] = None) -> Dict:
        try:
            results = {
                'short_term': [],
                'long_term': [],
                'related': []
            }

            # Search short-term memory
            if memory_type:
                results['short_term'] = [
                    item for item in self.memory['short_term']['recent_insights']
                    if item['type'] == memory_type
                ]

            # Search long-term memory
            if memory_type == MemoryTypes.GROWTH_PATTERN:
                results['long_term'] = self._search_patterns(query)
            elif memory_type == MemoryTypes.INSIGHT:
                results['long_term'] = self._search_insights(query)
            elif memory_type == MemoryTypes.COIN:
                results['long_term'] = self._search_crypto_data(query)

            return results

        except Exception as e:
            self.logger.error(f"Error recalling memory: {e}")
            return {'error': str(e)}

    def _load_patterns(self) -> Dict:
        try:
            if self.memory_files['patterns'].exists():
                with open(self.memory_files['patterns'], 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Error loading patterns: {e}")
            return {}

    def _store_insight(self, insight: Dict, importance: float, timestamp: str):
        try:
            insight_id = insight.get('id') or f"insight_{timestamp}"
            insights = self._load_insights()

            insight_data = {
                **insight,
                'timestamp': timestamp,
                'importance': importance,
                'verification_count': 0,
                'confidence': insight.get('confidence', 0.5)
            }

            insights[insight_id] = insight_data

            with open(self.memory_files['insights'], 'w') as f:
                json.dump(insights, f, indent=2)

            self.memory['stats']['total_insights'] += 1

        except Exception as e:
            self.logger.error(f"Error storing insight: {e}")

    def _store_crypto_data(self, data: Dict, timestamp: str):
        try:
            crypto_data = self._load_crypto_data()
            symbol = data.get('symbol', 'UNKNOWN')

            if symbol not in crypto_data:
                crypto_data[symbol] = {
                    'first_seen': timestamp,
                    'price_history': [],
                    'volume_history': [],
                    'patterns': [],
                    'insights': []
                }

            crypto_data[symbol]['last_updated'] = timestamp

            if 'price' in data:
                crypto_data[symbol]['price_history'].append({
                    'price': data['price'],
                    'timestamp': timestamp
                })
            if 'volume' in data:
                crypto_data[symbol]['volume_history'].append({
                    'volume': data['volume'],
                    'timestamp': timestamp
                })

            crypto_data[symbol]['price_history'] = \
                crypto_data[symbol]['price_history'][-1000:]
            crypto_data[symbol]['volume_history'] = \
                crypto_data[symbol]['volume_history'][-1000:]

            with open(self.memory_files['crypto'], 'w') as f:
                json.dump(crypto_data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Error storing crypto data: {e}")

    def _load_insights(self) -> Dict:
        try:
            if self.memory_files['insights'].exists():
                with open(self.memory_files['insights'], 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Error loading insights: {e}")
            return {}

    def _load_crypto_data(self) -> Dict:
        try:
            if self.memory_files['crypto'].exists():
                with open(self.memory_files['crypto'], 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            self.logger.error(f"Error loading crypto data: {e}")
            return {}

    def _store_conversation(self, content: Any, context: Optional[Dict], timestamp: str):
        """Store conversation data in the database"""
        try:
            with sqlite3.connect(self.memory_files['conversations']) as conn:
                c = conn.cursor()
                c.execute("""
                    INSERT INTO conversations (timestamp, role, content, context)
                    VALUES (?, ?, ?, ?)
                """, (
                    timestamp,
                    content.get('role', 'unknown'),
                    content.get('content', ''),
                    json.dumps(context) if context else None
                ))
            self.logger.info("Stored conversation in memory")
        except Exception as e:
            self.logger.error(f"Error storing conversation in memory: {e}")

    async def save_state(self):
        """Save current memory state to disk"""
        try:
            self.logger.info("Saving memory state...")

            # Prepare state data
            state = {
                'short_term': self.memory['short_term'],
                'long_term': self.memory['long_term'],
                'patterns': self._load_patterns(),
                'insights': self._load_insights(),
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }

            # Save to disk
            save_path = os.path.join(self.memory_dir, 'memory_state.json')
            async with aiofiles.open(save_path, 'w') as f:
                await f.write(json.dumps(state, indent=2))

            self.logger.info(f"Memory state saved to {save_path}")

        except Exception as e:
            self.logger.error(f"Error saving memory state: {e}")

    async def load_state(self):
        """Load memory state from disk"""
        try:
            self.logger.info("Loading memory state...")

            load_path = os.path.join(self.memory_dir, 'memory_state.json')
            if not os.path.exists(load_path):
                self.logger.info("No previous state found")
                return

            async with aiofiles.open(load_path, 'r') as f:
                state = json.loads(await f.read())

            # Validate version
            if state.get('metadata', {}).get('version') != '1.0':
                self.logger.warning("State version mismatch, skipping load")
                return

            # Restore state
            self.memory['short_term'] = state['short_term']
            self.memory['long_term'] = state['long_term']

            self.logger.info("Memory state loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading memory state: {e}")

    async def initialize(self):
        """Initialize memory system"""
        try:
            self.logger.info("Initializing memory system...")

            # Create memory directory if needed
            os.makedirs(self.memory_dir, exist_ok=True)

            # Load previous state if exists
            await self.load_state()

            # Initialize memory structures
            if not hasattr(self, 'memory'):
                self.memory = self._initialize_memory()

            self.logger.info("Memory system initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing memory system: {e}")
            raise

    def cleanup(self):
        """Clean up old or unnecessary data"""
        try:
            # Clean up old conversations
            with sqlite3.connect(self.memory_files['conversations']) as conn:
                # Delete processed conversations older than 30 days
                conn.execute("""
                    DELETE FROM conversations 
                    WHERE processed = TRUE 
                    AND timestamp < datetime('now', '-30 days')
                """)

                # Keep only last 1000 unprocessed conversations
                conn.execute("""
                    DELETE FROM conversations 
                    WHERE processed = FALSE 
                    AND id NOT IN (
                        SELECT id FROM conversations 
                        WHERE processed = FALSE 
                        ORDER BY timestamp DESC 
                        LIMIT 1000
                    )
                """)

                # Clean up old metrics
                conn.execute("""
                    DELETE FROM metrics 
                    WHERE timestamp < datetime('now', '-90 days')
                """)

                conn.commit()

            # Clean up short-term memory
            self.memory['short_term']['recent_insights'] = \
                self.memory['short_term']['recent_insights'][-100:]

            # Reset current focus if it's too old
            if self.memory['short_term']['current_focus']:
                focus_time = datetime.fromisoformat(
                    self.memory['short_term']['current_focus'].get('timestamp', '2000-01-01')
                )
                if (datetime.now() - focus_time).days > 1:
                    self.memory['short_term']['current_focus'] = None

            self.logger.info("Memory cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            raise