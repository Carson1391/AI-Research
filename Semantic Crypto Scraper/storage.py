import aiosqlite
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path

@dataclass
class StorageConfig:
    """Configuration for data storage system"""
    data_dir: str = "data"
    decision_data_dir: str = "data/decision_data"
    ai_dataset_dir: str = "data/ai_dataset"
    db_path: str = "data/storage.db"
    # Categories for dataset organization
    decision_categories: Dict[str, List[str]] = field(default_factory=lambda: {
        'market_data': ['price_data', 'volume_data', 'market_cap'],
        'technical_data': ['moving_averages', 'oscillators', 'trend_lines'],
        'news_data': ['announcements', 'regulations', 'updates'],
        'social_data': ['reddit_sentiment', 'twitter_sentiment'],
        'vision_data': ['charts', 'patterns', 'screenshots']
    })

    ai_layers: List[str] = field(default_factory=lambda: ['early', 'middle', 'later'])
    data_types: List[str] = field(default_factory=lambda: ['abstract', 'concrete', 'graphs'])

class DataStorage:
    def __init__(self, config: StorageConfig):
        self.config = config
        self.logger = logging.getLogger('DataStorage')
        # Initialize paths
        self.data_dir = Path(config.data_dir)
        self.decision_dir = Path(config.decision_data_dir)
        self.dataset_dir = Path(config.ai_dataset_dir)
        self.db_path = Path(config.db_path)
        
        # Create directories
        for dir_path in [self.data_dir, self.decision_dir, self.dataset_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Create AI dataset structure
        for layer in config.ai_layers:
            for dtype in config.data_types:
                (self.dataset_dir / layer / dtype).mkdir(parents=True, exist_ok=True)
        
        self._setup_logging()
        # Initialize DB synchronously
        self._init_db_sync()

    async def store_data(self, content: Any, metadata: Dict[str, Any]) -> bool:
        """Store data with proper categorization"""
        try:
            timestamp = datetime.now().isoformat()
            data_type = metadata.get('type', 'general')
            
            # Determine storage location
            if metadata.get('purpose') == 'model_training':
                return await self._store_training_data(content, metadata)
            elif metadata.get('purpose') == 'decision_making':
                return await self._store_decision_data(content, metadata)
            else:
                return await self._store_general_data(content, metadata)
                
        except Exception as e:
            self.logger.error(f"Error storing data: {e}")
            return False

    async def _store_training_data(self, content: Any, metadata: Dict) -> bool:
        """Store data for model training"""
        try:
            layer = metadata.get('layer', 'early')
            data_type = metadata.get('data_type', 'concrete')
            
            # Save to appropriate dataset directory
            file_path = self.dataset_dir / layer / data_type / f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO training_data 
                    (file_path, metadata, timestamp) VALUES (?, ?, ?)
                """, (str(file_path), json.dumps(metadata), datetime.now().isoformat()))
                await db.commit()
            
            # Save actual data
            file_path.write_text(json.dumps({
                'content': content,
                'metadata': metadata,
                'timestamp': datetime.now().isoformat()
            }, indent=2))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing training data: {e}")
            return False

    def _init_db_sync(self):
        """Initialize database tables synchronously"""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as db:
                db.execute("""
                    CREATE TABLE IF NOT EXISTS training_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT NOT NULL,
                        metadata TEXT NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                """)
                db.execute("""
                    CREATE TABLE IF NOT EXISTS decision_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        category TEXT NOT NULL,
                        subcategory TEXT NOT NULL,
                        data TEXT NOT NULL,
                        metadata TEXT NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                """)
                db.execute("""
                    CREATE TABLE IF NOT EXISTS general_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        data_type TEXT NOT NULL,
                        data TEXT NOT NULL,
                        metadata TEXT NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                """)
                db.commit()
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")

    def _setup_logging(self):
        handler = logging.FileHandler('logs/storage.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    async def get_training_batch(self, batch_size: int = 32) -> List[Dict]:
        """Get a batch of training data"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute("""
                    SELECT file_path FROM training_data 
                    ORDER BY RANDOM() LIMIT ?
                """, (batch_size,)) as cursor:
                    files = await cursor.fetchall()
                    
            batch = []
            for (file_path,) in files:
                path = Path(file_path)
                if path.exists():
                    data = json.loads(path.read_text())
                    batch.append(data)
                    
            return batch
            
        except Exception as e:
            self.logger.error(f"Error getting training batch: {e}")
            return []