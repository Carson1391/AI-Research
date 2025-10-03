from pathlib import Path
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading
import time

class PersistentMemory:
    def __init__(self, base_dir: str = "data/persistent_memory"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_files = {
            'patterns': self.base_dir / 'patterns',
            'insights': self.base_dir / 'insights',
            'market_data': self.base_dir / 'market_data',
            'learned_behaviors': self.base_dir / 'behaviors'
        }
        
        for path in self.memory_files.values():
            path.mkdir(exist_ok=True)
        
        self.db_path = self.base_dir / 'memory.db'
        
        self.logger = logging.getLogger('PersistentMemory')
        self._setup_logging()
        
        self._setup_database()
        self._init_memory_indexes()
        
        self.save_thread = threading.Thread(target=self._background_save_loop, daemon=True)
        self.save_thread.start()
        
        self.logger.info("Persistent memory system initialized")

    def _setup_logging(self):
        handler = logging.FileHandler(self.base_dir / 'persistent_memory.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def _setup_database(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_index (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        memory_type TEXT NOT NULL,
                        category TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_accessed DATETIME,
                        access_count INTEGER DEFAULT 0,
                        confidence REAL DEFAULT 0,
                        metadata TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pattern_verification (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_id TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        success BOOLEAN,
                        confidence_change REAL,
                        context TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memory_type 
                    ON memory_index(memory_type, category)
                """)
        except Exception as e:
            self.logger.error(f"Database setup error: {e}")

    def _init_memory_indexes(self):
        self.memory_indexes = {
            'patterns': {},
            'insights': {},
            'market_data': {},
            'behaviors': {}
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT * FROM memory_index')
                for row in cursor:
                    memory_type = row[1]
                    if memory_type in self.memory_indexes:
                        self.memory_indexes[memory_type][row[0]] = {
                            'category': row[2],
                            'file_path': row[3],
                            'confidence': row[7],
                            'metadata': json.loads(row[8]) if row[8] else {}
                        }
        except Exception as e:
            self.logger.error(f"Error loading memory indexes: {e}")

    def store(self, memory_type: str, category: str, content: Any, 
             metadata: Optional[Dict] = None) -> bool:
        try:
            timestamp = datetime.now().isoformat()
            file_path = self._get_storage_path(memory_type, category, timestamp)
            
            with open(file_path, 'w') as f:
                json.dump({
                    'content': content,
                    'metadata': metadata or {},
                    'created_at': timestamp,
                    'last_modified': timestamp,
                    'access_count': 0,
                    'confidence': metadata.get('confidence', 0.5) if metadata else 0.5
                }, f, indent=2)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO memory_index 
                    (memory_type, category, file_path, metadata, confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    memory_type,
                    category,
                    str(file_path),
                    json.dumps(metadata) if metadata else None,
                    metadata.get('confidence', 0.5) if metadata else 0.5
                ))
            
            memory_id = file_path.stem
            self.memory_indexes[memory_type][memory_id] = {
                'category': category,
                'file_path': str(file_path),
                'confidence': metadata.get('confidence', 0.5) if metadata else 0.5,
                'metadata': metadata or {}
            }
            
            self.logger.info(f"Stored {memory_type}/{category} memory")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing memory: {e}")
            return False

    def retrieve(self, memory_type: str, category: Optional[str] = None,
                query: Optional[Dict] = None) -> List[Dict]:
        try:
            results = []
            
            sql = "SELECT * FROM memory_index WHERE memory_type = ?"
            params = [memory_type]
            
            if category:
                sql += " AND category = ?"
                params.append(category)
            
            if query:
                for key, value in query.items():
                    if key in ['confidence', 'access_count']:
                        sql += f" AND {key} >= ?"
                        params.append(value)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(sql, params)
                for row in cursor:
                    file_path = Path(row[3])
                    if file_path.exists():
                        with open(file_path, 'r') as f:
                            memory_data = json.load(f)
                            results.append({
                                'id': row[0],
                                'type': row[1],
                                'category': row[2],
                                'content': memory_data['content'],
                                'metadata': memory_data['metadata'],
                                'confidence': row[7],
                                'created_at': row[4],
                                'last_accessed': row[5]
                            })
                        
                        conn.execute("""
                            UPDATE memory_index 
                            SET access_count = access_count + 1,
                                last_accessed = CURRENT_TIMESTAMP
                            WHERE id = ?
                        """, (row[0],))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving memory: {e}")
            return []

    def _get_storage_path(self, memory_type: str, category: str, 
                         timestamp: str) -> Path:
        base_path = self.memory_files[memory_type]
        return base_path / category / f"memory_{timestamp}.json"

    def _background_save_loop(self):
        while True:
            try:
                time.sleep(3600)  # Sleep for an hour                
                self._verify_files()                
            except Exception as e:
                self.logger.error(f"Error in background save: {e}")
                time.sleep(300)

    def _verify_files(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT file_path FROM memory_index')
                for (file_path,) in cursor:
                    path = Path(file_path)
                    if not path.exists():
                        conn.execute('DELETE FROM memory_index WHERE file_path = ?',
                                   (file_path,))
                        self.logger.warning(f"Removed missing file from index: {file_path}")
        except Exception as e:
            self.logger.error(f"Error verifying files: {e}")

    def get_stats(self) -> Dict:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_memories,
                        AVG(confidence) as avg_confidence,
                        SUM(access_count) as total_accesses
                    FROM memory_index
                """)
                row = cursor.fetchone()
                
                return {
                    'total_memories': row[0],
                    'average_confidence': row[1],
                    'total_accesses': row[2],
                    'timestamp': datetime.now().isoformat()
                }
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }