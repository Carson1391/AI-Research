import sys
import asyncio
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
import torch

# Import components
from qwen_init import QwenVLManager
from storage import DataStorage, StorageConfig
from memory_system import MemorySystem
from persistent_memory import PersistentMemory
from learning_memory import LearningMemory
from memory_integration import IntegratedMemory, MemoryConfig
from crypto_scraper import CryptoScraper,ScraperConfig
from enhanced_chat import EnhancedChat
from enhanced_learning import EnhancedLearning

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# CUDA setup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class simple_formatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.INFO:
            self._style._fmt = "%(message)s"
        else:
            self._style._fmt = "%(levelname)s: %(message)s"
        return super().format(record)

class crypto_intelligence_system:
    def __init__(self):
        print("\n=== Initializing Crypto Intelligence System ===")
        try:
            # Setup directories
            base_dir = Path("data")
            dirs = {
                'memory_dir': str(base_dir / "memory"),
                'data_dir': str(base_dir)
            }
            
            # Create directories if they don't exist
            base_dir.mkdir(parents=True, exist_ok=True)
            Path(dirs['memory_dir']).mkdir(parents=True, exist_ok=True)
          
            # Initialize core AI
            print("\nInitializing Qwen VL Manager...")
            self.qwen = QwenVLManager()
            print("[OK] AI system initialized")
            
            # Initialize Integrated Memory System with MemoryConfig
            print("\nInitializing Memory Integration...")
            memory_config = MemoryConfig(
                base_dir=dirs['memory_dir'],
                memory_file=str(Path(dirs['memory_dir']) / "crypto_memory.json"))
            self.memory_integration = IntegratedMemory(config=memory_config)
            print("[OK] Memory Integration initialized")

            # Extract individual memory components from memory integration (optional)
            self.memory_system = self.memory_integration.memory_system
            self.persistent_memory = self.memory_integration.persistent_memory
            self.learning_memory = self.memory_integration.learning_memory

            # Initialize storage system with appropriate configuration
            print("\nInitializing Storage System...")
            storage_config = StorageConfig(data_dir=dirs['data_dir'])
            self.storage = DataStorage(config=storage_config, memory=self.memory_integration)
            print("[OK] Storage system initialized")

            # Initialize enhanced learning
            print("\nInitializing Enhanced Learning System...")
            self.learning = EnhancedLearning(data_interface=self.memory_integration,learning_memory=self.memory_integration.learning_memory)
            print("[OK] Enhanced Learning System initialized")
            
            # Initialize scraper configuration
            print("\nInitializing Scraper Configuration...")
            self.scraper_config = ScraperConfig()
            print("[OK] Scraper Configuration initialized")

            # Initialize scraper
            print("\nInitializing Scraper...")
            self.scraper = CryptoScraper(
            config=self.scraper_config,
            storage=self.storage,
            qwen_manager=self.qwen,
            memory=self.memory_integration,
            learning=self.learning)
            print("[OK] Scraper initialized")

            
            # Initialize chat interface
            print("\nInitializing Chat Interface...")
            self.chat = EnhancedChat(
                storage=self.storage,
                memory=self.memory_integration,
                qwen=self.qwen,
                learning=self.learning)
            print("[OK] Chat interface initialized")
            
            print("\nAll systems initialized successfully!")
        except Exception as e:
            print(f"\nError during initialization: {str(e)}")
            raise

    async def start_scraper(self):
        """Start the scraper asynchronously."""
        try:
            print("\nStarting Scraper...")
            await self.scraper.run()
        except Exception as e:
            print(f"Error starting scraper: {str(e)}")

    async def shutdown(self):
        """Gracefully shutdown all components."""
        print("\nShutting down systems...")
        try:
            # Stop the scraper
            if hasattr(self, 'scraper'):
                await self.scraper.close_browser()
            
            # Stop the system monitor
            if hasattr(self, 'monitor'):
                await self.monitor.close()  # Close monitoring tasks gracefully
            
            print("[OK] Systems shutdown complete")
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")

def main():
    """Main entry point."""
    try:
        # Initialize and start system
        system = crypto_intelligence_system()
        asyncio.run(system.start_scraper())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        if 'system' in locals():
            asyncio.run(system.shutdown())
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)