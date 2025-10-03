from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime
from typing import List, Optional, Dict
import logging
from memory_integration import IntegratedMemory

class EnhancedChat:
    def __init__(self,storage, memory: IntegratedMemory, qwen, learning):
        self.storage = storage
        self.memory = memory
        self.qwen = qwen
        self.learning = learning

        # Setup logging
        self.logger = self._setup_logging()
        
        # Chat state
        self.current_context = {}
        self.last_message_time = None
        
        self.logger.info("Chat system initialized")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for chat system"""
        logger = logging.getLogger("ChatSystem")
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Setup handler
        handler = logging.FileHandler('logs/chat.log')
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        return logger

    async def process_message(self, 
                            text: str, 
                            images: Optional[List[Dict]] = None) -> str:
        """Process a user message with optional images"""
        try:
            # Record message time
            self.last_message_time = datetime.now()
            
            # Format message for Qwen
            message = self._format_qwen_message(text, images)
            
            # Store in memory
            self.memory.store('conversation', {
                'role': 'user',
                'content': message,
                'timestamp': self.last_message_time.isoformat()
            })
            
            # Get response from Qwen
            if self.qwen:
                try:
                    response = await self._get_qwen_response(message)
                except Exception as e:
                    self.logger.error(f"Qwen error: {e}")
                    response = "I had trouble processing that. Could you try again?"
            else:
                response = "Chat model not initialized."
            
            # Store response in memory
            self.memory.store('conversation', {
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat()
            })
            
            # Learn from interaction
            if self.learning:
                await self.learning.learn_from_interaction(
                    user_message=message,
                    assistant_response=response,
                    context=self.current_context
                )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return "I encountered an error. Please try again."

    def _format_qwen_message(self, 
                           text: str, 
                           images: Optional[List[Dict]] = None) -> Dict:
        """Format message for Qwen model"""
        content = []
        
        # Add images if any
        if images:
            for img in images:
                content.append({
                    "type": "image",
                    "image": img.get("image"),
                    "min_pixels": img.get("min_pixels", 50176),
                    "max_pixels": img.get("max_pixels", 50176)
                })
        
        # Add text
        content.append({
            "type": "text",
            "text": text
        })
        
        return {
            "role": "user",
            "content": content
        }

    async def _get_qwen_response(self, message: Dict) -> str:
        """Get response from Qwen model"""
        try:
            response = await self.qwen.chat(message)
            return response
            
        except Exception as e:
            self.logger.error(f"Qwen response error: {e}")
            raise

    async def get_history(self, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        try:
            history = self.memory.get('conversation', limit=limit)
            return history
            
        except Exception as e:
            self.logger.error(f"Error getting history: {e}")
            return []

    async def handle_command(self, command: str) -> str:
        """Handle special commands"""
        try:
            cmd = command.lower().strip()
            
            if cmd == '/help':
                return self._get_help_text()
                
            elif cmd == '/history':
                history = await self.get_history()
                return self._format_history(history)
                
            elif cmd == '/clear':
                self.memory.clear('conversation')
                return "Conversation history cleared."
                
            elif cmd.startswith('/topic '):
                topic = cmd[7:].strip()
                self.current_context['topic'] = topic
                return f"Topic set to: {topic}"
                
            elif cmd == '/stats':
                return await self._get_learning_stats()
                
            else:
                return await self.process_message(command)
                
        except Exception as e:
            self.logger.error(f"Error handling command: {e}")
            return "Error processing command. Please try again."

    def _get_help_text(self) -> str:
        """Get help text for commands"""
        return """
Available Commands:
/help     - Show this help message
/history  - Show conversation history
/clear    - Clear conversation history
/topic    - Set conversation topic
/stats    - Show learning statistics
        """

    def _format_history(self, history: List[Dict]) -> str:
        """Format history for display"""
        lines = []
        for entry in history:
            role = entry['role'].title()
            content = entry['content']
            if isinstance(content, dict):
                content = content.get('text', str(content))
            timestamp = entry.get('timestamp', '')[:16]  # Just date and time
            lines.append(f"{timestamp} {role}: {content}")
        return "\n".join(lines)

    async def _get_learning_stats(self) -> str:
        """Get learning statistics"""
        try:
            if self.learning:
                stats = self.learning.get_stats()
                return (
                    f"Learning Stats:\n"
                    f"Insights gained: {stats['insights_gained']}\n"
                    f"Patterns learned: {stats['patterns_learned']}\n"
                    f"Success rate: {stats['success_rate']:.1%}"
                )
            return "Learning system not initialized."
            
        except Exception as e:
            self.logger.error(f"Error getting learning stats: {e}")
            return "Could not retrieve learning stats."

    def save_state(self):
        """Save chat state"""
        try:
            state = {
                'context': self.current_context,
                'last_message': self.last_message_time.isoformat() if self.last_message_time else None
            }
            
            with open('data/chat_state.json', 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving chat state: {e}")

    def load_state(self):
        """Load saved chat state"""
        try:
            if Path('data/chat_state.json').exists():
                with open('data/chat_state.json', 'r') as f:
                    state = json.load(f)
                    
                self.current_context = state.get('context', {})
                last_msg = state.get('last_message')
                if last_msg:
                    self.last_message_time = datetime.fromisoformat(last_msg)
                    
        except Exception as e:
            self.logger.error(f"Error loading chat state: {e}")