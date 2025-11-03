"""Conversation context management"""
from typing import List, Dict
from datetime import datetime


class ConversationContext:
    """Manage conversation history and context"""
    
    def __init__(self, max_history: int = 10):
        self.history: List[Dict] = []
        self.max_history = max_history
    
    def add_message(self, role: str, content: str):
        """Add message to history"""
        self.history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only recent history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_context_string(self, num_messages: int = 5) -> str:
        """Get recent conversation as string"""
        recent = self.history[-num_messages:] if len(self.history) > num_messages else self.history
        context = []
        for msg in recent:
            context.append(f"{msg['role']}: {msg['content']}")
        return "\n".join(context)
    
    def clear(self):
        """Clear conversation history"""
        self.history = []
