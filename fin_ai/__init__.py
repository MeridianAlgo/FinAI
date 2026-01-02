"""
Fin.AI - A trainable transformer-based language model
"""

__version__ = "0.1.0"
__author__ = "Fin.AI Team"

from fin_ai.model.config import FinAIConfig
from fin_ai.model.transformer import FinAIModel

__all__ = ["FinAIModel", "FinAIConfig"]
