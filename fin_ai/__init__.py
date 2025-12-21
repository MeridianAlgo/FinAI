"""
Fin.AI - A trainable transformer-based language model
"""

__version__ = "0.1.0"
__author__ = "Fin.AI Team"

from fin_ai.model.transformer import FinAIModel
from fin_ai.model.config import FinAIConfig

__all__ = ["FinAIModel", "FinAIConfig"]
