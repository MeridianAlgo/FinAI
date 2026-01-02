from fin_ai.model.config import FinAIConfig

# Legacy model available for backward compatibility
from fin_ai.model.transformer import FinAIModel as FinAIModelLegacy
from fin_ai.model.transformer_v2 import FinAIModelV2 as FinAIModel

__all__ = ["FinAIModel", "FinAIConfig", "FinAIModelLegacy"]
