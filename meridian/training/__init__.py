"""Meridian.AI training components."""

from meridian.training.ewc import ElasticWeightConsolidation
from meridian.training.trainer import MeridianTrainer, TrainingConfig

__all__ = ["MeridianTrainer", "TrainingConfig", "ElasticWeightConsolidation"]
