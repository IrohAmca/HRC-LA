"""
Needle In A Haystack (NIAH) Test Scenario.

Tests a model's ability to retrieve specific information hidden 
within long sequences of distractor tokens.
"""

from .datasets import MultiDepthNIAHDataset, NeedleInHaystackDataset
from .main import NIAHConfig, evaluate_by_depth, train_niah

__all__ = [
    "NeedleInHaystackDataset",
    "MultiDepthNIAHDataset", 
    "train_niah",
    "NIAHConfig",
    "evaluate_by_depth",
]
