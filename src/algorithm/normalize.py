# src/algorithm/normalize.py
"""
Turn raw values into 0–1.
normalize each raw feature using config.py
return another clean dictionary with the same keys
"""

from typing import Dict
from src.config import NORMALIZATION_RANGES
# importing boundaries that were determined in config.py


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def minmax_norm(value: float, min_value: float, max_value: float) -> float:
    if max_value <= min_value:
        return 0.0
    normalized = (value - min_value) / (max_value - min_value)
    return clamp01(normalized)


def normalize_features(raw_features: Dict[str, float]) -> Dict[str, float]:
    return {
        "shot_frequency": minmax_norm(
            raw_features["shot_frequency"],
            NORMALIZATION_RANGES.shot_frequency_min,
            NORMALIZATION_RANGES.shot_frequency_max,
        ),
        "object_entropy": minmax_norm(
            raw_features["object_entropy"],
            NORMALIZATION_RANGES.object_entropy_min,
            NORMALIZATION_RANGES.object_entropy_max,
        ),
        "interaction_density": minmax_norm(
            raw_features["interaction_density"],
            NORMALIZATION_RANGES.interaction_density_min,
            NORMALIZATION_RANGES.interaction_density_max,
        ),
        "human_presence_ratio": minmax_norm(
            raw_features["human_presence_ratio"],
            NORMALIZATION_RANGES.human_presence_ratio_min,
            NORMALIZATION_RANGES.human_presence_ratio_max,
        ),
    }