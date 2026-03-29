# src/algorithm/complexity.py
"""
Compute the weighted score.
It should return:
* total scene_complexity_score
* per-feature breakdown

That helps with:
debugging
interpretation
presentation
"""

from typing import Dict, Tuple
from src.config import COMPLEXITY_WEIGHTS 
# imports scoring formula that computes to 1.0


def compute_scene_complexity(norm_features: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    breakdown = {
        "shot_frequency": COMPLEXITY_WEIGHTS.shot_frequency * norm_features["shot_frequency"],
        "object_entropy": COMPLEXITY_WEIGHTS.object_entropy * norm_features["object_entropy"],
        "interaction_density": COMPLEXITY_WEIGHTS.interaction_density * norm_features["interaction_density"],
        "human_presence_ratio": COMPLEXITY_WEIGHTS.human_presence_ratio * norm_features["human_presence_ratio"],
    }

    score = sum(breakdown.values())
    return score, breakdown