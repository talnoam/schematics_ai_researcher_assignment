"""Configuration values for core deterministic and information-theory logic."""

DETERMINISTIC_CONFIDENCE: float = 1.0
DETERMINISTIC_MAX_PASSES: int = 5

ENTROPY_EPSILON: float = 1e-12
PROBABILITY_SUM_TOLERANCE: float = 1e-6

FRICTION_WEIGHT: float = 0.5
MIN_UTILITY_THRESHOLD: float = 0.05
AUTO_INFER_THRESHOLD: float = 0.85
