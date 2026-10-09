"""Simplified RapidChiplet-style evaluator for DNN chiplet studies."""

from .evaluator import evaluate_models, evaluate_one
from .model import load_models

__all__ = ["evaluate_models", "evaluate_one", "load_models"]
