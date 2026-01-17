"""
Models Package for Food Shelf Life Predictor
=============================================

This package contains the machine learning models for food classification
and shelf life prediction.
"""

from .classifier import (
    FoodClassifier,
    load_food_classifier,
    classify_food,
)

from .shelf_life_predictor import (
    ShelfLifePredictor,
    train_shelf_life_model,
    predict_shelf_life,
    evaluate_model,
)

__all__ = [
    "FoodClassifier",
    "load_food_classifier",
    "classify_food",
    "ShelfLifePredictor",
    "train_shelf_life_model",
    "predict_shelf_life",
    "evaluate_model",
]
