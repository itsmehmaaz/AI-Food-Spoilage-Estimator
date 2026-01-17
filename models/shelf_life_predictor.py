"""
Shelf Life Prediction Module
============================

This module implements the RandomForestRegressor for predicting remaining
shelf life in days based on food category and environmental conditions.

The model uses the following features:
- Food category (one-hot encoded): apple, banana, bread, milk
- Temperature (standardized): Storage temperature in Celsius
- Humidity (standardized): Relative humidity percentage
- Storage type (one-hot encoded): refrigerated, room_temperature, frozen

Academic Context:
- Random Forest: Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5-32.
- Feature importance analysis for interpretability
- MAE and RMSE metrics for evaluation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score

# Import from project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODELS_DIR,
    TARGET_CATEGORIES,
    STORAGE_TYPES,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_RANDOM_STATE,
    RF_MIN_SAMPLES_SPLIT,
    RF_MIN_SAMPLES_LEAF,
    RF_MODEL_FILENAME,
    SCALER_FILENAME,
    CV_FOLDS,
)
from utils.data_loader import (
    load_shelf_life_dataset,
    prepare_regression_data,
    encode_single_sample,
)


class ShelfLifePredictor:
    """
    Random Forest-based shelf life predictor.
    
    This class trains and uses a RandomForestRegressor to predict the
    remaining shelf life of food items based on their category and
    storage conditions.
    
    The model is trained on a research-based dataset that follows
    food science principles for shelf life modeling.
    
    Attributes:
        model: Trained RandomForestRegressor
        scaler: StandardScaler for numerical features
        feature_names: List of feature names for interpretability
        is_trained: Whether the model has been trained
    
    Example:
        >>> predictor = ShelfLifePredictor()
        >>> predictor.train()
        >>> days = predictor.predict("apple", temperature=4, humidity=60, storage="refrigerated")
        >>> print(f"Predicted shelf life: {days:.1f} days")
    """
    
    def __init__(self):
        """Initialize the shelf life predictor."""
        self.model = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            random_state=RF_RANDOM_STATE,
            n_jobs=-1,  # Use all available cores
            verbose=0
        )
        self.scaler = None
        self.feature_names = None
        self.is_trained = False
        self._training_metrics = {}
    
    def train(
        self,
        df: pd.DataFrame = None,
        save_model: bool = True
    ) -> Dict[str, float]:
        """
        Train the shelf life prediction model.
        
        Args:
            df: Optional DataFrame with training data (loads if not provided)
            save_model: Whether to save the trained model to disk
        
        Returns:
            Dictionary with training metrics:
                - train_mae: Mean Absolute Error on training set
                - test_mae: Mean Absolute Error on test set
                - train_rmse: Root Mean Squared Error on training set
                - test_rmse: Root Mean Squared Error on test set
                - cv_mae_mean: Cross-validation MAE mean
                - cv_mae_std: Cross-validation MAE standard deviation
        """
        print("=" * 60)
        print("Training Shelf Life Prediction Model")
        print("=" * 60)
        
        # Load and prepare data
        if df is None:
            df = load_shelf_life_dataset()
        
        X_train, X_test, y_train, y_test, feature_names, scaler = prepare_regression_data(df)
        
        self.feature_names = feature_names
        self.scaler = scaler
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {len(feature_names)}")
        
        # Train model
        print("\nFitting RandomForestRegressor...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Cross-validation
        print("Running cross-validation...")
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=CV_FOLDS,
            scoring='neg_mean_absolute_error'
        )
        cv_mae_mean = -cv_scores.mean()
        cv_mae_std = cv_scores.std()
        
        # Store metrics
        self._training_metrics = {
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "cv_mae_mean": cv_mae_mean,
            "cv_mae_std": cv_mae_std,
        }
        
        # Print results
        print("\n" + "-" * 40)
        print("Training Results:")
        print("-" * 40)
        print(f"Training MAE:  {train_mae:.2f} days")
        print(f"Test MAE:      {test_mae:.2f} days")
        print(f"Training RMSE: {train_rmse:.2f} days")
        print(f"Test RMSE:     {test_rmse:.2f} days")
        print(f"CV MAE:        {cv_mae_mean:.2f} ± {cv_mae_std:.2f} days")
        
        # Feature importance
        print("\nFeature Importance:")
        print("-" * 40)
        importance = self.model.feature_importances_
        for name, imp in sorted(zip(feature_names, importance), key=lambda x: -x[1]):
            print(f"  {name}: {imp:.4f}")
        
        # Save model
        if save_model:
            self.save()
        
        return self._training_metrics
    
    def predict(
        self,
        food_category: str,
        temperature: float,
        humidity: float,
        storage_type: str
    ) -> float:
        """
        Predict shelf life for given conditions.
        
        Args:
            food_category: Food type ('apple', 'banana', 'bread', 'milk')
            temperature: Storage temperature in Celsius
            humidity: Relative humidity percentage (30-90)
            storage_type: Storage type ('refrigerated', 'room_temperature', 'frozen')
        
        Returns:
            Predicted shelf life in days (float)
        
        Raises:
            ValueError: If model not trained or invalid inputs
        """
        if not self.is_trained:
            # Try to load saved model
            if not self.load():
                raise ValueError(
                    "Model not trained. Call train() or ensure saved model exists."
                )
        
        # Validate inputs
        if food_category not in TARGET_CATEGORIES:
            raise ValueError(
                f"Invalid food category: {food_category}. "
                f"Must be one of: {TARGET_CATEGORIES}"
            )
        
        if storage_type not in STORAGE_TYPES:
            raise ValueError(
                f"Invalid storage type: {storage_type}. "
                f"Must be one of: {STORAGE_TYPES}"
            )
        
        # Encode sample
        X = encode_single_sample(
            food_category, temperature, humidity, storage_type, self.scaler
        )
        
        # Predict
        prediction = self.model.predict(X)[0]
        
        # Ensure non-negative
        return max(0.5, prediction)
    
    def predict_batch(
        self,
        samples: list
    ) -> np.ndarray:
        """
        Predict shelf life for multiple samples.
        
        Args:
            samples: List of dicts with keys:
                - food_category, temperature, humidity, storage_type
        
        Returns:
            NumPy array of predictions
        """
        predictions = []
        for sample in samples:
            pred = self.predict(**sample)
            predictions.append(pred)
        return np.array(predictions)
    
    def save(self, path: Path = None) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Optional path for model file (default: models/trained_rf_model.joblib)
        """
        if path is None:
            path = MODELS_DIR / RF_MODEL_FILENAME
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "training_metrics": self._training_metrics,
        }
        
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    def load(self, path: Path = None) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            path: Optional path to model file
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if path is None:
            path = MODELS_DIR / RF_MODEL_FILENAME
        
        if not path.exists():
            print(f"Model file not found: {path}")
            return False
        
        try:
            model_data = joblib.load(path)
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_names = model_data["feature_names"]
            self._training_metrics = model_data.get("training_metrics", {})
            self.is_trained = True
            print(f"Model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_training_metrics(self) -> Dict[str, float]:
        """Get metrics from the last training run."""
        return self._training_metrics.copy()
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global predictor instance (lazy loaded)
_predictor_instance: Optional[ShelfLifePredictor] = None


def get_shelf_life_predictor() -> ShelfLifePredictor:
    """Get or create the global shelf life predictor."""
    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = ShelfLifePredictor()
        # Try to load existing model
        _predictor_instance.load()
    
    return _predictor_instance


def train_shelf_life_model(
    df: pd.DataFrame = None
) -> Tuple[ShelfLifePredictor, np.ndarray, np.ndarray]:
    """
    Train the shelf life model and return it with test data.
    
    This is a convenience function for the verification workflow.
    
    Args:
        df: Optional DataFrame with training data
    
    Returns:
        Tuple of (predictor, X_test, y_test)
    """
    predictor = ShelfLifePredictor()
    
    # Load data
    if df is None:
        df = load_shelf_life_dataset()
    
    # Get test data for evaluation
    X_train, X_test, y_train, y_test, _, _ = prepare_regression_data(df)
    
    # Train
    predictor.train(df, save_model=True)
    
    return predictor, X_test, y_test


def predict_shelf_life(
    food_category: str,
    temperature: float,
    humidity: float,
    storage_type: str,
    predictor: ShelfLifePredictor = None
) -> float:
    """
    Predict shelf life using the global predictor.
    
    Args:
        food_category: Food type
        temperature: Temperature in Celsius
        humidity: Humidity percentage
        storage_type: Storage type
        predictor: Optional predictor instance
    
    Returns:
        Predicted shelf life in days
    """
    if predictor is None:
        predictor = get_shelf_life_predictor()
    
    return predictor.predict(food_category, temperature, humidity, storage_type)


def evaluate_model(
    predictor: ShelfLifePredictor,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[float, float]:
    """
    Evaluate model on test data.
    
    Args:
        predictor: Trained predictor
        X_test: Test features
        y_test: True labels
    
    Returns:
        Tuple of (MAE, RMSE)
    """
    y_pred = predictor.model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return mae, rmse


if __name__ == "__main__":
    # Test the shelf life predictor
    print("=" * 60)
    print("Shelf Life Predictor Test")
    print("=" * 60)
    
    # Create and train predictor
    predictor = ShelfLifePredictor()
    metrics = predictor.train()
    
    # Test predictions
    print("\n" + "=" * 60)
    print("Sample Predictions")
    print("=" * 60)
    
    test_cases = [
        ("apple", 4, 60, "refrigerated"),
        ("apple", 25, 70, "room_temperature"),
        ("apple", -18, 50, "frozen"),
        ("banana", 4, 60, "refrigerated"),
        ("banana", 25, 70, "room_temperature"),
        ("bread", 4, 60, "refrigerated"),
        ("bread", 25, 70, "room_temperature"),
        ("milk", 4, 60, "refrigerated"),
        ("milk", 25, 70, "room_temperature"),
    ]
    
    print("\n{:<10} {:<8} {:<8} {:<18} {:<10}".format(
        "Food", "Temp(°C)", "Humidity", "Storage", "Days"
    ))
    print("-" * 60)
    
    for food, temp, humidity, storage in test_cases:
        days = predictor.predict(food, temp, humidity, storage)
        print("{:<10} {:<8} {:<8} {:<18} {:<10.1f}".format(
            food, temp, humidity, storage, days
        ))
