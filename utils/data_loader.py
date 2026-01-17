"""
Data Loader Module for Food Shelf Life Predictor
=================================================

This module handles loading, generating, and preparing the shelf life dataset
for training the regression model. The data generation follows established
food science principles for shelf life modeling.

Academic References:
- Q10 temperature coefficient for food spoilage: Labuza, T.P. (1982)
  "Shelf Life Dating of Foods"
- USDA FoodKeeper Application data for base shelf life values
- Arrhenius equation applications in food science

Key Functions:
- generate_shelf_life_dataset(): Creates research-based training data
- load_shelf_life_dataset(): Loads existing or generates new dataset
- compute_base_shelf_life(): Calculates average shelf life for refrigerated storage
- prepare_regression_data(): Prepares features and targets for model training
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# Import configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_DIR,
    MODELS_DIR,
    TARGET_CATEGORIES,
    CATEGORY_TO_INDEX,
    STORAGE_TYPES,
    STORAGE_TO_INDEX,
    BASE_SHELF_LIFE_REFRIGERATED,
    Q10_COEFFICIENT,
    FROZEN_MULTIPLIER,
    SAMPLES_PER_CATEGORY,
    TEMPERATURE_RANGE,
    HUMIDITY_RANGE,
    DATA_RANDOM_SEED,
    TEST_SIZE,
    SCALER_FILENAME,
)


def compute_shelf_life(
    food_category: str,
    temperature: float,
    humidity: float,
    storage_type: str
) -> float:
    """
    Compute expected shelf life in days using food science principles.
    
    The calculation uses the Q10 temperature coefficient model, which is
    a standard approach in food science for predicting how temperature
    affects food spoilage rates.
    
    Q10 Model:
        rate_ratio = Q10^((T2 - T1) / 10)
        shelf_life = base_shelf_life / rate_ratio
    
    Where:
        - Q10 ≈ 2.5 for typical food spoilage reactions
        - T1 = reference temperature (4°C for refrigerated)
        - T2 = actual storage temperature
    
    Humidity Factor:
        Higher humidity accelerates microbial growth. We apply a linear
        correction factor based on deviation from optimal humidity (50-60%).
    
    Args:
        food_category: One of 'apple', 'banana', 'bread', 'milk'
        temperature: Storage temperature in Celsius
        humidity: Relative humidity (30-90%)
        storage_type: 'refrigerated', 'room_temperature', or 'frozen'
    
    Returns:
        Predicted shelf life in days (float)
    
    References:
        - Labuza, T.P. (1982). Shelf Life Dating of Foods.
        - USDA FoodKeeper Application (2024)
    """
    # Get base shelf life for refrigerated storage (reference condition)
    base_shelf_life = BASE_SHELF_LIFE_REFRIGERATED[food_category]
    reference_temp = 4.0  # Reference temperature in Celsius
    
    if storage_type == "frozen":
        # Frozen storage extends shelf life significantly
        # Use frozen multiplier from FDA guidelines
        shelf_life = base_shelf_life * FROZEN_MULTIPLIER[food_category]
        # Temperature has minimal effect in frozen range
        return max(shelf_life, 1.0)
    
    # Calculate temperature effect using Q10 model
    # Q10 equation: rate = rate_ref * Q10^((T - T_ref) / 10)
    # Shelf life is inversely proportional to spoilage rate
    temp_diff = temperature - reference_temp
    rate_ratio = Q10_COEFFICIENT ** (temp_diff / 10.0)
    
    # Apply temperature correction to base shelf life
    shelf_life = base_shelf_life / rate_ratio
    
    # Apply humidity correction factor
    # Optimal humidity is around 50-60% for most foods
    # Higher humidity accelerates microbial growth
    optimal_humidity = 55.0
    humidity_diff = humidity - optimal_humidity
    
    # Each 10% increase in humidity above optimal reduces shelf life by ~10%
    humidity_factor = 1.0 - (humidity_diff / 100.0) * 0.5
    humidity_factor = max(0.3, min(humidity_factor, 1.2))  # Bound the factor
    
    shelf_life = shelf_life * humidity_factor
    
    # Ensure minimum shelf life of 0.5 days
    return max(shelf_life, 0.5)


def generate_shelf_life_dataset(
    samples_per_category: int = SAMPLES_PER_CATEGORY,
    save_path: Path = None,
    random_seed: int = DATA_RANDOM_SEED
) -> pd.DataFrame:
    """
    Generate a research-based shelf life dataset using food science models.
    
    This function creates training data by sampling environmental conditions
    and computing expected shelf life using established food science equations.
    Gaussian noise is added to simulate real-world variability.
    
    The generation process:
    1. For each food category, sample temperature, humidity, and storage type
    2. Compute theoretical shelf life using Q10 model
    3. Add realistic noise (±15% standard deviation)
    4. Save to CSV for reproducibility
    
    Args:
        samples_per_category: Number of samples to generate per food type
        save_path: Optional path to save the generated CSV
        random_seed: Random seed for reproducibility
    
    Returns:
        pd.DataFrame with columns:
            - food_category: str
            - temperature_celsius: float
            - humidity_percent: float
            - storage_type: str
            - days_to_spoilage: float (target variable)
    """
    np.random.seed(random_seed)
    
    data = []
    
    for category in TARGET_CATEGORIES:
        for _ in range(samples_per_category):
            # Sample storage type (with realistic distribution)
            # Most foods are stored refrigerated
            storage_probs = [0.5, 0.3, 0.2]  # refrigerated, room_temp, frozen
            storage_type = np.random.choice(STORAGE_TYPES, p=storage_probs)
            
            # Sample temperature based on storage type
            if storage_type == "refrigerated":
                temperature = np.random.uniform(2, 8)  # Typical fridge range
            elif storage_type == "room_temperature":
                temperature = np.random.uniform(18, 28)  # Room temp range
            else:  # frozen
                temperature = np.random.uniform(-20, -10)  # Freezer range
            
            # Sample humidity (affects refrigerated/room temp more)
            humidity = np.random.uniform(HUMIDITY_RANGE[0], HUMIDITY_RANGE[1])
            
            # Compute theoretical shelf life
            shelf_life = compute_shelf_life(
                category, temperature, humidity, storage_type
            )
            
            # Add realistic noise (coefficient of variation ~15%)
            noise_std = shelf_life * 0.15
            noisy_shelf_life = shelf_life + np.random.normal(0, noise_std)
            noisy_shelf_life = max(0.5, noisy_shelf_life)  # Minimum 0.5 days
            
            data.append({
                "food_category": category,
                "temperature_celsius": round(temperature, 1),
                "humidity_percent": round(humidity, 1),
                "storage_type": storage_type,
                "days_to_spoilage": round(noisy_shelf_life, 1)
            })
    
    df = pd.DataFrame(data)
    
    # Save to CSV if path provided
    if save_path is None:
        save_path = DATA_DIR / "food_shelf_life_data.csv"
    
    df.to_csv(save_path, index=False)
    print(f"Generated dataset with {len(df)} samples saved to {save_path}")
    
    return df


def load_shelf_life_dataset(
    filepath: Path = None,
    regenerate: bool = False
) -> pd.DataFrame:
    """
    Load the shelf life dataset from CSV, generating if necessary.
    
    Args:
        filepath: Path to the CSV file (default: data/food_shelf_life_data.csv)
        regenerate: If True, regenerate the dataset even if file exists
    
    Returns:
        pd.DataFrame with the shelf life data
    """
    if filepath is None:
        filepath = DATA_DIR / "food_shelf_life_data.csv"
    
    if regenerate or not filepath.exists():
        print("Generating new shelf life dataset...")
        return generate_shelf_life_dataset(save_path=filepath)
    
    print(f"Loading existing dataset from {filepath}")
    df = pd.read_csv(filepath)
    
    # Validate dataset structure
    required_columns = [
        "food_category", "temperature_celsius", 
        "humidity_percent", "storage_type", "days_to_spoilage"
    ]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df


def compute_base_shelf_life(df: pd.DataFrame) -> dict:
    """
    Compute average shelf life under refrigerated conditions for each category.
    
    This function implements requirement #4: compute base shelf life by averaging
    days-to-spoilage under refrigerated conditions (3-5°C) directly from the dataset.
    
    Args:
        df: DataFrame with shelf life data
    
    Returns:
        Dictionary mapping food category to average refrigerated shelf life
    """
    # Filter for refrigerated storage with temperature in 3-5°C range
    refrigerated_mask = (
        (df["storage_type"] == "refrigerated") &
        (df["temperature_celsius"] >= 3) &
        (df["temperature_celsius"] <= 5)
    )
    
    refrigerated_df = df[refrigerated_mask]
    
    # Calculate mean days to spoilage for each category
    base_shelf_life = {}
    
    for category in TARGET_CATEGORIES:
        category_data = refrigerated_df[refrigerated_df["food_category"] == category]
        if len(category_data) > 0:
            avg_days = category_data["days_to_spoilage"].mean()
            base_shelf_life[category] = round(avg_days, 1)
        else:
            # Fallback to configured value if no data in range
            base_shelf_life[category] = BASE_SHELF_LIFE_REFRIGERATED[category]
    
    return base_shelf_life


def encode_features(df: pd.DataFrame) -> tuple:
    """
    Encode categorical features and normalize numerical features.
    
    Feature encoding scheme:
    - food_category: One-hot encoded (4 categories)
    - storage_type: One-hot encoded (3 types)
    - temperature_celsius: Standardized (z-score normalization)
    - humidity_percent: Standardized (z-score normalization)
    
    Args:
        df: DataFrame with shelf life data
    
    Returns:
        Tuple of (X_encoded, y, feature_names, scaler)
    """
    # Prepare numerical features
    numerical_features = df[["temperature_celsius", "humidity_percent"]].values
    
    # Standardize numerical features
    scaler = StandardScaler()
    numerical_scaled = scaler.fit_transform(numerical_features)
    
    # One-hot encode categorical features
    category_encoded = pd.get_dummies(df["food_category"], prefix="cat")
    storage_encoded = pd.get_dummies(df["storage_type"], prefix="storage")
    
    # Combine all features
    X = np.hstack([
        numerical_scaled,
        category_encoded.values,
        storage_encoded.values
    ])
    
    # Target variable
    y = df["days_to_spoilage"].values
    
    # Feature names for interpretability
    feature_names = (
        ["temperature_scaled", "humidity_scaled"] +
        list(category_encoded.columns) +
        list(storage_encoded.columns)
    )
    
    return X, y, feature_names, scaler


def prepare_regression_data(
    df: pd.DataFrame = None,
    test_size: float = TEST_SIZE,
    random_state: int = DATA_RANDOM_SEED
) -> tuple:
    """
    Prepare data for training the shelf life regression model.
    
    This function:
    1. Loads or uses provided dataset
    2. Encodes features (one-hot for categories, normalized for numerical)
    3. Splits into train/test sets
    4. Saves the scaler for later use in inference
    
    Args:
        df: Optional DataFrame (will load if not provided)
        test_size: Fraction of data for testing (default 0.2)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    if df is None:
        df = load_shelf_life_dataset()
    
    # Encode features
    X, y, feature_names, scaler = encode_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Save scaler for inference
    scaler_path = MODELS_DIR / SCALER_FILENAME
    joblib.dump(scaler, scaler_path)
    print(f"Saved feature scaler to {scaler_path}")
    
    return X_train, X_test, y_train, y_test, feature_names, scaler


def encode_single_sample(
    food_category: str,
    temperature: float,
    humidity: float,
    storage_type: str,
    scaler: StandardScaler = None
) -> np.ndarray:
    """
    Encode a single sample for inference.
    
    Args:
        food_category: Food type (apple, banana, bread, milk)
        temperature: Temperature in Celsius
        humidity: Humidity percentage
        storage_type: Storage type (refrigerated, room_temperature, frozen)
        scaler: Fitted StandardScaler (will load from disk if not provided)
    
    Returns:
        Encoded feature array ready for model prediction
    """
    if scaler is None:
        scaler_path = MODELS_DIR / SCALER_FILENAME
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
        else:
            raise ValueError("Scaler not found. Please train the model first.")
    
    # Scale numerical features
    numerical = np.array([[temperature, humidity]])
    numerical_scaled = scaler.transform(numerical)
    
    # One-hot encode category
    category_encoded = np.zeros(len(TARGET_CATEGORIES))
    category_encoded[CATEGORY_TO_INDEX[food_category]] = 1
    
    # One-hot encode storage type
    storage_encoded = np.zeros(len(STORAGE_TYPES))
    storage_encoded[STORAGE_TO_INDEX[storage_type]] = 1
    
    # Combine features
    X = np.hstack([
        numerical_scaled.flatten(),
        category_encoded,
        storage_encoded
    ])
    
    return X.reshape(1, -1)


if __name__ == "__main__":
    # Test data generation and loading
    print("=" * 60)
    print("Food Shelf Life Dataset Generator")
    print("=" * 60)
    
    # Generate dataset
    df = load_shelf_life_dataset(regenerate=True)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nSample data:\n{df.head()}")
    
    # Compute base shelf life
    base_shelf_life = compute_base_shelf_life(df)
    print(f"\nBase shelf life (refrigerated 3-5°C):")
    for category, days in base_shelf_life.items():
        print(f"  {category}: {days} days")
    
    # Prepare regression data
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_regression_data(df)
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {feature_names}")
