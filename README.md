# Food Shelf Life Predictor

An AI-powered system that classifies food images and predicts remaining shelf life based on storage conditions.

## Features

- **Food Image Classification**: Uses EfficientNet-B0 to classify images into 4 categories: apple, banana, bread, milk
- **Shelf Life Prediction**: RandomForestRegressor predicts remaining days based on temperature, humidity, and storage type
- **Data-Driven Approach**: Shelf life calculations based on USDA FoodKeeper data and Q10 temperature coefficients
- **Interactive Web Interface**: Streamlit app for easy image upload and predictions

## Installation

1. **Clone/Download** the project to your local machine

2. **Create a virtual environment** (recommended):
```powershell
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies**:
```powershell
pip install -r requirements.txt
```

## Usage

### Run the Streamlit App

```powershell
streamlit run app.py
```

This will:
1. Open a browser window at `http://localhost:8501`
2. Allow you to upload food images
3. Adjust storage parameters (temperature, humidity, storage type)
4. Display classification results and predicted shelf life

### Command-Line Usage

```python
from models.classifier import load_food_classifier
from models.shelf_life_predictor import ShelfLifePredictor

# Classify a food image
classifier = load_food_classifier()
category, confidence, scores = classifier.classify("path/to/food_image.jpg")
print(f"Detected: {category} ({confidence:.1%})")

# Predict shelf life
predictor = ShelfLifePredictor()
predictor.train()  # First time only

days = predictor.predict(
    food_category="apple",
    temperature=4,       # Celsius
    humidity=60,         # Percent
    storage_type="refrigerated"
)
print(f"Shelf life: {days:.1f} days")
```

## Project Structure

```
Shelf Life Predictor/
├── app.py                 # Streamlit web interface
├── config.py              # Configuration and constants
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── data/
│   └── food_shelf_life_data.csv  # Generated training dataset
├── models/
│   ├── __init__.py
│   ├── classifier.py             # EfficientNet-B0 food classifier
│   ├── shelf_life_predictor.py   # RandomForest regressor
│   └── trained_rf_model.joblib   # Saved model (after training)
└── utils/
    ├── __init__.py
    ├── data_loader.py            # Dataset loading and generation
    └── preprocessing.py          # Image preprocessing
```

## Technical Details

### Food Classification
- **Model**: EfficientNet-B0 (pretrained on ImageNet)
- **Categories**: apple, banana, bread, milk
- **Input**: 224x224 RGB images
- **Output**: Category prediction with confidence score

### Shelf Life Prediction
- **Model**: RandomForestRegressor (scikit-learn)
- **Features**: 
  - Food category (one-hot encoded)
  - Temperature (normalized)
  - Humidity (normalized)
  - Storage type (one-hot encoded)
- **Target**: Days to spoilage
- **Metrics**: MAE and RMSE

### Data Science Approach
- Shelf life modeling uses the Q10 temperature coefficient (standard in food science)
- Base shelf life values from USDA FoodKeeper Application
- Temperature effects modeled using Arrhenius equation principles

## Evaluation Metrics

After training, the model reports:
- **MAE (Mean Absolute Error)**: Average prediction error in days
- **RMSE (Root Mean Squared Error)**: Error metric penalizing larger errors
- **Cross-Validation MAE**: Model generalization performance

## Dependencies

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- timm >= 0.9.0 (PyTorch Image Models)
- streamlit >= 1.28.0
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- Pillow >= 10.0.0

## License

This project is for educational and research purposes.

## References

- EfficientNet: Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019
- Food-101: Bossard et al., "Food-101 – Mining Discriminative Components", ECCV 2014
- Q10 Temperature Coefficient: Labuza, T.P. (1982). "Shelf Life Dating of Foods"
- USDA FoodKeeper Application Data
