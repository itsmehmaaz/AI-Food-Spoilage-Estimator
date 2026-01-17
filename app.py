"""
Food Shelf Life Predictor - Streamlit Application
==================================================

This is the main Streamlit web interface for the Food Shelf Life Predictor.
It allows users to:
1. Upload a food image for classification
2. Select storage conditions (temperature, humidity, storage type)
3. View the predicted remaining shelf life

The application uses:
- EfficientNet-B0 for food image classification
- RandomForestRegressor for shelf life prediction

Usage:
    streamlit run app.py

Academic Context:
This application demonstrates a complete machine learning pipeline for
food quality assessment, combining computer vision (image classification)
with regression modeling (shelf life prediction).
"""

import streamlit as st
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    TARGET_CATEGORIES,
    STORAGE_TYPES,
    UI_TEMP_MIN,
    UI_TEMP_MAX,
    UI_TEMP_DEFAULT,
    UI_HUMIDITY_MIN,
    UI_HUMIDITY_MAX,
    UI_HUMIDITY_DEFAULT,
    SHELF_LIFE_WARNING_THRESHOLD,
    SHELF_LIFE_DANGER_THRESHOLD,
    BASE_SHELF_LIFE_REFRIGERATED,
)

# Lazy imports for models to avoid slow initial load
_classifier = None
_predictor = None


def get_classifier():
    """Lazy load the food classifier."""
    global _classifier
    if _classifier is None:
        from models.classifier import load_food_classifier
        with st.spinner("Loading food classifier (first time may take a moment)..."):
            _classifier = load_food_classifier()
    return _classifier


def get_predictor():
    """Lazy load and train the shelf life predictor."""
    global _predictor
    if _predictor is None:
        from models.shelf_life_predictor import ShelfLifePredictor
        _predictor = ShelfLifePredictor()
        
        # Check if model exists, otherwise train
        if not _predictor.load():
            with st.spinner("Training shelf life model (first time only)..."):
                _predictor.train()
    return _predictor


def get_freshness_color(days: float) -> str:
    """Get color based on remaining shelf life."""
    if days <= SHELF_LIFE_DANGER_THRESHOLD:
        return "#FF4B4B"  # Red - danger
    elif days <= SHELF_LIFE_WARNING_THRESHOLD:
        return "#FFA500"  # Orange - warning
    else:
        return "#00C853"  # Green - fresh


def get_freshness_status(days: float) -> str:
    """Get status text based on remaining shelf life."""
    if days <= SHELF_LIFE_DANGER_THRESHOLD:
        return "⚠️ Use Immediately"
    elif days <= SHELF_LIFE_WARNING_THRESHOLD:
        return "⏰ Use Soon"
    elif days <= 7:
        return "✅ Fresh"
    else:
        return "🌟 Very Fresh"


def render_metrics_panel(classifier_metrics: dict = None, predictor_metrics: dict = None):
    """Render the model evaluation metrics panel."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Model Metrics")
    
    # Classifier metrics
    if classifier_metrics:
        st.sidebar.markdown("**Food Classifier:**")
        acc = classifier_metrics.get("accuracy", 0)
        st.sidebar.metric("Accuracy", f"{acc:.1%}")
    else:
        st.sidebar.markdown("**Food Classifier:**")
        st.sidebar.caption("Upload images to evaluate")
    
    # Predictor metrics
    if predictor_metrics:
        st.sidebar.markdown("**Shelf Life Predictor:**")
        mae = predictor_metrics.get("test_mae", 0)
        rmse = predictor_metrics.get("test_rmse", 0)
        col1, col2 = st.sidebar.columns(2)
        col1.metric("MAE", f"{mae:.2f} days")
        col2.metric("RMSE", f"{rmse:.2f} days")


def render_base_shelf_life_table():
    """Render the base shelf life reference table."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Base Shelf Life (Refrigerated)")
    st.sidebar.caption("Reference values at 3-5°C")
    
    for food, days in BASE_SHELF_LIFE_REFRIGERATED.items():
        st.sidebar.text(f"  {food.capitalize()}: {days} days")


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Food Shelf Life Predictor",
        page_icon="🍎",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #888;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">🍎 Food Shelf Life Predictor</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload a food image and specify storage conditions to predict remaining shelf life</p>',
        unsafe_allow_html=True
    )
    
    # Sidebar - Configuration
    st.sidebar.header("⚙️ Storage Conditions")
    
    # Temperature slider
    temperature = st.sidebar.slider(
        "Temperature (°C)",
        min_value=UI_TEMP_MIN,
        max_value=UI_TEMP_MAX,
        value=UI_TEMP_DEFAULT,
        help="Storage temperature in Celsius"
    )
    
    # Humidity slider
    humidity = st.sidebar.slider(
        "Humidity (%)",
        min_value=UI_HUMIDITY_MIN,
        max_value=UI_HUMIDITY_MAX,
        value=UI_HUMIDITY_DEFAULT,
        help="Relative humidity percentage"
    )
    
    # Storage type selector
    storage_type = st.sidebar.selectbox(
        "Storage Type",
        options=STORAGE_TYPES,
        format_func=lambda x: x.replace("_", " ").title(),
        help="Type of storage environment"
    )
    
    # Render base shelf life table
    render_base_shelf_life_table()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Upload Food Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png", "webp"],
            help="Upload an image of apple, banana, bread, or milk"
        )
        
        # Display uploaded image
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader("🔮 Prediction Results")
        
        if uploaded_file is not None:
            try:
                # Get image bytes
                image_bytes = uploaded_file.getvalue()
                
                # Classify food
                classifier = get_classifier()
                category, confidence, all_scores = classifier.classify(image_bytes)
                
                # Predict shelf life
                predictor = get_predictor()
                shelf_life = predictor.predict(
                    food_category=category,
                    temperature=temperature,
                    humidity=humidity,
                    storage_type=storage_type
                )
                
                # Display results
                st.markdown("---")
                
                # Classification result
                st.markdown("**Detected Food Category:**")
                st.markdown(f"### {category.upper()}")
                st.progress(confidence)
                st.caption(f"Confidence: {confidence:.1%}")
                
                # All category scores
                with st.expander("View all category scores"):
                    for cat, score in sorted(all_scores.items(), key=lambda x: -x[1]):
                        st.text(f"  {cat}: {score:.1%}")
                
                st.markdown("---")
                
                # Shelf life prediction
                st.markdown("**Predicted Shelf Life:**")
                
                color = get_freshness_color(shelf_life)
                status = get_freshness_status(shelf_life)
                
                st.markdown(
                    f"""
                    <div style="background-color: {color}20; border-left: 4px solid {color}; padding: 1rem; border-radius: 5px;">
                        <span class="metric-value" style="color: {color};">{shelf_life:.1f} days</span>
                        <br>
                        <span>{status}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.markdown("---")
                
                # Storage recommendation
                st.markdown("**Storage Conditions:**")
                st.text(f"  Temperature: {temperature}°C")
                st.text(f"  Humidity: {humidity}%")
                st.text(f"  Storage: {storage_type.replace('_', ' ').title()}")
                
                # Render metrics if available
                try:
                    predictor_metrics = predictor.get_training_metrics()
                    render_metrics_panel(predictor_metrics=predictor_metrics)
                except:
                    pass
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.exception(e)
        else:
            # Placeholder when no image uploaded
            st.info("👈 Upload an image to see predictions")
            
            # Show example categories
            st.markdown("**Supported Food Categories:**")
            cols = st.columns(4)
            icons = {"apple": "🍎", "banana": "🍌", "bread": "🍞", "milk": "🥛"}
            for i, cat in enumerate(TARGET_CATEGORIES):
                cols[i].markdown(f"### {icons[cat]}")
                cols[i].caption(cat.title())
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 0.9rem;">
        <p>
        Built with EfficientNet-B0 (food classification) and RandomForest (shelf life prediction)
        <br>
        Shelf life calculations based on USDA FoodKeeper data and Q10 temperature coefficients
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
