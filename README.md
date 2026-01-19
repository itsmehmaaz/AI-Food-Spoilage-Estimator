# 🍎 AI Food Spoilage Estimator & Shelf Life Predictor

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

> **An AI-powered ecological solution for reducing food waste through computer vision and thermal degradation modeling.**

---

## 📄 Project Abstract

The **AI Food Spoilage Estimator** is a comprehensive software solution designed to tackle the global issue of food waste. By combining **Computer Vision (CV)** for food identification with **Predictive Analytics (Machine Learning)** for shelf life estimation, the system provides users with actionable insights into when their food items will become unsafe for consumption. 

The system doesn't just "guess"; it utilizes established food science principles, including the **Q10 Temperature Coefficient** and **USDA FoodKeeper** averages, to provide scientifically grounded predictions tailored to specific storage environments (Refrigerated, Room Temperature, or Frozen).

---

## ⚠️ The Problem Statement

*   **Global Food Waste**: Approximately 1.3 billion tons of food are wasted annually, with a significant portion occurring at the consumer level due to confusion over "Use By" vs. "Best Before" dates.
*   **Storage Complexity**: Most consumers do not understand how small fluctuations in temperature (e.g., a fridge door being left open) or high humidity can drastically accelerate microbial growth and chemical degradation.
*   **Gap in Technology**: While smart fridges exist, they are expensive. There is a lack of accessible, software-first tools that can instantly analyze a food item and provide a remaining shelf life estimate based on real-time environmental data.

---

## 🎯 Project Objectives

1.  **Instant Identification**: Automate the classification of food items using state-of-the-art Deep Learning models.
2.  **Environmental Modeling**: Account for temperature and humidity variables to provide a dynamic (not static) shelf life prediction.
3.  **Science-Driven Insights**: Implement the Q10 temperature coefficient to model biological decay accurately.
4.  **Accessibility**: Deliver the solution via a lightweight web interface accessible on both desktop and mobile devices.
5.  **Sustainability**: Promote "Zero Hunger" (SDG 2) and "Responsible Consumption" (SDG 12) by preventing avoidable food spoilage.

---

## 🛠️ Technology Stack

### **Backend Core**
*   **Language**: Python 3.9+
*   **Web Frameworks**: 
    *   **Flask**: Powering the API and main production-grade web server.
    *   **Streamlit**: Providing an interactive dashboard for rapid prototyping and model metrics visualization.
*   **Data Science**: Pandas, NumPy (for complex mathematical modeling).

### **Artificial Intelligence & Machine Learning**
*   **Computer Vision**: Hybrid Deep Learning approach via Hugging Face Inference API.
    *   **Primary**: `Food-101` fine-tuned model (ResNet/ViT based) for prepared dish detection.
    *   **Fallback**: `EfficientNet-B0` (ImageNet) for raw ingredient classification.
*   **Regression Model**: Scikit-Learn `RandomForestRegressor`. This model interprets the detected category + temperature + humidity to output the "Days Remaining".

### **DevOps & Infrastructure**
*   **Containerization**: Docker & Docker Compose for platform-independent deployment.
*   **Environment management**: Python-dotenv for secure API token handling.

---

## 🔬 Scientific Methodology & "How It Works"

### **1. Image Classification (The "What")**
The system uses a **weighted ensemble/hybrid approach**:
- It first attempts to identify the food using a specialized Food-101 model.
- If the confidence score is below 70%, it falls back to an EfficientNet-B0 model trained on ImageNet to identify raw ingredients (like a single apple or a bottle of milk).
- This ensures high accuracy for both cooked meals and raw groceries.

### **2. The Q10 Spoilage Model (The "How Long")**
Food degradation is a temperature-dependent chemical and biological process. The system implements the **Q10 Temperature Coefficient**:

$$Q_{10} = \left(\frac{R_2}{R_1}\right)^{\frac{10}{T_2 - T_1}}$$

*   **Logic**: For many food items, the rate of spoilage increases by ~2.5x (Q10 factor) for every 10°C increase in temperature.
*   **Reference**: We use 4°C (Standard Fridge Temp) as our baseline $T_1$. If the user inputs 24°C (Room Temp), the "spoilage rate" is calculated as $2.5^2 = 6.25$, meaning the food spoils 6.25 times faster.

### **3. Humidity Correction**
Microbial growth (molds/bacteria) is accelerated by moisture. Our model applies a penalty/bonus factor:
- **Optimal Humidity**: 55%.
- **Impact**: Every 10% increase in humidity above the baseline reduces the remaining shelf life by approximately 10%.

---

## 📂 Project Architecture

```text
food-spoilage-estimator/
├── app_flask.py           # Production-ready Flask server
├── app.py                 # Interactive Streamlit dashboard
├── config.py              # Centralized configuration (Constants, Mappings, API URLs)
├── models/
│   ├── classifier.py      # Hybrid AI Vision logic (Hugging Face integration)
│   └── shelf_life_predictor.py # RandomForest ML Model
├── utils/
│   └── data_loader.py     # Research-based synthetic dataset generator
├── templates/
│   └── index.html         # Modern Glassmorphism UI (Flask version)
├── Dockerfile             # Container definition
├── docker-compose.yml     # Multi-container orchestration
└── requirements.txt       # Comprehensive dependency list
```

---

## 🚀 Deployment Guide

### **Standard Installation**
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/AI-Food-Spoilage-Estimator.git
    cd AI-Food-Spoilage-Estimator
    ```
2.  **Environment Setup**:
    - Obtain a free API Token from [Hugging Face](https://huggingface.co/settings/tokens).
    - Create a `.env` file and add: `HUGGINGFACE_API_TOKEN=hf_your_token_here`.
3.  **Run with Docker (Fastest)**:
    ```bash
    docker-compose up --build
    ```
    *Access at `http://localhost:5000`*

---

## 📊 Supported Food Categories & Baselines

| Category | Typical Fridge Life (Baseline) | Scientific Logic |
| :--- | :--- | :--- |
| **Banana** | 5 Days | High Ethylene production, sensitive to oxidation. |
| **Milk** | 7 Days | Bacterial lactic acid production (Pasteurization limit). |
| **Meat** | 4 Days | Protein degradation and microbial surface growth. |
| **Sushi** | 1 Day | Raw fish status (immediately high bacterial risk). |
| **Egg** | 28 Days | Natural protective shell coating (Cuticle). |

---

## 📈 Future Roadmap

*   **IoT Integration**: Connect with DHT11/22 sensors for real-time room temperature and humidity tracking.
*   **Mobile App (Flutter)**: Develop a mobile version for "point-and-click" grocery management.
*   **Barcode Scanning**: Integrate OpenFoodFacts API to fetch exact brand-specific ingredients.
*   **Multi-Item Detection**: Enable YOLOv8/v10 for detecting multiple food items in a single fridge photo.

---

## 🤝 Contributors & Academic Context

Created as part of an AI Research project exploring the intersection of **Ecological Sustainability** and **Deep Learning**. 

**References:**
- *USDA FoodKeeper Database (2024)*
- *Labuza, T.P. (1982). "Shelf Life Dating of Foods"*
- *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (ICML 2019)*

---

> [!TIP]
> **Pro Tip for PPT**: When presenting, highlight the **Q10 Coefficient** — it's the "Secret Sauce" that differentiates this project from simple image labelers by adding a layer of genuine food science.
