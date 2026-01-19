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

## ✨ Key Features

- **Hybrid AI Classification**: Uses both Food-101 and ImageNet models for maximum accuracy.
- **15 Food Categories**: Apple, Banana, Bread, Milk, Pasta, Pizza, Burger, Sushi, Meat, Fish, Egg, Vegetable, Fruit, Rice, Cake.
- **Dynamic Shelf Life Prediction**: RandomForest model predicts remaining days based on temperature, humidity, and storage type.
- **Science-Based**: Calculations based on USDA FoodKeeper data and Q10 temperature coefficients.
- **Interactive Dashboard**: Modern Glassmorphism UI for easy data input and visualization.
- **Docker Ready**: One-command deployment with `docker-compose up`.

---

## 🛠️ Technology Stack

- **Backend**: Python, Flask (Production API), Streamlit (Dashboard)
- **AI/Vision**: Hugging Face Inference API (`Food-101` & `EfficientNet-B0`)
- **ML/Analytics**: Scikit-Learn (RandomForestRegressor), Pandas, NumPy
- **Frontend**: HTML5, CSS3 (Glassmorphism), JavaScript
- **Deployment**: Docker, Docker Compose, Python-dotenv

---

## 🔬 Scientific Methodology & "How It Works"

### 1. Image Classification (The "What")
The system uses a **weighted ensemble/hybrid approach**:
- **Primary**: Queries `Kaludi/food-category-classification-v2.0` (Food-101 model) - Best for prepared dishes.
- **Fallback**: If confidence is low, queries `google/efficientnet-b0` (ImageNet) - Best for raw ingredients.
- **Result**: Returns the highest-confidence match from either model.

### 2. The Q10 Spoilage Model (The "How Long")
Food degradation is a temperature-dependent chemical and biological process. The system implements the **Q10 Temperature Coefficient**:

$$Q_{10} = \left(\frac{R_2}{R_1}\right)^{\frac{10}{T_2 - T_1}}$$

- **Logic**: For many food items, the rate of spoilage increases by ~2.5x (Q10 factor) for every 10°C increase in temperature.
- **Reference**: We use 4°C (Standard Fridge Temp) as our baseline.

### 3. Humidity Correction
Microbial growth is accelerated by moisture. Our model applies a penalty/bonus factor where every 10% increase in humidity above 55% reduces the remaining shelf life by approximately 10%.

---

## 📊 Supported Foods & Baselines

| Category | Typical Fridge Life | Example Items | Scientific Logic |
|----------|---------------------|---------------|------------------|
| **Apple** | 21 days | Whole apples, apple pie | High fiber, low oxidation rate |
| **Banana** | 5 days | Fresh bananas | High Ethylene production |
| **Bread** | 7 days | Loaves, toast, bagels | Moisture loss & retrogradation |
| **Milk** | 7 days | Dairy products | Bacterial lactic acid production |
| **Pasta** | 4 days | Spaghetti, lasagna, ramen | High water activity |
| **Pizza** | 4 days | Leftover pizza | Fat oxidation & microbial risk |
| **Burger** | 3 days | Hamburgers, hot dogs | Processed meat degradation |
| **Sushi** | 1 day | Sushi, sashimi | Raw fish; high bacterial risk |
| **Meat** | 4 days | Steak, pork, chicken | Protein degradation |
| **Fish** | 2 days | Salmon, tuna, seafood | Rapid enzymatic breakdown |
| **Egg** | 28 days | Eggs, omelettes | Natural protective shell coating |
| **Vegetable** | 7 days | Salads, fresh veggies | Respiration & transpiration |
| **Fruit** | 5 days | Berries, cut fruit | Cell wall breakdown |
| **Rice** | 5 days | Cooked rice, risotto | *Bacillus cereus* risk |
| **Cake** | 5 days | Cakes, desserts | High sugar stability |

---

## 🚀 Quick Start & Installation

### Option 1: Docker (Recommended)

1. **Clone the repo** and create `.env`:
   ```bash
   git clone https://github.com/yourusername/AI-Food-Spoilage-Estimator.git
   cd AI-Food-Spoilage-Estimator
   cp .env.example .env
   # Edit .env and add your Hugging Face token
   ```

2. **Run with Docker**:
   ```bash
   docker-compose up --build
   ```

3. **Open** [http://localhost:5000](http://localhost:5000)

### Option 2: Local Python

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set environment variable**:
   ```bash
   # Windows PowerShell
   $env:HUGGINGFACE_API_TOKEN = "hf_your_token_here"
   
   # Linux/Mac
   export HUGGINGFACE_API_TOKEN="hf_your_token_here"
   ```

3. **Run the apps**:
   - **Flask (Main UI)**: `python app_flask.py`
   - **Streamlit (Metrics)**: `streamlit run app.py`

---

## ⚙️ Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HUGGINGFACE_API_TOKEN` | Yes | Get free at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

---

## 📂 Project Structure

```text
food-spoilage-estimator/
├── app_flask.py           # Production-ready Flask server
├── app.py                 # Interactive Streamlit dashboard
├── config.py              # Centralized configuration (Constants, Mappings)
├── Dockerfile             # Container definition
├── docker-compose.yml     # Multi-container orchestration
├── requirements.txt       # Python dependencies
├── .env.example           # Environment template
├── models/
│   ├── classifier.py      # Hybrid AI Vision logic (Hugging Face)
│   └── shelf_life_predictor.py  # RandomForest ML Model
├── utils/
│   └── data_loader.py     # Research-based dataset generator
├── templates/
│   └── index.html         # Modern Glassmorphism UI
└── data/                  # Generated training data (gitignored)
```

---

## 📈 Future Roadmap

*   **IoT Integration**: Connect with real-time temperature/humidity sensors.
*   **Mobile App**: Develop a Flutter cross-platform mobile application.
*   **Barcode Scanning**: Fetch brand-specific data via OpenFoodFacts.
*   **Multi-Item Detection**: Enable YOLOv8 for detecting multiple items in one fridge scan.

---

## 🤝 References & Context

Created for educational and research purposes in AI-driven sustainability.

- **Food-101**: Bossard et al., "Food-101 – Mining Discriminative Components", ECCV 2014
- **Q10 Coefficient**: Labuza, T.P. (1982). "Shelf Life Dating of Foods"
- **USDA**: FoodKeeper Application Data (2024)

## 📜 License

MIT License - Free for educational and commercial use.

---

> [!TIP]
> **Pro Tip for PPT**: When presenting, highlight the **Q10 Coefficient** — it's the "Secret Sauce" that differentiates this project from simple image labelers by adding a layer of genuine food science.
