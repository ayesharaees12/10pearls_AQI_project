# 10pearls_AQI_project
🌍 Karachi AQI Forecasting – End-to-End MLOps Pipeline

An end-to-end MLOps-based Air Quality Index (AQI) forecasting system designed to monitor and predict air quality levels for Karachi, Pakistan.
The project integrates real-time data ingestion, automated feature engineering, model training, deployment, and visualization using modern MLOps tools.

🚀 Project Overview

This project implements a fully automated machine learning pipeline that:

Continuously collects real-time and historical AQI & weather data

Performs data preprocessing and feature engineering

Stores engineered features in a Hopsworks Feature Store

Automatically retrains the model using GitHub Actions

Serves predictions through an interactive Streamlit dashboard

The system provides live AQI monitoring and 72-hour air quality forecasts.

🏗️ Architecture
APIs (AQI & Weather)
        ↓
Data Preprocessing & Feature Engineering
        ↓
Hopsworks Feature Store
        ↓
Model Training & Registry
        ↓
GitHub Actions (Automation)
        ↓
Streamlit Dashboard (Deployment)

📊 Data Sources
🌫️ Air Quality Data

Source: OpenWeatherMap Air Pollution API

Data: AQI levels and pollutant concentrations

PM2.5, PM10, CO, NO₂, SO₂, O₃

Frequency: Real-time updates

🌦️ Meteorological Data

Source: Open-Meteo Historical Archive

Features:

Temperature

Wind speed

Humidity

Atmospheric pressure

🧠 Machine Learning

Model Used: Random Forest Regressor

Prediction Target: AQI Level (1–5 scale)

Evaluation Metrics:

RMSE

MAE

R² Score

The best-performing model is automatically registered and fetched from the model registry.

⚙️ MLOps & Automation
🔹 Feature Store

Tool: Hopsworks

Stores cleaned and engineered features

Automatically validates schema using input_example

🔹 CI/CD Automation

Tool: GitHub Actions

Feature Pipeline:

Runs every hour to ingest fresh data

Training Pipeline:

Runs daily to retrain and register the best model

📈 Dashboard (Streamlit)

🔗 Live App:
👉 https://10pearlsaqiproject-x44hvmjyqmc3qohlndqqrf.streamlit.app/

Dashboard Features:

Live AQI status with health category

72-hour AQI forecast visualization

Pollutant concentration breakdown

Past 30 days AQI distribution

Model performance metrics

Health recommendations based on AQI level

🛠️ Tech Stack

Programming: Python

ML: Scikit-learn

MLOps: Hopsworks, GitHub Actions

APIs: OpenWeatherMap, Open-Meteo

Visualization: Streamlit, Matplotlib

Deployment: Streamlit Cloud

📁 Project Structure
├── data_pipeline/
│   ├── fetch_data.py
│   ├── feature_engineering.py
│
├── training_pipeline/
│   ├── train_model.py
│   ├── evaluate_model.py
│
├── app/
│   ├── app.py
│
├── .github/workflows/
│   ├── feature_pipeline.yml
│   ├── training_pipeline.yml
│
├── README.md
└── requirements.txt

✅ Key Highlights

End-to-end production-ready MLOps pipeline

Real-time data ingestion & monitoring

Automated retraining & model versioning

Feature store integration

Deployed and publicly accessible dashboard

📌 Future Improvements

Predict actual AQI numeric values (instead of category only)

Add alert notifications for hazardous AQI levels

Extend system to multiple cities




Ayesha
Computational Mathematics Undergraduate
Interests: Data Science, Machine Learning, MLOps
