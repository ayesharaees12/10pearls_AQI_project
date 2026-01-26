import os
import warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import hopsworks
import requests
from datetime import datetime,timedelta,timezone
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
import tensorflow as tf
from pathlib import Path
import re
import json 
import shutil

os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
warnings.filterwarnings("ignore")
# CONFIG (change these if needed)



PROJECT_NAME = "aqi_quality_fs"
MODEL_NAME = "karachi_aqi_best_model"          # Your registered model name
FG_NAME = "karachi_aqi_features"
FG_VERSION = 7                            # Your feature group version
AQI_API_KEY = "6f8154a1a8bf4c5197fe70b0da282b35" # OpenWeather API key
LAT, LON = 24.8607, 67.0011                # Karachi coordinates
# ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not HOPSWORKS_API_KEY:
    st.error("HOPSWORKS_API_KEY not found in .env file")
    st.stop()

st.set_page_config(page_title="Pearls AQI Predictor", layout="wide", page_icon="🌬️")

# Title & Intro (matches your PDF)
st.title("Pearls AQI Predictor")
st.markdown("""
Let’s predict the **Air Quality Index (AQI)** in **Karachi** for the **next 3 days**, using a **100% serverless stack**.
""")
# --- DEBUG CHECKPOINTS ---
st.write("### 🔍 System Check")
checkpoint = st.empty()

# 1. Login
checkpoint.info("Step 1: Logging into Hopsworks...")
try:
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
    checkpoint.success("Step 1: Connected to Hopsworks!")
except Exception as e:
    st.error(f"Login Failed: {e}")
    st.stop()

# 2. Load Model from your local folder
checkpoint.info("Step 2: Loading local model.pkl...")
if os.path.exists("model.pkl") and os.path.exists("scaler.pkl"):
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    checkpoint.success("Step 2: Model & Scaler loaded from folder!")
else:
    st.error("Step 2 Failed: model.pkl or scaler.pkl not found in your aqi_folder!")
    st.stop()

# 3. Fetch Data
checkpoint.info("Step 3: Fetching data from Feature Store...")
try:
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)
    # Use .head(5) to test if it's a data size issue
    df_recent = fg.read().tail(10) 
    if df_recent.empty:
        st.error("Feature Group is empty!")
        st.stop()
    checkpoint.success(f"Step 3: Data fetched! ({len(df_recent)} rows)")
except Exception as e:
    st.error(f"Step 3 Failed: {e}")
    st.stop()

# If it reaches here, the UI will show!
checkpoint.empty() 
st.success("✅ System ready! Displaying Dashboard...")
# --- DASHBOARD UI ---
st.success("✅ System ready! Displaying Dashboard...")

# 1. LIVE AQI HEADER
st.subheader("📍 Current Air Quality (Live)")
def get_live_aqi():
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={AQI_API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if data.get("list"):
            return data["list"][0]["main"]["aqi"]
    except:
        return None
    return None
live_aqi = get_live_aqi() # Uses your OpenWeather function
if live_aqi:
    col1, col2 = st.columns(2)
    with col1:
        st.metric("AQI Index", live_aqi)
    with col2:
        colors = {1: "🟢 Good", 2: "🟡 Fair", 3: "🟠 Moderate", 4: "🔴 Poor", 5: "💀 Hazardous"}
        st.write(f"### Status: {colors.get(live_aqi, 'Unknown')}")

# 2. THE PREDICTION ENGINE
# 2. THE PREDICTION ENGINE
st.divider()
st.subheader("📈 72-Hour Forecast")

# We use the df_recent we just fetched in 'Step 3'
if not df_recent.empty:
    # Start from the last known data point
    last_row = df_recent.iloc[-1].copy()
    current_time = pd.to_datetime(last_row["datetime"])
    
    predictions = []
    
    # Define features exactly as used during model training
    feature_cols = [
        'pm2_5', 'pm10', 'nitrogen_dioxide', 'ozone', 'sulphor_dioxide',
        'carbon_monooxide', 'temp_c', 'humidity', 'wind_speed_kph', "day_of_week",
        'precipitation_mm', 'year', 'month', 'day', 'hour','temp_humid_interaction', 'wind_pollution_interaction',
        'aqi_lag_1', "aqi_roll_max_24h"
    ]

    # Run loop for 72 hours (3 days)
    current_time=datetime.now()
    for i in range(1, 74):
        # 1. Prepare input
        input_data = last_row[feature_cols].values.reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        
        # 2. Predict base AQI
        base_pred = model.predict(input_scaled)[0]
        
        # 3. Apply ±5% Random Variation
        # Uniform distribution between 0.95 and 1.05
        variation = np.random.uniform(0.95, 1.05)
        final_pred = base_pred * variation
        
        # 4. Store prediction
        target_time = current_time + timedelta(hours=i)
        predictions.append({
            "datetime": target_time, 
            "aqi": final_pred
        })
        
        # 5. Update last_row for the NEXT iteration (Recursive step)
        # We update the 'lag' and time features so the next prediction knows 
        # what happened in the previous hour
        last_row["aqi_lag_1"] = final_pred
        last_row["hour"] = target_time.hour
        last_row["day"] = target_time.day
        last_row["day_of_week"] = target_time.weekday()
        last_row["month"]=target_time.month
        last_row["year"]=target_time.year

    forecast_df = pd.DataFrame(predictions)

    # --- 5. CALENDAR-BASED DAILY FORECAST TABLE ---
# --- 5. CALENDAR-BASED DAILY FORECAST TABLE ---
st.divider()
st.subheader("🗓️ 3-Day Daily Outlook (Karachi)")

if not forecast_df.empty:
    # 1. Ensure datetime is correct and extract the Date
    forecast_df['Date'] = pd.to_datetime(forecast_df['datetime']).dt.date
  
    
    # 2. Group by Date to get the Daily Average
    daily_summary = forecast_df.groupby('Date')['aqi'].mean().reset_index()
    
    # 3. Define the Logic for Status labels
    def get_summary_text(val):
        rounded_aqi=round(val,2)
        if val <= 1.5: return f"{rounded_aqi}  🟢 Good"
        if val <= 2.5: return f"{rounded_aqi}  🟡 Fair"
        if val <= 3.5: return f"{rounded_aqi}  🟠 Moderate"
        if val <= 4.5: return f"{rounded_aqi} 🔴 Poor"
        return f"{rounded_aqi}  💀 Hazardous"

    # 4. Use the first 3 available dates from the forecast
    today_date = datetime.now().date()
    future_df = daily_summary[daily_summary["Date"] > today_date].copy()
        
        # 5. Take the first 3 days and apply the function
    final_table = future_df.head(3).copy()
        
        # IMPORTANT: This name must match the function name above!
    final_table["Air Quality Condition"] = final_table["aqi"].apply(get_summary_text)
        
        # 6. Display the Clean Table
    display_table = final_table[['Date', 'Air Quality Condition']]
            
    st.dataframe(
            display_table,
            hide_index=True,
            use_container_width=True
        )

        # 7. Trend Chart
    fig = px.line(forecast_df, x="datetime", y="aqi", 
                    title="Hourly Prediction Trend (Jan 2026)",
                    template="plotly_dark")
    fig.update_traces(line_color='#00d1b2')
    st.plotly_chart(fig, use_container_width=True)