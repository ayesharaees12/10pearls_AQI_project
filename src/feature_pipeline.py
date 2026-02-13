import time
import os
import requests
import pandas as pd
import numpy as np
import hopsworks
from datetime import datetime
import pytz

# ---------------------------------------------------------
#RETRY DECORATOR (The "Try Again" Logic)
# ---------------------------------------------------------
def retry_operation(func, max_retries=3, delay=10, operation_name="Operation"):
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as e:
            print(f"⚠️ {operation_name} failed (Attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                print(f"⏳ Waiting {delay}s before retrying...")
                time.sleep(delay)
            else:
                print(f"❌ {operation_name} failed after {max_retries} attempts.")
                raise e

# ---------------------------------------------------------
# 1. CONNECT TO HOPSWORKS
# ---------------------------------------------------------
def connect_to_hopsworks():
    hw_api_key = os.getenv("HOPSWORKS_API_KEY")
    if not hw_api_key:
        raise ValueError("HOPSWORKS_API_KEY is missing")
    
    project = hopsworks.login(api_key_value=hw_api_key, project="aqi_quality_fs")
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="karachi_aqi_features", version=7)
    return fg

# ─────────────────────────────────────────────────────────────
# 2. DATA FETCHER (OPENWEATHERMAP ONLY)
# ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------
# 2. DATA FETCHER (OPENWEATHERMAP ONLY)
# ---------------------------------------------------------
def get_live_weather_data():
    lat, lon = 24.8607, 67.0011
    aqi_key = os.getenv('AQI_API_KEY')
    
    if not aqi_key:
        raise ValueError("AQI_API_KEY is missing")
            
    # A. Fetch Pollution (Set timeout to prevent hanging)
    p_res = requests.get(f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={aqi_key}", timeout=10)
    
    # B. Fetch Weather
    w_res = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={aqi_key}", timeout=10)
    
    # C. Check for Errors (Critical Step)
    if p_res.status_code != 200 or w_res.status_code != 200:
        # If API fails, Raise an error so the 'retry_operation' can catch it and try again
        raise ConnectionError(f"API Failed. Status: {p_res.status_code}/{w_res.status_code}")
        
    # D. Parse Data (Only happens if status is 200)
    poll = p_res.json()['list'][0]
    wea = w_res.json()['main']
    wind = w_res.json()['wind']
            
    # Timezone Handling
    pk_tz = pytz.timezone('Asia/Karachi')
    current_dt = datetime.now(pk_tz).replace(tzinfo=None) 

    # E. Build the Row
    row = {
        'datetime': current_dt,
        'aqi': int(poll['main']['aqi']),
        'humidity': int(wea['humidity']),
        'pm2_5': float(poll['components']['pm2_5']),
        'pm10': float(poll['components']['pm10']),
        'nitrogen_dioxide': float(poll['components']['no2']),
        'ozone': float(poll['components']['o3']),
        'sulphor_dioxide': float(poll['components']['so2']),
        'carbon_monooxide': float(poll['components']['co']),
        'temp_c': float(wea['temp']),
        'wind_speed_kph': float(wind['speed'] * 3.6),
        'precipitation_mm': float(w_res.json().get('rain', {}).get('1h', 0.0)),
        'year': current_dt.year,
        'month': current_dt.month,
        'day': current_dt.day,
        'hour': current_dt.hour,
        'day_of_week': current_dt.weekday(),
        'temp_humid_interaction': float(wea['temp'] * wea['humidity']),
        'wind_pollution_interaction': float((wind['speed'] * 3.6) * poll['components']['pm2_5']),
        # IMPORTANT: Initialize these as 0. The calculate_features function will fill them later.
        'aqi_lag_1': 0,
        'aqi_roll_max_24h': 0
    }
        
    # F. Return the DataFrame
    return pd.DataFrame([row])
def calculate_features(fg, new_df):
    # Read last 48 hours of history to calculate lags
    print("📥 Fetching history for lag calculation...")
    history_df = fg.read(read_options={"use_arrow_flight": False}).tail(48)
    
    if history_df.empty:
        return new_data_df

    # Combine
    combined_df = pd.concat([history_df, new_df], axis=0, ignore_index=True)
    combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])
    combined_df = combined_df.sort_values('datetime').reset_index(drop=True)

    # Calculate Lags
    combined_df['aqi_lag_1'] = combined_df['aqi'].shift(1)
    combined_df['aqi_roll_max_24h'] = combined_df['aqi'].rolling(window=24, min_periods=1).max()
    
    # Return ONLY the new row (the last one)
    return combined_df.tail(1)

# ---------------------------------------------------------
# 4. MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    try:
        print("🚀 Pipeline Started...")

        # A. Login
        fg = retry_operation(connect_to_hopsworks, operation_name="Hopsworks Login")
        print("✅ Logged in.")

        # B. Get Data
        new_df = retry_operation(get_live_weather_data, operation_name="Weather Fetch")
        print(f"✅ Data Fetched: AQI {new_df['aqi'].values[0]}")

        # C. Feature Engineering
        final_df = calculate_features(fg, new_df)

        # D. Upload
        def upload_task():
            fg.insert(final_df, write_options={"wait_for_job": False})
        
        retry_operation(upload_task, max_retries=5, delay=20, operation_name="Upload")
        print("✅ Data Ingested Successfully!")

    except Exception as e:
        print(f"❌ PIPELINE CRASHED: {e}")
        exit(1) # This ensures GitHub sends you an email alert!
