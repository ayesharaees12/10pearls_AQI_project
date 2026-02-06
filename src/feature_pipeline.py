import os
import requests
import pandas as pd
import numpy as np
import hopsworks
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# 1. SETUP & LOGIN
# ─────────────────────────────────────────────────────────────
print("🚀 Starting Feature Pipeline...")

# Check Hopsworks Key
hw_api_key = os.getenv("HOPSWORKS_API_KEY")
if not hw_api_key:
    print("❌ ERROR: HOPSWORKS_API_KEY is missing.")
    exit(1)

# Login
try:
    project = hopsworks.login(api_key_value=hw_api_key, project="aqi_quality_fs")
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="karachi_aqi_features", version=7)
    print("✅ Connected to Hopsworks Feature Store.")
except Exception as e:
    print(f"❌ Critical Login Error: {e}")
    exit(1)

# ─────────────────────────────────────────────────────────────
# 2. DATA FETCHER (OPENWEATHERMAP ONLY)
# ─────────────────────────────────────────────────────────────
def get_live_weather_data():
    lat, lon = 24.8607, 67.0011
    aqi_key = os.getenv('AQI_API_KEY')
    
    if not aqi_key:
        print("❌ ERROR: AQI_API_KEY is missing.")
        return pd.DataFrame()

    print("🌤️ Fetching data from OpenWeatherMap...")
    try:
        # A. Fetch Pollution
        p_res = requests.get(f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={aqi_key}")
        # B. Fetch Weather
        w_res = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={aqi_key}")
        
        # Check for 200 OK
        if p_res.status_code != 200 or w_res.status_code != 200:
            print(f"❌ API Error: {p_res.text}")
            return pd.DataFrame()

        poll = p_res.json()['list'][0]
        wea = w_res.json()['main']
        wind = w_res.json()['wind']
        
        # Use Server Time (UTC)
        current_dt = pd.to_datetime(datetime.now())

        row = {
            'datetime': current_dt,
            'aqi': int(poll['main']['aqi']), # Already 1-5 scale
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
            # Date Features
            'year': current_dt.year,
            'month': current_dt.month,
            'day': current_dt.day,
            'hour': current_dt.hour,
            'day_of_week': current_dt.dayofweek,
            # Interactions
            'temp_humid_interaction': float(wea['temp'] * wea['humidity']),
            'wind_pollution_interaction': float((wind['speed'] * 3.6) * poll['components']['pm2_5'])
        }
        return pd.DataFrame([row])
        
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING LOGIC
# ─────────────────────────────────────────────────────────────
def calculate_advanced_features(combined_df):
    combined_df = combined_df.sort_values(by='datetime').reset_index(drop=True)
    
    combined_df['aqi_lag_1'] = combined_df['aqi'].shift(1)
    combined_df['aqi_change'] = combined_df['aqi'].diff()
    combined_df['aqi_pct_change'] = combined_df['aqi'].pct_change()
    combined_df['aqi_roll_max_24h'] = combined_df['aqi'].rolling(window=24, min_periods=1).max()
    combined_df['target_aqi_24h'] = np.nan
    
    return combined_df

# ─────────────────────────────────────────────────────────────
# 4. MAIN EXECUTION
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # A. Get New Data
    new_data_df = get_live_weather_data()
    
    if new_data_df.empty:
        print("❌ Failed: No data fetched.")
        exit(1) # Fail the GitHub Action so you get an email
        
    print(f"✅ Fetched new data for: {new_data_df['datetime'].values[0]}")

    # B. Get History (Needed for Lag Features)
    print("📥 Reading history to calculate lags...")
    try:
        history_df = fg.read() 
    except Exception as e:
        print(f"⚠️ History read warning: {e}")
        history_df = pd.DataFrame()

    # C. Safe Merge
    if not history_df.empty:
        history_df = history_df.reset_index(drop=True)
        new_data_df = new_data_df.reset_index(drop=True)

        common_cols = new_data_df.columns.intersection(history_df.columns).tolist()
        history_df = history_df[common_cols]
        new_data_df = new_data_df[common_cols]

        history_df['datetime'] = pd.to_datetime(history_df['datetime']).astype('int64')
        new_data_df['datetime'] = pd.to_datetime(new_data_df['datetime']).astype('int64')

    # D. Stitch
    if not history_df.empty:
        combined_df = pd.concat([history_df, new_data_df], axis=0, ignore_index=True)
    else:
        combined_df = new_data_df
    
    # E. Restore Datetime
    combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])
    combined_df = combined_df.drop_duplicates(subset=['datetime'], keep='last')

    # F. Calculate & Upload
    # ... (Keep all your previous code up to 'processed_df = ...')

    # F. Calculate & Upload
    print("⚙️ Calculating Features...")
    processed_df = calculate_advanced_features(combined_df)
    
    upload_df = processed_df.tail(1).copy() 

    # G. Final Checks
    upload_df['day_of_week'] = upload_df['day_of_week'].astype(int)
    upload_df['aqi'] = upload_df['aqi'].astype(int)
    upload_df['humidity'] = upload_df['humidity'].astype(int)
    upload_df['year'] = upload_df['year'].astype(int)
    upload_df['month'] = upload_df['month'].astype(int)
    upload_df['day'] = upload_df['day'].astype(int)
    upload_df['hour'] = upload_df['hour'].astype(int)
    
    print(f"📤 Uploading AQI: {upload_df['aqi'].values[0]}")
    
    # 🛡️ RETRY LOGIC (The Fix)
    import time
    max_retries = 3
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"🔄 Upload Attempt {attempt}/{max_retries}...")
            fg.insert(upload_df, write_options={"wait_for_job": False})
            print("✅ Success: Live data uploaded!")
            break # Stop looping if it works
            
        except Exception as e:
            print(f"⚠️ Upload failed on attempt {attempt}: {e}")
            
            if attempt < max_retries:
                print("⏳ Waiting 10 seconds before retrying...")
                time.sleep(10)
            else:
                print("❌ All attempts failed. Skipping this hour.")
                # We exit with 0 (Success) so GitHub doesn't send you a 'Failed' email
                # Missing one hour of data is okay.
                exit(0)
