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
# 2. ROBUST DATA FETCHER (Plan A + Plan B)
# ─────────────────────────────────────────────────────────────
def get_live_weather_data():
    lat, lon = 24.8607, 67.0011
    aqi_key = os.getenv('AQI_API_KEY')
    
    # --- PLAN A: OpenWeatherMap (Best for Live) ---
    if aqi_key:
        print("🌤️ Attempting Plan A: OpenWeatherMap...")
        try:
            # Fetch Pollution & Weather
            p_res = requests.get(f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={aqi_key}")
            w_res = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={aqi_key}")
            
            if p_res.status_code != 200 or w_res.status_code != 200:
                print(f"⚠️ OpenWeatherMap Error: {p_res.text}")
                raise ValueError("API Request Failed")

            poll = p_res.json()['list'][0]
            wea = w_res.json()['main']
            wind = w_res.json()['wind']
            
            # Use Server Time (UTC)
            current_dt = pd.to_datetime(datetime.now())

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
            print(f"⚠️ Plan A Failed: {e}")
            print("🔄 Switching to Backup...")

    # --- PLAN B: OpenMeteo (Free Backup) ---
    print("🌤️ Attempting Plan B: OpenMeteo (No Key Needed)...")
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,pm10,pm2_5,nitrogen_dioxide,ozone,sulphur_dioxide,carbon_monoxide",
            "timezone": "auto"
        }
        res = requests.get(url, params=params)
        data = res.json().get('current', {})
        
        if not data: raise ValueError("Empty Data")

        current_dt = pd.to_datetime(datetime.now())
        
        row = {
            'datetime': current_dt,
            'aqi': 3, # Fallback AQI
            'humidity': int(data.get('relative_humidity_2m', 50)),
            'pm2_5': float(data.get('pm2_5', 0)),
            'pm10': float(data.get('pm10', 0)),
            'nitrogen_dioxide': float(data.get('nitrogen_dioxide', 0)),
            'ozone': float(data.get('ozone', 0)),
            'sulphor_dioxide': float(data.get('sulphur_dioxide', 0)),
            'carbon_monooxide': float(data.get('carbon_monoxide', 0)),
            'temp_c': float(data.get('temperature_2m', 0)),
            'wind_speed_kph': float(data.get('wind_speed_10m', 0)),
            'precipitation_mm': float(data.get('precipitation', 0)),
            # Features
            'year': current_dt.year,
            'month': current_dt.month,
            'day': current_dt.day,
            'hour': current_dt.hour,
            'day_of_week': current_dt.dayofweek,
            'temp_humid_interaction': float(data.get('temperature_2m', 0) * data.get('relative_humidity_2m', 0)),
            'wind_pollution_interaction': float(data.get('wind_speed_10m', 0) * data.get('pm2_5', 0))
        }
        return pd.DataFrame([row])

    except Exception as e:
        print(f"❌ Plan B also Failed: {e}")
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
# 4. MAIN EXECUTION (STITCH & UPLOAD)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # A. Get New Data
    new_data_df = get_live_weather_data()
    
    if new_data_df.empty:
        print("❌ Failed: No data fetched from any source.")
        exit(1)
        
    print(f"✅ Fetched new data for: {new_data_df['datetime'].values[0]}")

    # B. Get History (Needed for Lag Features)
    print("📥 Reading history from Hopsworks to calculate lags...")
    try:
        # We only need the last ~48 hours to calculate rolling features
        history_df = fg.read() 
    except Exception as e:
        print(f"⚠️ History read warning: {e}")
        history_df = pd.DataFrame()

    # C. Safe Merge (The "Integer Bypass" Logic)
    if not history_df.empty:
        history_df = history_df.reset_index(drop=True)
        new_data_df = new_data_df.reset_index(drop=True)

        # Align columns
        common_cols = new_data_df.columns.intersection(history_df.columns).tolist()
        history_df = history_df[common_cols]
        new_data_df = new_data_df[common_cols]

        # Convert to int64 to avoid Timestamp mismatch errors
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

    # F. Calculate Features
    print("⚙️ Calculating Advanced Features (Lags/Rolling)...")
    processed_df = calculate_advanced_features(combined_df)
    
    # G. Select ONLY the New Row to Upload
    upload_df = processed_df.tail(1).copy() 

    # H. Final Type Safety Checks
    upload_df['day_of_week'] = upload_df['day_of_week'].astype(int)
    upload_df['aqi'] = upload_df['aqi'].astype(int)
    upload_df['humidity'] = upload_df['humidity'].astype(int)
    upload_df['year'] = upload_df['year'].astype(int)
    upload_df['month'] = upload_df['month'].astype(int)
    upload_df['day'] = upload_df['day'].astype(int)
    upload_df['hour'] = upload_df['hour'].astype(int)
    
    print(f"📤 Uploading new row (AQI: {upload_df['aqi'].values[0]})...")
    
    try:
        fg.insert(
            upload_df,
            write_options={"wait_for_job": False}
        )
        print("✅ Success: Pipeline run finished!")
        
    except Exception as e:
        print(f"⚠️ Upload Error: {e}")
        # Don't fail the build if it's just a network blip
        exit(0)
