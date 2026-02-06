import os
import requests
import pandas as pd
import hopsworks
from datetime import datetime

# 1. Connect to Hopsworks
api_key = os.getenv("HOPSWORKS_API_KEY")
project = hopsworks.login(api_key_value=api_key, project="aqi_quality_fs")
fs = project.get_feature_store()
fg = fs.get_feature_group(name="karachi_aqi_features", version=7)

# 2. Fetch LIVE Data (OpenWeatherMap + OpenMeteo Backup)
def get_live_weather_data():
    # Try OpenWeatherMap first (Best for Live)
    aqi_key = os.getenv('AQI_API_KEY')
    lat, lon = 24.8607, 67.0011
    
    try:
        # A. Fetch Pollution (Live)
        pol_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={aqi_key}"
        p_res = requests.get(pol_url).json()
        
        # B. Fetch Weather (Live)
        wea_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={aqi_key}"
        w_res = requests.get(wea_url).json()
        
        if not p_res.get('list') or 'main' not in w_res:
            raise ValueError("OpenWeatherMap returned empty data.")

        # C. Parse Data
        poll = p_res['list'][0]
        comps = poll['components']
        wea = w_res['main']
        wind = w_res['wind']
        
        current_dt = pd.to_datetime(datetime.now()) # Use Server Time (UTC)

        row = {
            'datetime': current_dt,
            'aqi': int(poll['main']['aqi']),
            'humidity': int(wea['humidity']),
            'pm2_5': float(comps['pm2_5']),
            'pm10': float(comps['pm10']),
            'nitrogen_dioxide': float(comps['no2']),
            'ozone': float(comps['o3']),
            'sulphor_dioxide': float(comps['so2']),
            'carbon_monooxide': float(comps['co']),
            'temp_c': float(wea['temp']),
            'wind_speed_kph': float(wind['speed'] * 3.6),
            'precipitation_mm': float(w_res.get('rain', {}).get('1h', 0.0)),
            
            # Date Features
            'year': current_dt.year,
            'month': current_dt.month,
            'day': current_dt.day,
            'hour': current_dt.hour,
            'day_of_week': current_dt.dayofweek,
            
            # Interactions
            'temp_humid_interaction': float(wea['temp'] * wea['humidity']),
            'wind_pollution_interaction': float((wind['speed'] * 3.6) * comps['pm2_5'])
        }
        
        return pd.DataFrame([row])

    except Exception as e:
        print(f"⚠️ Live Fetch Error: {e}")
        return pd.DataFrame()

# 3. Execution Logic
if __name__ == "__main__":
    print("🚀 Starting Hourly Feature Pipeline...")
    
    new_data = get_live_weather_data()
    
    if not new_data.empty:
        print(f"✅ Fetched Live Data for: {new_data['datetime'].iloc[0]}")
        
        # Insert into Hopsworks
        fg.insert(new_data, write_options={"wait_for_job": False})
        print("✅ Live data uploaded to Feature Store.")
        
    else:
        print("❌ Failed to fetch live data. Exiting.")
