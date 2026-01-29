import os
import hopsworks
import pandas as pd
import requests
import numpy as np
from datetime import datetime

# 1. Connect to Hopsworks
project = hopsworks.login(
    project="aqi_quality_fs", 
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
fs = project.get_feature_store()
fg = fs.get_feature_group(name="karachi_aqi_features", version=7)

# 2. Fetch Real-Time Data
def get_weather_data():
    api_key = os.getenv('AQI_API_KEY')
    lat, lon = 24.8607, 67.0011
    
    # API Call 1: Pollution
    poll_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    poll_res = requests.get(poll_url).json()
    
    # API Call 2: Weather
    weath_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={api_key}"
    weath_res = requests.get(weath_url).json()

    if not poll_res.get('list') or 'main' not in weath_res:
        print("API Error")
        return pd.DataFrame()

    poll_data = poll_res['list'][0]
    comps = poll_data['components']
    
    row = {
        'datetime': pd.to_datetime(poll_data['dt'], unit='s'),
        'aqi': float(poll_data['main']['aqi']),
        'pm2_5': float(comps['pm2_5']),
        'pm10': float(comps['pm10']),
        'nitrogen_dioxide': float(comps['no2']),
        'ozone': float(comps['o3']),
        'sulphor_dioxide': float(comps['so2']),
        'carbon_monooxide': float(comps['co']),
        'temp_c': float(weath_res['main']['temp']),
        'humidity': float(weath_res['main']['humidity']),
        'wind_speed_kph': float(weath_res['wind']['speed'] * 3.6),
        'precipitation_mm': float(weath_res.get('rain', {}).get('1h', 0.0)),
        
        'year': pd.to_datetime(poll_data['dt'], unit='s').year,
        'month': pd.to_datetime(poll_data['dt'], unit='s').month,
        'day': pd.to_datetime(poll_data['dt'], unit='s').day,
        'hour': pd.to_datetime(poll_data['dt'], unit='s').hour,
        'day_of_week': pd.to_datetime(poll_data['dt'], unit='s').dayofweek,
        
        'temp_humid_interaction': float(weath_res['main']['temp'] * weath_res['main']['humidity']),
        'wind_pollution_interaction': float((weath_res['wind']['speed'] * 3.6) * comps['pm2_5'])
    }
    
    return pd.DataFrame([row])

# 3. Feature Logic
def calculate_advanced_features(combined_df):
    combined_df = combined_df.sort_values(by='datetime').reset_index(drop=True)
    
    combined_df['aqi_lag_1'] = combined_df['aqi'].shift(1)
    combined_df['aqi_change'] = combined_df['aqi'].diff()
    combined_df['aqi_pct_change'] = combined_df['aqi'].pct_change()
    combined_df['aqi_roll_max_24h'] = combined_df['aqi'].rolling(window=24, min_periods=1).max()
    combined_df['target_aqi_24h'] = np.nan
    
    return combined_df

if __name__ == "__main__":
    # A. Get New Data
    new_data_df = get_weather_data()
    
    if new_data_df.empty:
        print("Failed: No data fetched.")
    else:
        print(f"Fetched new data for: {new_data_df['datetime'].values[0]}")

        # B. Get History
        print("Reading history from Hopsworks...")
        try:
            history_df = fg.read()
        except Exception as e:
            print(f"History read warning: {e}")
            history_df = pd.DataFrame()

        # --- C. THE PERMANENT FIX: Universal DateTime Sanitizer ---
        # This block forces both dataframes to have the exact same time format (Naive UTC)
        # It prevents the 'Dimension 1' error by rebuilding the columns from scratch.
        if not history_df.empty:
            # 1. Reset Index to be safe
            history_df = history_df.reset_index(drop=True)
            
            # 2. Filter columns FIRST to avoid shape mismatches
            relevant_cols = new_data_df.columns.tolist()
            history_df = history_df[relevant_cols]
            
            # 3. Force History to Naive UTC
            history_df['datetime'] = pd.to_datetime(history_df['datetime'], utc=True).dt.tz_localize(None)

        # 4. Force New Data to Naive UTC (Matches History)
        new_data_df['datetime'] = pd.to_datetime(new_data_df['datetime'], utc=True).dt.tz_localize(None)
        # ------------------------------------------------------------

        # D. Stitch
        if not history_df.empty:
            combined_df = pd.concat([history_df, new_data_df], axis=0, ignore_index=True)
        else:
            combined_df = new_data_df
            
        combined_df = combined_df.drop_duplicates(subset=['datetime'], keep='last')

        # E. Calculate & Upload
        print("Calculating Features...")
        processed_df = calculate_advanced_features(combined_df)
        upload_df = processed_df.tail(1)

        print(f"Uploading row. AQI: {upload_df['aqi'].values[0]}")
        
        fg.insert(
            upload_df,
            write_options={"wait_for_job": False}
        )
        print("Success: Pipeline run finished!")
   


    
          
      

        
       
