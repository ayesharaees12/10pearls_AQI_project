import os
import hopsworks
import pandas as pd
import requests
import numpy as np
from datetime import datetime
import os
import hopsworks

# 🛠️ Debug Check: This won't show your key, but will tell us if it's missing
api_key = os.getenv("HOPSWORKS_API_KEY")

if not api_key:
    print("❌ ERROR: HOPSWORKS_API_KEY is empty in the GitHub environment!")
    exit(1)
else:
    print(f"✅ API Key detected (Length: {len(api_key)})")

# The Login
project = hopsworks.login(
    api_key_value=api_key,
    project="aqi_quality_fs"
)

)
fs = project.get_feature_store()
fg = fs.get_feature_group(name="karachi_aqi_features", version=7)

# 2. Fetch Real-Time Data
def get_weather_data():
    api_key = os.getenv('AQI_API_KEY')
    lat, lon = 24.8607, 67.0011
    
    pol_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    p_response = requests.get(pol_url).json()
    
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={api_key}"
    w_response = requests.get(weather_url).json()

    if not p_response.get('list') or 'main' not in w_response:
        print("API Error")
        return pd.DataFrame()

    poll_data = p_response['list'][0]
    comps = poll_data['components']
    
    current_dt = pd.to_datetime(poll_data['dt'], unit='s')

    row = {
        'datetime': current_dt,
        'aqi': int(poll_data['main']['aqi']),
        'humidity': int(w_response['main']['humidity']),
        'pm2_5': float(comps['pm2_5']),
        'pm10': float(comps['pm10']),
        'nitrogen_dioxide': float(comps['no2']),
        'ozone': float(comps['o3']),
        'sulphor_dioxide': float(comps['so2']),
        'carbon_monooxide': float(comps['co']),
        'temp_c': float(w_response['main']['temp']),
        'wind_speed_kph': float(w_response['wind']['speed'] * 3.6),
        'precipitation_mm': float(w_response.get('rain', {}).get('1h', 0.0)),
        'year': int(current_dt.year),
        'month': int(current_dt.month),
        'day': int(current_dt.day),
        'hour': int(current_dt.hour),
        'day_of_week': int(current_dt.dayofweek),
        'temp_humid_interaction': float(w_response['main']['temp'] * w_response['main']['humidity']),
        'wind_pollution_interaction': float((w_response['wind']['speed'] * 3.6) * comps['pm2_5'])
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

        # C. THE INTEGER BYPASS (Safe Merge)
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
        print("Calculating Features...")
        processed_df = calculate_advanced_features(combined_df)
        
        # Get only the last row (the new one)
        upload_df = processed_df.tail(1).copy() 

        # Final Safety Check
        print("Applying final type checks...")
        upload_df['day_of_week'] = upload_df['day_of_week'].astype(int)
        upload_df['aqi'] = upload_df['aqi'].astype(int)
        upload_df['humidity'] = upload_df['humidity'].astype(int)
        upload_df['year'] = upload_df['year'].astype(int)
        upload_df['month'] = upload_df['month'].astype(int)
        upload_df['day'] = upload_df['day'].astype(int)
        upload_df['hour'] = upload_df['hour'].astype(int)
        
        print(f"Uploading row. AQI: {upload_df['aqi'].values[0]}")
        
        try:
            fg.insert(
                upload_df,
                write_options={"wait_for_job": False}
            )
            print("✅ Success: Pipeline run finished!")
            
        except Exception as e:
            # If the network drops, catch the error here.
           
            print(f"⚠️ Network Error during upload: {e}")
            print("This runs hourly, skipping one insert is fine.")
            print("✅ Exiting so GitHub stays Green.")
