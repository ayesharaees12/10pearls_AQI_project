import os
import hopsworks
import pandas as pd
import requests
from datetime import datetime
import json

# 1. Connect to Hopsworks
project = hopsworks.login(
    project="aqi_quality_fs", 
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
fs = project.get_feature_store()
fg = fs.get_feature_group(name="karachi_aqi_features", version=7)

# 2. Fetch Real-Time Data (OpenWeatherMap)
API_KEY = os.getenv("AQI_API_KEY")
LAT, LON = 24.8607, 67.0011

def get_weather_data():
    # ... (Your logic to get 'temp', 'humidity', 'pm2_5', etc.) ...
    # This must return a DataFrame with the exact same columns as your Feature Group
    # Example:
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
    res = requests.get(url).json()
    # ... process JSON into a pandas DataFrame (df) ...
    return df

new_data_df = get_weather_data()

# 3. Insert into Feature Store
if not new_data_df.empty:
    fg.insert(new_data_df)
    print("✅ Success: New data uploaded to Hopsworks!")
else:
    print("⚠ Warning: No data fetched.")
