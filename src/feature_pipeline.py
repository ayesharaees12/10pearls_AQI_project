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
    api_key = os.getenv('AQI_API_KEY')
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat=24.86&lon=67.00&appid={api_key}"
    
    response = requests.get(url)
    data = response.json()
    
    # Extract data
    raw_list = data.get('list', [])
    if not raw_list:
        return pd.DataFrame()

    df = pd.DataFrame(raw_list)
    
    # Flatten components (pm2_5, no2, etc.)
    components = pd.json_normalize(df['components'])
    df = pd.concat([df, components], axis=1)
    
    # Create human-readable aqi and datetime
    df['aqi'] = df['main'].apply(lambda x: x['aqi'])
    # Crucial: Convert to datetime object for Hopsworks Event Time
    df['datetime'] = pd.to_datetime(df['dt'], unit='s')
    
    # Cleanup: Drop nested columns that cause issues in Feature Stores
    df = df.drop(columns=['main', 'components', 'dt'])
    
    return df

