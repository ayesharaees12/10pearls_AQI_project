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

import pandas as pd
import requests

def get_weather_data():
    api_key = os.getenv('AQI_API_KEY')
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat=24.86&lon=67.00&appid={api_key}"
    
    response = requests.get(url)
    data = response.json()
    
    # 1. Create the 'df' variable
    # This extracts the data from the OpenWeather JSON structure
    df = pd.DataFrame(data['list']) 
    
    # 2. Extract specific components (example)
    df['aqi'] = df['main'].apply(lambda x: x['aqi'])
    df['datetime'] = pd.to_datetime(df['dt'], unit='s')
    
    # Now 'df' exists and can be returned added to Hopsworks!")
    return df
else:
    print("⚠ Warning: No data fetched.")
