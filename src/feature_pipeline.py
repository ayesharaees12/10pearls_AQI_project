import os
import hopsworks
import pandas as pd
import requests
from datetime import datetime

# 1. Connect to Hopsworks
project = hopsworks.login(
    project="aqi_quality_fs", 
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
fs = project.get_feature_store()
fg = fs.get_feature_group(name="karachi_aqi_features", version=7)

# 2. Fetch Real-Time Data (Simple Version)
def get_weather_data():
    api_key = os.getenv('AQI_API_KEY')
    lat, lon = 24.8607, 67.0011
    
    # Call 1: Pollution (for PM2.5, NO2, AQI)
    poll_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    poll_res = requests.get(poll_url).json()
    
    # Call 2: Weather (for Temp, Humidity, Wind)
    # This is necessary because Pollution API doesn't give Temp/Wind
    weath_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={api_key}"
    weath_res = requests.get(weath_url).json()

    # Safety Check
    if not poll_res.get('list') or 'main' not in weath_res:
        print("API Error or Empty Response")
        return pd.DataFrame()

    # Extract Data
    poll_data = poll_res['list'][0]
    comps = poll_data['components']
    
    # Create the DataFrame row
    # Notice: WE REMOVED THE ROLLING/LAG COLUMNS HERE
    row = {
        'datetime': pd.to_datetime(poll_data['dt'], unit='s'),
        'aqi': poll_data['main']['aqi'],
        'pm2_5': comps['pm2_5'],
        'pm10': comps['pm10'],
        'nitrogen_dioxide': comps['no2'],
        'ozone': comps['o3'],
        'sulphor_dioxide': comps['so2'],
        'carbon_monooxide': comps['co'],
        'temp_c': weath_res['main']['temp'],
        'humidity': weath_res['main']['humidity'],
        'wind_speed_kph': weath_res['wind']['speed'] * 3.6,
        'precipitation_mm': weath_res.get('rain', {}).get('1h', 0),
        
        # Simple Date Features (Safe to keep!)
        'year': pd.to_datetime(poll_data['dt'], unit='s').year,
        'month': pd.to_datetime(poll_data['dt'], unit='s').month,
        'day': pd.to_datetime(poll_data['dt'], unit='s').day,
        'hour': pd.to_datetime(poll_data['dt'], unit='s').hour,
        'day_of_week': pd.to_datetime(poll_data['dt'], unit='s').dayofweek,
        
        # Simple Math Features (Safe to keep!)
        'temp_humid_interaction': weath_res['main']['temp'] * weath_res['main']['humidity'],
        'wind_pollution_interaction': (weath_res['wind']['speed'] * 3.6) * comps['pm2_5']
    }
    
    return pd.DataFrame([row])

if __name__ == "__main__":
    df = get_weather_data()
    
    if not df.empty:
        # Since we changed the columns (removed rolling), 
        # Hopsworks might complain if the Feature Group schema doesn't match.
        # But for insertion, it often just fills missing columns with Nulls.
        fg.insert(df, write_options={"wait_for_job": False})
        print("Success: Data pushed to Hopsworks!")
    else:
        print("Failed: No data fetched.")


