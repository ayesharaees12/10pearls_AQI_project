import os, time, requests, hopsworks
import pandas as pd
from datetime import datetime
import pytz
import hopsworks
print("Hopsworks Version:", hopsworks.__version__)

# --- CONFIG ---
HW_API_KEY = os.getenv("HOPSWORKS_API_KEY")
AQI_API_KEY = os.getenv("AQI_API_KEY")
LAT, LON = 24.8607, 67.0011

# --- HELPER: RETRY LOGIC ---
def retry(func, retries=3, delay=10):
    for i in range(retries):
        try: return func()
        except Exception as e:
            print(f"⚠️ Attempt {i+1} failed: {e}")
            time.sleep(delay)
    raise Exception(f"❌ Failed after {retries} retries")

# --- 1. GET DATA ---
def get_data():
    if not AQI_API_KEY: raise ValueError("Missing AQI_API_KEY")
    
    # Fetch Data
    p = requests.get(f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={AQI_API_KEY}", timeout=10).json()['list'][0]
    w = requests.get(f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&units=metric&appid={AQI_API_KEY}", timeout=10).json()
    
    # Timezone & Parse
    dt = datetime.now(pytz.timezone('Asia/Karachi')).replace(tzinfo=None)
    
    return pd.DataFrame([{
        'datetime': dt,
        'aqi': int(p['main']['aqi']),
        'humidity': int(w['main']['humidity']),
        'pm2_5': float(p['components']['pm2_5']),
        'pm10': float(p['components']['pm10']),
        'nitrogen_dioxide': float(p['components']['no2']),
        'ozone': float(p['components']['o3']),
        'sulphor_dioxide': float(p['components']['so2']),
        'carbon_monooxide': float(p['components']['co']),
        'temp_c': float(w['main']['temp']),
        'wind_speed_kph': float(w['wind']['speed'] * 3.6),
        'precipitation_mm': float(w.get('rain', {}).get('1h', 0.0)),
        'year': dt.year, 'month': dt.month, 'day': dt.day, 'hour': dt.hour,
        'day_of_week': int(dt.weekday()), # Forced Int
        'temp_humid_interaction': float(w['main']['temp'] * w['main']['humidity']),
        'wind_pollution_interaction': float(w['wind']['speed'] * 3.6 * p['components']['pm2_5'])
    }])

# --- 2. PROCESS & UPLOAD ---
def run_pipeline():
    print("🚀 Pipeline Started...")
    
    # Login
    project = hopsworks.login(api_key_value=HW_API_KEY)
    fg = project.get_feature_store().get_feature_group(name="karachi_aqi_features", version=7)
    
    # Get New Data
    new_df = retry(get_data)
    new_row = new_df.iloc[0].to_dict()
    
    # Safe Merge History (Prevents Shape Mismatch)
    try:
        hist_df = fg.read(read_options={"use_arrow_flight": False}).tail(48)
        if not hist_df.empty:
            hist_df['datetime'] = pd.to_datetime(hist_df['datetime']).dt.tz_localize(None)
            hist_df = hist_df[new_df.columns.intersection(hist_df.columns)] # Align cols
            data = hist_df.to_dict('records') + [new_row]
        else: data = [new_row]
    except: data = [new_row]

    # Calculate Features
    df = pd.DataFrame(data).sort_values('datetime').reset_index(drop=True)
    df['aqi_lag_1'] = df['aqi'].shift(1)
    df['aqi_roll_max_24h'] = df['aqi'].rolling(24, min_periods=1).max()
    df['aqi_change'] = df['aqi'].diff()
    df['aqi_pct_change'] = df['aqi'].pct_change()
    df['target_aqi_24h'] = df['aqi'].shift(-24)

    # Final Clean (Fill NaNs & Cast Types)
    final_row = df.tail(1).fillna(0)
    int_cols = ['day_of_week', 'aqi', 'humidity', 'year', 'month', 'day', 'hour']
    final_row[int_cols] = final_row[int_cols].astype('int64')

    print(f"✅ Uploading AQI: {final_row['aqi'].values[0]}")
    retry(lambda: fg.insert(final_row, write_options={"wait_for_job": False}))
    print("✅ Success!")

if __name__ == "__main__":
    try: run_pipeline()
    except Exception as e: 
        print(f"❌ Error: {e}")
        exit(1)
