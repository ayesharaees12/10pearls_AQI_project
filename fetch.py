import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
import time

# Your API keys (replace with real ones)
AQI_API_KEY = "6f8154a1a8bf4c5197fe70b0da282b35"
CITY = "Karachi"

# ================= GEO =================
def fetch_coordinates():
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={CITY}&limit=1&appid={AQI_API_KEY}"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    return data[0]["lat"], data[0]["lon"]

# ================= AQI =================
def fetch_aqi(lat, lon, start_ts, end_ts):
    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={lat}&lon={lon}&start={start_ts}&end={end_ts}&appid={AQI_API_KEY}"
    )
    r = requests.get(url)
    r.raise_for_status()
    return r.json().get("list", [])

# ================= WEATHER (SMART) =================
def fetch_weather_open_meteo(lat, lon, start_date, end_date):
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
        "&timezone=UTC"
    )
    r = requests.get(url)
    r.raise_for_status()
    return r.json()["hourly"]
# ================= MAIN PIPELINE =================
lat, lon = fetch_coordinates()

end_date = datetime.now(timezone.utc).date()
start_date = end_date - timedelta(days=180)

aqi_data = fetch_aqi(
    lat,
    lon,
    int(datetime.combine(start_date,datetime.min.time()).timestamp()),
    int(datetime.combine(end_date,datetime.max.time()).timestamp())
)

# Map AQI by UNIX timestamp
aqi_map = {item["dt"]: item for item in aqi_data}

weather=fetch_weather_open_meteo(
    lat,
    lon,
    start_date.isoformat(),
    end_date.isoformat()
)
rows = [] 

for i,time_str in enumerate(weather["time"]):
    ts=int(
        datetime.strptime(time_str,"%Y-%m-%dT%H:%M")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    aqi_match = aqi_map.get(ts)
    if not aqi_match:
        continue
        
        
    rows.append({
        "timestamp": time_str.replace("T"," "),
        "aqi": aqi_match["main"]["aqi"],
        "pm25": aqi_match["components"]["pm2_5"],
        "pm10": aqi_match["components"]["pm10"],
        "no2": aqi_match["components"]["no2"],
        "o3": aqi_match["components"]["o3"],
        "so2": aqi_match["components"]["so2"],
        "co": aqi_match["components"]["co"],
        "temp_c": weather["temperature_2m"][i],
        "humidity": weather["relative_humidity_2m"][i],
        "wind_speed_kph": weather["wind_speed_10m"][i],
        "precipitation_mm": weather["precipitation"][i]
    })
      

# ================= SAVE =================
df = pd.DataFrame(rows)
df.to_csv("data/first_aqi_data.csv", index=False)
print("✅ Dataset Ready:", df.shape)
