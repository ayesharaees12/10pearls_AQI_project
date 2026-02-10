import hopsworks
import pandas as pd

API_KEY = "DGdCrO6zZkBc42ad.L1OpxZ0oXVXn9Md1YPllyCIu8XOj0Q3epPmrVZeoJikQ9mNdJTe9uqtsZAQQ93FJ"
PROJECT_NAME = "aqi_quality_fs"
FG_NAME = "karachi_aqi_features"
FG_VERSION = 7 # Bump version if previous v3 is broken / has wrong schema

project = hopsworks.login(
    host="c.app.hopsworks.ai",
    project=PROJECT_NAME,
    api_key_value=API_KEY
)

fs = project.get_feature_store()

# Use get_or_create  but bump version if schema is wrong
fg_new = fs.get_or_create_feature_group(
    name=FG_NAME,
    version=FG_VERSION,
    description="Karachi hourly AQI features",
    primary_key=["datetime"],
    event_time="datetime",
    online_enabled=False,
   
)

print(f"Using feature group: {FG_NAME} v{FG_VERSION}")

# Load your CSV
csv_path = "data/karachi_aqi_features.csv"
print(f"Reading from: {csv_path}")
df = pd.read_csv(csv_path)

df.columns = df.columns.str.lower().str.replace(r'[\s\-]', '_', regex=True)

# Convert to real datetime → this becomes TIMESTAMP in Hopsworks
df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

# Drop rows with invalid dates (NaT)
df = df.dropna(subset=['datetime'])



print("Columns in DataFrame:", df.columns.tolist())
print("datetime dtype:", df["datetime"].dtype)          # Should show datetime64[ns]
print("datetime sample:", df["datetime"].head(3).tolist())
df['day_of_week'] = pd.to_numeric(df['day_of_week'], errors='coerce').astype('Int64')   # Int64 allows nulls if any

fg_new.insert(
    df,
    write_options={
        "wait_for_job": True,
        "overwrite": False,   
        "upsert": True        # good for time-series / append-like behavior
    }
)

print(f"Successfully inserted {len(df)} rows.")