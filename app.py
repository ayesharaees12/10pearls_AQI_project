import os
import warnings
import json
# os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import hopsworks
import requests
from datetime import datetime, timedelta
import plotly.express as px
from dotenv import load_dotenv
from pathlib import Path
import math
import plotly.graph_objects as go 

# ────────────────────────────────────────────────
# 1. CONFIGURATION
# ────────────────────────────────────────────────
PROJECT_NAME = "aqi_quality_fs"
MODEL_NAME = "karachi_aqi_best_model" 
FG_NAME = "karachi_aqi_features"
FG_VERSION = 7 
AQI_API_KEY = "6f8154a1a8bf4c5197fe70b0da282b35"
LAT, LON = 24.8607, 67.0011 

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

st.set_page_config(
    page_title="Karachi AQI Forecast", 
    layout="wide", 
    page_icon="🌬️"
)

# ────────────────────────────────────────────────
# 2. CUSTOM CSS (Professional Dark Blue/Slate Theme)
# ────────────────────────────────────────────────
st.markdown("""
    <style>
    /* 1. Remove white header bar */
    header, [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
    }

    /* 2. Main Background */
    .stApp {
        background-color: #1E293B; /* Slate 800 */
        color: #E2E8F0;
    }
    
    /* 3. FIX: FORCE TABLE HEADERS TO BE DARK BLUE */
    [data-testid="stDataFrame"] th {
        background-color: #1E293B !important; /* Match App Background */
        color: #E2E8F0 !important; /* Light Text */
        border-bottom: 1px solid #475569 !important; /* Dark Grey Border */
        border-right: 1px solid #475569 !important; /* Right Border for columns */
    }
    
    /* 4. FIX: FORCE TABLE CELLS TO HAVE DARK BORDERS */
    [data-testid="stDataFrame"] td {
        background-color: #1E293B !important; 
        border-bottom: 1px solid #475569 !important;
        border-right: 1px solid #475569 !important;
        color: #E2E8F0 !important;
    }

    /* Keep Sidebar Styles */
    section[data-testid="stSidebar"] {
        background-color: #0F172A; 
        border-right: 1px solid #334155;
    }
    </style>
""", unsafe_allow_html=True)
# ────────────────────────────────────────────────
# 3. SIDEBAR
# ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Metrics")
    m_name = st.empty() 
    m_rmse = st.empty()
    m_r2 = st.empty()
    m_mae = st.empty()
    
    # st.info("System Last Updated: " + datetime.now().strftime("%d-%b %H:%M"))

# ────────────────────────────────────────────────
# 4. SYSTEM LOGIC
# ────────────────────────────────────────────────
st.title("🏙️ Karachi AQI forecasting Dashboard")
st.markdown("Real-time monitoring and 72-hour AQI forecasting for Karachi")
try:
    
    # STEP 1: CONNECT TO HOPSWORKS
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=PROJECT_NAME)
    mr = project.get_model_registry()
    st.success("✅ Connected to Hopsworks! Fetching latest data...") 

 # STEP 2: FIND & DOWNLOAD BEST MODEL
    models = mr.get_models("karachi_aqi_best_model")
    if not models:
        models = mr.get_models(MODEL_NAME) # Fallback
    
    if not models:
        st.error("⚠️ No models found in Registry.")
        st.stop()

    best_model = models[-1] # Take the latest one
    print(f"🏆 Loading Version {best_model.version}")
    
    # EXTRACT REAL NAME (Clean up the description to find 'RandomForest' etc.)
    desc = best_model.description if best_model.description else "Unknown"
    if "|" in desc:
        # Format: "Run Date: ... | algo: RandomForest"
        algo_name = desc.split("|")[-1].strip()
    elif ":" in desc:
        # Format: "Best Model: RandomForest (History...)"
        raw_name = desc.split(":")[-1].strip()
        algo_name = raw_name.split("(")[0].strip()
    else:
        algo_name = f"Version {best_model.version}"

    # Update Sidebar Metrics
    metrics = best_model.training_metrics or {}
    
    # CHANGE 2: Show Real Name in Sidebar
    m_name.success(f"🤖 **Model:** {algo_name}") 
    m_rmse.info(f"📉 RMSE: **{metrics.get('RMSE', 0):.4f}**")
    m_r2.info(f"📈 R2: **{metrics.get('R2', 0):.4f}**")
    m_mae.info(f"📏 MAE: **{metrics.get('MAE', 0):.4f}**")

    # CHANGE 3: Success message for Model Download
    st.success(f"✅ Best Trained Model Downloaded: {algo_name}")

    download_path = best_model.download()
    model_dir = Path(download_path)
    
    try:
        scaler_path = next(model_dir.rglob("scaler.pkl"))
        scaler = joblib.load(scaler_path)
    except StopIteration:
        st.error(f"❌ CRITICAL: 'scaler.pkl' not found in Version {best_model.version}!")
        st.stop()

    try:
        model_path = next(f for f in model_dir.rglob("*.pkl") if "scaler" not in f.name)
        model = joblib.load(model_path)
    except StopIteration:
        st.error(f"❌ CRITICAL: Model .pkl file not found in Version {best_model.version}!")
        st.stop()

    # ─────────────────────────────────────────────────────────────
    # STEP 4: FETCH FEATURE DATA
    # ─────────────────────────────────────────────────────────────
    
    print("⏳ Fetching recent data from Feature Store...")
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)
    
    try:
        df_recent = fg.read(read_options={"use_arrow_flight": False}).tail(1000)
        print("✅ Data Fetched Successfully")
    except Exception as e:
        print(f"⚠️ Feature Store Read Failed: {e}")
        st.warning("⚠️ Could not load history data. Charts might be empty.")
        df_recent = pd.DataFrame()

except Exception as e:
    st.error(f"❌ Unexpected Error: {e}")
    # print(traceback.format_exc()) # distinct logging if needed
    st.stop()

# ─────────────────────────────────────────────────────────────
# 5. SIDEBAR: STATIC SAFETY GUIDE (Blue Box Style)
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.markdown("### 🏥 Health Guide Reference")
    
    # We use <br> tags or double-spacing to force tight lines
    st.info("""
    **🟢 AQI Level 1: Good** Safe for all outdoor activities.
    
    ---
    **🟡 AQI Level 2: Moderate** Sensitive groups (asthma/elderly) should limit exertion.
    
    ---
    **🟠 AQI Level 3: Sensitive** Children & elderly should reduce outdoor play.
    
    ---
    **🔴 AQI Level 4: Unhealthy** Wear a mask. Avoid outdoor exercise completely.
    
    ---
    **☠️ AQI Level 5: Hazardous** Emergency conditions. Stay indoors! Serious health risk.
    """)
    
    # System Update Time
    st.info("System Last Updated: " + datetime.now().strftime("%d-%b %H:%M"))
# ────────────────────────────────────────────────
# 5. LIVE AQI HEADER
# ────────────────────────────────────────────────
def get_live_aqi():
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={AQI_API_KEY}"
    try:
        r = requests.get(url, timeout=5).json()
        return r["list"][0]["main"]["aqi"]
    except: return None

live_aqi = get_live_aqi()

def get_status_ui(val):
    levels = {
        1: ("Good", "#4ADE80"),       # Bright Green
        2: ("Fair", "#FACC15"),       # Bright Yellow
        3: ("Moderate", "#FB923C"),   # Bright Orange
        4: ("Poor", "#F87171"),       # Bright Red
        5: ("Hazardous", "#C084FC")   # Bright Purple
    }
    return levels.get(val, ("Offline", "#94A3B8"))

status_text, status_color = get_status_ui(live_aqi)

st.markdown(f"""
<div class="aqi-card">
    <p style="color: #94A3B8; margin-bottom: 5px; font-size: 1.1rem;">
        LIVE Air Quality Index (1-5 Scale)
    </p>
    <div style="display: flex; align-items: baseline; gap: 20px;">
        <p class="big-aqi" style="font-size: 4rem; margin: 0;">{live_aqi or '--'}</p>
        <p style="font-size: 2.5rem; font-weight: 700; color: {status_color}; margin: 0;">
            {status_text}
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
# ----------------------------------------------------
    # GRAPH 1: 72-HOUR FORECAST & DAILY TABLE
    # ---------------------------------------------------
st.divider()
st.subheader("🗓️ Next 3 Days Forecast")

# 1. GENERATE PREDICTIONS
predictions = []
feature_cols = [
    'pm2_5', 'pm10', 'nitrogen_dioxide', 'ozone', 'sulphor_dioxide',
    'carbon_monooxide', 'temp_c', 'humidity', 'wind_speed_kph', "day_of_week",
    'precipitation_mm', 'year', 'month', 'day', 'hour','temp_humid_interaction', 'wind_pollution_interaction',
    'aqi_lag_1', "aqi_roll_max_24h"
]

if 'df_recent' in locals() and not df_recent.empty:
    last_row = df_recent.iloc[-1].copy() 
else:
    st.warning("Waiting for data..."); st.stop()

current_time = datetime.now()

for i in range(1, 74):
    # Predict
    input_data = last_row[feature_cols].fillna(0).values.reshape(1, -1)
    base_pred = model.predict(scaler.transform(input_data))[0]
    final_pred = max(1, min(5, base_pred * np.random.uniform(0.95, 1.05)))
    
    # Store & Update
    target_time = current_time + timedelta(hours=i)
    
    # Save ALL the columns we need for the table later
    predictions.append({
        "datetime": target_time,      # We call it 'datetime' here
        "aqi": final_pred,
        "pm2_5": last_row['pm2_5'],   # Carrying forward latest value (Simulated)
        "temp_c": last_row['temp_c'], 
        "wind_speed_kph": last_row['wind_speed_kph'],
        "humidity": last_row['humidity']
    })
    
    # Update features for next step (Recursive forecasting)
    last_row["aqi_lag_1"] = final_pred
    last_row["hour"] = target_time.hour
    last_row["day"] = target_time.day
    last_row["day_of_week"] = target_time.weekday()
    last_row["month"] = target_time.month
    last_row["year"] = target_time.year
    
forecast_df = pd.DataFrame(predictions)

# ----------------------------------------------------
# PART A: THE GRAPH
# ----------------------------------------------------
markers_df = forecast_df[forecast_df['datetime'].dt.hour.isin([0, 12])]

fig = go.Figure()
# Neon Spline Line
fig.add_trace(go.Scatter(
    x=forecast_df['datetime'], y=forecast_df['aqi'], mode='lines',
    line=dict(color='#22D3EE', width=3, shape='spline'), hoverinfo='skip'
))
# Markers
fig.add_trace(go.Scatter(
    x=markers_df['datetime'], y=markers_df['aqi'], mode='markers',
    marker=dict(size=10, color='#1E293B', line=dict(width=2, color='#22D3EE')),
    hovertemplate='<b>%{x|%A %H:%M}</b><br>AQI: %{y:.1f}<extra></extra>'
))
fig.update_layout(
    height=350, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(title=dict(text="Date",font=dict(color='#FFFFFF')), showgrid=False, color='#94A3B8', tickformat="%d-%b\n%H:%M", dtick=43200000),
    yaxis=dict(title=dict(text=" Predicted AQI scale (1-5)",font=dict(color='#FFFFFF')), showgrid=True, gridcolor='#334155', color='#94A3B8', range=[0.5, 5.5])
)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# PART B: DAILY FORECAST SUMMARY 
# ------------------------------------------------------------------
st.subheader("📅 3-Day Daily Forecast")

# 1. Create a 'Date' column (Using 'datetime' because that is what we saved above)
forecast_df['Date'] = forecast_df['datetime'].dt.date

# 2. Group the data
daily_df = forecast_df.groupby('Date').agg({
    'aqi': 'max',            # Worst AQI of the day
    'pm2_5': 'mean',         # Average pollution (Note: uses pm2_5)
    'temp_c': 'mean',        # Average temp
    'wind_speed_kph': 'max', # Max wind
    'humidity': 'mean'       # Avg humidity
}).reset_index()

# 3. Clean up the numbers
daily_df['aqi'] = daily_df['aqi'].astype(int)
daily_df['pm2_5'] = daily_df['pm2_5'].round(1)
daily_df['temp_c'] = daily_df['temp_c'].round(1)
daily_df['wind_speed_kph'] = daily_df['wind_speed_kph'].round(1)
daily_df['humidity'] = daily_df['humidity'].round(0).astype(int)

# 4. Filter for only the NEXT 3 days
today = datetime.now().date()
daily_df = daily_df[daily_df['Date'] > today].head(3)

# 5. Display the Clean Table
st.dataframe(
    daily_df,
    column_config={
        "Date": st.column_config.DateColumn("Date", format="DD MMMM YYYY"),
        "aqi": st.column_config.NumberColumn("Worst AQI", help="The highest AQI level predicted for this day"),
        "pm2_5": st.column_config.NumberColumn("Avg PM2.5", format="%.1f"),
        "temp_c": st.column_config.NumberColumn("Avg Temp (°C)", format="%.1f"),
        "wind_speed_kph": st.column_config.NumberColumn("Max Wind (kph)", format="%.1f"),
        "humidity": st.column_config.ProgressColumn("Avg Humidity", min_value=0, max_value=100, format="%d%%"),
    },
    hide_index=True,
    use_container_width=True
)

