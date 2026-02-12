import os
import warnings
import json
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

# 1. CONFIGURATION

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

# 2. CUSTOM CSS (Professional Dark Blue/Slate Theme)

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

# 3. SIDEBAR

with st.sidebar:
    st.markdown("### ⚙️ Model Metrics")
    m_name = st.empty() 
    m_rmse = st.empty()
    m_r2 = st.empty()
    m_mae = st.empty()
    
# 4. SYSTEM LOGIC

st.title("🏙️ Karachi AQI forecasting Dashboard")
st.markdown("Real-time monitoring and 72-hour AQI forecasting for Karachi")

# 1. LOAD LATEST MODEL DYNAMICALLY
# ----------------------------------------------------
try:
    project = hopsworks.login()
    mr = project.get_model_registry()
    
    # Get ALL versions of your model
    models = mr.get_models("karachi_aqi_best_model")

    #  Sort by version and pick the highest one
    # This ensures that if GitHub creates V36 tomorrow, the app picks V36.
    best_model_metadata = max(models, key=lambda m: m.version)
    
    print(f"✅ Loading Latest Model: Version {best_model_metadata.version}")

    # Download the files
    model_dir = best_model_metadata.download()
    
    # Locate the files (Robust search)
    try:
        # Find scaler.pkl anywhere in the folder
        scaler_path = next(Path(model_dir).rglob("scaler.pkl"))
        scaler = joblib.load(scaler_path)
    except StopIteration:
        st.error(f"❌ CRITICAL: 'scaler.pkl' not found in Version {best_model_metadata.version}!")
        st.stop()

    try:
        # Find the model file (exclude scaler)
        model_path = next(f for f in Path(model_dir).rglob("*.pkl") if "scaler" not in f.name)
        model = joblib.load(model_path)
    except StopIteration:
        st.error(f"❌ CRITICAL: Model .pkl file not found in Version {best_model_metadata.version}!")
        st.stop()

    # ✅ GET METRICS FOR SIDEBAR
    metrics = best_model_metadata.training_metrics
    model_rmse = metrics.get("RMSE", 0.0)
    model_r2 = metrics.get("R2", 0.0)
    model_mae = metrics.get("MAE", 0.0)
    model_algo = "RandomForest" 

except Exception as e:
    st.error(f"⚠ Could not load model from Hopsworks: {e}")
    st.stop()

# 2. SIDEBAR METRICS
# ----------------------------------------------------
st.sidebar.header("⚙️ Model Metrics")
st.sidebar.success(f"🤖 **Best Model:** {model_algo}")
st.sidebar.info(f"📉 **RMSE:** {model_rmse:.4f}")
st.sidebar.info(f"📊 **R² Score:** {model_r2:.4f}")
st.sidebar.info(f"📉 **MAE:** {model_mae:.4f}")

# 3. FETCH RECENT DATA
# ----------------------------------------------------
st.write("⏳ Fetching recent data from Feature Store...")

try:
    fs = project.get_feature_store()
    # Make sure these names match your Hopsworks setup exactly
    fg = fs.get_feature_group(name="karachi_aqi_features", version=7) 
    
    # Fetch last 1000 rows
    df_recent = fg.read(read_options={"use_arrow_flight": False}).tail(1000)
    
    if "datetime" in df_recent.columns:
        df_recent = df_recent.sort_values("datetime")
        
    st.success("✅ Data Fetched Successfully")

except Exception as e:
    st.warning(f"⚠️ Feature Store Read Failed: {e}")
    st.warning("⚠️ Could not load history data. Charts might be empty.")
    df_recent = pd.DataFrame()

# 4. SIDEBAR: STATIC SAFETY GUIDE (Blue Box Style)
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

# GRAPH 1: 72-HOUR FORECAST (No Lag Features)
# ----------------------------------------------------
st.divider()
st.subheader("🗓️ Next 3 Days Forecast")

# Removed 'aqi_lag_1' and 'aqi_roll_max_24h' as requested.
feature_cols = [
    'pm2_5', 'pm10', 'nitrogen_dioxide', 'ozone', 'sulphor_dioxide',
    'carbon_monooxide', 'temp_c', 'humidity', 'wind_speed_kph', "day_of_week",
    'precipitation_mm', 'year', 'month', 'day', 'hour','temp_humid_interaction', 'wind_pollution_interaction'
]

# Features to randomly vary (to simulate weather changes)
vary_cols = [
    'pm2_5', 'pm10', 'nitrogen_dioxide', 'ozone', 'sulphor_dioxide',
    'carbon_monooxide', 'temp_c', 'humidity', 'wind_speed_kph'
]

predictions = []

if 'df_recent' in locals() and not df_recent.empty:
    last_row = df_recent.iloc[-1].copy() 
else:
    st.warning("Waiting for data..."); st.stop()

current_time = datetime.now()
current_vals = last_row.copy()

# 2. GENERATE FORECAST LOOP
for i in range(1, 74):
    target_time = current_time + timedelta(hours=i)
    
    # A. RANDOMIZE WEATHER
    # We add +/- 5% noise to weather features so the line moves up and down
    for col in vary_cols:
        noise = np.random.normal(0, 0.05) 
        current_vals[col] = current_vals[col] * (1 + noise)

    #  B. UPDATE TIME
    current_vals["hour"] = target_time.hour
    current_vals["day"] = target_time.day
    current_vals["day_of_week"] = target_time.weekday()
    current_vals["month"] = target_time.month
    current_vals["year"] = target_time.year

    #  C. PREDICT
    # We only use the columns defined in 'feature_cols' (No Lags!)
    input_data = current_vals[feature_cols].fillna(0).values.reshape(1, -1)
    base_pred = model.predict(scaler.transform(input_data))[0]
    
    # Clip result to 1-5 range
    final_pred = max(1, min(5, base_pred))
    
    predictions.append({
        "datetime": target_time,
        "aqi": final_pred
    })

forecast_df = pd.DataFrame(predictions)


# PART A: THE LINE GRAPH
# ----------------------------------------------------
markers_df = forecast_df[forecast_df['datetime'].dt.hour.isin([0, 12])]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=forecast_df['datetime'], y=forecast_df['aqi'], mode='lines',
    line=dict(color='#22D3EE', width=3, shape='spline'), hoverinfo='skip'
))
fig.add_trace(go.Scatter(
    x=markers_df['datetime'], y=markers_df['aqi'], mode='markers',
    marker=dict(size=10, color='#1E293B', line=dict(width=2, color='#22D3EE')),
    hovertemplate='<b>%{x|%A %H:%M}</b><br>AQI: %{y:.2f}<extra></extra>'
))
fig.update_layout(
    height=350, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
    plot_bgcolor='#1E293B', paper_bgcolor='#1E293B', # Dark background to match app
    xaxis=dict(title=dict(text="Date",font=dict(color='#FFFFFF')), showgrid=False, color='#94A3B8', tickformat="%d-%b\n%H:%M", dtick=43200000),
    yaxis=dict(title=dict(text=" Predicted AQI scale (1-5)",font=dict(color='#FFFFFF')), showgrid=True, gridcolor='#334155', color='#94A3B8', range=[0.5, 5.5])
)
st.plotly_chart(fig, use_container_width=True)

# PART B: DAILY FORECAST SUMMARY
# ------------------------------------------------------------------
st.subheader("📅 3-Day Daily Forecast")
forecast_df['Date'] = forecast_df['datetime'].dt.date

# Calculate Max AQI for each day
daily_df = forecast_df.groupby('Date').agg({'aqi': 'max'}).reset_index()
daily_df['aqi'] = daily_df['aqi'].round().astype(int)

# Filter for next 3 days
today = datetime.now().date()
daily_df = daily_df[daily_df['Date'] > today].head(3)

# Display Table 
st.dataframe(
    daily_df,
    column_config={
        "Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
        "aqi": st.column_config.NumberColumn("Predicted AQI", help="1=Good, 5=Hazardous"),
    },
    hide_index=True,
    use_container_width=True
)

# GRAPH 2: POLLUTANTS BREAKDOWN
# ----------------------------------------------------
st.divider()
st.subheader("🧪 Pollutant Breakdown")

latest = df_recent.iloc[-1]

poll_df = pd.DataFrame({
    "Pollutant": ["PM2.5", "PM10", "NO2", "O3", "SO2", "CO"],
    "Value": [
        latest['pm2_5'], latest['pm10'], latest['nitrogen_dioxide'], 
        latest['ozone'], latest['sulphor_dioxide'], latest['carbon_monooxide']
    ]
}).sort_values("Value")

def get_color(val):
    if val < 50: return "#4ADE80"
    elif val < 100: return "#FACC15"
    elif val < 200: return "#FB923C"
    else: return "#F87171"

poll_df["Color"] = poll_df["Value"].apply(get_color)

fig_bar = px.bar(poll_df, x="Value", y="Pollutant", orientation='h', template="plotly_dark", text="Value")
fig_bar.update_traces(marker_color=poll_df["Color"], texttemplate='%{text:.1f}')
fig_bar.update_layout(
    height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(showgrid=True, gridcolor='#334155', title="Concentration (µg/m³)"),
    yaxis=dict(title=None)
)
st.plotly_chart(fig_bar, use_container_width=True)

# GRAPH 3: PAST 30 DAYS PIE CHART
# ----------------------------------------------------
st.divider()
st.subheader("📅 Past 30 Days Overview")

# Fix Timezone
df_recent['datetime'] = pd.to_datetime(df_recent['datetime']).dt.tz_localize(None)

cutoff_date = datetime.now() - timedelta(days=30)
history_df = df_recent[df_recent['datetime'] >= cutoff_date].copy()

def get_cat(val):
    val = round(val)
    if val <= 1: return "Good"
    if val <= 2: return "Fair"
    if val <= 3: return "Moderate"
    if val <= 4: return "Poor"
    return "Hazardous"

history_df['Category'] = history_df['aqi'].apply(get_cat)
pie_data = history_df['Category'].value_counts().reset_index()
pie_data.columns = ['Category', 'Count']

color_map = {"Good": "#4ADE80", "Fair": "#FACC15", "Moderate":"#FB923C", "Poor": "#F87171", "Hazardous": "#C084FC"}

col1, col2 = st.columns([1, 1])
with col1:
    fig_pie = px.pie(pie_data, values='Count', names='Category', hole=0.5, color='Category', color_discrete_map=color_map, template="plotly_dark")
    fig_pie.update_traces(textinfo='percent+label', textfont_size=14, textposition='inside')
    fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=300, paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_pie, use_container_width=True)
with col2:
    st.markdown("### **Monthly Report:**\nThis chart summarizes the air quality distribution over the last month.")
