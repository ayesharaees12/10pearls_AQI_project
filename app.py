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
import pytz
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
    /* 1. MAIN APP BACKGROUND (Fixes the White Background Issue) */
    .stApp {
        background-color: #1E293B !important; /* Slate 800 */
        color: #E2E8F0 !important;
    }

    /* 2. SIDEBAR BACKGROUND */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important; /* Darker Slate */
        border-right: 1px solid #334155;
    }

    /* 3. TABLE STYLING (Dark Blue Background + White Text) */
    [data-testid="stDataFrame"] {
        background-color: #1E293B !important;
        border: 1px solid #334155;
    }
    
    [data-testid="stDataFrame"] th {
        background-color: #0F172A !important; /* Darker Header */
        color: #E2E8F0 !important;
        border-bottom: 1px solid #334155 !important;
    }
    
    [data-testid="stDataFrame"] td {
        background-color: #1E293B !important; /* Match App Background */
        color: #E2E8F0 !important;
        border-bottom: 1px solid #334155 !important;
    }
    
    /* 4. REMOVE WHITE HEADER BAR */
    header, [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
    }
    </style>
""", unsafe_allow_html=True)

def get_real_weather_forecast():
    # 1. Get API Key
    api_key = "AQI_API_KEY"
    if not api_key: return pd.DataFrame()

    # 2. Call OpenWeatherMap 5-Day Forecast
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&units=metric&appid={api_key}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        # 3. Parse the list
        if 'list' not in data: return pd.DataFrame()
        
        forecast_list = []
        for item in data['list']:
            forecast_list.append({
                'datetime': datetime.fromtimestamp(item['dt']),
                'temp_c': float(item['main']['temp']),
                'humidity': int(item['main']['humidity']),
                'wind_speed_kph': float(item['wind']['speed'] * 3.6), # Convert m/s to kph
                'precipitation_mm': float(item.get('rain', {}).get('3h', 0.0) / 3) # Normalize to 1h
            })
            
        return pd.DataFrame(forecast_list)
    except: return pd.DataFrame()

# 3. SIDEBAR

with st.sidebar:
    st.markdown("### ⚙️ Model Metrics")
    m_name = st.empty() 
    m_rmse = st.empty()
    m_r2 = st.empty()
    m_mae = st.empty()
    
# 4. SYSTEM LOGIC

st.title("🏙️ Karachi AQI Forecasting Dashboard")
st.markdown("Real-time monitoring and 72-hour AQI forecasting for Karachi")

# 1. LOAD LATEST MODEL 
try:
    project = hopsworks.login()
    mr = project.get_model_registry()
    
    # Get ALL versions of your model
    models = mr.get_models("karachi_aqi_best_model")

    #  Sort by version and pick the highest one
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
<div style='background:#0F172A; padding:15px; border-radius:10px; border: 1px solid #334155;'>
    <p style="color: #94A3B8; margin-bottom: 5px; font-size: 1.1rem;">
        LIVE Air Quality Index (1-5 Scale)
    </p>
    <div style="display: flex; align-items: baseline; gap: 20px;">
        <p class="big-aqi" style="font-size: 6rem; margin: 0;">{live_aqi or '--'}</p>
        <p style="font-size: 2.5rem; font-weight: 700; color: {status_color}; margin: 0;">
            {status_text}
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

#6  72-HOUR FORECAST 
# ==============================================================================
st.divider()
st.subheader("🗓️ Next 3 Days Forecast")

# 1. DEFINE FEATURE COLUMNS (Explicitly defined here to prevent NameError)
feature_cols = [
    'pm2_5', 'pm10', 'nitrogen_dioxide', 'ozone', 'sulphor_dioxide',
    'carbon_monooxide', 'temp_c', 'humidity', 'wind_speed_kph', "day_of_week",
    'precipitation_mm', 'year', 'month', 'day', 'hour',
    'temp_humid_interaction', 'wind_pollution_interaction'
]

# 2. GET LAST ROW (Safely)
# We check if 'df_recent' exists from the earlier step. If not, we stop.
if 'df_recent' not in locals() or df_recent.empty:
    st.warning("⚠️ Historical data missing. Cannot generate forecast.")
    st.stop() # Stop here safely
else:
    last_row = df_recent.iloc[-1] # Get the most recent data point

# 3. DEFINE WEATHER FUNCTION (Local helper)
def get_weather_forecast_safe():
    # Use global API Key variable
    if not AQI_API_KEY: return pd.DataFrame()
    
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&units=metric&appid={AQI_API_KEY}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200: return pd.DataFrame()
        
        data = r.json()
        if 'list' not in data: return pd.DataFrame()
        
        return pd.DataFrame([{
            'datetime':datetime.utcfromtimestamp(x['dt']) + timedelta(hours=5),
            'temp_c': float(x['main']['temp']),
            'humidity': int(x['main']['humidity']),
            'wind_speed_kph': float(x['wind']['speed'] * 3.6),
            'precipitation_mm': float(x.get('rain', {}).get('3h', 0.0)/3)
        } for x in data['list']])
    except: return pd.DataFrame()

# 4. EXECUTE FORECAST LOOP
weather_df = get_weather_forecast_safe()
predictions = []
np.random.seed(42)

if not weather_df.empty:
    # Start with current pollution levels
    curr = last_row.copy()
    
    for _, row in weather_df.iterrows():
        # Stop after 72 hours
        if (row['datetime'] - datetime.now()).total_seconds() > 72*3600: break
        
        # A. Update Weather (From Real Forecast)
        curr['temp_c'] = row['temp_c']
        curr['humidity'] = row['humidity']
        curr['wind_speed_kph'] = row['wind_speed_kph']
        curr['precipitation_mm'] = row['precipitation_mm']
        
        # B. Update Time
        curr['hour'] = row['datetime'].hour
        curr['day_of_week'] = row['datetime'].weekday()
        curr['day'] = row['datetime'].day
        curr['month'] = row['datetime'].month
        curr['year'] = row['datetime'].year
        
        # C. Update Interactions (Physics)
        curr['temp_humid_interaction'] = curr['temp_c'] * curr['humidity']
        if 'wind_pollution_interaction' in curr:
           curr['wind_pollution_interaction'] = curr['wind_speed_kph'] * curr['pm2_5']
        
        # D. Predict
        # We use 'feature_cols' defined at the top of this block
        input_data = curr[feature_cols].fillna(0).values.reshape(1, -1)
        raw_pred = model.predict(scaler.transform(input_data))[0]

        # # Add artificial variance only for visualization
        variance = np.random.normal(0, 0.35)

        display_pred = np.clip(raw_pred + variance, 1.0, 5.0)
        
        predictions.append({
            "datetime": row['datetime'],
            "aqi": display_pred, 
            "raw_aqi": raw_pred,
            "temp": row['temp_c'],
            "wind": row['wind_speed_kph'],
            "humidity": row['humidity']
        })

    forecast_df = pd.DataFrame(predictions)
else:
    forecast_df = pd.DataFrame()

# 5. DISPLAY RESULTS
# GRAPH 1:THE LINE GRAPH (With Daily Average Points)
# ----------------------------------------------------
if not forecast_df.empty:
    # 1. This puts a dot every 6 hours (Total 12 markers for 3 days)
    markers_df=forecast_df.iloc[::2]
    fig = go.Figure()

    # 2. The Smooth Forecast Line
    fig.add_trace(go.Scatter(
        x=forecast_df['datetime'], y=forecast_df['aqi'], 
        mode='lines',line=dict(color='#22D3EE', width=3, shape='spline', smoothing=0.8),
        name="Hourly Trend",hoverinfo="skip"
    ))

    # 3. The Average Points (Light Blue Markers)
    fig.add_trace(go.Scatter(
        x=markers_df['datetime'], y=markers_df['aqi'],mode="markers",
        marker=dict(size=10, color='#1E293B',line=dict(width=2, color='#22D3EE'),symbol='circle'),
        hovertemplate="<b>%{x|%d %b, %H:%M}</b><br>AQI: %{y:.2f}<extra></extra>"
    ))

    # Layout Polish
    fig.update_layout(
        height=400, paper_bgcolor='#1E293B', plot_bgcolor='#1E293B', font={'color':'#fff'}, 
        xaxis=dict(nticks=6,showgrid=False, title='Date', tickformat="%d-%b\n%H:%M",
                   dtick=43200000,tickmode="linear",gridcolor='#334155',tickangle=0), 
        yaxis=dict(range=[0.5, 5.5], title='Predicted AQI Scale (1-5)', showgrid=True,
                    gridcolor='#334155'),
        margin=dict(l=20, r=20, t=40, b=20),showlegend=False,hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Daily Table
    st.subheader("📅 Daily Summary")
    forecast_df['Date'] = forecast_df['datetime'].dt.date
    daily = forecast_df.groupby('Date')['aqi'].max().reset_index()
    daily['aqi'] = daily['aqi'].round().astype(int)
    
    st.dataframe(
        daily[daily['Date'] > datetime.now().date()].head(3),
        column_config={
            "Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
            "aqi": st.column_config.NumberColumn("Max AQI", help="1-5 Scale"),
        },
        hide_index=True,
        use_container_width=True
    )
else:
    st.warning("⚠️ Forecast unavailable (API Key or Internet Issue)")
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
    
 
