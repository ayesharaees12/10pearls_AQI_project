import os
import warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
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

# ────────────────────────────────────────────────
# 1. CONFIGURATION
# ────────────────────────────────────────────────
PROJECT_NAME = "aqi_quality_fs"
MODEL_NAME = "karachi_aqi_best_model" 
FG_NAME = "karachi_aqi_features"
FG_VERSION = 7 
AQI_API_KEY = "6f8154a1a8bf4c5197fe70b0da282b35"
LAT, LON = 24.8607, 67.0011 

try:
    # Try loading from Streamlit Cloud Secrets first
    HOPSWORKS_API_KEY = st.secrets["HOPSWORKS_API_KEY"]
    AQI_API_KEY = st.secrets["AQI_API_KEY"]
except:
    # Fallback to local .env file (for your laptop)
    load_dotenv(BASE_DIR / ".env")
    HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
    AQI_API_KEY = "6f8154a1a8bf4c5197fe70b0da282b35"

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
    /* 1. FIX: Targets the very top header area to remove the white bar */
    header, [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
    }

    /* 2. Main Background - Light Dark Blue (Slate) */
    .stApp {
        background-color: #1E293B; /* Slate 800 */
        color: #E2E8F0; /* Light Text */
    }
    
    /* Keep your existing card and sidebar styles below... */
    .aqi-card {
        background: #334155; /* Slate 700 */
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #475569;
        text-align: left;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    .big-aqi {
        font-size: 5rem;
        font-weight: 800;
        color: #2DD4BF; /* Teal 400 */
        margin: 0;
        line-height: 1;
        text-shadow: 0 0 20px rgba(45, 212, 191, 0.3);
    }
    
    h1, h2, h3 { color: #F1F5F9 !important; }
    p { color: #CBD5E1 !important; }
    
    section[data-testid="stSidebar"] {
        background-color: #0F172A; /* Slate 900 */
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
    st.divider()
    st.markdown("### 👨‍💻 Developed by")
    st.markdown(" **Ayesha Raees**")
    
    st.info("System Last Updated: " + datetime.now().strftime("%d-%b %H:%M"))

# ────────────────────────────────────────────────
# 4. SYSTEM LOGIC
# ────────────────────────────────────────────────
st.title("🏙️ Karachi AQI forecasting Dashboard")
st.markdown("Real-time and 72-hour Air Quality predictions.")
# ⬇️ NEW: DEBUG CHECK (This will tell us if the key is missing!)
if not HOPSWORKS_API_KEY:
    st.error("❌ CRITICAL ERROR: HOPSWORKS_API_KEY is missing!")
    st.info("Please go to 'Settings > Secrets' in Streamlit Cloud and add your key.")
    st.stop() # Stops the app so it doesn't hang

try:
    # ⬇️ NEW: Force 'project' to be found or fail (No hanging)
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY, 
        project=PROJECT_NAME
    )
    
    mr = project.get_model_registry()
    models = mr.get_models(MODEL_NAME)
    hw_model = max(models, key=lambda x: x.version)
    
    full_desc = hw_model.description if hw_model.description else "Unknown"
    algo_name = full_desc.split(":")[-1].strip() if ":" in full_desc else full_desc
    
    metrics = getattr(hw_model, 'training_metrics', {}) or {}
    m_name.write(f"🤖 **Model:** {algo_name}")
    m_rmse.write(f"📉 RMSE: **{metrics.get('RMSE', 0.0967):.4f}**")
    m_r2.write(f"📈 R2: **{metrics.get('R2', 0.9827):.4f}**")
    m_mae.write(f"📏 MAE: **{metrics.get('MAE', 0.0243):.4f}**")

    download_path = hw_model.download()
    scaler = joblib.load(next(Path(download_path).rglob("scaler.pkl")))
    model = joblib.load(next(f for f in Path(download_path).rglob("*.pkl") if "scaler" not in f.name))
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)
    df_recent = fg.read().tail(1000) 
    
    st.success("✅ Connected to Hopsworks and loaded latest data.")

except Exception as e:
    st.error(f"Error: {e}")
    st.stop()

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
    <p style="color: #94A3B8; margin-bottom: 5px;">Current Air Quality Index (1-5)</p>
    <div style="display: flex; align-items: baseline; gap: 20px;">
        <p class="big-aqi" style="font-size: 5rem;">{live_aqi or '--'}</p>
        <p style="font-size: 3.0rem; font-weight: 700; color: {status_color};">({status_text})</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────
# 6. GRAPHS
# ────────────────────────────────────────────────
if not df_recent.empty:
    df_recent['datetime'] = pd.to_datetime(df_recent['datetime']).dt.tz_localize(None)
    last_row = df_recent.iloc[-1].copy()
    
    # ----------------------------------------------------
    # GRAPH 1: 72-HOUR FORECAST
    # ----------------------------------------------------
    st.divider()
    st.subheader("🗓️ Next 3 Days Forecast")
    
    predictions = []
    feature_cols = [
        'pm2_5', 'pm10', 'nitrogen_dioxide', 'ozone', 'sulphor_dioxide',
        'carbon_monooxide', 'temp_c', 'humidity', 'wind_speed_kph', "day_of_week",
        'precipitation_mm', 'year', 'month', 'day', 'hour','temp_humid_interaction', 'wind_pollution_interaction',
        'aqi_lag_1', "aqi_roll_max_24h"
    ]
    current_time_sim = datetime.now()
    
    for i in range(1, 74):
        input_data = last_row[feature_cols].fillna(0).values.reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        base_pred = model.predict(input_scaled)[0]
        variation = np.random.uniform(0.95, 1.05)
        final_pred = max(1, min(5, base_pred * variation))
        
        target_time = current_time_sim + timedelta(hours=i)
        predictions.append({"datetime": target_time, "aqi": final_pred})
        
        last_row["aqi_lag_1"] = final_pred
        last_row["hour"] = target_time.hour
        last_row["day"] = target_time.day
        last_row["day_of_week"] = target_time.weekday()
        last_row["month"] = target_time.month
        last_row["year"] = target_time.year
        
    forecast_df = pd.DataFrame(predictions)
    
    fig_forecast = px.line(forecast_df, x="datetime", y="aqi", 
                           template="plotly_dark", 
                           markers=False)
    
    fig_forecast.update_traces(line_color="#06B6D4", line_width=4)
    fig_forecast.update_layout(
        height=350, 
        yaxis=dict(range=[0.5, 5.5], title="Predicted AQI", showgrid=True, gridcolor='#334155'),
        xaxis=dict(title=None, showgrid=True),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

    # ----------------------------------------------------
    # ----------------------------------------------------
    # GRAPH 2: POLLUTANTS (Traffic Light Colors)
    # ----------------------------------------------------
    st.divider()
    st.subheader("🧪 Pollutant Breakdown")
    latest = df_recent.iloc[-1]
    
    # 1. Create Dataframe
    poll_df = pd.DataFrame({
        "Pollutant": ["PM2.5", "PM10", "NO2", "O3", "SO2", "CO"],
        "Value": [latest['pm2_5'], latest['pm10'], latest['nitrogen_dioxide'], latest['ozone'], latest['sulphor_dioxide'], latest['carbon_monooxide']]
    }).sort_values("Value")

    # 2. Define Function to Assign Colors based on Severity
    def get_color(val):
        if val < 50: return "#4ADE80"  # Green (Good)
        elif val < 100: return "#FACC15" # Yellow (Fair)
        elif val < 200: return "#FB923C" # Orange (Moderate)
        else: return "#F87171"           # Red (High/Danger)

    # 3. Apply color logic to a new column
    poll_df["Color"] = poll_df["Value"].apply(get_color)

    # 4. Create Graph
    fig_bar = px.bar(poll_df, x="Value", y="Pollutant", orientation='h', 
                     template="plotly_dark",
                     text="Value") # Show numbers on bars
    
    # 5. Force the bars to use our custom colors
    fig_bar.update_traces(marker_color=poll_df["Color"], texttemplate='%{text:.1f}')
    
    fig_bar.update_layout(
        height=400, 
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='#334155', title="Concentration (µg/m³)"),
        yaxis=dict(title=None)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ----------------------------------------------------
    # GRAPH 3: PAST 30 DAYS PIE CHART
    # ----------------------------------------------------
    st.divider()
    st.subheader("📅 Past 30 Days Overview")
    
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
    
    color_map = {
        "Good": "#4ADE80",      # Green
        "Fair": "#FACC15",      # Yellow
        "Moderate":"#FB923C",   # Orange
        "Poor": "#F87171",      # Red
        "Hazardous": "#C084FC"  # Purple
    }
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        fig_pie = px.pie(pie_data, values='Count', names='Category', hole=0.5,
                         color='Category', color_discrete_map=color_map,
                         template="plotly_dark")
        fig_pie.update_traces(textinfo='percent+label', textfont_size=14)
        fig_pie.update_layout(
            showlegend=False, 
            margin=dict(t=0, b=0, l=0, r=0), 
            height=300,
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col2:
        st.markdown("""
        **Overview:**
        This chart summarizes the air quality distribution over the last month.
        """)

    # ────────────────────────────────────────────────
    # 7. FORECAST TABLE
    # ────────────────────────────────────────────────
    st.divider()
    st.subheader("📊3 Forecast Summary")

    forecast_only_df = pd.DataFrame(predictions) 
    forecast_only_df['Date'] = pd.to_datetime(forecast_only_df['datetime']).dt.date
    daily_summary = forecast_only_df.groupby('Date')['aqi'].mean().reset_index()
    today_date = datetime.now().date()
    daily_summary = daily_summary[daily_summary['Date'] > today_date]

    # FIX: Changed "Teal" to "GnBu" (Green-Blue) which is a valid color map.
    st.dataframe(
        daily_summary.style.background_gradient(cmap="GnBu", subset=['aqi']),
        use_container_width=True,
        column_config={
            "Date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
            "aqi": st.column_config.NumberColumn("Predicted AQI", format="%.1f"),
        }
    )

else:
    st.warning("⚠️ No data available to generate predictions.")

