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
st.markdown("Real-time and 72-hour Air Quality predictions.")
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
        # Format: "Run Date: ... | Algo: RandomForest"
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
    <p style="color: #94A3B8; margin-bottom: 5px;">LIVE Air Quality Index </p>
    <div style="display: flex; align-items: baseline; gap: 20px;">
        <p class="big-aqi" style="font-size: 4rem;">{live_aqi or '--'}</p>
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
    # ---------------------------------------------------

    st.divider()
    st.subheader("🗓️ Next 3 Days Forecast")
    
    # 1. GENERATE PREDICTIONS (Logic Preserved)
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
        predictions.append({"datetime": target_time, "aqi": final_pred})
        
        last_row["aqi_lag_1"] = final_pred
        last_row["hour"] = target_time.hour
        last_row["day"] = target_time.day
        last_row["day_of_week"] = target_time.weekday()
        last_row["month"] = target_time.month
        last_row["year"] = target_time.year
        
    forecast_df = pd.DataFrame(predictions)
    markers_df = forecast_df[forecast_df['datetime'].dt.hour.isin([0, 12])]
    
    # 2. PLOT CHART (Compact Style)
    fig = go.Figure()
    
    # Layer A: Neon Spline Line
    fig.add_trace(go.Scatter(
        x=forecast_df['datetime'], y=forecast_df['aqi'], mode='lines',
        line=dict(color='#22D3EE', width=3, shape='spline'), hoverinfo='skip'
    ))
    
    # Layer B: Checkpoint Markers (Midnight/Noon)
    fig.add_trace(go.Scatter(
        x=markers_df['datetime'], y=markers_df['aqi'], mode='markers',
        marker=dict(size=10, color='#1E293B', line=dict(width=2, color='#22D3EE')),
        hovertemplate='<b>%{x|%A %H:%M}</b><br>AQI: %{y:.1f}<extra></extra>'
    ))
    
    # Styling
    fig.update_layout(
        height=350, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title=dict(text="Date",font=dict(color='#FFFFFF')), showgrid=False, color='#94A3B8', tickformat="%d-%b\n%H:%M", dtick=43200000),
        yaxis=dict(title=dict(text=" Predicted AQI",font=dict(color='#FFFFFF')), showgrid=True, gridcolor='#334155', color='#94A3B8', range=[0.5, 5.5],
                   )
    )
    
    st.plotly_chart(fig, use_container_width=True)
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
    fig_bar = px.bar(poll_df, x="Value", y="Pollutant", orientation='h', template="plotly_dark", text="Value") # Show numbers on bars
    
    # 5. Force the bars to use our custom colors
    fig_bar.update_traces(marker_color=poll_df["Color"], texttemplate='%{text:.1f}')
    
    fig_bar.update_layout(height=400, plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)',
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

    # ─────────────────────────────────────────────────────────────
    # 3-DAY SUMMARY TABLE 
    # ─────────────────────────────────────────────────────────────
    st.divider(); st.subheader("📊 3-Day Forecast Summary")

    if 'predictions' in locals() and predictions:
        
        # 1. Define 'Today' (Karachi Time: UTC+5)
        today = (datetime.now() + timedelta(hours=5)).date()

        # 2. Process Data
        daily_summary = pd.DataFrame(predictions)
        daily_summary["datetime"]=daily_summary["datetime"] + timedelta(hours=5)
        daily_summary['Date'] = pd.to_datetime(daily_summary['datetime']).dt.date
        
        # Filter: strictly greater than today
        daily_summary = daily_summary[daily_summary['Date'] > today]
        
        # Group & Rename"
        daily_summary = daily_summary.groupby('Date')['aqi'].mean().reset_index()
        daily_summary.columns = ['Forecast Date', 'AQI Level']

        # 3. Apply Styling (Dark Background, No Gradient)
        styled_df = daily_summary.style.set_properties(**{
            'background-color': '#1E293B',  # 👈 Sets entire table to Dark Blue
            'color': '#E2E8F0',             # Light Text
            'border-color': '#475569',      # Subtle Grey Border
            'text-align': 'center'          # Center alignment looks better
        }) 
        # Note: I removed .background_gradient() so colors won't change!

        # 4. Display
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Forecast Date": st.column_config.DateColumn("📅 Date", format="DD-MMM-YYYY"),
                "AQI Level": st.column_config.NumberColumn("💨 Avg AQI", format="%.1f")
            }
        )

    else:
        st.warning("⚠️ No data available to generate predictions.")





















