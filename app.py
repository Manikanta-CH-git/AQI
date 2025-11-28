import streamlit as st
import pandas as pd
import plotly.express as px
from supabase import create_client, Client
import joblib 
import xgboost as xgb
from datetime import timedelta

# ==================================================
# ☁ SUPABASE CONFIG
# ==================================================
if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
else:
    # Fallback for local testing if secrets.toml isn't set up
    # Replace these with your actual keys if running locally without secrets
    SUPABASE_URL = "YOUR_SUPABASE_URL" 
    SUPABASE_KEY = "YOUR_SUPABASE_KEY"

# Initialize Supabase client
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    st.error(f"Failed to initialize Supabase client: {e}")
    st.stop()

# ==================================================
# ⚙ PAGE SETTINGS
# ==================================================
st.set_page_config(page_title="Air Quality Monitoring System", layout="wide")

# ==================================================
# 🌈 GLOBAL CSS
# ==================================================
st.markdown("""
<style>
    /* Main Color Bar */
    .aqi-bar-container { display: flex; height: 45px; border-radius: 10px; overflow: hidden; margin-top: 10px; position: relative; }
    .seg { flex: 1; text-align: center; font-weight: bold; padding-top: 12px; color: white; font-family: sans-serif; font-size: 14px; }
    .good { background: #00e400; } .moderate { background: #ffff00; color: black !important; }
    .poor { background: #ff7e00; } .unhealthy { background: #ff0000; }
    .veryunhealthy { background: #8f3f97; } .hazardous { background: #7e0023; }
    
    /* Pointer */
    .pointer-container { position: relative; width: 100%; height: 25px; margin-top: -5px; z-index: 2; }
    .aqi-pointer { position: absolute; top: 0; font-size: 24px; color: #333; transform: translateX(-50%); transition: left 0.5s ease-out; }

    .ticks { width: 100%; display: flex; justify-content: space-between; margin-top: 0px; font-size: 12px; color: #aaa; }
    .big-aqi-value { font-size: 48px; font-weight: 800; text-align: center; margin-top: 15px; transition: color 0.5s ease; }
    .status-text { font-size: 24px; text-align: center; margin-bottom: 10px; font-weight: bold; }
    .prediction-card { background-color: #1E1E1E; padding: 15px; border-radius: 10px; border: 1px solid #333; margin-bottom: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ==================================================
# 📥 HELPER: FETCH DATA
# ==================================================
def get_latest_data(table_name, limit=200):
    try:
        response = supabase.table(table_name).select("*").order("id", desc=True).limit(limit).execute()
        return response.data if response.data else []
    except: return []

# ==================================================
# 📥 HELPER: ROBUST DATA FETCH (Live + History)
# ==================================================
def get_combined_data(limit=3000):
    try:
        resp_live = supabase.table("realtime_data").select("*").order("id", desc=True).limit(limit).execute()
        rows_live = resp_live.data if resp_live.data else []
        
        if len(rows_live) < 1000:
            resp_hist = supabase.table("sensor_data").select("*").order("id", desc=True).limit(2000).execute()
            rows_hist = resp_hist.data if resp_hist.data else []
            rows_live.extend(rows_hist)
            
        return rows_live
    except: return []

# ==================================================
# 🧭 SIDEBAR
# ==================================================
refresh_seconds = st.sidebar.slider("⏱ Auto Refresh (Seconds)", 1, 60, 5)

choice = st.sidebar.radio("Navigation", ["Live Dashboard", "Log Analysis", "Future Forecasting"])

# ==================================================
# 🟢 LIVE DASHBOARD (Auto-Refreshes based on slider)
# ==================================================
@st.fragment(run_every=refresh_seconds)
def show_live_monitor():
    rows = get_latest_data("realtime_data", 50)
    if not rows: st.info("Waiting for data..."); return

    df = pd.DataFrame(rows)
    
    # 1. Rename Column
    if "created_at" in df.columns: df.rename(columns={"created_at": "Timestamp"}, inplace=True)
    elif "updated_at" in df.columns: df.rename(columns={"updated_at": "Timestamp"}, inplace=True)
    else: st.error("Missing timestamp column."); return

    # 2. TIMEZONE FIX (Force Convert UTC -> Asia/Kolkata)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    
    # 3. Numeric Fix
    cols = ['aqi', 'temperature', 'humidity']
    for c in cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(subset=["Timestamp"])
    if df.empty: st.warning("No valid data."); return

    # Sort so the last row is truly the latest time
    df = df.sort_values("Timestamp")
    latest = df.iloc[-1]

    
    aqi = int(latest["aqi"]) if not pd.isna(latest["aqi"]) else 0
    temp = latest.get("temperature", 0)
    hum = latest.get("humidity", 0)

    if aqi <= 50: status, color = "Good", "#00e400"
    elif aqi <= 100: status, color = "Moderate", "#ffff00"
    elif aqi <= 150: status, color = "Poor", "#ff7e00"
    elif aqi <= 200: status, color = "Unhealthy", "#ff0000"
    elif aqi <= 300: status, color = "Very Unhealthy", "#8f3f97"
    else: status, color = "Hazardous", "#7e0023"

    # POINTER CALCULATION
    seg_width = 100 / 6
    if aqi <= 50: pointer_pos = (aqi / 50) * seg_width
    elif aqi <= 100: pointer_pos = seg_width + ((aqi - 50) / 50) * seg_width
    elif aqi <= 150: pointer_pos = (seg_width * 2) + ((aqi - 100) / 50) * seg_width
    elif aqi <= 200: pointer_pos = (seg_width * 3) + ((aqi - 150) / 50) * seg_width
    elif aqi <= 300: pointer_pos = (seg_width * 4) + ((aqi - 200) / 100) * seg_width
    else: pointer_pos = (seg_width * 5) + min(((aqi - 300) / 100) * seg_width, seg_width - 2)
    pointer_pos = max(1, min(99, pointer_pos))

    st.title("Air Quality Monitoring System")
    col1, col2, col3 = st.columns(3)
    col1.metric("AQI", aqi)
    col2.metric("Temperature (°C)", temp)
    col3.metric("Humidity (%)", hum)

    # FIXED TIME DISPLAY
    time_str = latest['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')
    st.caption(f"Last Updated: {time_str}")

    st.markdown(f"""
    <div class="status-text">Current Status: {status}</div>
    <div class="aqi-bar-container">
        <div class="seg good">Good</div><div class="seg moderate">Moderate</div>
        <div class="seg poor">Poor</div><div class="seg unhealthy">Unhealthy</div>
        <div class="seg veryunhealthy">Very Unhealthy</div><div class="seg hazardous">Hazardous</div>
    </div>
    <div class="pointer-container">
        <div class="aqi-pointer" style="left: {pointer_pos}%;">▼</div>
    </div>
    <div class="ticks"><span>0</span><span>50</span><span>100</span><span>150</span><span>200</span><span>300</span><span>300+</span></div>
    <div class="big-aqi-value" style="color:{color};">{aqi} AQI</div>
    """, unsafe_allow_html=True)

    st.subheader("Live Trend")
    try:
        trimmed_df = df.tail(20)   # 👈 Only last 20 rows

        chart_df = trimmed_df[["Timestamp", "aqi", "temperature", "humidity"]].copy()
        chart_df = chart_df.set_index("Timestamp")

        fig = px.line(
            chart_df,
            x=chart_df.index,
            y=["aqi", "temperature", "humidity"],
            markers=True,
            
        )

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Value",
            legend_title="Parameters",
            height=450
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to plot combined graph: {e}")

# ==================================================
# 📁 LOG ANALYSIS
# ==================================================
def show_history():
    st.title("Log Analysis")
    rows = get_latest_data("sensor_data", 1000)
    if not rows:
        st.warning("No data.")
        return

    df = pd.DataFrame(rows)

    # Rename timestamp column
    if "created_at" in df.columns:
        df.rename(columns={"created_at": "Timestamp"}, inplace=True)
    elif "updated_at" in df.columns:
        df.rename(columns={"updated_at": "Timestamp"}, inplace=True)

    # Fix data types
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    for c in ['aqi', 'temperature', 'humidity']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Timezone correction
    try:
        df["Timestamp"] = df["Timestamp"].dt.tz_convert("Asia/Kolkata")
    except:
        df["Timestamp"] = df["Timestamp"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")

    # Clean + Sort
    df_sorted = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    # ----- Move ID column to the front -----
    if "id" in df_sorted.columns:
        cols = ["id"] + [c for c in df_sorted.columns if c != "id"]
        df_sorted = df_sorted[cols]

    st.subheader("Historical Trends")
    fig = px.line(df_sorted, x="Timestamp", y=["aqi", "temperature", "humidity"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Data Table")
    st.dataframe(df_sorted.set_index("Timestamp"), use_container_width=True)


# ==================================================
# 🔮 FUTURE FORECASTING (High Accuracy Direct Mode)
# ==================================================
def show_future():
    st.title("Future Forecasting")
    st.markdown("Predicting next **60 minutes** using **6-Step Direct AI Models**")
    
    # Refresh button
    if st.button("🚀 Run AI Prediction"):
        with st.spinner("Analyzing historical patterns..."):
            
            # 1. Load the "Six Models" File
            MODEL_FILE = 'aqi_six_models.joblib'
            try:
                model_suite = joblib.load(MODEL_FILE)
            except FileNotFoundError:
                st.error(f"❌ '{MODEL_FILE}' not found. Please upload the file from Colab.")
                return

            # 2. Get Data (Need enough for 60 mins of lag)
            rows = get_combined_data(3000)
            if not rows: st.warning("Collecting data..."); return

            # 3. Process Data
            df = pd.DataFrame(rows)
            if "created_at" in df.columns: t_col = "created_at"
            elif "updated_at" in df.columns: t_col = "updated_at"
            else: t_col = df.columns[0]

            df[t_col] = pd.to_datetime(df[t_col], utc=True, errors='coerce')
            try: df[t_col] = df[t_col].dt.tz_convert("Asia/Kolkata")
            except: df[t_col] = df[t_col].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata")
            
            for c in ['aqi', 'temperature', 'humidity']:
                if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

            # Clean and Resample to 10 mins
            df = df.sort_values(t_col).dropna(subset=[t_col, 'aqi'])
            df = df.drop_duplicates(subset=[t_col])
            
            # --- GAP DETECTION ---
            # If there is a massive time gap (e.g. device off for hours), 
            # drop old data so we don't interpolate across 5 hours.
            # We filter for only data in the last 2 hours.
            cutoff_time = df[t_col].max() - timedelta(hours=2)
            df = df[df[t_col] > cutoff_time]
            
            df = df.set_index(t_col)
            
            # Resample to 10T to match training format
            df_10min = df.resample('10T').mean(numeric_only=True).interpolate().reset_index()

            # We need at least 4 rows to make a decent guess (Cold Start Handling)
            if len(df_10min) < 4: 
                st.warning(f"❄️ Cold Start: Not enough continuous history yet. Please wait {40 - len(df_10min)*10} minutes.")
                return

            # 4. Construct Feature Vector (The "Now" State)
            last_known = df_10min.iloc[-1]
            last_time = last_known[t_col]
            current_aqi_real = float(last_known['aqi']) # This is your 55
            
            # Handle missing lags for Cold Start (Backfill if lag_6 is missing)
            # If we don't have 60 mins of data, assume lag_6 was same as current
            lag_1 = float(df_10min.iloc[-1]['aqi'])
            lag_2 = float(df_10min.iloc[-2]['aqi']) if len(df_10min) > 1 else lag_1
            lag_3 = float(df_10min.iloc[-3]['aqi']) if len(df_10min) > 2 else lag_2
            lag_6 = float(df_10min.iloc[-6]['aqi']) if len(df_10min) > 5 else lag_3

            input_features = {
                'aqi_lag_1': lag_1,
                'aqi_lag_2': lag_2,
                'aqi_lag_3': lag_3,
                'aqi_lag_6': lag_6,
                'rolling_mean': float(df_10min.iloc[-3:].mean()['aqi']) 
            }

            # 5. Predict Next 6 Steps
            preds = []
            
            # --- BIAS CORRECTION PRE-CALCULATION ---
            # Predict step 1 first to see how far off the model is
            test_input = pd.DataFrame([{
                'hour': (last_time + timedelta(minutes=10)).hour,
                'minute': (last_time + timedelta(minutes=10)).minute,
                'aqi_lag_1': input_features['aqi_lag_1'],
                'aqi_lag_2': input_features['aqi_lag_2'],
                'aqi_lag_3': input_features['aqi_lag_3'],
                'aqi_lag_6': input_features['aqi_lag_6'],
                'rolling_mean': input_features['rolling_mean']
            }])
            raw_pred_t1 = model_suite[1].predict(test_input)[0]
            
            # Calculate the "Anchoring Offset"
            # If Current is 55 and Model predicts 75, Offset is -20.
            # We apply this offset to everything to force the chart to connect.
            bias_offset = current_aqi_real - raw_pred_t1
            
            # We dampen the offset slightly so it gradually trusts the model more over time
            # (Optional, but here we just apply full offset for visual continuity)
            
            for step in range(1, 7):
                future_time = last_time + timedelta(minutes=10 * step)
                
                current_input = pd.DataFrame([{
                    'hour': future_time.hour,
                    'minute': future_time.minute,
                    'aqi_lag_1': input_features['aqi_lag_1'],
                    'aqi_lag_2': input_features['aqi_lag_2'],
                    'aqi_lag_3': input_features['aqi_lag_3'],
                    'aqi_lag_6': input_features['aqi_lag_6'],
                    'rolling_mean': input_features['rolling_mean']
                }])
                
                model = model_suite[step]
                raw_pred = model.predict(current_input)[0]
                
                # Apply Anchoring
                corrected_pred = raw_pred + bias_offset
                
                if corrected_pred < 0: corrected_pred = 0
                
                preds.append({
                    "Time": future_time.strftime("%H:%M"), 
                    "Forecast AQI": round(corrected_pred, 1)
                })

            # 6. Display Results
            pred_df = pd.DataFrame(preds)
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader("📈 Forecast Trend")
                
                # Dynamic y-axis
                min_aqi = pred_df["Forecast AQI"].min()
                max_aqi = pred_df["Forecast AQI"].max()
                y_range = [max(0, min_aqi - 5), max_aqi + 5]
                
                fig = px.line(pred_df, x="Time", y="Forecast AQI", markers=True, template="plotly_dark")
                
                # Color zones
                fig.add_hrect(y0=0, y1=50, fillcolor="green", opacity=0.1, layer="below")
                fig.add_hrect(y0=50, y1=100, fillcolor="yellow", opacity=0.1, layer="below")
                fig.add_hrect(y0=100, y1=200, fillcolor="orange", opacity=0.1, layer="below")
                
                fig.update_traces(line_color="#00CC96", line_width=4)
                fig.update_layout(yaxis_range=y_range)
                
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                st.write("**Prediction Data**")
                st.dataframe(pred_df, hide_index=True)


# ==================================================
# ROUTING
# ==================================================
if choice == "Live Dashboard": show_live_monitor()
elif choice == "Log Analysis": show_history()
elif choice == "Future Forecasting": show_future()