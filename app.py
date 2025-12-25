import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Irrigation Prediction", page_icon="💧", layout="wide")

# -------------------- DARK THEME STYLE --------------------
st.markdown("""
<style>
body { background-color:#0e1117; color:white; }
[data-testid="stSidebar"] { background-color:#161a23; }
.block-container { background-color:#0e1117; padding:20px; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD ARTIFACTS --------------------
model = joblib.load("irrigation_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder_dict = joblib.load("encoder_dict.pkl")
target_le = joblib.load("target_encoder.pkl")

# -------------------- HEADER --------------------
st.markdown("<h1 style='text-align:center; color:#38bdf8;'>💧 Irrigation Need Prediction Dashboard 🌱</h1>", unsafe_allow_html=True)
st.write("Smart farming assistant based on soil, climate & crop conditions.")

# -------------------- SIDEBAR MENU --------------------
page = st.sidebar.radio("📍 Navigation", ["📊 Dashboard", "📈 Visualization", "🔮 Prediction"])

# -------------------- DATASET COLUMN ORDER --------------------
features = [
    "Soil_Type","Soil_pH","Soil_Moisture","Organic_Carbon","Electrical_Conductivity",
    "Temperature_C","Humidity","Rainfall_mm","Sunlight_Hours","Wind_Speed_kmh",
    "Crop_Type","Crop_Growth_Stage","Season","Irrigation_Type","Water_Source",
    "Field_Area_hectare","Mulching_Used","Previous_Irrigation_mm","Region"
]

# ===========================================================
# 📊 DASHBOARD
# ===========================================================
if page == "📊 Dashboard":
    st.header("📊 Datase#t Overview")
    file = st.file_uploader("📎 Upload CSV to View", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.success("Dataset Loaded Successfully!")
        st.dataframe(df, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.write("📌 Summary")
            st.write(df.describe())
        with col2:
            st.write("🔎 Data Types")
            st.write(df.dtypes)
elif page == "📈 Charts & Visualization":
    st.header("📈 Data Visualization Dashboard")

    uploaded = st.file_uploader("📎 Upload CSV to Generate Charts", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)

        # Soil Moisture Distribution
        st.subheader("🌱 Soil Moisture Distribution")
        fig = px.histogram(df, x="Soil_Moisture", title="Soil Moisture Levels")
        st.plotly_chart(fig, use_container_width=True)

        # Rainfall vs Irrigation Need
        st.subheader("🌧️ Rainfall vs Irrigation Need")
        if "Irrigation_Need" in df.columns:
            fig2 = px.scatter(df, x="Rainfall_mm", y="Soil_Moisture",
                              color="Irrigation_Need", title="Rainfall Impact on Irrigation")
            st.plotly_chart(fig2, use_container_width=True)

        # Temperature vs Humidity
        st.subheader("🌡 Temperature vs Humidity")
        fig3 = px.line(df, y=["Temperature_C", "Humidity"], title="Weather Trend")
        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.warning("⚠️ Please upload a dataset first.")

 





# ===================== PREDICTION PAGE =====================
elif page == "🔮 Prediction":
    st.header("🔮 Predict Irrigation Need")

    # --- Input Form (19 Features) ---
    col1, col2, col3 = st.columns(3)

    with col1:
        Soil_Type = st.selectbox("🧱 Soil Type", ["Clay", "Silt", "Sandy"])
        Soil_pH = st.number_input("⚗ Soil pH", 0.0, 14.0, 6.5)
        Soil_Moisture = st.number_input("💧 Soil Moisture (%)", 0.0, 100.0, 35.0)
        Organic_Carbon = st.number_input("🌿 Organic Carbon", 0.0, 5.0, 1.2)
        Electrical_Conductivity = st.number_input("🔌 Electrical Conductivity", 0.0, 10.0, 2.0)
        Mulching_Used = st.selectbox("🪵 Mulching Used", ["Yes", "No"])

    with col2:
        Temperature_C = st.number_input("🌡 Temperature (°C)", 0.0, 60.0, 25.0)
        Humidity = st.number_input("💨 Humidity (%)", 0.0, 100.0, 50.0)
        Rainfall_mm = st.number_input("🌧 Rainfall (mm)", 0.0, 3000.0, 120.0)
        Sunlight_Hours = st.number_input("☀ Sunlight Hours", 0.0, 15.0, 6.0)
        Wind_Speed_kmh = st.number_input("🍃 Wind Speed (km/h)", 0.0, 200.0, 12.0)
        Previous_Irrigation_mm = st.number_input("💦 Previous Irrigation (mm)", 0.0, 200.0, 30.0)

    with col3:
        Crop_Type = st.selectbox("🌾 Crop Type", ["Wheat", "Maize", "Cotton"])
        Crop_Growth_Stage = st.selectbox("🌱 Crop Growth Stage", ["Sowing", "Vegetative", "Flowering", "Harvest"])
        Season = st.selectbox("📅 Season", ["Rabi", "Kharif", "Zaid"])
        Irrigation_Type = st.selectbox("🚰 Irrigation Type", ["Rainfed", "Canal", "Drip"])
        Water_Source = st.selectbox("🌊 Water Source", ["Reservoir", "Groundwater", "River"])
        Field_Area_hectare = st.number_input("🌍 Field Area (hectare)", 0.1, 100.0, 5.0)
        Region = st.selectbox("📍 Region", ["North", "South", "Central"])


    # --- Create Input DataFrame in correct order ---
    input_data = pd.DataFrame([[
        Soil_Type, Soil_pH, Soil_Moisture, Organic_Carbon, Electrical_Conductivity,
        Temperature_C, Humidity, Rainfall_mm, Sunlight_Hours, Wind_Speed_kmh,
        Crop_Type, Crop_Growth_Stage, Season, Irrigation_Type, Water_Source,
        Field_Area_hectare, Mulching_Used, Previous_Irrigation_mm, Region
    ]], columns=[
        "Soil_Type","Soil_pH","Soil_Moisture","Organic_Carbon","Electrical_Conductivity",
        "Temperature_C","Humidity","Rainfall_mm","Sunlight_Hours","Wind_Speed_kmh",
        "Crop_Type","Crop_Growth_Stage","Season","Irrigation_Type","Water_Source",
        "Field_Area_hectare","Mulching_Used","Previous_Irrigation_mm","Region"
    ])

    # --- Apply Encoding ---
    for col in input_data.select_dtypes(include="object").columns:
        le = encoder_dict[col]        # load specific encoder
        input_data[col] = le.transform(input_data[col])

    # --- Scale numeric data ---
    input_scaled = scaler.transform(input_data)

    # --- Predict ---
    if st.button("🚀 Predict Irrigation Need"):
        pred = model.predict(input_scaled)[0]
        result = {0:"Low", 1:"Medium", 2:"High"}.get(pred, "Unknown")

        if result == "Low":
            st.success("💧 Irrigation Need: **LOW** | Soil moisture is currently sufficient.")
        elif result == "Medium":
            st.warning("🚿 Irrigation Need: **MEDIUM** | Irrigation will be required soon.")
        else:
            st.error("💦 Irrigation Need: **HIGH** | Immediate irrigation needed!")
