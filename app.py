import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("traffic_model.pkl")

# Page config
st.set_page_config(page_title="🚦 Smart Traffic AI", layout="wide")

# 🌙 Theme Toggle
theme = st.toggle("🌙 Dark Mode")

if theme:
    bg_color = "#0E1117"
    text_color = "white"
else:
    bg_color = "#f5f7fa"
    text_color = "black"

# 🎨 Custom CSS
st.markdown(f"""
<style>
.stApp {{
    background-color: {bg_color};
    color: {text_color};
}}

.card {{
    background: rgba(255,255,255,0.1);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 10px;
}}

h1 {{
    text-align: center;
}}

.stButton>button {{
    background: linear-gradient(to right, #00c6ff, #0072ff);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    height: 3em;
    width: 100%;
}}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🚦 Smart Traffic AI Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# Layout
col1, col2 = st.columns(2)

# 📍 Location
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📍 Location Details")

    latitude = st.number_input("Latitude", value=19.07)
    longitude = st.number_input("Longitude", value=72.87)

    st.map(pd.DataFrame({'lat':[latitude], 'lon':[longitude]}))

    vehicle_count = st.number_input("Vehicle Count", min_value=0)
    traffic_speed = st.number_input("Traffic Speed", min_value=0.0)
    road_occupancy = st.slider("Road Occupancy (%)", 0.0, 100.0, 50.0)

    st.markdown("</div>", unsafe_allow_html=True)

# 🚦 Traffic
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🚦 Traffic Conditions")

    traffic_light = st.selectbox("Traffic Light", ["Red", "Yellow", "Green"])
    traffic_map = {"Red": 0, "Yellow": 1, "Green": 2}
    traffic_light = traffic_map[traffic_light]

    weather = st.selectbox("Weather", ["Clear", "Rainy", "Foggy", "Snowy"])
    weather_map = {"Clear": 0, "Rainy": 1, "Foggy": 2, "Snowy": 3}
    weather = weather_map[weather]

    accident = st.selectbox("Accident", ["No", "Yes"])
    accident_map = {"No": 0, "Yes": 1}
    accident = accident_map[accident]

    sentiment = st.slider("Sentiment Score", -1.0, 1.0, 0.0)
    ride_demand = st.number_input("Ride Demand", min_value=0)
    parking = st.number_input("Parking Availability", min_value=0)

    st.markdown("</div>", unsafe_allow_html=True)

# ⚡ Additional
st.markdown("---")
st.subheader("⚡ Additional Parameters")

col3, col4 = st.columns(2)

with col3:
    emission = st.number_input("Emission Levels")
    energy = st.number_input("Energy Consumption")

with col4:
    hour = st.slider("Hour", 0, 23, 12)
    day = st.slider("Day", 1, 31, 15)
    month = st.slider("Month", 1, 12, 6)

# 🚀 Predict
st.markdown("---")

if st.button("🚀 Predict Traffic"):

    features = np.array([[latitude, longitude, vehicle_count, traffic_speed,
                          road_occupancy,
                          traffic_light, weather, accident, sentiment,
                          ride_demand, parking, emission, energy,
                          hour, day, month]])

    prediction = model.predict(features)

    # 🎯 KPI Cards
    st.subheader("📊 Prediction Result")

    colA, colB, colC = st.columns(3)

    if prediction[0] == 0:
        colA.metric("Traffic Level", "Low 🟢")
    elif prediction[0] == 1:
        colA.metric("Traffic Level", "Medium 🟡")
    else:
        colA.metric("Traffic Level", "High 🔴")

    colB.metric("Vehicle Count", vehicle_count)
    colC.metric("Speed (km/h)", traffic_speed)

    # 📊 Chart
    chart_data = pd.DataFrame({
        'Feature': ['Vehicle Count', 'Speed', 'Occupancy'],
        'Value': [vehicle_count, traffic_speed, road_occupancy]
    })

    st.bar_chart(chart_data.set_index('Feature'))