import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from predict import predict_risk

# PAGE CONFIG

st.set_page_config(
    page_title="AeroGuard AI – Intelligent Engine Risk Monitoring System",
    page_icon="✈️",
    layout="wide",
)

# HEADER

st.title("✈️ AeroGuard AI")
st.subheader("Intelligent Engine Risk Monitoring System")
st.markdown("---")

# LABELS & COLORS

labels = {
    0: "🟢 Low Risk",
    1: "🟡 Medium Risk",
    2: "🔴 High Risk"
}

colors = {
    0: "green",
    1: "orange",
    2: "red"
}

# MUST match training order
feature_names = [
    "setting_1", "setting_2", "setting_3",
    "sensor_2", "sensor_3", "sensor_4",
    "sensor_7", "sensor_8", "sensor_9",
    "sensor_11", "sensor_12", "sensor_13",
    "sensor_14", "sensor_15", "sensor_17",
    "sensor_20", "sensor_21"
]

# INPUT MODE

option = st.radio(
    "Select Input Mode:",
    ["Upload CSV (Last 30 Cycles)", "Manual Entry (Single Cycle)"]
)

# GAUGE FUNCTION

def show_gauge(prob_value, risk_class):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_value * 100,
        title={'text': "Failure Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': colors[risk_class]},
            'steps': [
                {'range': [0, 40], 'color': "lightgreen"},
                {'range': [40, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightcoral"},
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)
# CSV MODE

if option == "Upload CSV (Last 30 Cycles)":

    st.info("Upload a CSV file containing exactly 30 rows × 17 columns")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        if df.shape != (30, 17):
            st.error("File must contain exactly 30 rows and 17 columns.")
        else:

            df.columns = feature_names

            risk, probs = predict_risk(df.values)

            st.success(f"Predicted Risk Level: {labels[risk]}")

            # PROBABILITY TABLE
        
            st.subheader("Prediction Confidence (%)")

            prob_df = pd.DataFrame({
                "Risk Level": ["Low", "Medium", "High"],
                "Probability (%)": (probs * 100).round(2)
            })

            st.dataframe(prob_df, use_container_width=True)
            # GAUGE

            show_gauge(probs[risk], risk)

            # SENSOR TREND GRAPH        

            st.subheader("Sensor Trend (Last 30 Cycles)")

            selected_sensor = st.selectbox(
                "Select Sensor to Visualize",
                feature_names
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=df[selected_sensor],
                mode="lines+markers"
            ))

            fig.update_layout(
                xaxis_title="Cycle",
                yaxis_title="Sensor Value",
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

            # ALERT SYSTEM
           

            if risk == 2:
                st.error("🚨 CRITICAL ALERT: Immediate Maintenance Required!")
            elif risk == 1:
                st.warning("⚠️ Warning: Schedule Inspection Soon.")
            else:
                st.success("✅ Engine Operating Normally.")

# MANUAL MODE

if option == "Manual Entry (Single Cycle)":

    st.info("Enter single cycle sensor values (17 features)")

    manual_data = []
    cols = st.columns(4)

    for i, name in enumerate(feature_names):
        value = cols[i % 4].number_input(
            name,
            value=0.0,
            format="%.4f"
        )
        manual_data.append(value)

    if st.button("Predict Risk"):

        risk, probs = predict_risk(manual_data)

        st.success(f"Predicted Risk Level: {labels[risk]}")

        # Probability table
        prob_df = pd.DataFrame({
            "Risk Level": ["Low", "Medium", "High"],
            "Probability (%)": (probs * 100).round(2)
        })

        st.dataframe(prob_df, use_container_width=True)

        # Gauge
        show_gauge(probs[risk], risk)

        # Alert
        if risk == 2:
            st.error("🚨 CRITICAL ALERT: Immediate Maintenance Required!")
        elif risk == 1:
            st.warning("⚠️ Warning: Schedule Inspection Soon.")
        else:
            st.success("✅ Engine Operating Normally.")
