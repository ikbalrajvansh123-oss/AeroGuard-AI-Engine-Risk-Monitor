# ✈️ AeroGuard AI  
## Intelligent Engine Risk Monitoring System

AeroGuard AI is an end-to-end predictive maintenance system designed to classify aircraft engine failure risk using multivariate time-series sensor data.

The system leverages an LSTM-based deep learning model trained on 30-cycle sliding window sequences and provides an interactive monitoring dashboard for real-time risk assessment.

---
### Online Deploy Link (https://aeroguardai.streamlit.app/)

## 🚀 Project Overview

Modern aircraft engines generate large volumes of sensor data.  
AeroGuard AI uses this data to predict engine failure risk levels:

- 🟢 Low Risk  
- 🟡 Medium Risk  
- 🔴 High Risk  

The goal is to enable proactive maintenance and reduce unexpected failures.

---

## 🧠 Model Architecture

- Model Type: LSTM (Long Short-Term Memory)
- Input: 30 cycles × 17 features
- Hidden Size: 64
- Layers: 2
- Output: 3-class classification
- Activation: Softmax
- Accuracy: ~95%

Sliding window sequence modeling is used to capture temporal dependencies in engine behavior.

---

## 📊 Features

✔ LSTM-based time-series classification  
✔ 30-cycle sliding window prediction  
✔ Probability-based confidence scores  
✔ Risk threshold alert system  
✔ Interactive Streamlit dashboard  
✔ Sensor trend visualization  
✔ CSV and Manual input support  

---

## 🏗️ Project Structure

Aeroguard-AI/
├── LSTM_model/                 
│   ├── data_loader.py
│   ├── scaling_data.py
│   ├── split_data.py
│   ├── create_sequence.py
│   ├── model.py
│   ├── classify_risk.py
│   └── train.py
│
├── models/
│   ├── risk_lstm_model.pth
│   ├── scaler.pkl
│   └── metadata.pkl
│
├── app.py
├── predict.py
├── README.md
└── requirements.txt

