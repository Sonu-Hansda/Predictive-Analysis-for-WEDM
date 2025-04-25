import streamlit as st
import pandas as pd
import joblib

st.title("ML Based Predictive Modeling of Wire EDM Performance Metrics - Sonu Hansda")
st.write("""
Demonstrating the working of three machine learning models for predicting:  
1. **Cutting Rate (CR)**  
2. **Surface Roughness (SR)**  
Models used:
- **Support Vector Regression (SVR)**  
- **Linear Regression (LR)**  
- **Decision Tree Regression (DT)**
""")

st.sidebar.header("Input Parameters")

def user_input_features():
    ton = st.sidebar.slider('Pulse On Time (Ton)', 0.4, 1.4, 0.8)
    toff = st.sidebar.slider('Pulse Off Time (Toff)', 14, 46, 20)
    ip = st.sidebar.slider('Peak Current (IP)', 70, 210, 100)
    sv = st.sidebar.slider('Servo Voltage (SV)', 16, 80, 40)
    wf = st.sidebar.slider('Wire Feed Rate (WF)', 2, 12, 5)
    wt = st.sidebar.slider('Wire Tension (WT)', 450, 1600, 700)

    data = {
        'Ton': [ton],
        'Toff': [toff],
        'IP': [ip],
        'SV': [sv],
        'WF': [wf],
        'WT': [wt]
    }
    return pd.DataFrame(data)

input_data = user_input_features()

st.subheader("Selected Input Parameters")
st.write(input_data)

models = {
    "Decision Tree": {
        "CR": joblib.load('models/decision_tree_cr.pkl'),
        "SR": joblib.load('models/decision_tree_sr.pkl')
    },
    "Linear Regression": {
        "CR": joblib.load('models/linear_regression_cr.pkl'),
        "SR": joblib.load('models/linear_regression_sr.pkl')
    },
    "SVR": {
        "CR": joblib.load('models/support_vector_regressor_cr.pkl'),
        "SR": joblib.load('models/support_vector_regressor_sr.pkl')
    }
}

cr_predictions = {model: models[model]["CR"].predict(input_data)[0] for model in models.keys()}

sr_predictions = {model: models[model]["SR"].predict(input_data)[0] for model in models.keys()}

st.subheader("Model Predictions")

st.write("### Cutting Rate (CR)")
for model, prediction in cr_predictions.items():
    st.write(f"**{model}:** {prediction:.2f}")

st.write("### Surface Roughness (SR)")
for model, prediction in sr_predictions.items():
    st.write(f"**{model}:** {prediction:.2f}")

st.subheader("Comparison of Predictions")
st.write("### Cutting Rate (CR)")
st.bar_chart(pd.DataFrame({
    'Model': list(cr_predictions.keys()),
    'CR Prediction': list(cr_predictions.values())
}).set_index('Model'))

st.write("### Surface Roughness (SR)")
st.bar_chart(pd.DataFrame({
    'Model': list(sr_predictions.keys()),
    'SR Prediction': list(sr_predictions.values())
}).set_index('Model'))
