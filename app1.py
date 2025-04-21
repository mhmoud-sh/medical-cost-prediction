import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os
from datetime import datetime

st.set_page_config(page_title="Medical Cost SHAP Dashboard", layout="wide")

# Load model and preprocessor
model = joblib.load("C:\\Users\\DELL\\Desktop\\medical-cost-prediction\\models\\xgb_model.pkl")
preprocessor = joblib.load("C:\\Users\\DELL\\Desktop\\medical-cost-prediction\\models\\preprocessor.pkl")

# Sidebar: Upload or use default dataset
st.sidebar.title("⚙️ SHAP Dashboard Options")
data_source = st.sidebar.radio("Choose Dataset", ("Default (train)", "Upload CSV"))

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()
else:
    df = pd.read_csv("C:\\Users\\DELL\\Desktop\\medical-cost-prediction\\data\\medical.csv")

st.title("💡 Medical Cost Prediction SHAP Dashboard")

# Preprocess data
X = df.drop("charges", axis=1)
y = df["charges"]
X_transformed = preprocessor.transform(X)
feature_names = preprocessor.get_feature_names_out()
X_df = pd.DataFrame(X_transformed, columns=feature_names)

# Compute SHAP values
explainer = shap.Explainer(model)
shap_values = explainer(X_df)

# --- SHAP Global Importance (Bar) ---
with st.container():
    st.subheader("📊 SHAP Feature Importance (Bar)")
    st.caption("Global feature contributions to model predictions.")
    fig_bar = plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=10)
    st.pyplot(fig_bar)

# --- SHAP Beeswarm Plot ---
with st.container():
    st.subheader("🐝 SHAP Beeswarm Plot")
    st.caption("Visualizes the distribution and impact of features across all samples.")
    fig_beeswarm = plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=10)
    st.pyplot(fig_beeswarm)

# --- Individual Prediction Explanation ---
with st.container():
    st.subheader("🔍 Individual Prediction Explanation (Waterfall)")
    sample_index = st.sidebar.slider("Choose a sample index", 0, len(X_df)-1, 0)
    st.caption(f"Detailed explanation for prediction on sample index: {sample_index}")
    fig_waterfall = plt.figure(figsize=(8, 6))
    shap.plots.waterfall(shap_values[sample_index])
    st.pyplot(fig_waterfall)

# --- Custom Input Prediction ---
with st.container():
    st.subheader("🧠 Custom Prediction & Explanation")
    st.markdown("Enter your own values to see predicted medical cost and explanation:")

    # User input fields
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", options=["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    children = st.slider("Number of Children", 0, 5, 0)
    smoker = st.selectbox("Smoker", options=["yes", "no"])
    region = st.selectbox("Region", options=["southwest", "southeast", "northwest", "northeast"])

    if st.button("Predict Medical Cost"):
        input_dict = {
            "age": [age],
            "sex": [sex],
            "bmi": [bmi],
            "children": [children],
            "smoker": [smoker],
            "region": [region]
        }

        input_df = pd.DataFrame(input_dict)

        # Preprocess input and predict
        input_transformed = preprocessor.transform(input_df)
        prediction = model.predict(input_transformed)[0]

        st.success(f"💰 Estimated Medical Cost: **${prediction:,.2f}**")

        # SHAP Explanation
        st.markdown("### 📌 SHAP Explanation for Custom Input")
        shap_input = pd.DataFrame(input_transformed, columns=feature_names)
        shap_values_input = explainer(shap_input)

        fig_custom = plt.figure(figsize=(8, 6))
        shap.plots.waterfall(shap_values_input[0])
        st.pyplot(fig_custom)

        # Save prediction to CSV
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **input_dict,
            "predicted_charges": [prediction]
        }
        log_df = pd.DataFrame(log_entry)

        log_file = "custom_predictions_log.csv"
        if os.path.exists(log_file):
            log_df.to_csv(log_file, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_file, mode="w", header=True, index=False)

        st.success("✅ Prediction logged to `custom_predictions_log.csv`")
