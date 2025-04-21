# app.py

import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os
import datetime

# === App Config ===
st.set_page_config(page_title="Medical Cost SHAP Dashboard", layout="wide")
st.title("💡 Medical Cost Prediction SHAP Dashboard")

# === Create logs folder if not exists ===
os.makedirs("logs", exist_ok=True)
log_file = os.path.join("logs", "custom_predictions_log.csv")

# === Load model and preprocessor ===
model = joblib.load("models/xgb_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# === Sidebar Dataset Option ===
st.sidebar.title("SHAP Dashboard Options")
data_source = st.sidebar.radio("Choose Dataset", ("Default (train)", "Upload CSV"))

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()
else:
    df = pd.read_csv("data/medical.csv")

# === Prepare Data ===
X = df.drop("charges", axis=1)
y = df["charges"]
X_transformed = preprocessor.transform(X)
feature_names = preprocessor.get_feature_names_out()
X_df = pd.DataFrame(X_transformed, columns=feature_names)

# === SHAP Values ===
explainer = shap.Explainer(model)
shap_values = explainer(X_df)

# === SHAP Summary Plot (Bar) ===
st.subheader("SHAP Summary Plot (Top 10 Features)")
fig_bar, ax_bar = plt.subplots()
shap.plots.bar(shap_values, max_display=10, show=False)
st.pyplot(fig_bar)

# === SHAP Beeswarm Plot ===
st.subheader("SHAP Beeswarm Plot")
fig_bee, ax_bee = plt.subplots()
shap.plots.beeswarm(shap_values, max_display=10, show=False)
st.pyplot(fig_bee)

# --- SHAP Waterfall Plot (for a single example)
st.subheader("SHAP Waterfall Plot (First Prediction)")
fig_water, ax_water = plt.subplots()
shap.plots.waterfall(shap_values[0], max_display=10, show=False)
st.pyplot(fig_water)

# === Individual Prediction Explanation ===
st.subheader("🔍 Individual Prediction Explanation")
sample_index = st.slider("Choose a sample index", 0, len(X_df)-1, 0)
plt.figure(figsize=(8, 6))
shap.plots.waterfall(shap_values[sample_index])
st.pyplot(plt.gcf())

# === Custom Input Prediction ===
st.subheader("🧮 Try a Custom Prediction")

with st.form("custom_form"):
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

    submit = st.form_submit_button("Predict Medical Cost")

if submit:
    user_input = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])
    transformed_input = preprocessor.transform(user_input)
    prediction = model.predict(transformed_input)

    st.success(f"💰 Predicted Medical Cost: ${prediction[0]:,.2f}")

    # Save prediction to logs
    log_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "predicted_charge": prediction[0],
        "timestamp": datetime.datetime.now()
    }])
    if os.path.exists(log_file):
        log_data.to_csv(log_file, mode='a', header=False, index=False)
    else:
        log_data.to_csv(log_file, index=False)

# === View Past Predictions Log ===
st.subheader("📜 Past Custom Predictions")
if os.path.exists(log_file):
    logs = pd.read_csv(log_file)
    st.dataframe(logs.sort_values("timestamp", ascending=False))
else:
    st.info("No predictions logged yet.")

# === Download SHAP Waterfall Plot ===
st.subheader("📥 Download SHAP Waterfall Plot")
shap_path = "shap_explanation.png"
plt.figure(figsize=(8, 6))
shap.plots.waterfall(shap_values[sample_index], show=False)
plt.savefig(shap_path)
with open(shap_path, "rb") as f:
    st.download_button("Download Plot as PNG", f, file_name="shap_waterfall.png")
