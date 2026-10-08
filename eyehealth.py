import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

st.set_page_config(page_title="Eye Health Classification", layout="wide")

st.title("Eye Health Classification Project")
st.write(
    "Multi-class/binary classification project predicting eye health status and diabetic retinopathy using patient metrics."
)

# Load Data
uploaded_file = st.sidebar.file_uploader("Upload patient_data.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Awaiting CSV file upload. Using dummy data for demonstration.")
    # Fallback structure matching the dataset structure
    df = pd.DataFrame(
        {
            "name": [f"Patient {i}" for i in range(1, 101)],
            "age": np.random.randint(30, 80, 100),
            "has_eye_disease": np.random.choice([True, False], 100),
            "has_diabetic_retinopathy": np.random.choice([True, False], 100),
            "sugar_percentage": np.random.uniform(4.0, 15.0, 100),
            "glucose_percentage": np.random.uniform(70.0, 200.0, 100),
            "cholesterol_percentage": np.random.uniform(100.0, 300.0, 100),
            "obesity_percentage": np.random.uniform(15.0, 40.0, 100),
            "blood_pressure": [
                f"{sbp}/{dbp}"
                for sbp, dbp in zip(
                    np.random.randint(90, 140, 100),
                    np.random.randint(60, 90, 100),
                )
            ],
            "heart_rate": np.random.randint(60, 100, 100),
        }
    )

# Raw Data Display
if st.checkbox("Show Raw Data"):
    st.dataframe(df.head())

# Data Preprocessing & Feature Engineering
df_proc = df.copy()

# Map boolean target columns to binary integers
bool_map = {True: 1, False: 0}
if df_proc["has_eye_disease"].dtype == "bool":
    df_proc["has_eye_disease"] = df_proc["has_eye_disease"].map(bool_map)
if df_proc["has_diabetic_retinopathy"].dtype == "bool":
    df_proc["has_diabetic_retinopathy"] = df_proc[
        "has_diabetic_retinopathy"
    ].map(bool_map)

# Drop identifier column
if "name" in df_proc.columns:
    df_proc.drop(columns=["name"], inplace=True)

# Split blood pressure into systolic and diastolic
if "blood_pressure" in df_proc.columns:
    df_proc[["systolic_bp", "diastolic_bp"]] = (
        df_proc["blood_pressure"].str.split("/", expand=True).astype(int)
    )
    df_proc.drop(columns=["blood_pressure"], inplace=True)

# Feature Engineering
df_proc["pulse_pressure"] = df_proc["systolic_bp"] - df_proc["diastolic_bp"]
df_proc["map_bp"] = (df_proc["systolic_bp"] + 2 * df_proc["diastolic_bp"]) / 3
df_proc["age_glucose_ratio"] = df_proc["age"] * df_proc["glucose_percentage"]
df_proc["age_sys_bp"] = df_proc["age"] * df_proc["systolic_bp"]
df_proc["metabolic_risk_score"] = (
    (df_proc["glucose_percentage"] * 0.4)
    + (df_proc["cholesterol_percentage"] * 0.4)
    + (df_proc["obesity_percentage"] * 0.2)
)

# Target Selection & Training Setup
st.subheader("Model Training & Evaluation")

target_col = st.selectbox(
    "Select Target Column:",
    ["has_eye_disease", "has_diabetic_retinopathy"],
)

# Define X and y
y = df_proc[target_col]
X = df_proc.drop(
    columns=["has_eye_disease", "has_diabetic_retinopathy"], errors="ignore"
)

if st.button("Train XGBoost Model"):
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X, y)

    y_pred = model.predict(X)

    st.success("Model trained successfully!")
    st.write(f"**Accuracy:** {accuracy_score(y, y_pred):.4f}")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Confusion Matrix:**")
        st.dataframe(pd.DataFrame(confusion_matrix(y, y_pred)))

    with col2:
        st.write("**Classification Report:**")
        st.text(classification_report(y, y_pred))