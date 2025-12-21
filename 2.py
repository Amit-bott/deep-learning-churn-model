import streamlit as st
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# Page Config
# -------------------------------------------------mlkj
st.set_page_config(
    page_title="ANN Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

# -------------------------------------------------
# Load Model
# -------------------------------------------------
model = load_model("model.h5")

# -------------------------------------------------
# Load & Prepare Data (for encoders & accuracy)
# -------------------------------------------------
df = pd.read_csv("Churn_Modelling.csv")
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

X = df.drop("Exited", axis=1)
y = df["Exited"]

# Encode Gender
le = LabelEncoder()
X["Gender"] = le.fit_transform(X["Gender"])

# OneHot Geography
ct = ColumnTransformer(
    transformers=[("geo", OneHotEncoder(drop="first"), ["Geography"])],
    remainder="passthrough"
)
X = ct.fit_transform(X)

# Scaling
sc = StandardScaler()
X = sc.fit_transform(X)

# Model accuracy
y_pred = (model.predict(X) > 0.5).astype(int)
accuracy = accuracy_score(y, y_pred)

# -------------------------------------------------
# UI Header
# -------------------------------------------------
st.title("🏦 Bank Customer Churn Prediction")
st.caption("Deep Learning ANN Model | Real-time Prediction")

st.metric("📊 Model Accuracy", f"{accuracy*100:.2f}%")

st.divider()

# -------------------------------------------------
# Input Section
# -------------------------------------------------
st.subheader("🔍 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    credit_score = st.number_input("Credit Score", 300, 900, 650)
    age = st.number_input("Age", 18, 100, 35)
    tenure = st.slider("Tenure (Years)", 0, 10, 5)
    balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female"])
    products = st.selectbox("Number of Products", [1, 2, 3, 4])
    credit_card = st.selectbox("Has Credit Card", [0, 1])
    active_member = st.selectbox("Is Active Member", [0, 1])
    salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# -------------------------------------------------
# Prediction History
# -------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------------------------
# Predict Button
# -------------------------------------------------
if st.button("🔮 Predict Churn", use_container_width=True):

    gender_val = 1 if gender == "Male" else 0

    input_df = pd.DataFrame([[geography, credit_score, gender_val, age,
                              tenure, balance, products,
                              credit_card, active_member, salary]],
                            columns=["Geography", "CreditScore", "Gender",
                                     "Age", "Tenure", "Balance",
                                     "NumOfProducts", "HasCrCard",
                                     "IsActiveMember", "EstimatedSalary"])

    input_transformed = ct.transform(input_df)
    input_scaled = sc.transform(input_transformed)

    probability = model.predict(input_scaled)[0][0]

    churn_percent = probability * 100

    if probability > 0.5:
        st.error(f"❌ Customer is likely to CHURN ({churn_percent:.2f}%)")
        result = "Churn"
    else:
        st.success(f"✅ Customer will NOT churn ({100 - churn_percent:.2f}%)")
        result = "No Churn"

    # Save history
    st.session_state.history.append({
        "Geography": geography,
        "Age": age,
        "Balance": balance,
        "Salary": salary,
        "Churn Probability (%)": round(churn_percent, 2),
        "Prediction": result
    })

# -------------------------------------------------
# Show Prediction History
# -------------------------------------------------
if st.session_state.history:
    st.subheader("📜 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)
