import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_assets():
    model = load_model("models/churn_model.h5")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_assets()

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["Home", "Single Prediction", "Batch Prediction", "Insights"]
)

# -------------------------
# HOME
# -------------------------
if menu == "Home":
    st.title("📊 Customer Churn Prediction System")

    st.markdown("""
    ### 🔍 Overview
    This system predicts **bank customer churn** using a **Deep Learning ANN model**.

    ### 🧠 Model Details
    - Artificial Neural Network
    - Binary classification
    - Output: Churn Probability

    ### 🚀 Features
    ✔ Single customer prediction  
    ✔ Bulk CSV prediction  
    ✔ Interactive analytics  
    ✔ Production-ready UI  
    """)

# -------------------------
# SINGLE PREDICTION
# -------------------------
elif menu == "Single Prediction":
    st.title("🔮 Single Customer Prediction")

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", 300, 900, 650)
        age = st.number_input("Age", 18, 100, 35)
        tenure = st.slider("Tenure", 0, 10, 5)
        balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)

    with col2:
        products = st.selectbox("Number of Products", [1, 2, 3, 4])
        has_card = st.selectbox("Has Credit Card", ["Yes", "No"])
        active = st.selectbox("Is Active Member", ["Yes", "No"])
        salary = st.number_input("Estimated Salary", 0.0, 200000.0, 60000.0)

    if st.button("Predict Churn"):
        input_data = np.array([[
            credit_score, age, tenure, balance, products,
            1 if has_card == "Yes" else 0,
            1 if active == "Yes" else 0,
            salary
        ]])

        scaled = scaler.transform(input_data)
        prob = model.predict(scaled)[0][0]

        st.metric("Churn Probability", f"{prob:.2%}")

        if prob > 0.5:
            st.error("❌ Customer is likely to churn")
        else:
            st.success("✅ Customer will stay")

# -------------------------
# BATCH PREDICTION
# -------------------------
elif menu == "Batch Prediction":
    st.title("📁 Batch Prediction")

    file = st.file_uploader("Upload CSV", type="csv")

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        scaled = scaler.transform(df)
        probs = model.predict(scaled).flatten()

        df["Churn_Probability"] = probs
        df["Prediction"] = np.where(probs > 0.5, "Yes", "No")

        st.success("Prediction completed")
        st.dataframe(df)

        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            "churn_predictions.csv"
        )

# -------------------------
# INSIGHTS
# -------------------------
elif menu == "Insights":
    st.title("📈 Churn Insights")

    file = st.file_uploader("Upload Prediction CSV", type="csv")

    if file:
        df = pd.read_csv(file)

        churn_rate = (df["Prediction"] == "Yes").mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.2f}%")

        fig = px.histogram(
            df,
            x="Churn_Probability",
            color="Prediction",
            title="Churn Probability Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.caption("© Deep Learning ANN | Customer Churn Prediction")
