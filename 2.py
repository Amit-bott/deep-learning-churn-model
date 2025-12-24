# import streamlit as st
# import numpy as np
# import pandas as
# import tensorflow
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.metrics import accuracy_score


# st.set_page_config(
#     page_title="ANN Churn Prediction",
#     page_icon="🏦",
#     layout="centered"
# )
# # pd.read_csv("Churn_Modelling_ANN.h5")
# model = pd.read_csv("Churn_Modelling_ANN.h5")

# df = pd.read_csv("Churn_Modelling.csv")
# df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])

# X = df.drop("Exited", axis=1)
# y = df["Exited"]

# le = LabelEncoder()
# X["Gender"] = le.fit_transform(X["Gender"])

# ct = ColumnTransformer(
#     transformers=[("geo", OneHotEncoder(drop="first"), ["Geography"])],
#     remainder="passthrough"
# )
# X = ct.fit_transform(X)


# sc = StandardScaler()
# X = sc.fit_transform(X)

# y_pred = (model.predict(X) > 0.5).astype(int)
# accuracy = accuracy_score(y, y_pred)


# st.title("🏦 Bank Customer Churn Prediction")
# st.caption("Deep Learning ANN Model | Real-time Prediction")

# st.metric("📊 Model Accuracy", f"{accuracy*100:.2f}%")

# st.divider()


# st.subheader("🔍 Enter Customer Details")

# col1, col2 = st.columns(2)
# with col1:
#     geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
#     credit_score = st.number_input("Credit Score", 300, 900, 650)
#     age = st.number_input("Age", 18, 100, 35)
#     tenure = st.slider("Tenure (Years)", 0, 10, 5)
#     balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)

# with col2:
#     gender = st.selectbox("Gender", ["Male", "Female"])
#     products = st.selectbox("Number of Products", [1, 2, 3, 4])
#     credit_card = st.selectbox("Has Credit Card", [0, 1])
#     active_member = st.selectbox("Is Active Member", [0, 1])
#     salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

# if "history" not in st.session_state:
#     st.session_state.history = []


# if st.button("🔮 Predict Churn", use_container_width=True):

#     gender_val = 1 if gender == "Male" else 0

#     input_df = pd.DataFrame([[geography, credit_score, gender_val, age,
#                               tenure, balance, products,
#                               credit_card, active_member, salary]],
#                             columns=["Geography", "CreditScore", "Gender",
#                                      "Age", "Tenure", "Balance",
#                                      "NumOfProducts", "HasCrCard",
#                                      "IsActiveMember", "EstimatedSalary"])

#     input_transformed = ct.transform(input_df)
#     input_scaled = sc.transform(input_transformed)

#     probability = model.predict(input_scaled)[0][0]

#     churn_percent = probability * 100

#     if probability > 0.5:
#         st.error(f"❌ Customer is likely to CHURN ({churn_percent:.2f}%)")
#         result = "Churn"
#     else:
#         st.success(f"✅ Customer will NOT churn ({100 - churn_percent:.2f}%)")
#         result = "No Churn"


#     st.session_state.history.append({
#         "Geography": geography,
#         "Age": age,
#         "Balance": balance,
#         "Salary": salary,
#         "Churn Probability (%)": round(churn_percent, 2),
#         "Prediction": result
#     })


# if st.session_state.history:
#     st.subheader("📜 Prediction History")
#     history_df = pd.DataFrame(st.session_state.history)
#     st.dataframe(history_df, use_container_width=True)








import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# LOAD MODEL & SCALER
# -----------------------------
@st.cache_resource
def load_assets():
    # model = load_model("model.h5")
    model = load_model("models/churn_Modelling.csv.h5")
    # scaler = joblib.load("scaler.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_assets()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["Home", "Single Prediction", "Batch Prediction", "Insights"]
)

# -----------------------------
# HOME PAGE
# -----------------------------
if menu == "Home":
    st.title("📊 Customer Churn Prediction Dashboard")

    st.markdown("""
    ### 🔍 Project Overview
    This application predicts **customer churn** using a **Deep Learning (ANN) model**.
    It helps businesses identify customers likely to leave and take preventive action.

    ### 🧠 Model
    - Artificial Neural Network (ANN)
    - Trained on historical customer behavior data
    - Outputs churn probability and prediction

    ### 🚀 Features
    - Single customer prediction
    - Bulk CSV prediction
    - Visual analytics & KPIs
    - Production-ready UI
    """)

    st.success("✔ Model loaded successfully")

# -----------------------------
# SINGLE PREDICTION
# -----------------------------
elif menu == "Single Prediction":
    st.title("🔮 Single Customer Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.number_input("Credit Score", 300, 900, 650)
        age = st.number_input("Age", 18, 100, 35)
        tenure = st.number_input("Tenure (Years)", 0, 10, 3)

    with col2:
        balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
        products = st.number_input("No. of Products", 1, 4, 1)
        salary = st.number_input("Estimated Salary", 0.0, 200000.0, 60000.0)

    with col3:
        has_card = st.selectbox("Has Credit Card", ["Yes", "No"])
        active = st.selectbox("Is Active Member", ["Yes", "No"])
        gender = st.selectbox("Gender", ["Male", "Female"])

    if st.button("🔍 Predict Churn"):
        data = np.array([[
            credit_score,
            age,
            tenure,
            balance,
            products,
            1 if has_card == "Yes" else 0,
            1 if active == "Yes" else 0,
            salary
        ]])

        data_scaled = scaler.transform(data)
        prob = model.predict(data_scaled)[0][0]

        churn = "Yes" if prob > 0.5 else "No"

        st.subheader("📌 Result")
        st.metric("Churn Probability", f"{prob:.2%}")
        st.success(f"Churn Prediction: {churn}")

# -----------------------------
# BATCH PREDICTION
# -----------------------------
elif menu == "Batch Prediction":
    st.title("📁 Batch Prediction (CSV Upload)")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data")
        st.dataframe(df.head())

        df_scaled = scaler.transform(df)
        probs = model.predict(df_scaled).flatten()

        df["Churn_Probability"] = probs
        df["Churn_Prediction"] = (probs > 0.5).map({True: "Yes", False: "No"})

        st.subheader("✅ Prediction Results")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download Results",
            csv,
            "churn_predictions.csv",
            "text/csv"
        )

# -----------------------------
# INSIGHTS
# -----------------------------
elif menu == "Insights":
    st.title("📈 Churn Insights")

    st.info("Upload batch data to see insights")

    uploaded_file = st.file_uploader("Upload CSV with Predictions", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        churn_rate = (df["Churn_Prediction"] == "Yes").mean() * 100

        col1, col2 = st.columns(2)
        col1.metric("Churn Rate", f"{churn_rate:.2f}%")
        col2.metric("Total Customers", len(df))

        fig = px.histogram(
            df,
            x="Churn_Probability",
            color="Churn_Prediction",
            nbins=20,
            title="Churn Probability Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("© Deep Learning Churn Model | Streamlit Dashboard")
