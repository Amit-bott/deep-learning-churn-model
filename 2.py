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
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import shap

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📉",
    layout="wide"
)

# ---------------------------
# Load Model & Data
# ---------------------------
@st.cache_resource
def load_model():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("Churn_Modelling.csv")

model = load_model()
scaler = load_scaler()
data = load_data()

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.title("🧑 Customer Details")

credit = st.sidebar.slider("Credit Score", 350, 850, 650)
age = st.sidebar.slider("Age", 18, 92, 35)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
balance = st.sidebar.number_input("Balance", 0.0, 300000.0, 50000.0)
salary = st.sidebar.number_input("Estimated Salary", 10000.0, 200000.0, 50000.0)

input_df = pd.DataFrame({
    "CreditScore": [credit],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "EstimatedSalary": [salary]
})

scaled_input = scaler.transform(input_df)

# ---------------------------
# Prediction
# ---------------------------
prediction = model.predict(scaled_input)[0][0]
churn_prob = float(prediction)

# ---------------------------
# Header
# ---------------------------
st.title("📊 Customer Churn Prediction – Deep Learning Dashboard")
st.markdown("**AI-powered system to identify customers likely to leave the bank**")

# ---------------------------
# KPI Cards
# ---------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Customers", data.shape[0])

with col2:
    st.metric("Churn Rate", f"{data.Exited.mean()*100:.2f}%")

with col3:
    st.metric("Model Confidence", f"{churn_prob*100:.2f}%")

# ---------------------------
# Prediction Result
# ---------------------------
st.subheader("🔮 Prediction Result")

if churn_prob > 0.5:
    st.error(f"⚠ Customer is likely to CHURN ({churn_prob*100:.2f}%)")
else:
    st.success(f"✅ Customer will NOT churn ({churn_prob*100:.2f}%)")

# ---------------------------
# Probability Gauge
# ---------------------------
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=churn_prob * 100,
    title={"text": "Churn Probability (%)"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "red"},
        "steps": [
            {"range": [0, 50], "color": "lightgreen"},
            {"range": [50, 100], "color": "pink"}
        ],
    }
))
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Data Analysis Section
# ---------------------------
st.subheader("📈 Dataset Insights")

col4, col5 = st.columns(2)

with col4:
    fig_age = px.histogram(data, x="Age", color="Exited", title="Age vs Churn")
    st.plotly_chart(fig_age, use_container_width=True)

with col5:
    fig_balance = px.box(data, x="Exited", y="Balance", title="Balance Distribution")
    st.plotly_chart(fig_balance, use_container_width=True)

# ---------------------------
# SHAP Explainability
# ---------------------------
st.subheader("🧠 Model Explainability (SHAP)")

explainer = shap.Explainer(model, scaled_input)
shap_values = explainer(scaled_input)

st.write("Feature contribution to churn prediction:")
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(bbox_inches='tight')
