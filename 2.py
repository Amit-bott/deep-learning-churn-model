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







# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# ————————————————
# Load Data & Model
# ————————————————
@st.cache_data
def load_data():
    data = pd.read_csv("Churn_Modelling.csv")
    return data

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

data = load_data()
model = load_model()

# ————————————————
# Sidebar - User Inputs
# ————————————————
st.sidebar.header("Select Customer Features")

def user_input_features():
    credit_score = st.sidebar.slider("Credit Score", int(data.CreditScore.min()), int(data.CreditScore.max()), int(data.CreditScore.mean()))
    age = st.sidebar.slider("Age", int(data.Age.min()), int(data.Age.max()), int(data.Age.mean()))
    tenure = st.sidebar.slider("Tenure", int(data.Tenure.min()), int(data.Tenure.max()), int(data.Tenure.mean()))
    balance = st.sidebar.number_input("Balance", float(data.Balance.min()), float(data.Balance.max()), float(data.Balance.mean()))
    estimated_salary = st.sidebar.number_input("Estimated Salary", float(data.EstimatedSalary.min()), float(data.EstimatedSalary.max()), float(data.EstimatedSalary.mean()))
    
    return pd.DataFrame({
        "CreditScore":[credit_score],
        "Age":[age],
        "Tenure":[tenure],
        "Balance":[balance],
        "EstimatedSalary":[estimated_salary]
    })

input_df = user_input_features()

# ————————————————
# Main Dashboard
# ————————————————
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Enter customer info in the left sidebar to predict churn probability.")

# Show first few rows of the dataset
if st.checkbox("Show raw data"):
    st.write(data.head())

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)[:,1]

st.subheader("Prediction")
st.write("🔴 **Churn**" if prediction[0]==1 else "🟢 **Not Churn**")

st.subheader("Churn Probability")
fig = px.bar(x=["Chance to Churn"], y=prediction_proba, labels={"x":"Metric", "y":"Probability"})
st.plotly_chart(fig)

# Feature Analysis
st.subheader("Feature Distributions")
for col in ["CreditScore","Age","Balance"]:
    fig2 = px.histogram(data, x=col, color="Exited", nbins=30, title=f"{col} distribution by churn")
    st.plotly_chart(fig2)
