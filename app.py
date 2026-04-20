import streamlit as st
import pandas as pd
import joblib

# MUST BE FIRST
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="centered"
)

# Title
st.title("💳 Credit Risk Prediction App")
st.write("Financial Inclusion Based Risk Model")

# Load model
@st.cache_resource
def load_model():
    model = joblib.load("credit_risk_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, le = load_model()

st.divider()

# Inputs
st.subheader("Enter Details")

age = st.slider("Age", 18, 60)
female = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
educ = st.selectbox("Education", [0, 1], format_func=lambda x: "Not Educated" if x == 0 else "Educated")
account = st.selectbox("Bank Account", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
digital = st.selectbox("Digital Payment", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
saved = st.selectbox("Savings", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
borrowed = st.selectbox("Borrowed", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
internet = st.selectbox("Internet Use", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

score = account + digital + saved + borrowed + internet

st.write("Inclusion Score:", score)

# Predict
if st.button("Predict Risk"):

    data = pd.DataFrame([{
        "age": age,
        "female": female,
        "educ": educ,
        "account": account,
        "anydigpayment": digital,
        "saved": saved,
        "borrowed": borrowed,
        "internet_use": internet,
        "inclusion_score": score
    }])

    pred = model.predict(data)[0]
    label = le.inverse_transform([pred])[0]

    if label == "Low Risk":
        st.success("Low Risk ✅")
    elif label == "Medium Risk":
        st.warning("Medium Risk ⚠️")
    else:
        st.error("High Risk 🚨")

st.divider()
st.caption("Financial Inclusion Project | 2025")