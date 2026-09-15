import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

@st.cache_resource
def load_pipeline():
    model_name = "titanic_svm_model.pkl" if os.path.exists("titanic_svm_model.pkl") else "model.pkl"
    return joblib.load(model_name)

pipeline = load_pipeline()

st.title("🚢 Titanic Survival Predictor")
st.markdown("Enter passenger information to predict survival probability using our trained ML pipeline.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", options=[1, 2, 3], format_func=lambda x: f"Class {x}")
    sex = st.selectbox("Gender", options=["male", "female"], format_func=lambda x: x.capitalize())
    age = st.slider("Age", min_value=1.0, max_value=80.0, value=28.0, step=0.5)
    fare = st.number_input("Ticket Fare ($)", min_value=0.0, max_value=600.0, value=32.2, step=1.0)

with col2:
    sibsp = st.number_input("Siblings / Spouses aboard", min_value=0, max_value=8, value=0, step=1)
    parch = st.number_input("Parents / Children aboard", min_value=0, max_value=6, value=0, step=1)
    embarked = st.selectbox(
        "Port of Embarkation", 
        options=["S", "C", "Q"],
        format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x]
    )

st.write("")
predict_btn = st.button("Predict Survival Status", use_container_width=True, type="primary")

if predict_btn:
    input_df = pd.DataFrame([{
        'Age': age,
        'Fare': fare,
        'SibSp': sibsp,
        'Parch': parch,
        'Sex': sex,
        'Pclass': pclass,
        'Embarked': embarked
    }])

    prediction = pipeline.predict(input_df)[0]
    
    st.divider()
    if prediction == 1:
        st.success("### 🎉 Prediction: Survived")
        st.balloons()
    else:
        st.error("### 🕯️ Prediction: Did Not Survive")