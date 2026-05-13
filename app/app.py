import streamlit as st
import numpy as np
import joblib

model = joblib.load("models/model.pkl")

st.title("Titanic Survival Prediction 🚢")

pclass = st.selectbox("Pclass", [1,2,3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80)
fare = st.slider("Fare", 0, 500)

if sex == "male":
    sex = 0
else:
    sex = 1

if st.button("Predict"):
    data = np.array([[pclass, sex, age, fare]])
    result = model.predict(data)

    if result[0] == 1:
        st.success("Survived ✅")
    else:
        st.error("Did not survive ❌")


joblib.dump(model, "models/model.pkl")