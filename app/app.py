import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Set up page
st.set_page_config(page_title="Titanic Survival Predictor", layout="wide")
st.title("🚢 Titanic Survival Prediction Dashboard")

# 1. Define GitHub Raw Data Paths
TRAIN_DATA_URL = "https://githubusercontent.com"
TEST_DATA_URL = "https://githubusercontent.com"
MODEL_URL = "https://githubusercontent.com"

# 2. Load Datasets using caching
@st.cache_data
def load_github_data(url):
    return pd.read_csv(url)

try:
    df_train = load_github_data(TRAIN_DATA_URL)
    
    # --- DASHBOARD TAB ---
    st.subheader("📊 Passenger Insights Dashboard")
    
    # Metrics Row
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Passengers Analyzed", len(df_train))
    
    if 'Survived' in df_train.columns:
        survival_rate = (df_train['Survived'].sum() / len(df_train)) * 100
        col2.metric("Overall Survival Rate", f"{survival_rate:.1f}%")
        col3.metric("Total Survivors", int(df_train['Survived'].sum()))

    # Layout for charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.write("### Survival Rate by Gender")
        if 'Sex' in df_train.columns and 'Survived' in df_train.columns:
            gender_survival = df_train.groupby('Sex')['Survived'].mean()
            st.bar_chart(gender_survival)
            
    with chart_col2:
        st.write("### Passenger Age Distribution")
        if 'Age' in df_train.columns:
            st.bar_chart(df_train['Age'].value_counts().sort_index())

    st.write("### Raw Training Dataset Preview")
    st.dataframe(df_train.head(10), use_container_width=True)

    # --- PREDICTION FORM ---
    st.markdown("---")
    st.subheader("🔮 Predict Individual Survival")
    st.write("Enter traveler details below to test the trained model:")
    
    pred_col1, pred_col2, pred_col3 = st.columns(3)
    with pred_col1:
        pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
        age = st.slider("Age", 0, 100, 29)
    with pred_col2:
        sex = st.selectbox("Sex", ["male", "female"])
        fare = st.number_input("Ticket Fare ($)", min_value=0.0, value=32.0)
    with pred_col3:
        sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, value=0)
        parch = st.number_input("Parents/Children Aboard", min_value=0, value=0)

    if st.button("Calculate Survival Probability"):
        # Put your feature processing matching your model_training.py here!
        st.info("Features ready for prediction. Connect your model file to get the live outputs.")

except Exception as e:
    st.error(f"Could not load data from GitHub. Check your files. Error: {e}")
