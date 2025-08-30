import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ================================
# Load and Train Model (cached)
# ================================
@st.cache_resource
def load_model():
    df = pd.read_csv("diabetes.csv")

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
    grid = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    return grid.best_estimator_, scaler

model, scaler = load_model()

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Prediction using SVM")
st.write("This app predicts the likelihood of diabetes based on medical details.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.2f")
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Prediction button
if st.button("🔍 Predict"):
    user_input = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])
    user_input_scaled = scaler.transform(user_input)
    prediction = model.predict(user_input_scaled)[0]
    prob = model.predict_proba(user_input_scaled)[0][1]

    if prediction == 1:
        st.error(f"🚨 The person is LIKELY to develop diabetes.\nConfidence: {prob*100:.2f}%")
    else:
        st.success(f"✅ The person is NOT likely to develop diabetes.\nConfidence: {prob*100:.2f}%")

st.info("Dataset: PIMA Indians Diabetes Database. ML model: Support Vector Machine (SVM).")