import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

st.title("""Screening Application for Diabetes""")
gender = st.selectbox('Select Sex', ("Male", "Female")),
age = st.slider('Your Age:', min_value = 1, max_value = 100, step=5 ),
hypertension = st.selectbox('Do you have hypertension:', ("Yes", "No")),
bmi = st.slider('Your BMI:', min_value= 1, max_value = 100, step=10),
heart_disease = st.selectbox('Have you contracted Heart Disease?:',    ("Yes", "No")),
HbA1c_level = st.slider('What is your HbA1c Level (Average Blood Sugar Levels for the last two to three months):', 1,10, 1),
blood_glucose_level = st.slider('What is your Blood Glucose Level (Blood Sugar level):', (50, 300, 5))
def prediction():
    data=CustomData(
        gender = st.selectbox('Select Sex', ("Male", "Female")),
        age = st.slider('Your Age:', min_value = 1, max_value = 100, step=5 ),
        hypertension = st.selectbox('Do you have hypertension:', ("Yes", "No")),
        bmi = st.slider('Your BMI:', min_value= 1, max_value = 100, step=10),
        heart_disease = st.selectbox('Have you contracted Heart Disease?:',    ("Yes", "No")),
        HbA1c_level = st.slider('What is your HbA1c Level (Average Blood Sugar Levels for the last two to three months):', 1,10, 1),
        blood_glucose_level = st.slider('What is your Blood Glucose Level (Blood Sugar level):', (50, 300, 5))
    )
    pred_df=data.get_data_as_frame()    

    predict_pipeline=PredictPipeline()
    results=predict_pipeline.predict(pred_df)

    
    return prediction

results = prediction





    

# Web Application
def app():
    st.subheader('Prediction: ')
    print(results)
app()
