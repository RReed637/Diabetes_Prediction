import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
st.title("""Screening Application for Diabetes""")
def prediction():
    data=CustomData(
        gender = main.gender,
        age = main.age,
        hypertension = main.hypertension,
        bmi = st.sidebar.slider('Your BMI:', min_value= 1, max_value = 100, step=10),
        heart_disease = st.sidebar.selectbox('Have you contracted Heart Disease?:',    ("Yes", "No")),
        HbA1c_level = st.sidebar.slider('What is your HbA1c Level (Average Blood Sugar Levels for the last two to three months):', 1,10, 1),
        blood_glucose_level = st.sidebar.slider('What is your Blood Glucose Level (Blood Sugar level):', (50, 300, 5))
    )
    pred_df=data.get_data_as_frame()    
    predict_pipeline=PredictPipeline()
    results=predict_pipeline.predict(pred_df)
    
    print(results)
    if(results[0]==0):
        return 'not diabetic'
    else:
        return 'diabetic'

def main():
    st.title('Diabetes Prediction Application')
    gender = st.sidebar.selectbox('Select Sex', ("Male", "Female")),
    age = st.sidebar.slider('Your Age:', min_value = 1, max_value = 100, step=5 ),
    hypertension = st.sidebar.selectbox('Do you have hypertension:', ("Yes", "No")),
    bmi = st.sidebar.slider('Your BMI:', min_value= 1, max_value = 100, step=10),
    heart_disease = st.sidebar.selectbox('Have you contracted Heart Disease?:',    ("Yes", "No")),
    HbA1c_level = st.sidebar.slider('What is your HbA1c Level (Average Blood Sugar Levels for the last two to three months):', 1,10, 1),
    blood_glucose_level = st.sidebar.slider('What is your Blood Glucose Level (Blood Sugar level):', (50, 300, 5))

    Diagnosis=''

    if st.button('Diabetes Test Result'):
        Diagnosis = prediction([gender,age,hypertension,bmi,heart_disease,HbA1c_level,blood_glucose_level])
    st.success(Diagnosis)

if __name__=='__main__':
    main()


    
