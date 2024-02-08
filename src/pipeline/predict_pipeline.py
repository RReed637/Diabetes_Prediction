import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifact\model.pkl'
            preprocessor_path='artifact\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
class CustomData:
    def __init__(self,
        gender: str,
        age: int,
        hypertension: str,
        heart_disease: str,
        bmi: int,
        HbA1c_level: int,
        blood_glucose_level: int):

        self.gender = gender
        self.age = age
        self.hypertension = hypertension
        self.heart_disease = heart_disease
        self.bmi = bmi
        self.HbA1c_level = HbA1c_level
        self.blood_glucose_level = blood_glucose_level
    
    def get_data_as_data_frame(self):
        try:
           user_data = {
               'gender': [self.gender],
                'age': [self.age],
                'hypertension' : [self.hypertension],
                'heart_disease' : [self.heart_disease],
                'bmi' : [self.bmi],
                'HbA1c_level' : [self.HbA1c_level],
                'blood_glucose_level' : [self.blood_glucose_level]
           }     
           return pd.DataFrame(user_data)
        
        except Exception as e:
            raise CustomException(e,sys)
