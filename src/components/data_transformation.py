import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, minmax_scale 
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
@dataclass
class DatatransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifact', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DatatransformationConfig()

    def get_data_transformer_object(self):
        '''This function is responsible for data transformation'''
        try: 
            numerical_columns = [
                "age",
                "bmi","HbA1c_level",
                "blood_glucose_level"
            ]
            cat_columns = ["gender","hypertension",
                "heart_disease",]

            num_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))
                    ]
                
            )

            cat_pipeline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler",StandardScaler(with_mean=False))
                ]
                
            )
            
            logging.info(f"Categorical columns: {cat_columns}")
            
            logging.info(f"Numerical Columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, cat_columns)
                ]

            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
        

            logging.info("Read train and test data completed")

            logging.info("obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name="diabetes"
            
            numerical_columns = [
                "age",
                "bmi","HbA1c_level",
                "blood_glucose_level"
            ]

            majority_class = train_df[train_df[target_column_name] == 0]
            minority_class = train_df[train_df[target_column_name] == 1]

            minority_upsampled = resample(minority_class, replace=True)
            balanced_train_df = pd.concat([majority_class,minority_upsampled])

            input_feature_train_df=balanced_train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=balanced_train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)]
            
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info("Preprocessing object saved")

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )
            
            

        except Exception as e:
            raise CustomException(e,sys)
        
            
