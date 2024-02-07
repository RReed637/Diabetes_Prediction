import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DatatransformationConfig
from src.components.model_trainer import ModeltrainingConfig
from src.components.model_trainer import ModelTrainer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifact', "train.csv")
    test_data_path:str=os.path.join('artifact', "test.csv")
    raw_data_path:str=os.path.join('artifact', "raw.csv")
    #creates folders for splitted data
class DataIngestion:
    def __init__(self):
        self.Ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('Notebook\Data\diabetes_prediction_dataset.csv') #reads in data
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.Ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.Ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42) #splits data

            train_set.to_csv(self.Ingestion_config.train_data_path,index=False,header=True) # creates csv for training set

            test_set.to_csv(self.Ingestion_config.test_data_path,index=False,header=True) # creates csv for test set

            logging.info("Ingestion of Data Completed")

            return(
                self.Ingestion_config.train_data_path,
                self.Ingestion_config.test_data_path,
                )
        except Exception as e:
            
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    ModelTrainer=ModelTrainer()
    print(ModelTrainer.initiate_model_trainer(train_arr,test_arr))

