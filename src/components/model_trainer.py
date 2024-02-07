import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from src.exception import CustomException
from src.logger import logging
import warnings

from src.utils import save_object,evaluate_models

@dataclass
class ModeltrainingConfig:
    trained_model_file_path=os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModeltrainingConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest Classifier": RandomForestClassifier(),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "CatBoostClassifier": CatBoostClassifier(verbose=False),
                "XGBClassifier":XGBClassifier(),
                "Naive Bayes": GaussianNB(),
            }
            params={
                "Decision Tree": {
                    'criterion':['gini', 'entropy', 'log_loss'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest Classifier":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoostClassifier":{
                    'learning_rate':[.1, .01, .05, .001],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "CatBoostClassifier":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                    
                },

                "XGBClassifier":{
                    'learning_rate':[.1, .01, .5, .001],
                    'n_estimators':[8,16,32,64,128,256]
                },

                "Naive Bayes":{
                    'var_smoothing': np.logspace(0, -9, num=200)
                }
            }

            
            
            
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models, param=params)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]


            logging.info(f"Found best model on training and testing datasets")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            Classification_Report = classification_report(y_test,predicted)
            return Classification_Report
        except Exception as e:
            raise CustomException(e,sys)

