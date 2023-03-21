import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor



from src.exception import CustomException
from src.logger import logging

from src.utils import  save_obj,evalaute_models


@dataclass
class ModeltrainerConfig:
    trained_model_file_path=os.path.join("artifact","model.pkl")

class Modeltrainer:
    def __init__(self) :
        self.Model_trainer_config=ModeltrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info(f"splitting training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],

            )
            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor()
            }
            

            model_report:dict=evalaute_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test
                                             ,models=models)
            
            ## to get the best model score from dict

            best_model_score=max(sorted(model_report.values()))

            ## to get best model name from dict

            best_model_name= list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("no best model found")
            logging.info(f"best found model both on training and testing dataset")

            save_obj(
                file_path=self.Model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_sqaure=r2_score(y_test,predicted)

            return r2_sqaure

        
        except Exception as e:
            raise CustomException(e,sys)