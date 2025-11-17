import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
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
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:    
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        '''
            Method to decide the best model out of given set of models.
        '''
        try:
            logging.info('Spli the training and test data')
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params ={
                "Random Forest" :{
                    # parameter 'criterion' to measure the splif of the data and to choose the best
                    # function among the list below.
                    "criterion" : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best', 'random']
                    # 'max_features':['sqrt', 'log2']
                },
                "Decision Tree" :{
                    # parameter 'criterion' to measure the splif of the data and to choose the best
                    # function among the list below.
                    "criterion" : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best', 'random']
                    # 'max_features':['sqrt', 'log2']
                },
                "Gradient Boosting":{
                    # Each new tree tries to correct the errors of the previous trees.
                    # 'learning_rate' determines how big a step we take toward the correct
                    # answer eachtime.
                    'learning_rate': [.1, .01, .05, .001],
                    # Instead of training every tree with full dataset, train it on a random subset
                    # (without replacement)
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # It defines how many weak learners(usually decision trees) will be built
                    # and combined to form the final stron model.
                    'n_estimators': [8,16,32,64,128,256]
                },
                'Linear Regression': {},
                'XGBRegressor':{
                    'learning_rate': [0.01, 0.05, 0.1],
                    'n_estimators': [8,16,32,64,128,256]
                },
                'CatBoosting Regressor': {
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30,50,100]
                },
                'AdaBoost Regressor': {
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators':[8,16,32,64,128,256]
                }
            }

            model_report:dict = evaluate_models(
                X_train=X_train, 
                y_train=y_train, 
                X_test=X_test, 
                y_test=y_test, 
                models=models, 
                params=params)
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)  
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('No best model found')
            logging.info(f"Best model found in both training and testing dataset is: {best_model_name}")

            save_object(self.model_trainer_config.trained_model_path,best_model)

            predicted = best_model.predict(X_test)
            r2score = r2_score(y_test, predicted)

            return r2score,best_model_name

        except Exception as e:
            raise CustomException(e, sys)
        


