import os
import sys
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


import dill

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)        
    except Exception as e:
        raise CustomException(e, sys)
    
def save_object(file_path, obj):
    try:
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report={}
        print(f"Length of models is {len(models)}")
        for i in range(len(list(models))):
            model=list(models.values())[i]
            param=list(params.values())[i]

            gs=GridSearchCV(model,param, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            y_test_predicted = model.predict(X_test)
            r2score = r2_score(y_test,y_test_predicted)

            report[list(models.keys())[i]] = r2score

        return report
    except Exception as e:
        raise CustomException(e, sys)
        