import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    
    def predict(self, features):
        '''
         Method to predict the target value witht the input values passed.
        '''
        try:
            model_path = 'artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)

            return prediction
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, 
                 longitude: float,
                 latitude:float,
                 housing_median_age:int,
                 total_rooms:int,
                 total_bedrooms:float,
                 population:int,
                 households:int,
                 median_income:float,
                 ocean_proximity:str):
        self.longitude = longitude
        self.latitude = latitude
        self.housing_median_age = housing_median_age
        self.total_rooms = total_rooms
        self.total_bedrooms = total_bedrooms
        self.population = population
        self.households = households
        self.median_income = median_income
        self.ocean_proximity = ocean_proximity
    
    def get_data_as_data_frame(self):
        '''
         To get the input data as as dataframe.
        '''
        try:
            house_price_input_dict ={
                "longitude" : [self.longitude],
                "latitude" : [self.latitude],
                "housing_median_age" : [self.housing_median_age],
                "total_rooms" : [self.total_rooms],
                "total_bedrooms" : [self.total_bedrooms],
                "population" : [self.population],
                "households" : [self.households],
                "median_income" : [self.median_income],
                "ocean_proximity" : [self.ocean_proximity]
            }
            return pd.DataFrame(house_price_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
        

    
