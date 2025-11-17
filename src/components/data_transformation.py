import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

#Column transformer to create the pipeline of operations to be performed.
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

class DataTransformationConfig:
    preprocessor_object_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.preprocessor_config = DataTransformationConfig()
    
    def get_datatransformer_object(self):
        '''
         This function is responsible for data transformation. 
        '''
        try:
            numerical_columns = ["longitude", 
                                 "latitude", 
                                 "housing_median_age",
                                 "total_rooms",
                                 "total_bedrooms",
                                 "population",
                                 "households",
                                 "median_income"                                 
                                 ]
            categorical_columns = [
                "ocean_proximity"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info(f"Categorical Columns: {categorical_columns}")
            logging.info(f"Numerical Columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor;    

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info('Obtaining preprocessing object')

            preprocessing_obj=self.get_datatransformer_object()

            target_column_name = "median_house_value"

            input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
            target_featuer_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f'input_feature_train_df.columns:{input_feature_train_df.columns}') 

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_featuer_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                self.preprocessor_config.preprocessor_object_file_path, 
                preprocessing_obj
                )
            logging.info('Saved the preprocessing object under artifacts folder.')

            return (
                train_arr, 
                test_arr, 
                self.preprocessor_config.preprocessor_object_file_path
                )
        except Exception as e:
            raise CustomException(e,sys)

    
        






