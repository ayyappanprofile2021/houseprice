import os
import sys  

from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', "train.csv")
    test_data_path:str = os.path.join('artifacts', "test.csv")
    raw_data_path:str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestionconfig = DataIngestionConfig()


    def initiatedataingestion(self):
        logging.info('Starting data ingestion...')
        try:
            df=pd.read_csv('notebook\\HousePrice.csv')
            logging.info("Data copied into dataframe object successfully.")
            logging.info(f'training data path:{self.ingestionconfig.train_data_path}')
            os.makedirs(os.path.dirname(self.ingestionconfig.train_data_path), exist_ok=True)
            df.to_csv(self.ingestionconfig.raw_data_path, index=False, header=True)
            logging.info(f"Dataframe stored into the raw file in {self.ingestionconfig.raw_data_path}")
            train_set, test_set =train_test_split(df,test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestionconfig.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestionconfig.test_data_path, index=False, header=True)
            logging.info('Train and test data split and stored successfully.')
            return(
                self.ingestionconfig.train_data_path,
                self.ingestionconfig.test_data_path
            )
        except Exception as e:
            logging.info('Error Occurred')
            raise CustomException(e, sys)
        
if __name__=="__main__":
    dataingestion = DataIngestion()
    train_path, test_path = dataingestion.initiatedataingestion()
    
    datatransformation = DataTransformation()
    train_array, test_array, _ = datatransformation.initiate_data_transformation(train_path=train_path, test_path=test_path)
    
    modeltrainer = ModelTrainer()
    r2score, bestmodelname =  modeltrainer.initiate_model_training(train_array=train_array, test_array=test_array)
    print(f"Best model is:{bestmodelname} and its r2score is :{r2score}")

    