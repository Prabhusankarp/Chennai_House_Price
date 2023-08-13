import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation


@dataclass
class DataIngestionconfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw.csv')


## Create a class for Data Ingestion

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()


    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')
        try:
            df=pd.read_csv(os.path.join('notebooks/data','train-chennai-sale.csv'))
            logging.info('Dataset read as pandas DataFrame')
            #df.drop_duplicates(keep='first',inplace=True)
            #logging.info('Drop duplicates')
            #df = whitespace_remover(df)

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Train Test Split')

            train_set,test_set = train_test_split(df,test_size=0.20,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)
        
'''
## run data_ingestion

if __name__=='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

# python src/components/data_ingestion.py
'''


## run data_ingestion
'''
if __name__=='__main__':
    obj=DataIngestion()
    train_data_path,test_data_path=obj.initiate_data_ingestion()
    data_tranformation=DataTransformation()
    train_arr,test_arr,_=data_tranformation.initiate_data_transformation(train_data_path,test_data_path)

# python src/components/data_ingestion.py

'''