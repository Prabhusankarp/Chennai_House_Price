import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 AREA:str,
                 INT_SQFT:float,
                 N_BEDROOM:float,
                 N_BATHROOM:float,
                 N_ROOM:float,
                 SALE_COND:str,
                 PARK_FACIL:str,
                 UTILITY_AVAIL:str,
                 STREET:str,
                 MZZONE:str,
                 BUILDTYPE:str,
                 DATE_BUILD,
                 DATE_SALE
                 ):
        
        self.AREA=AREA
        self.INT_SQFT=INT_SQFT
        self.N_BEDROOM=N_BEDROOM
        self.N_BATHROOM=N_BATHROOM
        self.N_ROOM=N_ROOM
        self.SALE_COND=SALE_COND
        self.PARK_FACIL = PARK_FACIL
        self.UTILITY_AVAIL = UTILITY_AVAIL
        self.STREET = STREET
        self.MZZONE = MZZONE
        self.BUILDTYPE = BUILDTYPE
        self.DATE_BUILD = DATE_BUILD
        self.DATE_BUILD=pd.DatetimeIndex([self.DATE_BUILD])
        self.DATE_SALE=DATE_SALE
        self.DATE_SALE=pd.DatetimeIndex([self.DATE_SALE])
        self.building_age=(self.DATE_SALE.year - self.DATE_BUILD.year)[0]
        


    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'AREA':[self.AREA],
                'INT_SQFT':[self.INT_SQFT],
                'N_BEDROOM':[self.N_BEDROOM],
                'N_BATHROOM':[self.N_BATHROOM],
                'N_ROOM':[self.N_ROOM],
                'SALE_COND':[self.SALE_COND],
                'PARK_FACIL':[self.PARK_FACIL],
                'UTILITY_AVAIL':[self.UTILITY_AVAIL],
                'STREET':[self.STREET],
                'MZZONE':[self.MZZONE],
                'BUILDTYPE':[self.BUILDTYPE],
                'building_age':[self.building_age]
                
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)
