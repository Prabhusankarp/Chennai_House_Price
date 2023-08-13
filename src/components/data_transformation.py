import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config  = DataTransformationconfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')

            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['AREA','SALE_COND','PARK_FACIL','BUILDTYPE','UTILITY_AVAIL','STREET','MZZONE']
            numerical_cols = ['INT_SQFT','N_BEDROOM','N_ROOM','building_age','N_BATHROOM']  

            AREA = ['Karapakkam', 'Adyar', 'Chrompet','Velachery','KK Nagar','Anna Nagar','T Nagar']
            SALE_COND = ['Partial','Family','AbNormal','Normal Sale','AdjLand']
            PARK_FACIL = ['No','Yes']
            BUILDTYPE=['House','Others','Commercial'] 
            UTILITY_AVAIL = ['ELO','NoSeWa','NoSewr','AllPub']  
            STREET = ['No Access','Paved','Gravel']  
            MZZONE = ['I','A','C','RM','RH','RL']  
                            
                              

            logging.info('Pipeline Initiated')

            ## Numerical Pipelines

            num_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('scaler',StandardScaler())
                ]
            )

            ## Categorical Pipeline

            cat_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[AREA,SALE_COND,PARK_FACIL,BUILDTYPE,UTILITY_AVAIL,STREET,MZZONE])),
                ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            logging.info('Pipeline Completed')
            
            
            return preprocessor


        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            correct_area={"Chrompt":"Chrompet","Chrmpet":"Chrompet","Chormpet":"Chrompet","TNagar":"T Nagar",
              "Karapakam":"Karapakkam","KKNagar":"KK Nagar","Velchery":"Velachery",
              "Ana Nagar":"Anna Nagar","Ann Nagar":"Anna Nagar","Adyr":"Adyar"}
            Typo_error={"Adj Land":"AdjLand","Ab Normal":"AbNormal","Partiall":"Partial","PartiaLl":"Partial",
           "Noo":"No","Other":"Others","Comercial":"Commercial","All Pub":"AllPub","NoSewr ": "NoSewr" ,
            "Pavd":"Paved","NoAccess":"No Access"}

            train_df["AREA"]=train_df["AREA"].replace(correct_area)
            train_df['SALE_COND']=train_df['SALE_COND'].replace(Typo_error)
            train_df['PARK_FACIL']=train_df['PARK_FACIL'].replace(Typo_error)
            train_df['BUILDTYPE']=train_df['BUILDTYPE'].replace(Typo_error)
            train_df['UTILITY_AVAIL']=train_df['UTILITY_AVAIL'].replace(Typo_error)
            train_df['STREET']=train_df['STREET'].replace(Typo_error)

            test_df["AREA"]=test_df["AREA"].replace(correct_area)
            test_df['SALE_COND']=test_df['SALE_COND'].replace(Typo_error)
            test_df['PARK_FACIL']=test_df['PARK_FACIL'].replace(Typo_error)
            test_df['BUILDTYPE']=test_df['BUILDTYPE'].replace(Typo_error)
            test_df['UTILITY_AVAIL']=test_df['UTILITY_AVAIL'].replace(Typo_error)
            test_df['STREET']=test_df['STREET'].replace(Typo_error)

            logging.info('Mapping the typo errors')

            # converting 'DATE_SALE' to date format. 

            train_df["DATE_SALE"] = pd.to_datetime(train_df["DATE_SALE"],format='%d-%M-%Y')
            train_df["DATE_BUILD"] = pd.to_datetime(train_df["DATE_BUILD"],format='%d-%M-%Y')

            test_df["DATE_SALE"] = pd.to_datetime(test_df["DATE_SALE"],format='%d-%M-%Y')
            test_df["DATE_BUILD"] = pd.to_datetime(test_df["DATE_BUILD"],format='%d-%M-%Y')

            logging.info('Date convertion')

            # Converting N_BEDROOM,N_BATHROOM the datatypes from float into int datatype

            #train_df['N_BEDROOM'] = train_df['N_BEDROOM'].astype(int)
            #train_df['N_BATHROOM'] = train_df['N_BATHROOM'].astype(int)

            #test_df['N_BEDROOM'] = test_df['N_BEDROOM'].astype(int)
            #test_df['N_BATHROOM'] = test_df['N_BATHROOM'].astype(int)

            # calculating the age of the building

            train_df['building_age'] = (pd.DatetimeIndex(train_df['DATE_SALE']).year) - (pd.DatetimeIndex(train_df['DATE_BUILD']).year)
            test_df['building_age'] = (pd.DatetimeIndex(test_df['DATE_SALE']).year) - (pd.DatetimeIndex(test_df['DATE_BUILD']).year)

            # Dropping unwanted columns
            target_column_name = 'SALES_PRICE'
            
            drop_columns = [target_column_name,"PRT_ID","REG_FEE","COMMIS",'DATE_SALE','DATE_BUILD',
                'DIST_MAINROAD','QS_ROOMS','QS_BATHROOM','QS_BEDROOM','QS_OVERALL']



            logging.info(f'Train Dataframe head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe head: \n{test_df.head().to_string()}')


            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Unwanted columns Dropped')

            ## Transforming using preprocessor obj
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )


        except Exception as e:
            logging.info('Exception occured in the initiate_data_transformation')
            raise CustomException(e,sys)
