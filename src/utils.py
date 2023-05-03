from src.exception import SensorException
from src.logger import logging
from src.config import mongo_client
import os, sys
import pandas as pd

def get_collection_as_dataframe(database_name:str, collection_name:str)->pd.DataFrame:
    try:
        logging.info("Connecting to mongodb.....")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info("Data Extraction completed...")
        print(df.shape)
    except Exception as e:
        raise SensorException(e, sys)

def convert_column_to_float(df:pd.DataFrame, exclude_column:list)->pd.DataFrame:
    try:
        for col in df.columns:
            if col not in exclude_column:
                df[col] = df[col].astype('float')

        return df
    except Exception as e:
        raise SensorException(e, sys)

def write_to_ymal(file_path:str, data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)
        with open(file_path,"w") as file_writer:
            ymal.dump(data,file_writer)
    except Exception as e:
        SensorException(e, sys)