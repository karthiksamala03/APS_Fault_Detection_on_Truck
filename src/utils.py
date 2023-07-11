from src.exception import SensorException
from src.logger import logging
from src.config import mongo_client
import os, sys
import pandas as pd
import numpy as np
import dill

def get_collection_as_dataframe(database_name:str, collection_name:str)->pd.DataFrame:
    """
    Description : This function return collection as Dataframe
    ==========================================================
    Params:
    database_name : database name
    collection_name : collection name 
    ==========================================================
    return: Pandas DataFrame
    """
    try:
        logging.info(f"Reading Database {database_name} and collection {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"{df.columns}")
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
            logging.info(f"_id column droped successfully")
        logging.info(f"Rows and Columns in DataFrame {df.shape}")

        return df 

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

def save_numpy_arr_data(file_path:str, array:np.array):
    try:
        dir = os.path.dirname(file_path)
        os.makedirs(dir, exist_ok=True)
        with open(file_path,"wb") as file_obj:
            np.save(file=file_obj, arr=array)

    except Exception as e:
        raise SensorException(e, sys)

def save_object(file_path:str, obj:object)-> None:
    try:
        dir = os.path.dirname(file_path)
        os.makedirs(dir, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise SensorException(e, sys)

def load_numpy_arr_data(file_path:str)->np.array:
    try:
        if not os.path.exists(path=file_path):
            raise Exception(f"The file : {file_path} not exist")
        with open(file_path, "rb") as file_obj:
            return np.load(file=file_obj)
    except Exception as e:
        raise SensorException(e, sys)

def load_object(file_path:str)->object:
    try:
        if not os.path.exists(path=file_path):
            raise Exception(f"The object: {file_path} not exist")
        with open(file_path, "rb") as file_obj:
            return dill.load(file=file_obj)
    except Exception as e:
        raise SensorException(e, sys)
