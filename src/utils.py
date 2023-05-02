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

# if __name__=="__main__":
#     get_collection_as_dataframe("aps", "sensor")