from src.exception import SensorException
from src.logger import logging
from src import utils
import os,sys
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            logging.info(f"{'>>' * 20} Data Ingestion {'<<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_ingestion(self,)-> DataIngestionArtifact:
        try:
            # logging.info("loading data from csv file")
            # df:pd.DataFrame = pd.read_csv("aps_failure_training_set1.csv")

            logging.info("Exporting collection data as pandas Dataframe")
            #Exporting collection data as pandas Dataframe
            df:pd.DataFrame = utils.get_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name, 
                collection_name=self.data_ingestion_config.collection_name)

            logging.info("replacing na with np.NAN")
            df.replace(to_replace='na', value=np.NAN, inplace=True)

            logging.info("Save Data in Feature Store")
            future_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(name=future_store_dir,exist_ok=True)
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path, index=False, header=True)

            logging.info("Creating Dataset folder")
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(name=dataset_dir, exist_ok=True)
            

            logging.info("Performing Trian Test split")
            train_df, test_df = train_test_split(df, test_size=self.data_ingestion_config.test_size, random_state=42)

            logging.info("Save Train and Test data in dataset folder")
            train_df.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)
            test_df.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)

            data_ingestion_artifact = DataIngestionArtifact(feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
             train_file_path=self.data_ingestion_config.train_file_path,
              test_file_path=self.data_ingestion_config.test_file_path)

            logging.info(f"Data Ingestion Artifact : {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise SensorException(e, sys)
