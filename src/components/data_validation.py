from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.exception import SensorException
from src.logger import logging
from src import utils
from src.config import TARGET_COLUMN
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import os,sys

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact, data_validation_config:DataValidationConfig):
        try:
            logging.info(f" {'>>' * 20} Data Validation {'<<' *20} ")
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self.validation_error = dict()
        except Exception as e:
            raise SensorException(e, sys)

    def drop_missing_values(self,df:pd.DataFrame, report_key_name:str)->pd.DataFrame:
        try:
            threshold = self.data_validation_config.missing_threshold

            logging.info(f"select columns which contains null values more than {threshold}")
            null_report = df.isna().sum()/df.shape[0]
            droping_column_names = null_report[null_report>20].index

            logging.info(f"columns to drop {list(droping_column_names)}")
            self.validation_error[report_key_name] = list(droping_column_names)
            df.drop(list(droping_column_names),axis=1, inplace=True)

            #return None if no column left
            if len(df)==0:
                return None
            return df
        except Exception as e:
            raise SensorException(e, sys)
    
    def is_required_columns_exists(self,base_df:pd.DataFrame, current_df:pd.DataFrame,report_key_name:str)->bool:
        try:
            base_columns = base_df.columns
            missing_columns = []
            for base_column in base_columns:
                if base_column not in current_df.columns:
                    logging.info(f"column : {base_column} not avaiable")
                    missing_columns.append(base_column)

            if len(missing_columns)>0:
                self.validation_error[report_key_name]=missing_columns
                return False
            return True

        except Exception as e:
            raise SensorException(e, sys)

    def data_drift(self,base_df:pd.DataFrame, current_df:pd.DataFrame, report_key_name:str):
        try:
            base_columns = base_df.columns
            current_columns = current_df.columns
            drift_report = dict()    
            for base_column in base_columns:
                base_data, current_data = base_df[base_column], current_df[base_column]

                #Null hypothesis is that both data drawn from same distribution
                logging.info(f"{base_column} : base_datatype={base_data.dtype} , current_datatype={current_data.dtype}")
                distribution = ks_2samp(data1=base_data, data2=current_data)

                # Same distribution if pvalue > 0.05 else Diff distribution
                if distribution.pvalue > 0.05:
                    drift_report[base_column]={
                        'p-value' : float(distribution.pvalue),
                        'same distribution' : True
                    }
                else:
                    drift_report[base_column]={
                        'p-value' : float(distribution.pvalue),
                        'same distribution' : False
                    }

            self.validation_error[report_key_name]=drift_report
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_validation(self,)->DataValidationArtifact:
        try:
            logging.info("reading base dataframe")
            base_df = pd.read_csv("aps_failure_training_set1.csv")

            logging.info("replacing na values in base dataframe")
            base_df.replace("na", np.NAN, inplace=True)

            logging.info("droping null values with in base DataFrame")
            base_df = self.drop_missing_values(df=base_df, report_key_name="missing_values_within_base_dataset")

            logging.info("reading train Dataframe")
            train_df = pd.read_csv(filepath_or_buffer=self.data_ingestion_artifact.train_file_path)

            logging.info("reading test Dataframe")
            test_df = pd.read_csv(filepath_or_buffer=self.data_ingestion_artifact.test_file_path)

            logging.info("Droping null values in Training DataFrame")
            train_df = self.drop_missing_values(df=train_df, report_key_name="missing_values_within_train_dataset")
            logging.info("Droping null vlaues in Test DataFrame")
            test_df = self.drop_missing_values(df=test_df, report_key_name="missing_values_within_test_dataset")

            logging.info(f"converting required columns to float")
            exclude_column=[TARGET_COLUMN]
            base_df = utils.convert_column_to_float(df=base_df, exclude_column=exclude_column)
            train_df = utils.convert_column_to_float(df=train_df, exclude_column=exclude_column)
            test_df = utils.convert_column_to_float(df=test_df, exclude_column=exclude_column)

            logging.info("Is required columns available in train df")
            train_df_column_status = self.is_required_columns_exists(base_df=base_df, current_df=train_df, report_key_name="missing_value_within_the_train_df")
            logging.info("Is required columns available in train df")
            test_df_column_status = self.is_required_columns_exists(base_df=base_df, current_df=test_df, report_key_name="missing_value_within_the_test_df")

            if train_df_column_status:
                logging.info("As all columns are available in train df hence detecting data drift")
                self.data_drift(base_df=base_df, current_df=train_df, report_key_name="data_drift_with_train_dataset")
            if test_df_column_status:
                logging.info("As all columns are available in test df hence detecting data drift")
                self.data_drift(base_df=base_df, current_df=test_df, report_key_name="data_drift_with_test_dataset")

            #write report to ymal file
            logging.info("write report to ymal file")
            utils.write_to_ymal(file_path=self.data_validation_config.report_file_path, data=self.validation_error)

            data_validation_artifact = DataValidationArtifact(report_file_path=self.data_validation_config.report_file_path)
            logging.info(f"{data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise SensorException(e, sys)




