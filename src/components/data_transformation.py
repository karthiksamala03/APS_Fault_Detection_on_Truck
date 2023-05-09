from src.exception import SensorException
from src.logger import logging
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact
import os, sys
from src import utils
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTETomek
from src.config import TARGET_COLUMN

class DataTransformation:
    def __init__(self, data_transformation_config:DataTransformationConfig,
                    data_ingestion_artifact:DataIngestionArtifact):
        try:
            logging.info(f"{'>>'* 20} Data Transformation {'<<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise SensorException(e, sys)

    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        try:
            simple_imputer = SimpleImputer(strategy='constant', fill_value=0)
            robust_scaler = RobustScaler()
            pipeline = Pipeline(steps=[
                ('imputer' , simple_imputer),
                ('scaler' , simple_imputer)
            ])
            return pipeline
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            #reading training and testing file
            logging.info(f"reading train dataframe")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info(f"reading test dataframe")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info(f"selecting input features from train and test dataframes")
            input_features_train_df = train_df.drop(TARGET_COLUMN, axis=1)
            input_features_test_df = test_df.drop(TARGET_COLUMN, axis=1)

            logging.info(f"selecting target features from train and test dataframes")  
            target_features_train_df = train_df[TARGET_COLUMN]
            target_feature_test_df = test_df[TARGET_COLUMN]

            logging.info(f"Performing LabelEncoding")
            labelencoder = LabelEncoder()
            labelencoder.fit(target_features_train_df)

            #transformation on target column
            target_feature_train_arr = labelencoder.transform(target_features_train_df)
            target_feature_test_arr = labelencoder.transform(target_feature_test_df)

            #Creating Transformation Pipeline
            transformation_pipeline = DataTransformation.get_data_transformer_object()
            transformation_pipeline.fit(input_features_train_df)

            #Tranforming input features
            input_feature_train_arr = transformation_pipeline.transform(input_features_train_df)
            input_feature_test_arr = transformation_pipeline.transform(input_features_test_df)

            logging.info(f"Performing Sampling on both Train and Test data")
            smt = SMOTETomek(random_state=42)
            logging.info(f"Before resampling in trainging set Input:{input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")
            input_feature_train_arr,target_feature_train_arr = smt.fit_resample(input_feature_train_arr, target_feature_train_arr)
            logging.info(f"After resampling in training set Input:{input_feature_train_arr.shape} Target:{target_feature_train_arr.shape}")

            logging.info(f"Before resampling in testing set Input:{input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")
            input_feature_test_arr,target_feature_test_arr = smt.fit_resample(input_feature_test_arr, target_feature_test_arr)
            logging.info(f"After resampling in testing set Input:{input_feature_test_arr.shape} Target:{target_feature_test_arr.shape}")

            #concate train and test array
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            #Save numpy array
            logging.info("Saving train and test arrrays")
            utils.save_numpy_arr_data(file_path=self.data_transformation_config.transformed_train_path, array=train_arr)
            utils.save_numpy_arr_data(file_path=self.data_transformation_config.transformed_test_path, array=test_arr)

            #Saving Transformation pipeline
            logging.info("Saving Transformation Pipeline")
            utils.save_object(file_path=self.data_transformation_config.transform_object_path, obj=transformation_pipeline)

            #Saving LabelEncoder
            logging.info("Saving LabelEncoder")
            utils.save_object(file_path=self.data_transformation_config.target_encoder_path, obj=labelencoder)

            data_transformation_artifact = DataTransformationArtifact(
                transform_object_path=self.data_transformation_config.transform_object_path,
                 transformed_train_path=self.data_transformation_config.transformed_train_path,
                  transformed_test_path=self.data_transformation_config.transformed_test_path,
                   target_encoder_path=self.data_transformation_config.target_encoder_path)

            logging.info(f"Data Transformation Object {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise SensorException(e, sys)
