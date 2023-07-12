from src.exception import SensorException
from src.logger import logging
import os, sys
from datetime import datetime
FILE_NAME = "sensor.csv"
TRAIN_FILE_NAME = "trian.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJECT_FILE_NAME="transformer.pkl"
TARGET_ENCODER_OBJECT_FILE_NAME="target_encoder.pkl"
MODEL_FILE_NAME = "model.pkl"

class TrainingPipelineConfig:
    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),"artifact",f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception as e:
            raise SensorException(e, sys)

class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig()):
        try:
            self.database_name="aps"
            self.collection_name="sensor"
            self.data_ingestion_dir=os.path.join(training_pipeline_config.artifact_dir,"data_ingestion")
            self.feature_store_file_path=os.path.join(self.data_ingestion_dir,"feature_store",FILE_NAME)
            self.train_file_path=os.path.join(self.data_ingestion_dir,"dataset",TRAIN_FILE_NAME)
            self.test_file_path=os.path.join(self.data_ingestion_dir,"dataset",TEST_FILE_NAME)
            self.test_size=0.2
        except Exception as e:
            raise SensorException(error_message=e, error_detail=sys)


class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig()):
        try:
            self.datavalidation_dir = os.path.join(training_pipeline_config.artifact_dir,"data_validation")
            self.report_file_path = os.path.join(self.datavalidation_dir,"report_file_path")
            self.missing_threshold:float = 0.2
            self.base_file_path="aps_failure_training_set1.csv"
        except Exception as e:
            raise SensorException(e, sys)

class DataTransformationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        try:
            self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir,"data_transformation")
            self.transform_object_path = os.path.join(self.data_transformation_dir,"transformer",TRANSFORMER_OBJECT_FILE_NAME)
            self.transformed_train_path=os.path.join(self.data_transformation_dir,"transformed",TRAIN_FILE_NAME.replace("csv","npz"))
            self.transformed_test_path=os.path.join(self.data_transformation_dir,"transformed",TEST_FILE_NAME.replace("csv","npz"))
            self.target_encoder_path = os.path.join(self.data_transformation_dir,"target_encoder",TARGET_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise SensorException(e, sys)
            
class ModelTrainerConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, "model_trainer")
        self.model_path = os.path.join(self.model_trainer_dir, "Model", MODEL_FILE_NAME)
        self.excepted_score = 0.7
        self.overfitting_threshold = 0.1

