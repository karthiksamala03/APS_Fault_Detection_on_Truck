import os, sys
from src.exception import SensorException
from src.logger import logging
from src.entity import config_entity, artifact_entity
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation


training_pipeline_config = config_entity.TrainingPipelineConfig()

try:
    #data ingestion
    data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
    data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
    data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
    print(data_ingestion_artifact)

    #data validation
    data_validation_config = config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
    data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact, data_validation_config=data_validation_config)
    data_validation_artifact = data_validation.initiate_data_validation()
    print(data_validation_artifact)
    
    #data trnasformation
    data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
    data_transformation = DataTransformation(data_transformation_config=data_transformation_config, data_ingestion_artifact=data_ingestion_artifact)
    data_transformation_artifact = data_transformation.initiate_data_transformation()
    print(data_transformation_artifact)
    
except Exception as e:
    raise SensorException(e, sys)