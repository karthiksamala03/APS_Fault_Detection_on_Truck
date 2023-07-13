from src.entity import config_entity, artifact_entity
from src.logger import logging
from src.exception import SensorException
from src import utils
from src.predictor import ModelResolver
import os, sys
from src.config import TARGET_COLUMN
from sklearn.metrics import f1_score
import pandas as pd

class ModelEvaluation:
    def __init__(self, 
                model_evaluation_config:config_entity.ModelEvaluationConfig,
                model_ingestion_artifact:artifact_entity.ModelIngestionArtifact,
                model_transformation_artifact:artifact_entity.ModelTransformationArtifact,
                model_trainer_artifact:artifact_entity.ModelTrainerArtifact ):
        try:
            logging.info(f"{'>>'*20}  Model Evaluation {'<<'*20}")
            self.model_evaluation_config = model_evaluation_config
            self.model_ingestion_artifact = model_ingestion_artifact
            self.model_transformation_artifact = model_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = model_resolver()

        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_evaluation(self,)->artifact_entity.ModelEvaluationArtifact:
        try:
            # if saved model folder has model, then will compare 
            # which model is best, the model trained or the model from the saved model folder
            logging.info("if saved model folder has model, then will compare "
                "which model is best, the model trained or the model from the saved model folder")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path == None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                                                                                improved_accuracy= None)
                logging.info(f"Model_Evaluation_artifact : {model_eval_artifact}")
                return model_eval_artifact

            # Finding location of Model, Transformer and target encoder
            logging.info("Finding location of Model, Transformer and target encoder")
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()
            target_encoder_path  = self.model_resolver.get_latest_target_encoder_path()

            # Loading Previously trained Transformer, model and Target encoder objects
            logging.info(f"Loading Previously trained Transformer, model and Target encoder objects")
            transformer = utils.load_object(file_path=transformer_path)
            model = utils.load_object(file_path=model_path)
            target_encoder = load_object(file_path=target_encoder_path)

            # Loading Currently Trained Transformer, model and Target encoder objects
            logging.info("Loading Currently Trained Transformer, model and Target encoder objects")
            current_transformer = utils.load_object(file_path=self.model_transformation_artifact.transform_object_path)
            current_model = utils.load_object(file_path=self.model_trainer_artifact.model_path)
            current_target_encoder = utils.load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_df = test_df[TARGET_COLUMN]
            y_true = target_encoder.transform(target_df)

            # Accuracy using previous trained model
            input_feature_names = list(transformer.feature_names_in_)
            input_arr = transformer.transform(test_df[input_feature_names])
            y_pred = model.predict(input_arr)
            print(f"Prediction using previous model :{target_encoder.inverse_transform(y_pred[:5])}")
            previous_model_score = f1_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Accuracy using previous trained model : {previous_model_score}")

            # Accuracy using Current trained model
            input_feature_names = list(current_transformer.feature_names_in_)
            input_arr = current_transformer.transform(test_df[input_feature_names])
            y_pred = current_model.predict(input_arr)
            y_true = current_target_encoder.transform(target_df)
            print(f"Prediction using current model :{current_target_encoder.inverse_transform(y_pred[:5])}")
            current_model_score = f1_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"Accuracy using current trained model : {current_model_score}")

            if current_model_score<=previous_model_score:
                logging.info(f"Current trained model is not better than previous model")
                raise Exception("Current trained model is not better than previous model")

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
            improved_accuracy=current_model_score-previous_model_score)
            logging.info(f"model Evalution Artifact :{model_eval_artifact}")
            return model_eval_artifact
            
        except Exception as e:
            raise SensorException(e, sys)



