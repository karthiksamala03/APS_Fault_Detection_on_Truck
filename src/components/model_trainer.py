from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from src.logger import logging
from src.exception import SensorException
from src import utils
import os, sys
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

class ModelTrainer:
    def __init__(self, model_trainer_config:ModelTrainerConfig, 
                data_transformation_artifact:DataTransformationArtifact ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise SensorException(e, sys)

    def train_model(self, X, y):
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(X, y)
            return xgb_clf
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_trainer(self,):
        try:
            logging.info(f"{'>>'* 20} Model Trainer {'<<'*20}")
            logging.info("loading train and test array")
            train_arr = utils.load_numpy_arr_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_arr_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info("Splitting input and target feature from both train and test arr")
            x_train, y_train = train_arr[:,:-1], train_arr[:,-1]
            x_test, y_test = test_arr[:,:-1], test_arr[:,-1]

            logging.info("training the model")
            model = self.train_model(X=x_train, y=y_train)

            logging.info("Calculating f1 train score")
            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_true=y_train, y_pred=yhat_train)

            logging.info("Calculating f1 test score")
            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_true=y_test, y_pred=yhat_test)

            logging.info(f"f1 train score : {f1_train_score} and f1 test score : {f1_test_score}")

            # check for over fitting , under fitting or excepted score
            logging.info("checking if model is under fitting or not ")
            if f1_test_score < self.model_trainer_config.excepted_score:
                raise Exception(f"Model is not good as it is not able to give excepted accuracy: \
                    {self.model_trainer_config.excepted_score} Model actual score {f1_test_score} ")

            logging.info("Checking if model is Overfitting or not ")
            diff = abs(f1_train_score - f1_test_score)

            if diff>self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and test score diff : {diff} is more than overfitting threshold \
                    :{self.model_trainer_config.overfitting_threshold}")

            # Saving the trained model
            logging.info("Saving Model Object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            # Prepare artifacts
            logging.info("Prepare model trainer artifacts")
            model_trainer_artifact = ModelTrainerArtifact(model_path=self.model_trainer_config.model_path, 
                                                        f1_train_score=f1_train_score, f1_test_score=f1_test_score)

            logging.info(f"Model Trainer Atifact : {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise SensorException(e, sys)


