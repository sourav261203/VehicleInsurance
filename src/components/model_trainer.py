import sys
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.exception import CustomException
from src.logger import log
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.estimator import MyModel


class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        Initialize ModelTrainer class with data transformation artifact and configuration.

        :param data_transformation_artifact: Output from data transformation stage
        :param model_trainer_config: Configuration parameters for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.ndarray, test: np.ndarray) -> Tuple[RandomForestClassifier, ClassificationMetricArtifact]:
        """
        Trains a RandomForestClassifier and returns the trained model along with evaluation metrics.

        :param train: Training dataset (numpy array)
        :param test: Testing dataset (numpy array)
        :return: Tuple containing trained model and metric artifact
        """
        try:
            log.info("Splitting train and test data into features and target variables.")
            x_train, y_train = train[:, :-1], train[:, -1]
            x_test, y_test = test[:, :-1], test[:, -1]

            log.info("Initializing RandomForestClassifier with specified parameters.")
            model = RandomForestClassifier(
                n_estimators = self.model_trainer_config._n_estimators,
                min_samples_split = self.model_trainer_config._min_samples_split,
                min_samples_leaf = self.model_trainer_config._min_samples_leaf,
                max_depth = self.model_trainer_config._max_depth,
                criterion = self.model_trainer_config._criterion,
                random_state = self.model_trainer_config._random_state
            )

            log.info("Training the RandomForest model...")
            model.fit(x_train, y_train)

            log.info("Generating predictions and computing evaluation metrics.")
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            log.info(f"Model Evaluation - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

            # Creating metric artifact
            metric_artifact = ClassificationMetricArtifact(
                f1_score=f1,
                precision_score=precision,
                recall_score=recall
            )
            return model, metric_artifact

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Initiates the model training process.

        :return: ModelTrainerArtifact containing trained model path and evaluation metrics
        """
        try:
            log.info("Loading transformed training and testing datasets.")
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            log.info("Training the model and obtaining evaluation metrics.")
            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)

            log.info("Loading preprocessing object.")
            preprocessing_obj = load_object(self.data_transformation_artifact.transformed_object_file_path)

            log.info("Validating trained model accuracy against expected threshold.")
            train_accuracy = accuracy_score(train_arr[:, -1], trained_model.predict(train_arr[:, :-1]))
            if train_accuracy < self.model_trainer_config.expected_accuracy:
                log.warning(f"Model accuracy {train_accuracy:.4f} is below the expected threshold {self.model_trainer_config.expected_accuracy:.4f}.")
                raise CustomException("Trained model does not meet the required accuracy threshold.")

            log.info("Saving the trained model along with preprocessing pipeline.")
            final_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, final_model)

            log.info(f"Model saved successfully at {self.model_trainer_config.trained_model_file_path}")

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact
            )

        except Exception as e:
            raise CustomException(e, sys) from e
