import sys
import logging
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from sklearn.metrics import f1_score
from src.exception import CustomException
from src.logger import log
from src.constants import TARGET_COLUMN
from src.utils.main_utils import load_object
from src.entity.s3_estimator import ProjEstimator
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: Optional[float]
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    """
    This class handles model evaluation by comparing a trained model with the best production model in S3.
    """

    def __init__(self, model_eval_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        """
        Initialize ModelEvaluation class.
        """
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            log.error(f"Error initializing ModelEvaluation: {e}")
            raise CustomException(e, sys)

    def get_best_model(self) -> Optional[ProjEstimator]:
        """
        Fetches the best production model from S3 if available.
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            proj_estimator = ProjEstimator(bucket_name=bucket_name, model_path=model_path)

            if proj_estimator.is_model_present(model_path=model_path):
                log.info(f"Best model found at {model_path} in {bucket_name}.")
                return proj_estimator
            log.info("No existing production model found.")
            return None
        except Exception as e:
            log.error(f"Error retrieving best model: {e}")
            raise CustomException(e, sys)

    def _preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies preprocessing steps: gender mapping, dummy variables, renaming, and ID column removal.
        """
        try:
            log.info("Applying preprocessing steps...")

            # Gender Mapping
            if "Gender" in df.columns:
                df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1}).astype(int)

            # Drop ID column if present
            if "_id" in df.columns:
                df = df.drop("_id", axis=1)

            # Create Dummy Variables
            df = pd.get_dummies(df, drop_first=True)

            # Rename columns for consistency
            rename_dict = {
                "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
                "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
            }
            df = df.rename(columns=rename_dict)

            # Convert categorical dummy columns to integer type
            for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
                if col in df.columns:
                    df[col] = df[col].astype(int)

            log.info("Feature preprocessing completed.")
            return df

        except Exception as e:
            log.error(f"Error during preprocessing: {e}")
            raise CustomException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Evaluates the trained model against the best production model.
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            X, y = test_df.drop(columns=[TARGET_COLUMN]), test_df[TARGET_COLUMN]

            log.info("Test data loaded. Beginning preprocessing...")

            X = self._preprocess_features(X)

            # Load trained model
            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            log.info(f"Trained Model F1 Score: {trained_model_f1_score}")

            # Compare with best production model (if available)
            best_model = self.get_best_model()
            best_model_f1_score = None

            if best_model is not None:
                log.info("Evaluating production model performance...")
                y_hat_best_model = best_model.predict(X)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                log.info(f"Production Model F1 Score: {best_model_f1_score}")

            # Determine if the trained model should replace the best model
            tmp_best_model_score = best_model_f1_score or 0
            is_model_accepted = trained_model_f1_score > tmp_best_model_score
            difference = trained_model_f1_score - tmp_best_model_score

            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=is_model_accepted,
                difference=difference
            )

            log.info(f"Evaluation Result: {result}")
            return result

        except Exception as e:
            log.error(f"Error during model evaluation: {e}")
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Initiates the model evaluation process and returns an evaluation artifact.
        """
        try:
            log.info("Starting model evaluation...")

            evaluate_model_response = self.evaluate_model()

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
            )

            log.info(f"Model evaluation completed: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            log.error(f"Error during model evaluation initiation: {e}")
            raise CustomException(e, sys)
