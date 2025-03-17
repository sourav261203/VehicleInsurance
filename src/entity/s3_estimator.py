import sys
import logging
from pandas import DataFrame
from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import CustomException
from src.logger import log
from src.entity.estimator import MyModel


class ProjEstimator:
    """
    A class to manage model saving, retrieval from an S3 bucket, and making predictions.
    """

    def __init__(self, bucket_name: str, model_path: str):
        """
        Initializes Proj1Estimator.

        Args:
            bucket_name (str): The name of the S3 bucket.
            model_path (str): The path to the model file in the S3 bucket.
        """
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.s3 = SimpleStorageService()
        self.loaded_model: MyModel = None

    def is_model_present(self, model_path: str) -> bool:
        """
        Checks if the model exists in the S3 bucket.

        Args:
            model_path (str): Path of the model in S3.

        Returns:
            bool: True if the model exists, False otherwise.
        """
        try:
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=model_path)
        except CustomException as e:
            log.error(f"Error checking model presence in S3: {e}")
            return False

    def load_model(self) -> MyModel:
        """
        Loads the model from the S3 bucket.

        Returns:
            MyModel: The loaded model.
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.s3.load_model(self.model_path, bucket_name=self.bucket_name)
                log.info(f"Model loaded successfully from {self.model_path}")
            return self.loaded_model
        except Exception as e:
            raise CustomException(f"Error loading model from {self.model_path}: {e}", sys)

    def save_model(self, from_file: str, remove: bool = False) -> None:
        """
        Saves the model to the S3 bucket.

        Args:
            from_file (str): Path of the local model file.
            remove (bool, optional): If True, deletes the local file after upload. Defaults to False.
        """
        try:
            self.s3.upload_file(
                from_filename=from_file,
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove
            )
            log.info(f"Model uploaded to {self.model_path} in {self.bucket_name}")
        except Exception as e:
            raise CustomException(f"Error saving model to S3: {e}", sys)

    def predict(self, dataframe: DataFrame) -> DataFrame:
        """
        Uses the loaded model to make predictions on the provided DataFrame.

        Args:
            dataframe (DataFrame): Input data for prediction.

        Returns:
            DataFrame: Model predictions.
        """
        try:
            if self.loaded_model is None:
                self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise CustomException(f"Error during prediction: {e}", sys)
