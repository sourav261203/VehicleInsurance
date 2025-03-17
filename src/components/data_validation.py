import json
import sys
import os

import pandas as pd
from pandas import DataFrame

from src.exception import CustomException
from src.logger import log
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        Initialize DataValidation class.

        :param data_ingestion_artifact: Reference to data ingestion artifact.
        :param data_validation_config: Configuration for data validation.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys) from e

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        Validates the number of columns in the dataframe.

        :param dataframe: Input dataframe.
        :return: True if column count matches schema, False otherwise.
        """
        try:
            expected_column_count = len(self._schema_config["columns"])
            actual_column_count = len(dataframe.columns)
            status = expected_column_count == actual_column_count
            log.info(f"Expected columns: {expected_column_count}, Actual columns: {actual_column_count}")
            return status
        except Exception as e:
            raise CustomException(e, sys) from e

    def validate_column_existence(self, df: DataFrame) -> bool:
        """
        Validates the existence of required numerical and categorical columns.

        :param df: Input dataframe.
        :return: True if all required columns exist, False otherwise.
        """
        try:
            dataframe_columns = set(df.columns)
            missing_columns = {
                "numerical": [col for col in self._schema_config["numerical_columns"] if col not in dataframe_columns],
                "categorical": [col for col in self._schema_config["categorical_columns"] if col not in dataframe_columns],
            }

            for col_type, missing in missing_columns.items():
                if missing:
                    log.warning(f"Missing {col_type} columns: {missing}")

            return not any(missing_columns.values())  # True if no missing columns, False otherwise.
        except Exception as e:
            raise CustomException(e, sys) from e

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        """
        Reads a CSV file and returns a dataframe.

        :param file_path: Path to the CSV file.
        :return: DataFrame containing the data.
        """
        try:
            log.info(f"Reading data from {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Initiates the data validation process.

        :return: DataValidationArtifact containing validation results.
        """
        try:
            log.info("Starting data validation process.")

            # Read train and test datasets
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            validation_errors = []

            # Column count validation
            for df_type, df in {"Training": train_df, "Testing": test_df}.items():
                if not self.validate_number_of_columns(df):
                    validation_errors.append(f"{df_type} dataset has incorrect number of columns.")

                if not self.validate_column_existence(df):
                    validation_errors.append(f"{df_type} dataset is missing required columns.")

            validation_status = len(validation_errors) == 0
            validation_message = "No validation errors found." if validation_status else " ".join(validation_errors)

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_message,
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            # Ensure the directory for validation_report_file_path exists
            os.makedirs(os.path.dirname(self.data_validation_config.validation_report_file_path), exist_ok=True)

            # Save validation status and message to a JSON file
            with open(self.data_validation_config.validation_report_file_path, "w") as report_file:
                json.dump({"validation_status": validation_status, "message": validation_message}, report_file, indent=4)

            log.info("Data validation completed successfully.")
            log.info(f"Validation Report: {data_validation_artifact}")

            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
