import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import CustomException
from src.logger import log
from src.data_access.project_data import VehicleInsuranceData

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig()):
        """
        :param data_ingestion_config: Configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
            log.info("Initialized DataIngestion class successfully.")
        except Exception as e:
            raise CustomException(e, sys) from e

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Exports data from MongoDB and saves it as a CSV file.

        Returns:
        --------
        DataFrame: The exported data.
        """
        try:
            log.info("Starting data export from MongoDB.")
            my_data = VehicleInsuranceData()
            dataframe = my_data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )

            if dataframe.empty:
                raise CustomException("Exported data is empty. Check MongoDB collection.", sys)

            log.info(f"Data export successful. Shape of dataframe: {dataframe.shape}")

            # Save to feature store
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)

            log.info(f"Data successfully saved to feature store: {feature_store_file_path}")
            return dataframe

        except Exception as e:
            raise CustomException(e, sys) from e

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Splits the data into training and testing sets and saves them as CSV files.

        Parameters:
        -----------
        dataframe: DataFrame
            The input DataFrame to be split.
        """
        try:
            log.info("Splitting data into training and testing sets.")

            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=42
            )

            log.info(f"Data split complete. Train shape: {train_set.shape}, Test shape: {test_set.shape}")

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)

            # Save train and test datasets
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            log.info("Train and test datasets successfully saved.")

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Orchestrates the data ingestion process: exporting, splitting, and saving data.

        Returns:
        --------
        DataIngestionArtifact: Contains paths to train and test datasets.
        """
        try:
            log.info("Initiating data ingestion process.")

            # Export data
            dataframe = self.export_data_into_feature_store()
            log.info("Data export completed successfully.")

            # Split data
            self.split_data_as_train_test(dataframe)
            log.info("Data split completed successfully.")

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )

            log.info(f"Data ingestion process completed. Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
