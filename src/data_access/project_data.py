import sys
import pandas as pd
import numpy as np
from typing import Optional

from src.configuration.mongo_db_connection import MongoDBClient
from src.constants import DATABASE_NAME
from src.exception import CustomException
from src.logger import log

class VehicleInsuranceData:
    """
    A class to export MongoDB records as a pandas DataFrame.
    """

    def __init__(self) -> None:
        """
        Initializes the MongoDB client connection.
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
            log.info(f"MongoDB connection initialized for database: {DATABASE_NAME}")
        except Exception as e:
            log.error(f"Error initializing MongoDB connection: {str(e)}")
            raise CustomException(e, sys)

    def export_collection_as_dataframe(self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        """
        Exports an entire MongoDB collection as a pandas DataFrame.

        Parameters:
        ----------
        collection_name : str
            The name of the MongoDB collection to export.
        database_name : Optional[str], default=None
            Name of the database (optional). Defaults to DATABASE_NAME.

        Returns:
        -------
        pd.DataFrame
            DataFrame containing the collection data, with '_id' column removed and 'na' values replaced with NaN.

        Raises:
        ------
        MyException
            If an error occurs during data extraction.
        """
        try:
            db_name = database_name if database_name else DATABASE_NAME
            collection = self.mongo_client.client[db_name][collection_name]

            log.info(f"Fetching data from MongoDB | Database: {db_name}, Collection: {collection_name}")
            
            # Convert MongoDB collection to DataFrame
            df = pd.DataFrame(list(collection.find()))
            log.info(f"Data fetched successfully | Records: {len(df)}")

            # Remove MongoDB's default `_id` column
            if "_id" in df.columns:
                df.drop(columns=["_id"], inplace=True)

            # Replace "na" with NaN for proper handling of missing values
            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            log.error(f"Error fetching data from collection '{collection_name}' in database '{db_name}': {str(e)}")
            raise CustomException(e, sys)
