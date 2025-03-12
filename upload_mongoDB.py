import os
import sys
import json
import certifi
import pandas as pd
import numpy as np
import pymongo
from dotenv import load_dotenv
from src.exception import CustomException
from src.logger import log

# Load environment variables
load_dotenv(dotenv_path=".env")

# Get MongoDB URL from .env file
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
CA_CERT = certifi.where()


class VehicleInsuranceETL:
    def __init__(self):
        """Initialize MongoDB client."""
        try:
            if not MONGO_DB_URL:
                raise ValueError("MongoDB URL is missing in environment variables.")
            
            self.client = pymongo.MongoClient(MONGO_DB_URL,tlsCAFile=certifi.where())

            log.info("MongoDB connection established successfully.")
        except Exception as e:
            raise CustomException(e, sys)

    def csv_to_json(self, path: str) -> list:
        """Converts a CSV file to a list of JSON records."""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"CSV file not found: {path}")

            data = pd.read_csv(path)
            data.reset_index(drop=True, inplace=True)

            records = list(json.loads(data.T.to_json()).values())
            log.info(f"Successfully converted {len(records)} records from CSV to JSON.")

            return records
        except Exception as e:
            raise CustomException(e, sys)

    def load_to_mongodb(self, records: list, database: str, collection: str) -> int:
        """Inserts records into MongoDB and returns the number of inserted records."""
        try:
            if not records:
                log.warning("No records to insert into MongoDB.")
                return 0

            db = self.client[database]
            col = db[collection]
            result = col.insert_many(records)

            log.info(f"Inserted {len(result.inserted_ids)} records into MongoDB collection: {collection}.")
            return len(result.inserted_ids)
        except Exception as e:
            raise CustomException(e, sys)
        finally:
            self.client.close()  # Ensure the MongoDB connection is closed


if __name__ == "__main__":
    FILE_PATH = 'Data\Vehicle_Insurance_data.csv'
    DATABASE = "MLOPS"
    COLLECTION = "VehicleInsuranceData"

    etl = VehicleInsuranceETL()
    
    try:
        records = etl.csv_to_json(FILE_PATH)
        num_inserted = etl.load_to_mongodb(records, DATABASE, COLLECTION)
        print(f"Inserted {num_inserted} records into MongoDB.")
    except Exception as e:
        log.error(f"An error occurred: {e}")
