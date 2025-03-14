import os
import sys
import pymongo
import certifi
from pymongo.errors import PyMongoError
from dotenv import load_dotenv

from src.exception import CustomException
from src.logger import log
from src.constants import DATABASE_NAME, MONGODB_URL_KEY

# Load the CA certificate to ensure a secure MongoDB connection
ca = certifi.where()
load_dotenv()

class MongoDBClient:
    """
    A singleton class for establishing and managing MongoDB connections.

    Attributes:
    -----------
    client : MongoClient
        A shared MongoClient instance across all instances of MongoDBClient.
    database : Database
        The specific database instance that the client connects to.

    Methods:
    --------
    __init__(database_name: str) -> None
        Initializes the MongoDB connection using the provided database name.
    """

    client = None  # Shared MongoClient instance for connection reuse

    def __init__(self, database_name: str = DATABASE_NAME) -> None:
        """
        Initializes a MongoDB connection. Reuses the existing connection if already established.

        Parameters:
        -----------
        database_name : str, optional
            Name of the MongoDB database to connect to. Default is DATABASE_NAME.

        Raises:
        -------
        MyException
            If there is an issue connecting to MongoDB or the required environment variable is not set.
        """
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)

                if not mongo_db_url:
                    error_msg = f"Environment variable '{MONGODB_URL_KEY}' is not set."
                    log.error(error_msg)
                    raise CustomException(error_msg, sys)

                log.info(f"Connecting to MongoDB at: {mongo_db_url}")

                # Establish a secure MongoDB connection
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

                log.info("MongoDB connection established successfully.")

            # Assign the shared client to this instance
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name

            log.info(f"Connected to MongoDB database: {database_name}")

        except PyMongoError as e:
            error_msg = f"MongoDB connection error: {str(e)}"
            log.error(error_msg)
            raise CustomException(error_msg, sys)
        except Exception as e:
            log.error(f"Unexpected error: {str(e)}")
            raise CustomException(e, sys)


if __name__=="__main__":
    MongoDBClient(DATABASE_NAME)