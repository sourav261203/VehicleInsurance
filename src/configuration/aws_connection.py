import boto3
import os
from src.constants import AWS_SECRET_ACCESS_KEY_ENV_KEY, AWS_ACCESS_KEY_ID_ENV_KEY, REGION_NAME
from src.logger import log


class S3Client:
    """
    A Singleton class to manage AWS S3 connection using boto3.
    Ensures credentials are securely retrieved from environment variables.
    """
    _s3_client = None
    _s3_resource = None

    def __init__(self, region_name: str = REGION_NAME):
        """
        Initializes an S3 connection if not already established.
        Raises an exception if required AWS credentials are missing.
        """
        if not S3Client._s3_client or not S3Client._s3_resource:
            self._initialize_s3_connection(region_name)

        self.s3_client = S3Client._s3_client
        self.s3_resource = S3Client._s3_resource

    @classmethod
    def _initialize_s3_connection(cls, region_name: str):
        """Sets up the S3 client and resource if not already initialized."""
        log.info("Initializing AWS S3 connection...")

        access_key_id = os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY)
        secret_access_key = os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY)

        if not access_key_id:
            raise ValueError(f"Missing environment variable: {AWS_ACCESS_KEY_ID_ENV_KEY}")
        if not secret_access_key:
            raise ValueError(f"Missing environment variable: {AWS_SECRET_ACCESS_KEY_ENV_KEY}")

        try:
            cls._s3_resource = boto3.resource(
                's3',
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region_name
            )
            cls._s3_client = boto3.client(
                's3',
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=region_name
            )
            log.info("AWS S3 connection established successfully.")

        except Exception as e:
            log.error(f"Failed to initialize AWS S3 connection: {str(e)}", exc_info=True)
            raise RuntimeError("AWS S3 initialization failed. Check credentials and network.") from e
