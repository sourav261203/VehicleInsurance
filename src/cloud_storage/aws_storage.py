import boto3
from src.configuration.aws_connection import S3Client
from io import StringIO
from typing import Union, List
import os
import sys
from src.logger import log
from mypy_boto3_s3.service_resource import Bucket
from src.exception import CustomException
from botocore.exceptions import ClientError
from pandas import DataFrame, read_csv
import pickle


class SimpleStorageService:
    """
    A class for interacting with AWS S3 storage, providing methods for file management, 
    data uploads, and data retrieval in S3 buckets.
    """

    def __init__(self):
        """Initializes the SimpleStorageService instance with S3 resource and client from the S3Client class."""
        s3_client = S3Client()
        self.s3_resource = s3_client.s3_resource
        self.s3_client = s3_client.s3_client

    def s3_key_path_available(self, bucket_name: str, s3_key: str) -> bool:
        """Checks if a specified S3 key path exists in the bucket."""
        try:
            bucket = self.get_bucket(bucket_name)
            return any(bucket.objects.filter(Prefix=s3_key))
        except Exception as e:
            log.error(f"Error checking S3 key path: {s3_key} in bucket: {bucket_name}")
            raise CustomException(e, sys)

    @staticmethod
    def read_object(object_: object, decode: bool = True, make_readable: bool = False) -> Union[StringIO, str]:
        """Reads an S3 object with optional decoding and conversion to StringIO."""
        try:
            content = object_.get()["Body"].read()
            if decode:
                content = content.decode()
            return StringIO(content) if make_readable else content
        except Exception as e:
            log.error("Error reading S3 object")
            raise CustomException(e, sys)

    def get_bucket(self, bucket_name: str) -> Bucket:
        """Retrieves the S3 bucket object."""
        try:
            return self.s3_resource.Bucket(bucket_name)
        except Exception as e:
            log.error(f"Error retrieving bucket: {bucket_name}")
            raise CustomException(e, sys)

    def get_file_object(self, filename: str, bucket_name: str) -> Union[List[object], object]:
        """Retrieves the file object(s) from the specified S3 bucket."""
        try:
            bucket = self.get_bucket(bucket_name)
            file_objects = list(bucket.objects.filter(Prefix=filename))
            return file_objects[0] if len(file_objects) == 1 else file_objects
        except Exception as e:
            log.error(f"Error retrieving file: {filename} from bucket: {bucket_name}")
            raise CustomException(e, sys)

    def load_model(self, model_name: str, bucket_name: str, model_dir: str = None) -> object:
        """Loads a serialized model from the S3 bucket."""
        try:
            model_file = f"{model_dir}/{model_name}" if model_dir else model_name
            file_object = self.get_file_object(model_file, bucket_name)
            model_data = self.read_object(file_object, decode=False)

            if not model_data:
                raise CustomException(f"Model file {model_name} is empty or missing.", sys)

            log.info(f"Loaded model {model_name} from bucket {bucket_name}")
            return pickle.loads(model_data)
        except Exception as e:
            log.error(f"Error loading model: {model_name} from bucket: {bucket_name}")
            raise CustomException(e, sys)

    def create_folder(self, folder_name: str, bucket_name: str) -> None:
        """Creates a folder in the specified S3 bucket."""
        try:
            folder_key = f"{folder_name}/"
            self.s3_resource.Object(bucket_name, folder_key).load()
            log.info(f"Folder {folder_name} already exists in bucket {bucket_name}")
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                self.s3_client.put_object(Bucket=bucket_name, Key=folder_key)
                log.info(f"Created folder {folder_name} in bucket {bucket_name}")
            else:
                raise CustomException(e, sys)

    def upload_file(self, from_filename: str, to_filename: str, bucket_name: str, remove: bool = True):
        """Uploads a local file to S3 with an optional delete flag."""
        try:
            log.info(f"Uploading {from_filename} to {to_filename} in {bucket_name}")
            self.s3_resource.meta.client.upload_file(from_filename, bucket_name, to_filename)

            if remove:
                os.remove(from_filename)
                log.info(f"Removed local file {from_filename} after upload")
        except Exception as e:
            log.error(f"Error uploading file: {from_filename} to {to_filename} in bucket: {bucket_name}")
            raise CustomException(e, sys)

    def upload_df_as_csv(self, data_frame: DataFrame, local_filename: str, bucket_filename: str, bucket_name: str) -> None:
        """Uploads a DataFrame as a CSV file to S3."""
        try:
            data_frame.to_csv(local_filename, index=False, header=True)
            self.upload_file(local_filename, bucket_filename, bucket_name)
        except Exception as e:
            log.error(f"Error uploading DataFrame as CSV to bucket: {bucket_name}")
            raise CustomException(e, sys)

    def get_df_from_object(self, object_: object) -> DataFrame:
        """Converts an S3 object to a DataFrame."""
        try:
            content = self.read_object(object_, make_readable=True)
            return read_csv(content, na_values="na")
        except Exception as e:
            log.error("Error converting S3 object to DataFrame")
            raise CustomException(e, sys)

    def read_csv(self, filename: str, bucket_name: str) -> DataFrame:
        """Reads a CSV file from S3 and converts it to a DataFrame."""
        try:
            csv_obj = self.get_file_object(filename, bucket_name)
            return self.get_df_from_object(csv_obj)
        except Exception as e:
            log.error(f"Error reading CSV file: {filename} from bucket: {bucket_name}")
            raise CustomException(e, sys)
