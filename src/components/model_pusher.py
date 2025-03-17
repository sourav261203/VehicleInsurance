import sys
from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import CustomException
from src.logger import log
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import ProjEstimator


class ModelPusher:
    def __init__(self, model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        Initializes ModelPusher.

        :param model_evaluation_artifact: Contains model evaluation details including trained model path.
        :param model_pusher_config: Configuration details for model pushing.
        """
        self.s3 = SimpleStorageService()
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config
        self.proj_estimator = ProjEstimator(
            bucket_name=model_pusher_config.bucket_name,
            model_path=model_pusher_config.s3_model_key_path
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Uploads the trained model to an S3 bucket and returns a ModelPusherArtifact.

        :return: ModelPusherArtifact containing bucket name and S3 model path.
        :raises MyException: If any error occurs during the model push process.
        """
        log.info("Entered initiate_model_pusher method of ModelPusher class.")

        try:
            log.info("Uploading trained model to S3 bucket...")

            # Upload model to S3
            self.proj_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)

            # Create artifact after successful upload
            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path
            )

            log.info(f"Model successfully uploaded to S3 at {self.model_pusher_config.s3_model_key_path}.")
            log.info(f"Model Pusher Artifact: {model_pusher_artifact}")
            log.info("Exited initiate_model_pusher method of ModelPusher class.")

            return model_pusher_artifact

        except Exception as e:
            log.error(f"Error in initiate_model_pusher: {str(e)}", exc_info=True)
            raise CustomException(e, sys) from e
