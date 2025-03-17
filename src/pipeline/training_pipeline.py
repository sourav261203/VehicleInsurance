import sys
from src.exception import CustomException
from src.logger import log

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_pusher import ModelPusher

from src.entity.config_entity import (DataIngestionConfig,
                                          DataValidationConfig,
                                          DataTransformationConfig,
                                          ModelTrainerConfig,
                                          ModelEvaluationConfig,
                                          ModelPusherConfig)
                                          
from src.entity.artifact_entity import (DataIngestionArtifact,
                                            DataValidationArtifact,
                                            DataTransformationArtifact,
                                            ModelTrainerArtifact,
                                            ModelEvaluationArtifact,
                                            ModelPusherArtifact)



class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()


    
    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        This method of TrainPipeline class is responsible for starting data ingestion component
        """
        try:
            log.info("Entered the start_data_ingestion method of TrainPipeline class")
            log.info("Getting the data from mongodb")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            log.info("Got the train_set and test_set from mongodb")
            log.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys) from e
        

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        """
        Starts the data validation process as part of the TrainPipeline.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): The artifact containing paths to ingested training and test data.

        Returns:
            DataValidationArtifact: The artifact containing validation results.
        """
        log.info("Starting data validation in TrainPipeline...")

        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config
            )

            log.info("Initializing data validation process.")
            return data_validation.initiate_data_validation()

        except Exception as e:
            log.error(f"Error occurred during data validation: {e}", exc_info=True)
            raise CustomException(e, sys) from e
        
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact, 
                              data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        """
        Starts the data transformation process by initializing the DataTransformation component.
        
        Args:
            data_ingestion_artifact (DataIngestionArtifact): Contains paths for ingested training and test data.
            data_validation_artifact (DataValidationArtifact): Contains validation status and schema-related information.

        Returns:
            DataTransformationArtifact: Contains paths for transformed data and preprocessing objects.
        """
        log.info("Starting data transformation process.")

        try:
            # Initialize data transformation
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
                data_validation_artifact=data_validation_artifact
            )
            log.info("DataTransformation instance created successfully.")

            # Perform data transformation
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            log.info("Data transformation completed successfully.")

            return data_transformation_artifact

        except Exception as e:
            log.exception("Error occurred during data transformation.")
            raise CustomException(e, sys) from e
        

    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
            """
            This method of TrainPipeline class is responsible for starting model training
            """
            try:
                model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                            model_trainer_config=self.model_trainer_config
                                            )
                model_trainer_artifact = model_trainer.initiate_model_trainer()
                return model_trainer_artifact

            except Exception as e:
                raise CustomException(e, sys)
            
    def start_model_evaluation(self, data_ingestion_artifact: DataIngestionArtifact,
                               model_trainer_artifact: ModelTrainerArtifact) -> ModelEvaluationArtifact:
        """
        This method of TrainPipeline class is responsible for starting modle evaluation
        """
        try:
            model_evaluation = ModelEvaluation(model_eval_config=self.model_evaluation_config,
                                               data_ingestion_artifact=data_ingestion_artifact,
                                               model_trainer_artifact=model_trainer_artifact)
            model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
            return model_evaluation_artifact
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_model_pusher(self, model_evaluation_artifact: ModelEvaluationArtifact) -> ModelPusherArtifact:
        """
        This method of TrainPipeline class is responsible for starting model pushing
        """
        try:
            model_pusher = ModelPusher(model_evaluation_artifact=model_evaluation_artifact,
                                       model_pusher_config=self.model_pusher_config
                                       )
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            return model_pusher_artifact
        except Exception as e:
            raise CustomException(e, sys)


        

    def run_pipeline(self, ) -> None:
        """
        This method of TrainPipeline class is responsible for running complete pipeline
        """
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact=data_ingestion_artifact, data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            model_evaluation_artifact = self.start_model_evaluation(data_ingestion_artifact=data_ingestion_artifact,
                                                                    model_trainer_artifact=model_trainer_artifact)
            if not model_evaluation_artifact.is_model_accepted:
                log.info(f"Model not accepted.")
                return None
            model_pusher_artifact = self.start_model_pusher(model_evaluation_artifact=model_evaluation_artifact)
            
        except Exception as e:
            raise CustomException(e, sys)