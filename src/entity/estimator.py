import sys
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import log


class TargetValueMapping:
    """
    Class to map categorical target values to numerical representations and vice versa.
    """
    def __init__(self):
        self.mapping = {"yes": 0, "no": 1}

    def _asdict(self) -> dict:
        """
        Returns the dictionary representation of the mapping.
        """
        return self.mapping

    def reverse_mapping(self) -> dict:
        """
        Returns a reversed dictionary to map numeric values back to original categories.
        """
        return {v: k for k, v in self.mapping.items()}


class MyModel:
    """
    Wrapper class for preprocessing and trained model to enable prediction on new data.
    """
    def __init__(self, preprocessing_object: Pipeline, trained_model_object: object):
        """
        :param preprocessing_object: Preprocessing pipeline for data transformation.
        :param trained_model_object: Trained model for making predictions.
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe: pd.DataFrame) -> DataFrame:
        """
        Applies preprocessing transformations and predicts target values.

        :param dataframe: Input DataFrame containing features.
        :return: Array of predicted values.
        """
        try:
            log.info("Starting prediction process.")

            if dataframe.empty:
                log.warning("Received an empty DataFrame for prediction.")
                raise ValueError("Input DataFrame is empty. Prediction aborted.")

            # Apply preprocessing transformations
            log.info("Applying preprocessing transformations.")
            transformed_features = self.preprocessing_object.transform(dataframe)

            # Make predictions
            log.info("Generating predictions using the trained model.")
            predictions = self.trained_model_object.predict(transformed_features)

            return predictions

        except Exception as e:
            log.error(f"Error occurred in predict method: {str(e)}", exc_info=True)
            raise CustomException(e, sys) from e

    def __repr__(self) -> str:
        return f"MyModel(trained_model_object={type(self.trained_model_object).__name__})"

    def __str__(self) -> str:
        return f"Trained Model: {type(self.trained_model_object).__name__}"
