import os
import sys
import dill
import yaml
import numpy as np
from pandas import DataFrame

from src.exception import CustomException
from src.logger import log


def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.

    Parameters:
    ----------
    file_path : str
        Path to the YAML file.

    Returns:
    -------
    dict
        Parsed YAML content.
    """
    try:
        log.info(f"Reading YAML file from {file_path}")
        with open(file_path, "r", encoding="utf-8") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes content to a YAML file.

    Parameters:
    ----------
    file_path : str
        Path where the YAML file will be saved.
    content : object
        Data to write.
    replace : bool, optional
        If True, replaces the existing file. Defaults to False.
    """
    try:
        log.info(f"Writing YAML file to {file_path}")
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as file:
            yaml.dump(content, file, default_flow_style=False)
    except Exception as e:
        raise CustomException(e, sys) from e


def load_object(file_path: str) -> object:
    """
    Loads and returns a serialized object from a file.

    Parameters:
    ----------
    file_path : str
        Path to the serialized object file.

    Returns:
    -------
    object
        The deserialized object.
    """
    try:
        log.info(f"Loading object from {file_path}")
        if not os.path.exists(file_path):
            raise CustomException(f"File not found: {file_path}", sys)
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    """
    Saves an object using dill serialization.

    Parameters:
    ----------
    file_path : str
        Path where the object will be saved.
    obj : object
        The object to save.
    """
    log.info(f"Saving object to {file_path}")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        log.info(f"Object successfully saved to {file_path}")

    except Exception as e:
        raise CustomException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Saves a NumPy array to a file.

    Parameters:
    ----------
    file_path : str
        Path where the NumPy array will be saved.
    array : np.ndarray
        The NumPy array to save.
    """
    try:
        log.info(f"Saving NumPy array to {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, array, allow_pickle=True)
        log.info(f"NumPy array successfully saved to {file_path}")

    except Exception as e:
        raise CustomException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Loads a NumPy array from a file.

    Parameters:
    ----------
    file_path : str
        Path to the NumPy array file.

    Returns:
    -------
    np.ndarray
        The loaded NumPy array.
    """
    try:
        log.info(f"Loading NumPy array from {file_path}")
        if not os.path.exists(file_path):
            raise CustomException(f"File not found: {file_path}", sys)
        return np.load(file_path, allow_pickle=True)
    except Exception as e:
        raise CustomException(e, sys) from e
