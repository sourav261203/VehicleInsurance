import sys
import logging

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extracts detailed error information including file name, line number, and the error message.

    :param error: The exception that occurred.
    :param error_detail: The sys module to access traceback details.
    :return: A formatted error message string.
    """
    exc_type, _, exc_tb = error_detail.exc_info()  # Extract traceback details
    
    if exc_tb is None:  
        return f"An error occurred: {error}"  # Fallback if no traceback
    
    # Get deepest traceback frame (for nested errors)
    while exc_tb.tb_next is not None:
        exc_tb = exc_tb.tb_next

    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the filename
    line_number = exc_tb.tb_lineno  # Get the exact line number
    error_message = f"Error in file [{file_name}], line [{line_number}]: {str(error)}"

    logging.error(error_message)  # Log error
    return error_message

class CustomException(Exception):
    """
    Custom exception class for handling errors in a standardized way.
    """
    def __init__(self, error_message: str, error_detail: sys):
        """
        Initializes the CustomException with a detailed error message.

        :param error_message: A string describing the error.
        :param error_detail: The sys module to access traceback details.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        """Returns the formatted error message."""
        return self.error_message

        
