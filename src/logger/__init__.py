import logging
import os
from datetime import datetime


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.FileHandler(log_filename),  # Log to file
        logging.StreamHandler()  # Log to console
    ]
)

# Get the logger instance
log = logging.getLogger(__name__)
