import logging
import os
from datetime import datetime
from from_root import from_root

# Generate log filename with timestamp
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d-%H%M%S')}.log"

# Define log folder path
log_dir = os.path.join(from_root(), 'logs')

# Ensure log directory exists
os.makedirs(log_dir, exist_ok=True)

# Define log file path
LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE_PATH,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

