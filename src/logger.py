import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FOLDER = f"{datetime.now().strftime('%m_%d_%Y')}"
log_folder_path = os.path.join(os.getcwd(), 'logs', LOG_FOLDER)
os.makedirs(log_folder_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_folder_path, LOG_FILE)

logging.basicConfig(
    filename= LOG_FILE_PATH,
    format="[%(asctime)s] - %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO  
)


# if __name__ == '__main__':
#     logging.info('Logging Started...')


