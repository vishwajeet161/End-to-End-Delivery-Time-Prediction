import logging
import os, sys
from datetime import datetime

LOG_DIR = "logs"
LOG_DIR = os.path.join(os.getcwd(), LOG_DIR) # os.getcwd() finds current working directory

#code to create LOD_DIR if not exist 
os.makedirs(LOG_DIR, exist_ok=True)

# file extention is .log with name as timestamp 
# fromat - "log_yyyy_mm_dd_hh_MM_ss.log"
CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
file_name = f"log_{CURRENT_TIME_STAMP}.log"

log_file_path = os.path.join(LOG_DIR, file_name)

logging.basicConfig(filename=log_file_path,
                    filemode="w",
                    format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO )