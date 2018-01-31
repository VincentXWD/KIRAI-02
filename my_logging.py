import os
import sys
import logging


logger = logging.getLogger('padding_multi_control')
formatter = logging.Formatter('[Image padding] \
  %(levelname)-5s: %(asctime)s <PID>%(process)d %(message)s')

file_handler = logging.FileHandler('./log/padding_process.log')
file_handler.setFormatter(formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
