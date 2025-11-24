import logging
import os

# Create a logger
logger = logging.getLogger("U2 Time Series Prediction")
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Create a console handler and set its level to INFO
ch = logging.StreamHandler()
ch.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# Create a formatter and set it for the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)
