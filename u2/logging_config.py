import logging

from u2.defaults import LOG_LEVEL

# Create a logger
logger = logging.getLogger("U2 Time Series Prediction")
logger.setLevel(LOG_LEVEL)

# Create a console handler and set its level to INFO
ch = logging.StreamHandler()
ch.setLevel(LOG_LEVEL)

# Create a formatter and set it for the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)
