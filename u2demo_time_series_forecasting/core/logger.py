import logging

from u2demo_time_series_forecasting.constants.envs import settings

logger = logging.getLogger("u2")
logger.setLevel(settings.log_level)

_handler = logging.StreamHandler()
_handler.setLevel(settings.log_level)
_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(_handler)
