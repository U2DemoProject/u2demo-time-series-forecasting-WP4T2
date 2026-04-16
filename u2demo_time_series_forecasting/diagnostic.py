import torch
import torch.version  # required for pyright so `torch.version.cuda` is statically known

from u2demo_time_series_forecasting.core.logger import logger


def sanity_check() -> None:
    """
    Log diagnostic information about PyTorch, CUDA availability, and GPUs.

    This function writes information to the configured logger, including:
    - PyTorch version
    - Whether CUDA is available and its version
    - Number of detected GPUs and their names
    """
    logger.info(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        logger.info("CUDA is available.")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs detected: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("CUDA is not available.")
