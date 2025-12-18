import torch

from .logging_config import logger


def sanity_check() -> None:
    """
    Log diagnostic information about PyTorch, CUDA availability, and GPUs.

    This function writes information to the configured logger, including:
    - Whether CUDA is available and its version
    - Number of detected GPUs and their names
    """
    # Check PyTorch version

    # Check if CUDA is available
    if torch.cuda.is_available():
        logger.info("CUDA is available.")
        logger.info(f"CUDA version: {torch.version.cuda}")

        # Check if GPU is detected
        logger.info(f"Number of GPUs detected: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.info("CUDA is not available.")
