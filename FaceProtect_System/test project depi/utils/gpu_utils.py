import torch
import logging

logger = logging.getLogger("GPUUtils")

def get_device(requested_device="auto"):
    """Returns the best available device."""
    if requested_device == "cuda" and torch.cuda.is_available():
        logger.info("Using requested CUDA device.")
        return torch.device("cuda")
    elif requested_device == "cpu":
        logger.info("Using requested CPU device.")
        return torch.device("cpu")
    else:
        if torch.cuda.is_available():
            logger.info("Auto-selected CUDA device.")
            return torch.device("cuda")
        else:
            logger.info("CUDA not available. Auto-selected CPU device.")
            return torch.device("cpu")
