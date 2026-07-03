import logging
import os
import sys

def get_logger(name="FaceProtect", log_level="INFO", log_dir="outputs/logs"):
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        fh = logging.FileHandler(os.path.join(log_dir, f"{name.lower()}.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
