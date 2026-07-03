import yaml
import os
import sys
import logging

try:
    from utils.logger import get_logger
    logger = get_logger("ConfigLoader")
except ImportError:
    logger = logging.getLogger("ConfigLoader")

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
        
    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            sys.exit(1)
