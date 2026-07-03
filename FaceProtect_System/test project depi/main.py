import logging
from utils.config_loader import load_config
from utils.logger import get_logger
import extract_embeddings
import evaluate

def main():
    config = load_config("config.yaml")
    logger = get_logger("Main", log_level=config['logging']['log_level'], log_dir=config['logging']['log_dir'])
    
    logger.info("="*50)
    logger.info("FaceProtect: Phase 3 Pipeline (Recognition & Embedding)")
    logger.info("="*50)
    
    try:
        extract_embeddings.main()
    except Exception as e:
        logger.error(f"Failed during embedding extraction: {e}")
        return
        
    try:
        evaluate.main()
    except Exception as e:
        logger.error(f"Failed during evaluation: {e}")
        return
        
    logger.info("="*50)
    logger.info("Pipeline Completed Successfully.")
    logger.info("="*50)

if __name__ == "__main__":
    main()
