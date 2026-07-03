import sys
import logging
from utils.config_loader import load_config
from utils.logger import get_logger
from utils.file_utils import get_image_paths
from models.model_loader import load_model
from embeddings.embedding_extractor import EmbeddingExtractor

def main():
    config = load_config("config.yaml")
    logger = get_logger("ExtractEmbeddings", log_level=config['logging']['log_level'], log_dir=config['logging']['log_dir'])
    
    logger.info("--- Starting Embedding Extraction ---")
    
    data_dir = config['pipeline']['input_data_dir']
    image_paths = get_image_paths(data_dir)
    
    if not image_paths:
        logger.error(f"No images found in {data_dir}. Exiting.")
        sys.exit(1)
        
    model = load_model(config)
    extractor = EmbeddingExtractor(model, config)
    
    extractor.extract_and_save(image_paths)
    
    logger.info("--- Embedding Extraction Completed ---")

if __name__ == "__main__":
    main()
