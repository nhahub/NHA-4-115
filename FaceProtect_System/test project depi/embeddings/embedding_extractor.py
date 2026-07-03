import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from embeddings.embedding_utils import normalize_l2
from utils.file_utils import get_identity_from_path

logger = logging.getLogger("EmbeddingExtractor")

class EmbeddingExtractor:
    def __init__(self, model, config):
        self.model = model
        self.batch_size = config.get("models", {}).get("batch_size", 32)
        self.output_dir = config.get("embeddings", {}).get("output_dir", "outputs/embeddings")
        self.should_normalize = config.get("embeddings", {}).get("normalize", True)
        self.model_name = config.get("models", {}).get("selected_model", "unknown")
        
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_and_save(self, image_paths):
        """Extracts embeddings in batches and saves them to disk."""
        all_embeddings = []
        metadata = []
        
        logger.info(f"Starting extraction for {len(image_paths)} images with batch size {self.batch_size}")
        
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="Extracting"):
            batch_paths = image_paths[i:i+self.batch_size]
            batch_embeddings, valid_paths = self.model.extract(batch_paths)
            
            if len(batch_embeddings) == 0:
                continue
                
            all_embeddings.extend(batch_embeddings)
            
            for path in valid_paths:
                metadata.append({
                    "image_path": path,
                    "person_id": get_identity_from_path(path),
                    "model_name": self.model_name,
                    "embedding_dimension": len(batch_embeddings[0])
                })
                
        if not all_embeddings:
            logger.warning("No embeddings were extracted.")
            return None, None
            
        all_embeddings = np.array(all_embeddings)
        
        if self.should_normalize:
            logger.info("Normalizing embeddings...")
            all_embeddings = normalize_l2(all_embeddings)
            
        # Save to disk
        emb_path = os.path.join(self.output_dir, "embeddings.npy")
        meta_path = os.path.join(self.output_dir, "metadata.csv")
        
        np.save(emb_path, all_embeddings)
        pd.DataFrame(metadata).to_csv(meta_path, index=False)
        
        logger.info(f"Saved {len(all_embeddings)} embeddings to {emb_path}")
        logger.info(f"Saved metadata to {meta_path}")
        
        return all_embeddings, pd.DataFrame(metadata)
