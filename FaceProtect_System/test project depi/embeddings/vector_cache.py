import os
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("VectorCache")

class VectorCache:
    """Manages cached embeddings to avoid re-computing."""
    
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.emb_path = os.path.join(cache_dir, "embeddings.npy")
        self.meta_path = os.path.join(cache_dir, "metadata.csv")
        self.embeddings = None
        self.metadata = None

    def load(self):
        """Loads cached embeddings if they exist."""
        if os.path.exists(self.emb_path) and os.path.exists(self.meta_path):
            try:
                self.embeddings = np.load(self.emb_path)
                self.metadata = pd.read_csv(self.meta_path)
                logger.info(f"Loaded {len(self.embeddings)} embeddings from cache.")
                return True
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")
                return False
        return False
        
    def get_data(self):
        return self.embeddings, self.metadata
