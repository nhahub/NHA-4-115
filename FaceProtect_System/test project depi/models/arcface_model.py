from deepface import DeepFace
import logging

logger = logging.getLogger("ArcFaceModel")

class ArcFaceModel:
    def __init__(self, device):
        self.device = device
        self.model_name = "ArcFace"
        self.loaded = False

    def load(self):
        if not self.loaded:
            logger.info("Loading ArcFace model via DeepFace...")
            self.loaded = True
            logger.info("ArcFace model ready.")

    def extract(self, image_paths):
        self.load()
        
        embeddings = []
        valid_paths = []
        
        for path in image_paths:
            try:
                result = DeepFace.represent(img_path=path, model_name=self.model_name, enforce_detection=False)
                if len(result) > 0:
                    embeddings.append(result[0]["embedding"])
                    valid_paths.append(path)
            except Exception as e:
                logger.error(f"Error extracting embedding for {path}: {e}")
                
        return embeddings, valid_paths
