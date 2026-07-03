from deepface import DeepFace
import logging

logger = logging.getLogger("DeepFaceModel")

class DeepFaceModel:
    def __init__(self, device):
        self.device = device
        self.model_name = "VGG-Face"
        self.loaded = False

    def load(self):
        if not self.loaded:
            logger.info(f"Loading {self.model_name} model via DeepFace...")
            self.loaded = True
            logger.info(f"{self.model_name} model ready.")

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
