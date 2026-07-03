import logging
from models.arcface_model import ArcFaceModel
from models.deepface_model import DeepFaceModel

# NOTE:
# Avoid importing facenet_pytorch at module import time.
# It may not be installed; we import FaceNetModel only if selected.

from utils.gpu_utils import get_device

logger = logging.getLogger("ModelLoader")

def load_model(config):
    """Factory function to load the specified model."""
    model_name = config.get("models", {}).get("selected_model", "facenet").lower()
    device_type = config.get("models", {}).get("device", "auto")
    
    device = get_device(device_type)
    
    if model_name == "facenet":
        # Import only when needed, so the project can run without facenet_pytorch installed.
        from models.facenet_model import FaceNetModel
        return FaceNetModel(device=device)
    elif model_name == "arcface":
        return ArcFaceModel(device=device)
    elif model_name == "deepface":
        return DeepFaceModel(device=device)
    else:
        logger.error(
            f"Unsupported model selected: {model_name}. Defaulting to ArcFace for robustness."
        )
        return ArcFaceModel(device=device)

