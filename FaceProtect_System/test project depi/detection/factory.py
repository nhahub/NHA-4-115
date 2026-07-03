from detection.mtcnn_detector import MTCNNDetector
from utils.logging_utils import logger

def get_detector(model_name="mtcnn", device=None, **kwargs):
    """
    Factory method to get a face detector instance.
    """
    model_name = model_name.lower()
    
    if model_name == "mtcnn":
        return MTCNNDetector(device=device, **kwargs)
    elif model_name == "yolov8":
        try:
            from detection.yolo_detector import YOLOFaceDetector
            return YOLOFaceDetector(device=device, **kwargs)
        except ImportError:
            logger.error("YOLOv8-Face dependencies not satisfied.")
            raise
    elif model_name == "retinaface" or model_name == "high_accuracy":
        try:
            # We use OpenCV's YuNet as a high-accuracy, high-speed alternative 
            # if RetinaFace (which often requires complex setup) is not available.
            from detection.yunet_detector import YuNetDetector
            return YuNetDetector(device=device, **kwargs)
        except Exception as e:
            logger.error(f"High accuracy detector initialization failed: {e}")
            raise
    else:
        raise ValueError(f"Unknown model name: {model_name}")
