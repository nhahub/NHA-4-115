import torch
import numpy as np
import cv2
from detection.base_detector import BaseDetector
from utils.logging_utils import logger

class YOLOFaceDetector(BaseDetector):
    def __init__(self, model_path="yolov8n-face.pt", device=None, conf=0.5):
        from ultralytics import YOLO
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        # Note: If the model file doesn't exist, ultralytics will try to download it.
        # However, 'yolov8n-face.pt' might not be a default ultralytics model name.
        # Usually it's 'yolov8n.pt' for general detection or a custom face model.
        # For the sake of this implementation, we assume a compatible YOLO face model.
        # Alternatively, we use 'yolov8n-face' if available or provide instructions.
        self.model = YOLO(model_path)
        self.conf = conf
        logger.info(f"YOLOv8-Face Detector initialized on {self.device} with model {model_path}")

    def detect_faces(self, image: np.ndarray):
        results = self.model(image, device=self.device, conf=self.conf, verbose=False)
        
        boxes = []
        landmarks = []
        scores = []
        
        for result in results:
            if result.boxes is not None:
                boxes.extend(result.boxes.xyxy.cpu().numpy().tolist())
                scores.extend(result.boxes.conf.cpu().numpy().tolist())
                
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    # Keypoints are usually [left_eye, right_eye, nose, mouth_left, mouth_right]
                    # We need to ensure the format matches MTCNN for the aligner
                    kpts = result.keypoints.xy.cpu().numpy()
                    for kpt in kpts:
                        # Reshape or select the first 5 keypoints if more are present
                        landmarks.append(kpt[:5].tolist())
                else:
                    # If no landmarks, append None
                    for _ in range(len(result.boxes)):
                        landmarks.append(None)
                        
        return boxes, landmarks, scores
