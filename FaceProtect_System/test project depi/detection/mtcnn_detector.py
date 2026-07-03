import torch
import numpy as np
import cv2
from facenet_pytorch import MTCNN
from detection.base_detector import BaseDetector
from utils.logging_utils import logger

class MTCNNDetector(BaseDetector):
    def __init__(self, device=None, min_face_size=20, thresholds=[0.6, 0.7, 0.7]):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(
            keep_all=True, 
            device=self.device, 
            min_face_size=min_face_size,
            thresholds=thresholds
        )
        logger.info(f"MTCNN Detector initialized on {self.device}")

    def detect_faces(self, image: np.ndarray):
        # MTCNN expect RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # detect faces
        boxes, scores, landmarks = self.detector.detect(img_rgb, landmarks=True)
        
        if boxes is None:
            return [], [], []
            
        return boxes.tolist(), landmarks.tolist(), scores.tolist()
