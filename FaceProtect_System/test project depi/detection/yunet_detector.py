import cv2
import numpy as np
import os
import requests
from detection.base_detector import BaseDetector
from utils.logging_utils import logger

class YuNetDetector(BaseDetector):
    def __init__(self, model_path="face_detection_yunet_2023mar.onnx", device=None, conf_threshold=0.6, nms_threshold=0.3):
        self.model_path = model_path
        self._ensure_model_exists()
        
        # Initialize detector
        self.detector = cv2.FaceDetectorYN.create(
            model=self.model_path,
            config="",
            input_size=(320, 320),
            score_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            top_k=5000,
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )
        logger.info(f"YuNet Detector initialized from {model_path}")

    def _ensure_model_exists(self):
        if not os.path.exists(self.model_path):
            logger.info(f"Downloading YuNet model to {self.model_path}...")
            url = "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
            try:
                response = requests.get(url)
                with open(self.model_path, 'wb') as f:
                    f.write(response.content)
                logger.info("YuNet model downloaded successfully.")
            except Exception as e:
                logger.error(f"Failed to download YuNet model: {e}")
                raise

    def detect_faces(self, image: np.ndarray):
        h, w, _ = image.shape
        self.detector.setInputSize((w, h))
        
        _, faces = self.detector.detect(image)
        
        boxes = []
        landmarks = []
        scores = []
        
        if faces is not None:
            for face in faces:
                # face format: [x1, y1, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
                # x1, y1, w, h are at indices 0, 1, 2, 3
                # landmarks are at indices 4 to 13
                # score is at index 14
                
                box = [face[0], face[1], face[0] + face[2], face[1] + face[3]]
                boxes.append(box)
                
                # YuNet landmarks: [right_eye, left_eye, nose, right_mouth_corner, left_mouth_corner]
                # We need to reorder to match MTCNN: [left_eye, right_eye, nose, mouth_left, mouth_right]
                # YuNet indices: 
                # 4,5: right eye
                # 6,7: left eye
                # 8,9: nose
                # 10,11: right mouth
                # 12,13: left mouth
                
                l_eye = [face[6], face[7]]
                r_eye = [face[4], face[5]]
                nose = [face[8], face[9]]
                l_mouth = [face[12], face[13]]
                r_mouth = [face[10], face[11]]
                
                landmarks.append([l_eye, r_eye, nose, l_mouth, r_mouth])
                scores.append(face[14])
                
        return boxes, landmarks, scores
