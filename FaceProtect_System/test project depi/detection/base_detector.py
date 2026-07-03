from abc import ABC, abstractmethod
import numpy as np

class BaseDetector(ABC):
    """
    Abstract Base Class for Face Detectors.
    All detection models should implement this interface.
    """
    
    @abstractmethod
    def detect_faces(self, image: np.ndarray):
        """
        Detect faces in an image.
        
        Args:
            image: np.ndarray (BGR format)
            
        Returns:
            boxes: list of bounding boxes [x1, y1, x2, y2]
            landmarks: list of facial landmarks (e.g., eyes, nose, mouth)
            scores: list of confidence scores
        """
        pass
