import cv2
import os
import time
import numpy as np
from utils.logging_utils import logger
from detection.factory import get_detector
from alignment.face_aligner import FaceAligner

class FacePipeline:
    def __init__(self, detector_type="mtcnn", device=None, output_size=(224, 224)):
        self.detector = get_detector(detector_type, device=device)
        self.aligner = FaceAligner(desired_face_width=output_size[0], 
                                    desired_face_height=output_size[1])
        self.output_size = output_size
        logger.info(f"Face Pipeline initialized with {detector_type} detector")

    def process_image(self, image_path, save_path=None):
        """
        Process a single image: Detect -> Align -> Crop
        """
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read image at {image_path}")
            return []

        start_time = time.time()
        boxes, landmarks, scores = self.detector.detect_faces(image)
        detection_time = time.time() - start_time
        
        logger.info(f"Detected {len(boxes)} faces in {detection_time:.4f}s")
        
        processed_faces = []
        for i, (box, landmark) in enumerate(zip(boxes, landmarks)):
            aligned_face = self.aligner.align(image, landmark)
            if aligned_face is not None:
                processed_faces.append(aligned_face)
                
                if save_path:
                    base, ext = os.path.splitext(save_path)
                    current_save_path = f"{base}_{i}{ext}"
                    cv2.imwrite(current_save_path, aligned_face)
                    logger.debug(f"Saved aligned face to {current_save_path}")
            else:
                # If alignment fails, fallback to simple crop if box exists
                x1, y1, x2, y2 = map(int, box)
                face_crop = image[max(0, y1):min(image.shape[0], y2), 
                                  max(0, x1):min(image.shape[1], x2)]
                if face_crop.size > 0:
                    face_crop = cv2.resize(face_crop, self.output_size)
                    processed_faces.append(face_crop)
                    logger.warning(f"Alignment failed for face {i}, used simple crop fallback.")

        return processed_faces

    def visualize(self, image_path, processed_faces):
        """
        Display before and after images (Requires UI/Windowing environment)
        In some environments, this might just save a composite image.
        """
        original = cv2.imread(image_path)
        if original is None:
            return
            
        # Draw boxes on original for visualization
        # (This would be done inside process_image if we wanted to show them here)
        
        # For now, let's just return if we can't show windows
        # But we can create a grid and save it.
        if not processed_faces:
            logger.info("No faces to visualize")
            return

        # Create a collage
        h, w = original.shape[:2]
        canvas_h = max(h, self.output_size[1])
        canvas_w = w + (len(processed_faces) * self.output_size[0])
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        
        canvas[:h, :w] = original
        for i, face in enumerate(processed_faces):
            x_offset = w + (i * self.output_size[0])
            canvas[:self.output_size[1], x_offset:x_offset+self.output_size[0]] = face
            
        return canvas
