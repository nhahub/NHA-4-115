import cv2
import numpy as np
from utils.logging_utils import logger

class FaceAligner:
    def __init__(self, desired_face_width=224, desired_face_height=224, 
                 left_eye_pos=(0.35, 0.35)):
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height
        self.left_eye_pos = left_eye_pos
        logger.info(f"Face Aligner initialized with output size {desired_face_width}x{desired_face_height}")

    def align(self, image, landmarks):
        """
        Align face using eye coordinates.
        
        Args:
            image: Original BGR image
            landmarks: Facial landmarks (at least 2 for eyes)
                       Assuming landmarks format: [left_eye, right_eye, nose, mouth_left, mouth_right]
                       Matches MTCNN output.
        """
        if landmarks is None or len(landmarks) < 2:
            return None
            
        # Extract eye coordinates
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        # Compute the angle between the eye centroids
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Compute the desired right eye x-coordinate based on the desired left eye x-coordinate
        right_eye_x = 1.0 - self.left_eye_pos[0]
        
        # Determine the scale of the new image by taking the ratio of the distance 
        # between eyes in the current image to the ratio of distance between eyes in the desired image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desired_dist = (right_eye_x - self.left_eye_pos[0])
        desired_dist *= self.desired_face_width
        scale = desired_dist / dist
        
        # Compute center (between eyes) in the current image
        eyes_center = (float((left_eye[0] + right_eye[0]) // 2),
                       float((left_eye[1] + right_eye[1]) // 2))
        
        # Grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        
        # Update the translation component of the matrix
        tX = self.desired_face_width * 0.5
        tY = self.desired_face_height * self.left_eye_pos[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])
        
        # Apply the affine transformation
        (w, h) = (self.desired_face_width, self.desired_face_height)
        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        
        return output
