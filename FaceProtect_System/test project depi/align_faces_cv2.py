import cv2
import os
import numpy as np

# Load pre-trained Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

input_dir = r"C:\Users\m7md\OneDrive\Desktop\test project\augmented_faces"
output_dir = r"C:\Users\m7md\OneDrive\Desktop\test project\aligned_faces"
os.makedirs(output_dir, exist_ok=True)

def align_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    if len(faces) == 0:
        return None
        
    # Take the largest face
    (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    roi_gray = gray[y:y+h, x:x+w]
    
    eyes = eye_cascade.detectMultiScale(roi_gray)
    
    if len(eyes) >= 2:
        # Sort eyes by x-coordinate
        eyes = sorted(eyes, key=lambda e: e[0])
        # Eye centers relative to original image
        left_eye_center = (float(x + eyes[0][0] + eyes[0][2]/2), float(y + eyes[0][1] + eyes[0][3]/2))
        right_eye_center = (float(x + eyes[1][0] + eyes[1][2]/2), float(y + eyes[1][1] + eyes[1][3]/2))
        
        # Calculate angle
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Rotate
        center = (float(x + w/2), float(y + h/2))
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
        
        # Re-detect face on rotated image
        rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        new_faces = face_cascade.detectMultiScale(rotated_gray, 1.1, 5)
        if len(new_faces) > 0:
            (nx, ny, nw, nh) = sorted(new_faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            # Add padding
            pad = int(nw * 0.15)
            nx1 = max(0, nx - pad)
            ny1 = max(0, ny - pad)
            nx2 = min(rotated.shape[1], nx + nw + pad)
            ny2 = min(rotated.shape[0], ny + nh + pad)
            face_crop = rotated[ny1:ny2, nx1:nx2]
            return cv2.resize(face_crop, (224, 224))
            
    # If eye detection fails, just return the cropped face
    pad = int(w * 0.1)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(image.shape[1], x + w + pad)
    y2 = min(image.shape[0], y + h + pad)
    face_crop = image[y1:y2, x1:x2]
    return cv2.resize(face_crop, (224, 224))

print("Starting face detection and alignment (OpenCV)...")

count = 0
failed = 0
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        file_path = os.path.join(root, file)
        person_name = os.path.basename(root)
        
        image = cv2.imread(file_path)
        if image is None:
            continue
            
        aligned = align_face(image)
        if aligned is not None:
            person_out_dir = os.path.join(output_dir, person_name)
            os.makedirs(person_out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(person_out_dir, file), aligned)
            count += 1
        else:
            failed += 1
            
        if (count + failed) % 1000 == 0:
            print(f"Processed {count + failed} images... (Aligned: {count}, Failed: {failed})")

print(f"Alignment complete. Total aligned: {count}, Failed: {failed}")
