import cv2
import os
import numpy as np
import random

input_dir = r"C:\Users\m7md\OneDrive\Desktop\test project\aligned_faces"
output_dir = r"C:\Users\m7md\OneDrive\Desktop\test project\augmented_faces_v2"
os.makedirs(output_dir, exist_ok=True)

def augment_image(image):
    augs = []
    # 1. Original
    augs.append(('orig', image))
    
    # 2. Horizontal Flip
    augs.append(('flip', cv2.flip(image, 1)))
    
    # 3. Brightness adjustment
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Brighten
    v_bright = cv2.add(v, 30)
    bright = cv2.merge((h, s, v_bright))
    augs.append(('bright', cv2.cvtColor(bright, cv2.COLOR_HSV2BGR)))
    
    # Darken
    v_dark = cv2.subtract(v, 30)
    dark = cv2.merge((h, s, v_dark))
    augs.append(('dark', cv2.cvtColor(dark, cv2.COLOR_HSV2BGR)))
    
    # 4. Slight Rotation
    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, 10, 1.0)
    augs.append(('rot_p10', cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))))
    
    M = cv2.getRotationMatrix2D(center, -10, 1.0)
    augs.append(('rot_m10', cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))))
    
    return augs

print("Starting data augmentation...")
count = 0
for root, dirs, files in os.walk(input_dir):
    for file in files:
        file_path = os.path.join(root, file)
        person_name = os.path.basename(root)
        
        image = cv2.imread(file_path)
        if image is None:
            continue
            
        person_out_dir = os.path.join(output_dir, person_name)
        os.makedirs(person_out_dir, exist_ok=True)
        
        augmented_images = augment_image(image)
        for suffix, aug_img in augmented_images:
            name, ext = os.path.splitext(file)
            cv2.imwrite(os.path.join(person_out_dir, f"{name}_{suffix}{ext}"), aug_img)
        
        count += 1
        if count % 500 == 0:
            print(f"Augmented {count} original images...")

print(f"Augmentation complete. Total original images processed: {count}")
