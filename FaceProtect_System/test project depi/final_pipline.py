import cv2
import os
from pipeline.face_pipeline import FacePipeline
from utils.logging_utils import logger

def main():
    detector_type = "mtcnn" 
    
    pipeline = FacePipeline(detector_type=detector_type)

    input_dir = r"C:\Users\m7md\OneDrive\Desktop\test project depi\augmented_faces"
    output_dir = r"C:\Users\m7md\OneDrive\Desktop\test project depi\final_results"

    os.makedirs(output_dir, exist_ok=True)

    count = 0

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)

                save_path = os.path.join(output_dir, file)

                processed_faces = pipeline.process_image(image_path, save_path=save_path)

                count += 1
                if count % 500 == 0:
                    logger.info(f"Processed {count} images...")

    logger.info(f"DONE ✅ Processed {count} images")

if __name__ == "__main__":
    main()