import cv2
import os
from pipeline.face_pipeline import FacePipeline
from utils.logging_utils import logger

def main():
    detector_type = "mtcnn" 
    
    pipeline = FacePipeline(detector_type=detector_type)

    
    test_image_dir = r"C:\Users\m7md\OneDrive\Desktop\test project\augmented_faces_v2"

    
    test_image = None
    for root, dirs, files in os.walk(test_image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_image = os.path.join(root, file)
                break
        if test_image:
            break
            
    if not test_image:
        logger.error("No test image found in augmented_faces directory.")
        return

    logger.info(f"Testing pipeline on image: {test_image}")
    
    # Process image
    save_dir = "test_results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "aligned_face.jpg")
    
    processed_faces = pipeline.process_image(test_image, save_path=save_path)
    
    if processed_faces:
        logger.info(f"Successfully processed {len(processed_faces)} faces.")
        # Create visualization
        canvas = pipeline.visualize(test_image, processed_faces)
        if canvas is not None:
            cv2.imwrite(os.path.join(save_dir, "visualization.jpg"), canvas)
            logger.info(f"Visualization saved to {os.path.join(save_dir, 'visualization.jpg')}")
    else:
        logger.warning("No faces detected or processed.")

if __name__ == "__main__":
    main()
