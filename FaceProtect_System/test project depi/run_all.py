import os
from pipeline.face_pipeline import FacePipeline

input_dir = r"C:\Users\m7md\OneDrive\Desktop\test_project_depi\augmented_faces"
output_dir = r"C:\Users\m7md\OneDrive\Desktop\test_project_depi\final_faces"

pipeline = FacePipeline(detector_type="mtcnn")

for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            in_path = os.path.join(root, file)
            
            rel_path = os.path.relpath(in_path, input_dir)
            out_path = os.path.join(output_dir, rel_path)
            
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            pipeline.process_image(in_path, save_path=out_path)

print("DONE ✅")