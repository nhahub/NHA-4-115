import pandas as pd
import os
from PIL import Image
import io

# Path to the parquet file
parquet_path = r"C:\Users\m7md\Desktop\data\train-00000-of-00001.parquet"
output_dir = r"C:\Users\m7md\Desktop\data\lfw_extracted"

os.makedirs(output_dir, exist_ok=True)

print(f"Reading parquet file: {parquet_path}")
df = pd.read_parquet(parquet_path)

print(f"Columns in parquet: {df.columns.tolist()}")

# LFW filenames are usually like 'Name_Surname_0001.jpg'
def get_label_from_filename(filename):
    # Remove extension and the last part (number)
    base = os.path.basename(filename)
    name_parts = base.split('_')
    if len(name_parts) > 1:
        # Check if the last part is a number (common in LFW)
        if name_parts[-1].split('.')[0].isdigit():
            return '_'.join(name_parts[:-1])
    return base.split('.')[0]

for index, row in df.iterrows():
    try:
        filename = row['filename']
        image_data = row['image']
        
        label = get_label_from_filename(filename)
        
        person_dir = os.path.join(output_dir, label)
        os.makedirs(person_dir, exist_ok=True)
        
        # Keep original filename
        image_path = os.path.join(person_dir, os.path.basename(filename))
        
        # Handle image data
        if isinstance(image_data, dict) and 'bytes' in image_data:
            with open(image_path, 'wb') as f:
                f.write(image_data['bytes'])
        elif isinstance(image_data, bytes):
            with open(image_path, 'wb') as f:
                f.write(image_data)
        elif hasattr(image_data, 'save'): # PIL Image
            image_data.save(image_path)
            
        if index % 1000 == 0:
            print(f"Extracted {index} images...")
            
    except Exception as e:
        print(f"Error extracting image at index {index}: {e}")

print(f"Extraction complete. Images saved to {output_dir}")
