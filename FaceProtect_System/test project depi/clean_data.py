import os
import hashlib
from PIL import Image

data_dir = '/home/ubuntu/face_recognition_project/raw_data/lfw_extracted'
corrupted_files = []
duplicates = {}
hashes = {}

print(f"Starting data cleaning in {data_dir}...")

for root, dirs, files in os.walk(data_dir):
    for file in files:
        file_path = os.path.join(root, file)
        
        # 1. Check for corrupted images
        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception as e:
            print(f"Corrupted file found: {file_path} - {e}")
            corrupted_files.append(file_path)
            continue
            
        # 2. Check for duplicates using MD5 hash
        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            if file_hash in hashes:
                if file_hash not in duplicates:
                    duplicates[file_hash] = [hashes[file_hash]]
                duplicates[file_hash].append(file_path)
            else:
                hashes[file_hash] = file_path
        except Exception as e:
            print(f"Error hashing file {file_path}: {e}")

# Report and Clean
print(f"\nCleaning Report:")
print(f"Total files checked: {len(hashes) + len(corrupted_files) + sum(len(v)-1 for v in duplicates.values())}")
print(f"Corrupted files: {len(corrupted_files)}")
print(f"Duplicate sets: {len(duplicates)}")

for f in corrupted_files:
    os.remove(f)
    print(f"Removed corrupted file: {f}")

for h, paths in duplicates.items():
    # Keep the first one, remove others
    for p in paths[1:]:
        os.remove(p)
        print(f"Removed duplicate file: {p}")

print("\nData cleaning complete.")
