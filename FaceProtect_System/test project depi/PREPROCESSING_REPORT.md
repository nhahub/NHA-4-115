# Face Recognition Dataset Preprocessing Report

## 1. Overview
The dataset has been successfully downloaded, cleaned, aligned, and augmented. It is now ready for the next phase of training and embedding extraction.

## 2. Dataset Statistics

| Metric | Aligned Dataset (Original) | Augmented Dataset |
| :--- | :--- | :--- |
| **Total Images** | 11842 | 51456 |
| **Total Identities (Classes)** | 5130 | 3936 |
| **Average Images per Identity** | 2.31 | 13.07 |
| **Max Images for one Identity** | 529 | 1404 |

## 3. Preprocessing Steps Completed
- **Data Collection**: Downloaded LFW dataset from Hugging Face.
- **Cleaning**: Removed corrupted files and duplicates (MD5 hashing).
- **Alignment**: Detected faces and aligned eyes horizontally using OpenCV Haar Cascades. All images resized to **224x224**.
- **Augmentation**: Generated 6 variants for each original image (Original, Flip, Bright, Dark, Rot+10, Rot-10).
- **Pipeline**: Created a PyTorch-compatible `FaceDataset` and `DataLoader` for seamless model training.

## 4. Directory Structure
- `/raw_data/`: Original downloaded files.
- `/processed_data/aligned_faces/`: Cleaned and aligned faces (one folder per person).
- `/processed_data/augmented_faces/`: Fully augmented dataset (ready for training).
- `data_pipeline.py`: Python script to load the data into a model.

## 5. Next Steps
The next team member can use `data_pipeline.py` to feed this data into a Face Recognition model (e.g., FaceNet or ArcFace) to extract embeddings.
