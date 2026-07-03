# Face Detection & Alignment Module

This module is part of the **FaceProtect** AI Privacy Guard system. It handles the critical first stages of the face recognition pipeline: **Detection** and **Alignment**.

## Features

- **Multi-Model Support**: Easily switch between different detection backends:
  - **MTCNN**: Robust baseline with built-in landmark detection.
  - **YOLOv8-Face**: Optimized for high-speed real-time processing.
  - **RetinaFace (YuNet)**: State-of-the-art accuracy and speed using OpenCV's optimized YuNet.
- **Robust Alignment**: Geometric transformation based on eye coordinates to normalize face rotation and scale.
- **Standardized Output**: Produces 224x224 (configurable) cropped faces ready for embedding generation.
- **Batch Processing**: Designed for efficiency in large-scale data pipelines.
- **Visualization**: Built-in tools to compare "Before" and "After" results.

## Project Structure

```text
detection/
├── base_detector.py    # Abstract base class
├── mtcnn_detector.py   # facenet-pytorch implementation
├── yolo_detector.py    # Ultralytics YOLOv8 implementation
├── yunet_detector.py   # OpenCV YuNet (RetinaFace-class)
└── factory.py          # Model factory
alignment/
└── face_aligner.py     # Geometric alignment logic
pipeline/
└── face_pipeline.py    # Orchestration module
utils/
└── logging_utils.py    # Logging configuration
test_pipeline.py        # Demo script
```

## Quick Start

1. **Initialize the Pipeline**:
   ```python
   from pipeline.face_pipeline import FacePipeline
   pipeline = FacePipeline(detector_type="mtcnn")
   ```

2. **Process an Image**:
   ```python
   faces = pipeline.process_image("path/to/image.jpg", save_path="output/face.jpg")
   ```

3. **Visualize Results**:
   ```python
   canvas = pipeline.visualize("path/to/image.jpg", faces)
   import cv2
   cv2.imwrite("visualization.jpg", canvas)
   ```

## Model Comparison

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| **MTCNN** | Medium | Medium | Baseline / Generic tasks |
| **YOLOv8** | High | Medium | Real-time / Video |
| **YuNet** | High | High | Production accuracy / High speed |

---
**Note**: The module automatically handles model weight downloads for YuNet and YOLOv8-Face if they are not present.
