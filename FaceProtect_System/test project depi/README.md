# FaceProtect: B2B AI Privacy Guard & Smart Recommender System
## Stage 3: Face Recognition & Embedding

This is a production-level Face Recognition & Embedding pipeline built as Phase 3 of the FaceProtect project. It takes aligned face images, extracts high-dimensional feature embeddings, compares them, and mathematically calculates the optimal similarity threshold.

### Features
* **Modular Architecture**: Supports FaceNet, ArcFace, and DeepFace.
* **Auto-Optimization**: Iterates through distance thresholds to find the Best F1-Score, while computing FAR (False Acceptance Rate) and FRR (False Rejection Rate).
* **High Performance**: Batch processing and vector caching.
* **Rich Evaluation**: Auto-generates ROC curves and Confusion Matrices.

### Folder Structure
```
├── embeddings/        # Extraction logic and caching
├── evaluation/        # Metrics, threshold optimization, and plots
├── models/            # Model wrappers (FaceNet, ArcFace, DeepFace)
├── outputs/           # Output directory for embeddings, reports, logs
├── utils/             # Helper utilities (Config, Logger, GPU)
├── config.yaml        # Main configuration file
├── evaluate.py        # Standalone evaluation script
├── extract_embeddings.py # Standalone extraction script
├── main.py            # Orchestrator
└── requirements.txt   # Python dependencies
```

### Setup & Installation

1. Make sure Python 3.8+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Update `config.yaml` to ensure `input_data_dir` points to your aligned faces directory.

### Running the Project

To run the full pipeline automatically (Extraction -> Evaluation):
```bash
python main.py
```

### Evaluation Output
After running, check the `outputs/` folder:
- `outputs/embeddings/embeddings.npy`: Numerical vectors.
- `outputs/embeddings/metadata.csv`: Maps rows in `embeddings.npy` to paths and identities.
- `outputs/reports/evaluation_report.json`: Contains the optimal threshold, accuracy, precision, recall, F1, FAR, and FRR.
- `outputs/plots/roc_curve.png`: ROC curve visualization.
- `outputs/plots/confusion_matrix.png`: Confusion Matrix at the optimal threshold.
