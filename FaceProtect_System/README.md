# 🛡️ FaceProtect System - DEPI Final Project

## 📌 Overview
FaceProtect is an end-to-end Face Detection and Recognition pipeline developed as the final project for the Digital Egypt Builders Initiative (DEPI) - MLOps Track. The system is fully containerized and integrates MLOps best practices for model tracking, vector search, and seamless deployment.

## 🛠️ Tech Stack & Architecture
* **Machine Learning & Computer Vision:** ArcFace, DeepFace, YuNet.
* **Vector Database:** FAISS (Facebook AI Similarity Search) for fast and scalable embedding retrieval.
* **Backend & API:** Python / RESTful API setup.
* **Frontend:** Streamlit.
* **MLOps & Tracking:** MLflow (Experiment tracking, metric logging, and model registry).
* **Deployment & Containerization:** Docker & Docker Compose.

## 🚀 How to Run the Project
The entire system architecture is containerized to ensure reproducibility across different environments. To run the project locally:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/nhahub/NHA-4-115.git](https://github.com/nhahub/NHA-4-115.git)
   cd NHA-4-115/FaceProtect_System
   ```

2. **Start the services using Docker Compose:**
   ```bash
   docker-compose up --build -d
   ```

3. **Access the Application Interfaces:**
   Once the containers are running, you can access:
   * **Frontend (Streamlit):** `http://localhost:8501` 
   * **Backend API:** `http://localhost:8000`
   * **MLflow UI (Tracking):** `http://localhost:5000`

## 📊 MLOps & Experiment Tracking
All model experiments, performance metrics (like confusion matrices and ROC curves), and hyperparameters are tracked using **MLflow**. The local SQLite database logs the runs to ensure full traceability of the machine learning lifecycle.

## ⚠️ Note on Large Files (Git LFS)
Due to GitHub's standard file size limits, large generated artifacts such as the raw embeddings file (`embeddings.npy` > 150MB) and the FAISS index (`faiss.index`) are explicitly ignored in this repository via `.gitignore`. These files are either generated automatically during the pipeline execution or can be provided separately upon request.
