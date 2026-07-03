FROM python:3.11-slim
WORKDIR /app
COPY api_requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r api_requirements.txt
RUN apt-get update && apt-get install -y libxcb1 libgomp1 libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]