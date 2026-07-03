import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
import logging

logger = logging.getLogger("FaceNetModel")

class FaceNetModel:
    def __init__(self, device):
        self.device = device
        self.model = None
        
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def load(self):
        if self.model is None:
            logger.info(f"Loading FaceNet model onto {self.device}...")
            self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            logger.info("FaceNet loaded successfully.")

    def extract(self, image_paths):
        self.load()
        
        images = []
        valid_paths = []
        for path in image_paths:
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = self.transform(img)
                images.append(img_tensor)
                valid_paths.append(path)
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
                
        if not images:
            return [], []
            
        batch_tensor = torch.stack(images).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model(batch_tensor).cpu().numpy()
            
        return embeddings, valid_paths
