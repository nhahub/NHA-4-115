import os
import glob
import logging

logger = logging.getLogger("FileUtils")

def get_image_paths(directory):
    """Recursively gets all common image paths from a directory."""
    if not os.path.exists(directory):
        logger.error(f"Directory not found: {directory}")
        return []
    
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = []
    
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
        image_paths.extend(glob.glob(os.path.join(directory, '**', ext.upper()), recursive=True))
        
    logger.info(f"Found {len(image_paths)} images in {directory}")
    return sorted(image_paths)

def get_identity_from_path(path):
    """Extracts identity (folder name) from the path."""
    return os.path.basename(os.path.dirname(path))
