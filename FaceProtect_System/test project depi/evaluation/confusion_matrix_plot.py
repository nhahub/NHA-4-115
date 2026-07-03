import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import logging

logger = logging.getLogger("ConfusionMatrixPlot")

def plot_confusion_matrix(labels, preds, output_dir):
    """Generates and saves the Confusion Matrix."""
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Different", "Same"])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title('Face Verification Confusion Matrix')
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved Confusion Matrix to {out_path}")
