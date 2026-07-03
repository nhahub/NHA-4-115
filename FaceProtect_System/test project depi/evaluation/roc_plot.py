import matplotlib.pyplot as plt
import os
import logging

logger = logging.getLogger("ROCPlot")

def plot_roc_curve(fpr, tpr, roc_auc, output_dir):
    """Generates and saves the ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (1 - FRR)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved ROC curve to {out_path}")
