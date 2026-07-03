import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import logging

logger = logging.getLogger("ThresholdOptimizer")

class ThresholdOptimizer:
    def __init__(self, distances, labels):
        self.distances = np.array(distances)
        self.labels = np.array(labels)
        
    def optimize(self):
        """Finds the optimal threshold that maximizes F1 score and computes FAR/FRR."""
        fpr, tpr, thresholds = roc_curve(self.labels, -self.distances) # ROC expects scores, so we negate distance
        roc_auc = auc(fpr, tpr)
        
        best_f1 = 0
        best_thresh = 0
        best_metrics = {}
        
        min_dist = np.min(self.distances)
        max_dist = np.max(self.distances)
        test_thresholds = np.linspace(min_dist, max_dist, 100)
        
        for t in test_thresholds:
            preds = (self.distances <= t).astype(int)
            
            acc = accuracy_score(self.labels, preds)
            prec = precision_score(self.labels, preds, zero_division=0)
            rec = recall_score(self.labels, preds, zero_division=0)
            f1 = f1_score(self.labels, preds, zero_division=0)
            
            fp = np.sum((preds == 1) & (self.labels == 0))
            tn = np.sum((preds == 0) & (self.labels == 0))
            fn = np.sum((preds == 0) & (self.labels == 1))
            tp = np.sum((preds == 1) & (self.labels == 1))
            
            far = fp / (fp + tn) if (fp + tn) > 0 else 0
            frr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
                best_metrics = {
                    "threshold": float(t),
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1_score": float(f1),
                    "far": float(far),
                    "frr": float(frr),
                    "auc": float(roc_auc)
                }
                
        logger.info(f"Optimal Threshold: {best_thresh:.4f} (F1: {best_f1:.4f})")
        return best_metrics, fpr, tpr, roc_auc
